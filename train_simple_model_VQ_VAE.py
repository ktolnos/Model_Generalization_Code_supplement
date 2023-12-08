import argparse
import json
import pickle as pkl
from collections import defaultdict
from functools import partial
from types import SimpleNamespace

import haiku as hk
import jax as jx
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.lax import stop_gradient as SG
from tqdm import tqdm

import environments
from operator import getitem
from optimizers import adamw
from tree_utils import tree_stack, tree_unstack
from vq_vae import Encoder, next_obs_decoder, termination_decoder, reward_decoder, Q_function,\
    obs_mi_decoder, action_mi_decoder, embedding_prior

from jax import config
# config.update("jax_disable_jit", True)

activation_dict = {"relu": jx.nn.relu, "silu": jx.nn.silu, "elu": jx.nn.elu}

parser = argparse.ArgumentParser()
parser.add_argument("--seed", "-s", type=int, default=10)
parser.add_argument("--output", "-o", type=str, default="vqvae")
parser.add_argument("--config", "-c", type=str)
args = parser.parse_args()
key = jx.random.PRNGKey(args.seed)

with open(args.config, 'r') as f:
    config = json.load(f)
    config.update({"agent_type": "simple_model", "seed": args.seed})


def set_default(d, k, v):
    if k not in d:
        d[k] = v


set_default(config, "double_DQN", False)
set_default(config, "episodic_env", False)
set_default(config, "updates_per_step", 1)
set_default(config, "save_params", True)
config = SimpleNamespace(**config)

Environment = getattr(environments, config.environment)

env_config = config.env_config

min_denom = 0.000001


########################################################################
# Probability Helper Functions
########################################################################

def log_gaussian_probability(x, params):
    mu = params['mu']
    sigma = params['sigma']
    return -(jnp.log(sigma) + 0.5 * jnp.log(2 * jnp.pi) + 0.5 * ((x - mu) / sigma) ** 2)


def log_binary_probability(x, params):
    logit = params['logit']
    return jnp.where(x, jx.nn.log_sigmoid(logit), jx.nn.log_sigmoid(-logit))


def is_binary_correct(x, params):
    logit = params['logit']
    return jnp.array_equal(jnp.greater(logit, 0), x)


def cross_entropy_loss(logits, labels):
    logits = jx.nn.log_softmax(logits['logits'])
    loss = vmap(getitem)(logits, labels)
    loss = -loss.mean()
    return loss

########################################################################
# Losses
########################################################################

def get_single_batch_model_loss(model_functions, mi_functions, binary_obs):

    mi_batch_loss = get_single_batch_mi_loss(mi_functions, model_functions, binary_obs)
    def single_batch_model_loss(model_params, model_states, mi_params, curr_obs, action, reward, next_obs, terminal, metrics, key):
        encoder_network = model_functions['encoder']
        reward_network = model_functions['reward']
        termination_network = model_functions['termination']
        next_obs_network = model_functions['next_obs']

        encoder_params = model_params['encoder']
        encoder_state = model_states['encoder']
        reward_params = model_params['reward']
        termination_params = model_params['termination']
        next_obs_params = model_params['next_obs']

        key, subkey = jx.random.split(key)
        quantization, encoder_state = encoder_network(
            encoder_params, encoder_state, curr_obs, jnp.eye(num_actions)[action], next_obs, terminal, reward, True)  # is_training=True
        model_states['prior'] = model_states['prior'].at[quantization['encoding_indices']].set(
            model_states['prior'][quantization['encoding_indices']] + 1)  # categorical logits
        model_states['encoder'] = encoder_state
        key, subkey = jx.random.split(key)
        next_obs_sample, o_hat_dist = next_obs_network(next_obs_params, curr_obs, jnp.eye(num_actions)[action],
                                                       quantization['quantize'], subkey)

        if binary_obs:
            o_hat_log_probs = jnp.sum(log_binary_probability(next_obs, o_hat_dist), axis=-1)
        else:
            o_hat_log_probs = jnp.sum(log_gaussian_probability(next_obs, o_hat_dist), axis=-1)

        # no need to reconstruct state on terminal steps, just need to get reward and terminal right
        obs_prediction_loss = jnp.where(terminal, 0.0, -o_hat_log_probs)

        r_dist = reward_network(reward_params, curr_obs, jnp.eye(num_actions)[action], quantization['quantize'])
        reward_loss = -log_gaussian_probability(reward, r_dist)

        gamma_dist = termination_network(termination_params, curr_obs, jnp.eye(num_actions)[action],
                                         quantization['quantize'])
        termination_loss = -log_binary_probability(jnp.logical_not(terminal), gamma_dist)

        if config.mi_loss_weight != 0:
            # we want encoder to maximize this loss. More correct way would be to minimize MI directly, see IRENE
            mi_loss, metrics = mi_batch_loss(mi_params, model_params, model_states, curr_obs, action, reward,
                                             next_obs, terminal, metrics, key)
            mi_loss = -mi_loss
        else:
            mi_loss = 0

        loss = (config.reward_weight * reward_loss +
                config.termination_weight * termination_loss +
                config.obs_prediction_weight * obs_prediction_loss +
                quantization['loss'] +
                config.mi_loss_weight * mi_loss)

        num_terminal = jnp.sum(terminal)
        terminal_correct_count = jnp.sum(jnp.where(terminal, jnp.less(gamma_dist['logit'], 0), 0))
        num_non_terminal = jnp.sum(jnp.logical_not(terminal))
        non_terminal_correct_count = jnp.sum(jnp.where(jnp.logical_not(terminal), jnp.greater(gamma_dist['logit'], 0), 0))
        next_obs_accuracy = jnp.sum(jnp.all(next_obs_sample == next_obs, axis=-1)) / next_obs.shape[0]
        metrics['next_obs_loss'] += obs_prediction_loss
        metrics['termination_loss'] += termination_loss
        metrics['reward_loss'] += reward_loss
        metrics['terminal_correct_count'] += terminal_correct_count
        metrics['terminal'] += num_terminal
        metrics['non_terminal'] += num_non_terminal
        metrics['non_terminal_correct_count'] += non_terminal_correct_count
        metrics['next_obs_accuracy'] += next_obs_accuracy
        return loss, (model_states, metrics)

    return single_batch_model_loss


def get_single_sample_Q_loss(Q_function):
    def single_sample_Q_loss(Q_params, Q_target_params, curr_obs, action, reward, next_obs, continuation_prob, weight, embedding_indices):
        Q_curr = Q_function(Q_params, curr_obs)[action]
        if (config.double_DQN):
            Q_next = Q_function(SG(Q_target_params), next_obs)[jnp.argmax(Q_function(SG(Q_params), next_obs))]
        else:
            Q_next = jnp.max(Q_function(SG(Q_target_params), next_obs))
        return weight * (Q_curr - (reward + config.gamma * continuation_prob * Q_next)) ** 2

    return single_sample_Q_loss

def get_single_sample_env_Q_loss(env_Q_function, Q_function, mi_functions):
    embedding_prior_network = mi_functions['embedding_prior']

    def single_sample_env_Q_loss(env_Q_params, env_Q_target_params,
                                 Q_target_params, mi_params, curr_obs, action, reward, next_obs,
                                 continuation_prob, weight, embedding_indices):
        embedding_prior_params = mi_params['embedding_prior']
        env_Q_curr = env_Q_function(env_Q_params, curr_obs,
                                    jnp.eye(num_actions)[action].squeeze())[embedding_indices]
        next_action = jnp.argmax(Q_function(SG(Q_target_params), next_obs))
        next_action_onehot = jnp.eye(num_actions)[next_action]

        prior_mask = jnp.zeros((config.num_embeddings,), dtype=bool)
        if config.embedding_prior:
            next_embedding_categorical = SG(embedding_prior_network(
                embedding_prior_params, next_obs, next_action_onehot))
            prior_mask = jnp.where(
                jnp.greater(jx.nn.softmax(next_embedding_categorical['logits']), config.min_env_Q_prior),
                0, -jnp.inf)

        if config.double_DQN:
            env_Q_next = env_Q_function(SG(env_Q_target_params), next_obs, next_action_onehot)[
                jnp.argmax(prior_mask + env_Q_function(SG(env_Q_params), next_obs, next_action_onehot))]
        else:
            env_Q_next = jnp.max(prior_mask + env_Q_function(SG(env_Q_target_params), next_obs, next_action_onehot))
        return weight * (env_Q_curr - (-reward + config.gamma * continuation_prob * env_Q_next)) ** 2

    return single_sample_env_Q_loss

def get_single_batch_mi_loss(mi_functions, model_functions, binary_obs):
    def single_batch_mi_loss(mi_params, model_params, model_states, curr_obs, action, reward, next_obs, terminal, metrics, key):
        obs_mi_network = mi_functions['obs']
        act_mi_network = mi_functions['act']

        obs_mi_params = mi_params['obs']
        act_mi_params = mi_params['act']

        encoder_network = model_functions['encoder']
        encoder_params = model_params['encoder']
        encoder_state = model_states['encoder']

        embedding_prior_network = mi_functions['embedding_prior']
        embedding_prior_params = mi_params['embedding_prior']

        quantization, encoder_state = encoder_network(
            encoder_params, encoder_state, curr_obs, jnp.eye(num_actions)[action], next_obs, terminal, reward,
            False)  # is_training=False

        if config.obs_mi:
            obs_dist = obs_mi_network(obs_mi_params, quantization['quantize'], key)
            if binary_obs:
                obs_log_probs = jnp.sum(log_binary_probability(obs, obs_dist), axis=-1)
            else:
                obs_log_probs = jnp.sum(log_gaussian_probability(obs, obs_dist), axis=-1)
            obs_mi_prediction_loss = jnp.where(terminal, 0.0, -obs_log_probs)
        else:
            obs_mi_prediction_loss = 0


        if config.act_mi:
            act_dist = act_mi_network(act_mi_params, quantization['quantize'], key)
            act_loss = -jnp.sum(log_binary_probability(jnp.eye(num_actions)[action], act_dist), axis=-1)
            act_acc = jnp.mean(jnp.equal(action, jnp.argmax(act_dist['logit'], axis=-1))) / action.shape[0]
        else:
            act_loss = 0
            act_acc = 1

        if config.embedding_prior:
            embedding_categorical = embedding_prior_network(embedding_prior_params, curr_obs, jnp.eye(num_actions)[action])
            labels = quantization['encoding_indices']
            embedding_prior_loss = cross_entropy_loss(embedding_categorical, labels)
        else:
            embedding_prior_loss = 0

        mi_loss = obs_mi_prediction_loss + act_loss + embedding_prior_loss
        metrics['mi_loss'] += mi_loss
        metrics['mi_action_acc'] += act_acc
        metrics['embedding_prior_loss'] += embedding_prior_loss

        return mi_loss, metrics

    return single_batch_mi_loss


def get_single_model_rollout_func(model_functions, mi_functions, Q_function, env_Q_function, rollout_length, num_actions):
    def single_model_rollout_func(initial_obs, model_params, model_states, mi_params, Q_params, env_Q_params, key):
        next_obs_network = model_functions['next_obs']
        reward_network = model_functions['reward']
        termination_network = model_functions['termination']
        prior_func = model_functions['prior']

        next_obs_params = model_params['next_obs']
        reward_params = model_params['reward']
        termination_params = model_params['termination']
        prior_state = model_states['prior']
        encoder_params = model_params['encoder']
        encoder_state = model_states['encoder']

        embedding_prior_network = mi_functions['embedding_prior']
        embedding_prior_params = mi_params['embedding_prior']

        def loop_function(carry, data):
            obs, continuation_prob, weight, key = carry

            if (config.episodic_env):
                weight = weight * continuation_prob
            else:
                weight = 1.0

            last_obs = obs

            Q_curr = Q_function(Q_params, obs.astype(float))
            if (config.exploration_strat == "epsilon_greedy"):
                key, subkey = jx.random.split(key)
                randomize_action = jx.random.bernoulli(subkey, config.epsilon)
                key, subkey = jx.random.split(key)
                action = jnp.where(randomize_action, jx.random.choice(subkey, Q_curr.shape[0]), jnp.argmax(Q_curr))
            elif (config.exploration_strat == "softmax"):
                key, subkey = jx.random.split(key)
                action = jx.random.categorical(subkey, Q_curr / config.softmax_temp)
            else:
                raise ValueError("Unknown Exploration Strategy.")

            key, subkey = jx.random.split(key)


            if config.embedding_prior:
                embedding_categorical = embedding_prior_network(embedding_prior_params, obs, jnp.eye(num_actions)[action])
                quantized, embedding_indices = prior_func(embedding_categorical['logits'], encoder_params, encoder_state, subkey)

                prior_mask = jnp.where(
                    jnp.greater(jx.nn.softmax(embedding_categorical['logits']), config.min_env_Q_prior),
                    0, -jnp.inf)

                embedding_indices_env = jnp.argmax(prior_mask + env_Q_function(SG(env_Q_params), obs, jnp.eye(num_actions)[action]))
                quantized_env, _estate = quantize_fun.apply(encoder_params, encoder_state, embedding_indices_env)

                env_mask = jnp.less(jx.random.uniform(subkey), config.env_q_rollout_impact)
                embedding_indices = jnp.where(env_mask, embedding_indices_env, embedding_indices)
                quantized = jnp.where(env_mask, quantized_env, quantized)
            else:
                quantized, embedding_indices = prior_func(jnp.log(prior_state), encoder_params, encoder_state, subkey)

            r_dist = reward_network(reward_params, obs, jnp.eye(num_actions)[action], quantized)
            reward = r_dist["mu"]

            gamma_dist = termination_network(termination_params, obs, jnp.eye(num_actions)[action], quantized)

            key, subkey = jx.random.split(key)
            obs, _ = next_obs_network(next_obs_params, obs, jnp.eye(num_actions)[action], quantized, subkey)

            continuation_prob = jnp.exp(log_binary_probability(True, gamma_dist))
            return (obs, continuation_prob, weight, key), (last_obs, action, reward, obs, continuation_prob, weight,
                                                           embedding_indices)

        key, subkey = jx.random.split(key)
        _, sample_transitions = jx.lax.scan(loop_function, (initial_obs.astype(float), 1.0, 1.0, subkey), None,
                                            length=rollout_length)
        return sample_transitions

    return single_model_rollout_func


def get_agent_environment_interaction_loop_function(env, Q_function, model_functions, mi_functions, env_Q_function,
                                                    Q_opt_update, model_opt_update, mi_opt_update, env_Q_opt_update,
                                                    get_Q_params, get_model_params, get_mi_params, get_env_Q_params,
                                                    replay_buffer, num_iterations,
                                                    num_actions):
    batch_Q_loss = lambda *x: jnp.mean(
        vmap(get_single_sample_Q_loss(Q_function), in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0))(*x))

    batch_env_Q_loss = lambda *x: jnp.mean(
        #env_Q_params, env_Q_target_params,Q_target_params, curr_obs, action, reward, next_obs, continuation_prob, weight, quantization
        vmap(get_single_sample_env_Q_loss(env_Q_function, Q_function, mi_functions),
             in_axes=(None, None, None, None, 0, 0, 0, 0, 0, 0, 0))(*x))
    def batch_model_loss(*x):
        loss, (model_states, metrics) = get_single_batch_model_loss(model_functions, mi_functions, config.binary_obs)(*x)
        return jnp.mean(loss), (model_states, jx.tree_util.tree_map(jnp.mean, metrics))

    def batch_mi_loss(*x):
        loss, metrics = get_single_batch_mi_loss(mi_functions, model_functions, config.binary_obs)(*x)
        return jnp.mean(loss), jx.tree_util.tree_map(jnp.mean, metrics)

    Q_loss_grad = grad(batch_Q_loss)
    model_loss_grad = grad(batch_model_loss, has_aux=True)
    mi_loss_grad = grad(batch_mi_loss, has_aux=True)
    env_Q_loss_grad = grad(batch_env_Q_loss)

    batch_model_rollout = lambda *x: [jnp.reshape(y, (config.batch_size * config.rollout_length, -1)) for y in vmap(
        get_single_model_rollout_func(model_functions, mi_functions, Q_function, env_Q_function, config.rollout_length, num_actions),
        in_axes=(0, None, None, None, None, None, 0))(*x)]

    def agent_environment_interaction_loop_function(env_state, Q_opt_state, Q_target_params,  env_Q_opt_state, env_Q_target_params, model_opt_state,
                                                    model_states, mi_opt_state, buffer_state, opt_t, t, key, train):
        obs = env.get_observation(env_state)
        metrics = {
            'avg_reward': 0.0,
            'avg_Q': 0.0,
            'next_obs_loss': 0.0,
            'reward_loss': 0.0,
            'termination_loss': 0.0,
            'terminal_correct_count': 0.0,
            'terminal': 0.0,
            'non_terminal': 0.0,
            'non_terminal_correct_count': 0.0,
            'next_obs_accuracy': 0.0,
            'mi_loss': 0.0,
            'mi_action_acc': 0.0,
            'embedding_prior_loss': 0.0,
        }
        def loop_function(carry, data):
            env_state, Q_opt_state, Q_target_params, env_Q_opt_state, env_Q_target_params, model_opt_state, model_states, mi_opt_state, buffer_state, opt_t, t, metrics, obs, key = carry
            key, subkey = jx.random.split(key)
            Q_curr = Q_function(get_Q_params(Q_opt_state), obs.astype(float))
            metrics['avg_Q'] += jnp.max(Q_curr)
            if (config.exploration_strat == "epsilon_greedy"):
                key, subkey = jx.random.split(key)
                randomize_action = jx.random.bernoulli(subkey, config.epsilon)
                key, subkey = jx.random.split(key)
                action = jnp.where(randomize_action, jx.random.choice(subkey, Q_curr.shape[0]), jnp.argmax(Q_curr))
            elif (config.exploration_strat == "softmax"):
                key, subkey = jx.random.split(key)
                action = jx.random.categorical(subkey, Q_curr / config.softmax_temp)
            else:
                raise ValueError("Unknown Exploration Strategy.")
            last_obs = obs
            key, subkey = jx.random.split(key)
            env_state, obs, reward, terminal, _ = env.step(subkey, env_state, action)
            # reset if terminated
            key, subkey = jx.random.split(key)
            env_state = jx.tree_map(lambda x, y: jnp.where(terminal, x, y), env.reset(subkey)[0], env_state)
            buffer_state = replay_buffer.add(buffer_state, last_obs, action, reward, obs, terminal)
            if (train):
                if (config.use_target):
                    Q_target_params = jx.tree_map(lambda x, y: jnp.where(t % config.target_update_frequency == 0, x, y),
                                                  get_Q_params(Q_opt_state), Q_target_params)
                    if config.train_env_Q:
                        env_Q_target_params = jx.tree_map(lambda x, y: jnp.where(t % config.target_update_frequency == 0, x, y),
                                                      get_env_Q_params(env_Q_opt_state), env_Q_target_params)
                else:
                    Q_target_params = get_Q_params(Q_opt_state)
                    if config.train_env_Q:
                        env_Q_target_params = get_env_Q_params(env_Q_opt_state)

                def update_loop_function(carry, data):
                    Q_opt_state, Q_target_params, env_Q_opt_state, env_Q_target_params, model_opt_state, model_states, mi_opt_state, buffer_state, opt_t, metrics, key = carry
                    buffer_state, sample_transitions = replay_buffer.sample(buffer_state)
                    key, subkey = jx.random.split(key)
                    # subkeys = jx.random.split(subkey, num=config.batch_size)
                    model_grad, (model_states, metrics) = model_loss_grad(get_model_params(model_opt_state), model_states, get_mi_params(mi_opt_state), *sample_transitions, metrics, subkey)
                    model_opt_state = model_opt_update(opt_t, model_grad, model_opt_state)

                    # curr_obs, action, reward, next_obs, terminal)
                    mi_grad, metrics = mi_loss_grad(get_mi_params(mi_opt_state), get_model_params(model_opt_state),
                                                    model_states, *sample_transitions, metrics, subkey)
                    mi_opt_state = mi_opt_update(opt_t, mi_grad, mi_opt_state)

                    sample_obs = sample_transitions[0]

                    key, subkey = jx.random.split(key)
                    subkeys = jx.random.split(subkey, num=config.batch_size)
                    model_transitions = batch_model_rollout(sample_obs, get_model_params(model_opt_state), model_states,
                                                            get_mi_params(mi_opt_state),
                                                            get_Q_params(Q_opt_state),
                                                            get_env_Q_params(env_Q_opt_state), subkeys)

                    Q_grad = Q_loss_grad(get_Q_params(Q_opt_state), Q_target_params, *model_transitions)
                    Q_opt_state = Q_opt_update(opt_t, Q_grad, Q_opt_state)

                    if config.train_env_Q:
                        env_Q_grad = env_Q_loss_grad(get_env_Q_params(env_Q_opt_state), env_Q_target_params, Q_target_params, get_mi_params(mi_opt_state), *model_transitions)
                        env_Q_opt_state = env_Q_opt_update(opt_t, env_Q_grad, env_Q_opt_state)

                    opt_t += 1
                    return (Q_opt_state, Q_target_params, env_Q_opt_state, env_Q_target_params, model_opt_state, model_states, mi_opt_state, buffer_state, opt_t, metrics, key), None

                key, subkey = jx.random.split(key)
                carry = (Q_opt_state, Q_target_params, env_Q_opt_state, env_Q_target_params, model_opt_state, model_states, mi_opt_state, buffer_state, opt_t, metrics, subkey)
                (Q_opt_state, Q_target_params, env_Q_opt_state, env_Q_target_params, model_opt_state, model_states,
                 mi_opt_state, buffer_state, opt_t, metrics, _), _ = jx.lax.scan(
                    update_loop_function, carry, None, length=config.updates_per_step)
            metrics['avg_reward'] += reward
            t += 1
            return (
            env_state, Q_opt_state, Q_target_params, env_Q_opt_state, env_Q_target_params, model_opt_state, model_states, mi_opt_state, buffer_state, opt_t, t, metrics,
            obs, key), None

        key, subkey = jx.random.split(key)

        carry = (
        env_state, Q_opt_state, Q_target_params, env_Q_opt_state, env_Q_target_params, model_opt_state, model_states, mi_opt_state, buffer_state, opt_t, t, metrics, obs,
        subkey)

        (env_state, Q_opt_state, Q_target_params,  env_Q_opt_state, env_Q_target_params, model_opt_state, model_states, mi_opt_state, buffer_state, opt_t, t, metrics, obs,
         _), _ = jx.lax.scan(loop_function, carry, None, length=num_iterations)
        for mk, v in metrics.items():
            metrics[mk] = v / num_iterations

        return env_state, Q_opt_state, Q_target_params, env_Q_opt_state, env_Q_target_params, model_opt_state, model_states, mi_opt_state, buffer_state, opt_t, t, metrics

    return jit(agent_environment_interaction_loop_function, static_argnames=('train',))


def get_agent_eval_function_episodic(env, Q_function, model_functions, get_Q_params, get_model_params, num_iterations):
    def agent_eval_function_episodic(Q_opt_state, model_opt_state, key):
        key, subkey = jx.random.split(key)
        env_state, phi = env.reset(subkey)
        total_reward = 0.0
        nonterminal_steps = 0

        def loop_function(carry, data):
            env_state, Q_opt_state, total_reward, nonterminal_steps, phi, terminated, key = carry
            Q_curr = Q_function(get_Q_params(Q_opt_state), phi.astype(float))
            action = jnp.argmax(Q_curr)
            key, subkey = jx.random.split(key)

            env_state, phi, reward, terminal, _ = env.step(subkey, env_state, action)

            total_reward += reward * jnp.logical_not(terminated)
            nonterminal_steps += jnp.logical_not(terminated)

            terminated = jnp.logical_or(terminated, terminal)
            return (env_state, Q_opt_state, total_reward, nonterminal_steps, phi, terminated, key), None

        key, subkey = jx.random.split(key)
        carry = (env_state, Q_opt_state, total_reward, nonterminal_steps, phi, False, subkey)

        (env_state, Q_opt_state, total_reward, nonterminal_steps, phi, _, _), _ = jx.lax.scan(loop_function, carry,
                                                                                              None,
                                                                                              length=num_iterations)

        return total_reward

    return jit(agent_eval_function_episodic)


def get_agent_eval_function_continuing(env, Q_function, model_functions, get_Q_params, get_model_params,
                                       num_iterations):
    def agent_eval_function_continuing(Q_opt_state, model_opt_state, key):
        key, subkey = jx.random.split(key)
        env_state, obs = env.reset(subkey)
        total_reward = 0.0

        def loop_function(carry, data):
            env_state, Q_opt_state, model_opt_state, total_reward, obs, key = carry
            key, subkey = jx.random.split(key)
            Q_curr = Q_function(get_Q_params(Q_opt_state), obs)
            key, subkey = jx.random.split(key)
            action = jnp.argmax(Q_curr)
            key, subkey = jx.random.split(key)

            env_state, obs, reward, terminal, _ = env.step(subkey, env_state, action)

            total_reward += reward
            return (env_state, Q_opt_state, model_opt_state, total_reward, obs, key), None

        key, subkey = jx.random.split(key)
        carry = (env_state, Q_opt_state, model_opt_state, total_reward, obs, subkey)

        (env_state, Q_opt_state, model_opt_state, total_reward, obs, _), _ = jx.lax.scan(loop_function, carry, None,
                                                                                         length=num_iterations)

        return total_reward / num_iterations

    return jit(agent_eval_function_continuing)


########################################################################
# Replay Buffer
########################################################################

class replay_buffer:
    def __init__(self, buffer_size, batch_size, item_shapes, item_types):
        self.buffer_size = buffer_size
        self.types = item_types
        self.shapes = item_shapes
        self.batch_size = batch_size

    @partial(jit, static_argnums=(0,))
    def initialize(self, key):
        location = 0
        full = False
        buffers = [jnp.zeros([self.buffer_size] + list(s), dtype=t) for s, t in zip(self.shapes, self.types)]
        state = (location, full, buffers, key)
        return state

    @partial(jit, static_argnums=(0,))
    def add(self, state, *args):
        location, full, buffers, key = state
        # Append when the buffer is not full but overwrite when the buffer is full
        for i, (a, t) in enumerate(zip(args, self.types)):
            buffers[i] = buffers[i].at[location].set(jnp.asarray(a, dtype=t))
        full = jnp.where(location == self.buffer_size - 1, True, full)
        # Increment the buffer location
        location = (location + 1) % self.buffer_size
        state = (location, full, buffers, key)
        return state

    @partial(jit, static_argnums=(0, 2))
    def sample(self, state):
        location, full, buffers, key = state
        key, subkey = jx.random.split(key)
        indices = jx.random.randint(subkey, minval=0, maxval=jnp.where(full, self.buffer_size, location),
                                    shape=(self.batch_size,))

        sample = []
        for b in buffers:
            sample += [b.take(indices, axis=0)]

        state = (location, full, buffers, key)
        return state, sample


env = Environment(**env_config)
key, subkey = jx.random.split(key)
env_state, obs = env.reset(subkey)
num_actions = env.num_actions()

dummy_a = jnp.zeros((num_actions))
dummy_emb = jnp.zeros((config.latent_dim))

Q_net = hk.without_apply_rng(hk.transform(lambda obs: Q_function(config, num_actions)(obs)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
Q_params = [Q_net.init(subkey, obs.astype(float)) for subkey in subkeys]
Q_func = Q_net.apply

Q_target_params = tree_stack(Q_params)

reward_net = hk.without_apply_rng(hk.transform(lambda obs, a, emb: reward_decoder(config)(obs, a, emb)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
reward_params = [reward_net.init(subkey, obs.astype(float), dummy_a, dummy_emb) for subkey in subkeys]
reward_func = reward_net.apply

termination_net = hk.without_apply_rng(hk.transform(lambda obs, a, emb: termination_decoder(config)(obs, a, emb)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
termination_params = [termination_net.init(subkey, obs.astype(float), dummy_a, dummy_emb) for subkey in subkeys]
termination_func = termination_net.apply

next_obs_net = hk.without_apply_rng(
    hk.transform(lambda obs, a, emb, key: next_obs_decoder(config, obs.shape[-1])(obs, a, emb, key)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
next_obs_params = [next_obs_net.init(subkey, obs.astype(float), dummy_a, dummy_emb, subkey) for subkey in subkeys]
next_obs_func = next_obs_net.apply

encoder_net = hk.without_apply_rng(
    hk.transform_with_state(lambda obs, action, next_obs, reward, termination, is_training:
                            Encoder(config)(obs, action, next_obs, reward, termination, is_training)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
encoder_params_and_state =\
    [encoder_net.init(subkey,
                      obs.astype(float)[None, :],
                      dummy_a[None, :],
                      obs.astype(float)[None, :],
                      jnp.zeros((1, 1)),
                      jnp.ones((1, 1), dtype=bool),
                      jnp.ones((1, 1), dtype=bool)
                      ) for subkey in subkeys]
encoder_params = [ps[0] for ps in encoder_params_and_state]
encoder_state = [ps[1] for ps in encoder_params_and_state]
encoder_func = encoder_net.apply

quantize_fun = hk.without_apply_rng(
    hk.transform_with_state(lambda encoding_indices: Encoder(config).vq.quantize(encoding_indices)))


def prior_func(prior_logits, encoder_params, encoder_state, key):
    embedding_indices = jx.random.categorical(key, prior_logits)
    quantized, enc_state = quantize_fun.apply(encoder_params, encoder_state, embedding_indices)
    return jx.lax.stop_gradient(quantized), embedding_indices


prior_state = jnp.zeros((config.num_seeds, config.num_embeddings), dtype=float)


env_Q_net = hk.without_apply_rng(hk.transform(lambda obs, a: Q_function(config, config.num_embeddings)(jnp.hstack([obs, a]))))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
env_Q_params = [env_Q_net.init(subkey, obs.astype(float), dummy_a) for subkey in subkeys]
env_Q_func = env_Q_net.apply

env_Q_target_params = tree_stack(env_Q_params)

model_funcs = {"reward": reward_func, "termination": termination_func, "next_obs": next_obs_func,
               "encoder": encoder_func, "prior": prior_func}

model_params = [{"reward": rp, "termination": tp, "next_obs": no, "encoder": ep} for rp, tp, no, ep in
                zip(reward_params, termination_params, next_obs_params, encoder_params)]

model_states = tree_stack([{"encoder": es, "prior": ps} for es, ps in zip(encoder_state, prior_state)])



obs_mi_net = hk.without_apply_rng(
    hk.transform(lambda emb, key: obs_mi_decoder(config, obs.shape[-1])(emb, key)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
obs_mi_params = [obs_mi_net.init(subkey, dummy_emb, subkey) for subkey in subkeys]
obs_mi_func = obs_mi_net.apply

action_mi_net = hk.without_apply_rng(
    hk.transform(lambda emb, key: action_mi_decoder(config, num_actions)(emb, key)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
act_mi_params = [action_mi_net.init(subkey, dummy_emb, subkey) for subkey in subkeys]
act_mi_func = action_mi_net.apply

embedding_prior_net = hk.without_apply_rng(
    hk.transform(lambda obs, a: embedding_prior(config)(obs, a)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
embedding_prior_params = [embedding_prior_net.init(subkey, obs.astype(float), dummy_a) for subkey in subkeys]
embedding_prior_func = embedding_prior_net.apply

mi_funcs = {"obs": obs_mi_func, "act": act_mi_func, "embedding_prior": embedding_prior_func}
mi_params = [{"obs": op, "act": ap, "embedding_prior": epp } for op, ap, epp in zip(
    obs_mi_params, act_mi_params, embedding_prior_params)]

Q_opt_init, Q_opt_update, get_Q_params = adamw(config.Q_alpha, eps=config.eps_adam, b1=config.b1_adam,
                                               b2=config.b2_adam, wd=config.wd_adam)
Q_opt_states = tree_stack([Q_opt_init(p) for p in Q_params])
Q_opt_update = jit(Q_opt_update)

env_Q_opt_init, env_Q_opt_update, get_env_Q_params = adamw(config.Q_alpha, eps=config.eps_adam, b1=config.b1_adam,
                                                           b2=config.b2_adam, wd=config.wd_adam)
env_Q_opt_states = tree_stack([env_Q_opt_init(p) for p in env_Q_params])
env_Q_opt_update = jit(env_Q_opt_update)

model_opt_init, model_opt_update, get_model_params = adamw(config.model_alpha, eps=config.eps_adam, b1=config.b1_adam,
                                                           b2=config.b2_adam, wd=config.wd_adam)
model_opt_states = tree_stack([model_opt_init(p) for p in model_params])
model_opt_update = jit(model_opt_update)

mi_opt_init, mi_opt_update, get_mi_params = adamw(config.model_alpha, eps=config.eps_adam, b1=config.b1_adam,
                                                           b2=config.b2_adam, wd=config.wd_adam)
mi_opt_states = tree_stack([mi_opt_init(p) for p in mi_params])
mi_opt_update = jit(mi_opt_update)

buffer = replay_buffer(config.buffer_size, config.batch_size, (obs.shape, (), (), obs.shape, ()),
                       (float, int, float, float, bool))

key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
buffer_states = tree_stack([buffer.initialize(subkey) for subkey in subkeys])

interaction_loop = get_agent_environment_interaction_loop_function(env, Q_func, model_funcs, mi_funcs, env_Q_func,
                                                                   Q_opt_update,
                                                                   model_opt_update, mi_opt_update, env_Q_opt_update,
                                                                   get_Q_params, get_model_params, get_mi_params,
                                                                   get_env_Q_params,
                                                                   buffer, config.eval_frequency, num_actions)

if (config.episodic_env):
    eval_agent = jit(lambda *x: jnp.mean(vmap(
        get_agent_eval_function_episodic(env, Q_func, model_funcs, get_Q_params, get_model_params, config.eval_steps),
        in_axes=(None, None, 0))(*x)))
else:
    eval_agent = jit(lambda *x: jnp.mean(vmap(
        get_agent_eval_function_continuing(env, Q_func, model_funcs, get_Q_params, get_model_params, config.eval_steps),
        in_axes=(None, None, 0))(*x)))

multiseed_interaction_loop = jit(
    vmap(interaction_loop, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, 0, None), out_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, 0)),
    static_argnames='train')

multiseed_eval_agent = jit(vmap(eval_agent, in_axes=(0, 0, 0)))

key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
env_states = tree_stack([(lambda subkey: env.reset(subkey)[0])(s) for s in subkeys])

opt_t = 0
t = 0

metrics = defaultdict(list)

time_since_last_save = 0
for i in tqdm(range(config.num_steps // config.eval_frequency)):
    time = config.eval_frequency * i
    if (config.save_params and (time_since_last_save >= config.save_frequency)):
        with open(args.output + ".params", 'wb') as f:
            pkl.dump({
                'model': [get_model_params(model_opt_state) for model_opt_state in tree_unstack(model_opt_states)],
                'mi': [get_mi_params(mi_opt_state) for mi_opt_state in tree_unstack(mi_opt_states)],
                'Q': [get_Q_params(Q_opt_state) for Q_opt_state in tree_unstack(Q_opt_states)],
                'env_Q': [get_env_Q_params(env_Q_opt_state) for env_Q_opt_state in tree_unstack(env_Q_opt_states)]
            }, f)
        time_since_last_save = 0

    # Train step
    key, subkey = jx.random.split(key)
    subkeys = jx.random.split(subkey, num=config.num_seeds)
    env_states, Q_opt_states, Q_target_params, env_Q_opt_states, env_Q_target_params, model_opt_states, model_states, mi_opt_states, buffer_states, opt_t, t, train_metrics = multiseed_interaction_loop(
        env_states, Q_opt_states, Q_target_params, env_Q_opt_states, env_Q_target_params, model_opt_states, model_states, mi_opt_states, buffer_states, opt_t, t, subkeys,
        time >= config.training_start_time)

    # Evaluation step
    key, subkey = jx.random.split(key)
    subkeys = jx.random.split(subkey, num=config.num_seeds * config.eval_batch_size).reshape(config.num_seeds,
                                                                                             config.eval_batch_size, 2)
    reward_rate = multiseed_eval_agent(Q_opt_states, model_opt_states, subkeys)

    # Logging
    metrics["reward_rates"] += [reward_rate]
    metrics["eval_times"] += [time]
    for k, value in train_metrics.items():
        metrics[k] += [value]
    # log_dict = {"reward_rate": reward_rate, "time": time}
    # write_string = "| ".join([k + ": " + str(v) for k, v in log_dict.items()])
    # tqdm.write(write_string)
    time_since_last_save += config.eval_frequency

with open('out/' + args.output + ".out", 'wb') as f:
    pkl.dump({
        'config': config,
        'metrics': metrics
    }, f)

# save params once more at the end
if (config.save_params):
    with open('out/' + args.output + ".params", 'wb') as f:
        pkl.dump({
            'model': [get_model_params(model_opt_state) for model_opt_state in tree_unstack(model_opt_states)],
            'mi': [get_mi_params(mi_opt_state) for mi_opt_state in tree_unstack(mi_opt_states)],
            'Q': [get_Q_params(Q_opt_state) for Q_opt_state in tree_unstack(Q_opt_states)],
            'env_Q': [get_env_Q_params(env_Q_opt_state) for env_Q_opt_state in tree_unstack(env_Q_opt_states)]
        }, f)
    with open('out/' + args.output + ".states", 'wb') as f:
        pkl.dump({
            'model': [s for s in tree_unstack(model_states)]
        }, f)
