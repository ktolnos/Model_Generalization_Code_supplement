import haiku as hk
import jax.numpy as jnp
import jax as jx
from haiku.nets import VectorQuantizerEMA

from simple_model_networks import activation_dict, Q_function


class Encoder(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]
        self.latent_dim = config.latent_dim
        self.vq = VectorQuantizerEMA(
            self.latent_dim, config.num_embeddings, config.commitment_cost, decay=config.embedding_decay)
        self.embed_reward = config.embed_reward
        self.embed_termination = config.embed_termination

    def __call__(self, obs, action, next_obs, reward, termination, is_training):
        inputs = [obs, action, next_obs]
        if self.embed_reward:
            inputs.append(reward)
        if self.embed_termination:
            inputs.append(termination)
        x = jnp.hstack(inputs)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        x = hk.Linear(self.latent_dim)(x)
        x = self.vq(x, is_training)
        return x


class reward_decoder(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]
        self.use_emb = config.embed_reward

    def __call__(self, obs, action, embedding, key=None):
        inputs = [obs, action]
        if self.use_emb:
            inputs.append(embedding)
        x = jnp.hstack(inputs)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        mu = hk.Linear(1)(x).flatten().squeeze()
        sigma = jnp.ones(mu.shape)
        return {'mu': mu, 'sigma': sigma}


class termination_decoder(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]
        self.use_emb = config.embed_termination

    def __call__(self, obs, action, embedding, key=None):
        inputs = [obs, action]
        if self.use_emb:
            inputs.append(embedding)
        x = jnp.hstack(inputs)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        logit = hk.Linear(1)(x).flatten().squeeze()
        return {'logit': logit}


class next_obs_decoder(hk.Module):
    def __init__(self, config, obs_width, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.obs_width = obs_width
        self.binary_obs = config.binary_obs
        self.activation_function = activation_dict[config.activation]
        self.use_emb = config.embed_next_obs

    def __call__(self, obs, action, embedding, key):
        inputs = [obs, action]
        if self.use_emb:
            inputs.append(embedding)
        x = jnp.hstack(inputs)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        if (self.binary_obs):
            logit = hk.Linear(self.obs_width)(x)
            x = jx.random.bernoulli(key, logit)
            return x.astype(float), {'logit': logit}
        else:
            mu = hk.Linear(self.obs_width)(x)
            sigma = jnp.ones(mu.shape)
            x = mu + sigma * jx.random.normal(key, mu.shape)
            return x, {'mu': mu, 'sigma': sigma}


class action_mi_decoder(hk.Module):
    def __init__(self, config, num_actions, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]
        self.num_actions = num_actions

    def __call__(self, embedding, key=None):
        x = embedding
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        logit = hk.Linear(self.num_actions)(x)
        return {'logit': logit}


class obs_mi_decoder(hk.Module):
    def __init__(self, config, obs_width, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.obs_width = obs_width
        self.binary_obs = config.binary_obs
        self.activation_function = activation_dict[config.activation]

    def __call__(self, embedding, key=None):
        x = embedding
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        if self.binary_obs:
            logit = hk.Linear(self.obs_width)(x)
            return {'logit': logit}
        else:
            mu = hk.Linear(self.obs_width)(x)
            sigma = jnp.ones(mu.shape)
            return {'mu': mu, 'sigma': sigma}


class embedding_prior(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]
        self.num_embeddings = config.num_embeddings

    def __call__(self, obs, action):
        x = jnp.hstack([obs, action])
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        logit = hk.Linear(self.num_embeddings)(x)
        return {'logits': logit}
