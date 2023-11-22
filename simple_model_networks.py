import haiku as hk
import jax as jx
import jax.numpy as jnp

activation_dict = {"relu": jx.nn.relu, "silu": jx.nn.silu, "elu": jx.nn.elu}

class Q_function(hk.Module):
    def __init__(self, config, num_actions, name=None):
        super().__init__(name=name)
        self.num_hidden_units = config.num_hidden_units
        self.num_hidden_layers = config.num_hidden_layers
        self.activation_function = activation_dict[config.activation]
        self.num_actions = num_actions

    def __call__(self, obs):
        x = jnp.ravel(obs)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        Q = hk.Linear(self.num_actions)(x)
        return Q


class reward_function(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]

    def __call__(self, obs, action, key=None):
        x = jnp.concatenate([jnp.ravel(obs), action])
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        mu = hk.Linear(1)(x)[0]
        sigma = jnp.ones(mu.shape)
        return {'mu': mu, 'sigma': sigma}


class termination_function(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]

    def __call__(self, obs, action, key=None):
        x = jnp.concatenate([jnp.ravel(obs), action])
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        logit = hk.Linear(1)(x)[0]
        return {'logit': logit}


class next_obs_function(hk.Module):
    def __init__(self, config, obs_width, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.obs_width = obs_width
        self.binary_obs = config.binary_obs
        self.activation_function = activation_dict[config.activation]

    def __call__(self, obs, action, key):
        x = jnp.concatenate([jnp.ravel(obs), action])
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

