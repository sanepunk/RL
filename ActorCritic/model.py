from flax import nnx
import jax


class ActorCriticNetwork(nnx.Module):
    def __init__(self, obs_dim, action_space, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(obs_dim, 90, rngs=rngs)
        self.l2 = nnx.Linear(90, 70, rngs=rngs)
        self.policy_net = nnx.Linear(70, action_space, rngs=rngs)
        self.value_net = nnx.Linear(70, 1, rngs=rngs)

    def __call__(self, obs):
        x = nnx.relu(self.l1(obs))
        x = nnx.relu(self.l2(x))
        return x

    def policy(self, obs):
        return self.policy_net(self(obs))

    def sample_action(self, obs, rng: jax.random.PRNGKey, epsilon=0.3):
        action = self.policy(obs)
        index = (jax.random.categorical(rng, action)).astype(int)
        return index

    def value(self, obs):
        return self.value_net(self(obs))

    @nnx.vmap(in_axes=(None, 0), out_axes=0)
    def vmap_value(self, obs):
        return self.value(obs)

    @nnx.vmap(in_axes=(None, 0), out_axes=0)
    def vmap_policy(self, obs):
        return self.policy(obs)
