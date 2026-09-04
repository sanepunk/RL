from flax import nnx
import jax


class ActorCriticNetwork(nnx.Module):
    def __init__(self, obs_dim, action_space, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(obs_dim, 30, rngs=rngs)
        self.l2 = nnx.Linear(30, 50, rngs=rngs)
        self.l3 = nnx.Linear(50, 8, rngs=rngs)
        self.policy_net = nnx.Linear(8, action_space, rngs=rngs)
        self.value_net = nnx.Linear(8, action_space, rngs=rngs)

    def __call__(self, obs):
        x = nnx.relu(self.l1(obs))
        x = nnx.relu(self.l2(x))
        x = nnx.relu(self.l3(x))
        return x

    def policy(self, obs):
        return self.policy_net(self(obs))

    def epsilon_greedy_strategy(self, obs, rng: jax.random.PRNGKey, epsilon=0.3):
        action = self.policy(obs)
        choice = jax.random.uniform(rng) > epsilon
        rng, choice_rng = jax.random.split(rng)
        return int(
            jax.lax.select(
                choice,
                jax.random.randint(choice_rng, (), minval=0, maxval=action.shape[-1]),
                jax.numpy.argmax(action),
            )
        )

    def value(self, obs):
        return self.value_net(self(obs))
