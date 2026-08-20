from flax import nnx
import jax


class ActionValueNet(nnx.Module):
    def __init__(self, obs_dim, action_space, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(obs_dim, 50, rngs=rngs)
        self.l2 = nnx.Linear(50, 70, rngs=rngs)
        self.l3 = nnx.Linear(70, 40, rngs=rngs)
        self.l4 = nnx.Linear(40, action_space, rngs=rngs)

    def __call__(self, x):
        x1 = nnx.relu(self.l1(x))
        x2 = nnx.relu(self.l2(x1))
        x3 = nnx.relu(self.l3(x2))
        return self.l4(x3)

    def epsilon_greedy_strategy(
        self, obs, epsilon=0.7, rngs: jax.random.PRNGKey = jax.random.PRNGKey(22)
    ):
        choice_key, selection_key = jax.random.split(rngs)
        x = self.__call__(obs)
        choice = jax.random.uniform(choice_key) > epsilon
        return jax.lax.select(
            choice,
            jax.random.randint(selection_key, shape=(), minval=0, maxval=x.shape[-1]),
            jax.numpy.argmax(x),
        )
