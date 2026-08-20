from flax import nnx
import jax
from typing import Literal


class Agent(nnx.Module):
    def __init__(
        self,
        observation_dim,
        num_actions,
        rngs: nnx.Rngs,
        hidden_layers: int = 2,
        neurons: list[int] = [16, 16, 16],
        kernel_init=Literal["kaiming", "he", "lecun", "xavier"],
    ):
        super().__init__()
        # self.layers = []
        neurons = [16, 16, 16]
        if neurons is None:
            self.layers.append(nnx.Linear(observation_dim, num_actions))
            self.layers = nnx.List(self.layers)
        else:
            # rngs1, rngs2, rngs3 = nnx.split_rngs(rngs, splits = 3)
            self.layers = nnx.List(
                [
                    nnx.Linear(
                        observation_dim,
                        16,
                        rngs=rngs,
                        kernel_init=nnx.initializers.xavier_normal(),
                    ),
                    nnx.LayerNorm(num_features=16, rngs=rngs),
                    nnx.relu,
                    nnx.Linear(
                        16, 64, rngs=rngs, kernel_init=nnx.initializers.xavier_normal()
                    ),
                    nnx.LayerNorm(64, rngs=rngs),
                    nnx.relu,
                    nnx.Linear(
                        64,
                        num_actions,
                        rngs=rngs,
                        kernel_init=nnx.initializers.xavier_normal(),
                    ),
                ]
            )

    @staticmethod
    @nnx.jit
    def __step_fn(x, layer):
        x = layer(x)
        return x, x

    def __call__(self, x):
        # prev = x
        # for i, layers in enumerate(self.layers):
        #     if i > 0:
        #         x = layers((x+prev - jnp.mean(x+prev)) / jnp.std(x+prev)**2)
        #     else:
        #         x = layers(x)
        #     prev = x
        # x, _ = jax.l
        for layer in self.layers:
            x = layer(x)
        # ax.scan(Agent.__step_fn, x, self.layers)
        return x

    def select_categorical_action(self, x, rngs):
        # x, _ = jax.lax.scan(Agent.__step_fn, x, self.layers)
        # prev = x
        # for i, layers in enumerate(self.layers):
        #     if i > 0:
        #         x = layers((x+prev - jnp.mean(x+prev)) / jnp.std(x+prev)**2)
        #     else:
        #         x = layers(x)
        #     prev = x
        for layer in self.layers:
            x = layer(x)
        x = nnx.log_softmax(x)
        sampled_action = jax.random.categorical(rngs, x)
        return sampled_action
