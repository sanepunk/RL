from flax import nnx
import optax
from typing import Literal


def return_optimizer(
    model: nnx.Module,
    optimizer: Literal["adam", "adamw", "adamax", "sgd", "rmsprop"],
    learning_rate: float = 1e-3,
    gradNorm: float | None = None,
) -> nnx.Optimizer:
    # Select the optimizer based on the input argument
    optimizer_map = {
        "adam": optax.adam(learning_rate),
        "adamw": optax.adamw(learning_rate),
        "adamax": optax.adamax(learning_rate),
        "sgd": optax.sgd(learning_rate),
        "rmsprop": optax.rmsprop(learning_rate),
    }

    opt = optimizer_map.get(optimizer, optax.adam(learning_rate))

    if gradNorm:
        return nnx.Optimizer(
            model,
            optax.chain(
                # optax.exponential_decay(learning_rate, decay_rate=0.9, transition_steps=10, transition_begin=1),
                optax.clip_by_global_norm(gradNorm),
                optax.normalize_by_update_norm(),
                opt,
            ),
            wrt=nnx.Param,
        )
    return nnx.Optimizer(model, opt, wrt=nnx.Param)
