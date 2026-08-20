from gymnax import make
from flax import nnx
import jax
from jax import numpy as jnp


def get_env(name: str = "CartPole-v1"):
    env, env_params = make(name)
    return env, env_params


def loss(model, target_net, obs, next_obs, r, action, done, gamma):
    curr_q = model(obs)[action]
    next_q = jnp.max(target_net(next_obs)) * (1 - done)

    next_q = jax.lax.stop_gradient(next_q)
    return (r + gamma * next_q - curr_q) ** 2


def update(
    model, target_net, obs, next_obs, r, action, done, gamma, optimizer: nnx.Optimizer
):
    grads = nnx.grad(loss, argnums=0)(
        model, target_net, obs, next_obs, r, action, done, gamma
    )
    optimizer.update(model, grads)
