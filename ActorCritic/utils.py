from typing import Any

from model import ActorCriticNetwork
from flax import nnx
import jax
from jax import numpy as jnp


def calculate_returns(rewards, last_val, gamma: float = 0.99, mask=None):
    def step_fn(R, data):
        reward, mask = data
        R = reward + R * gamma * mask
        return R, R

    _, returns = jax.lax.scan(step_fn, last_val, (rewards, mask), reverse=True)
    return returns


def run_episode(
    model: ActorCriticNetwork, env, env_params, rng, steps: int = 500
) -> dict:
    # model.eval()
    rng, rng_reset = jax.random.split(rng)
    obs, state = env.reset(rng_reset, env_params)

    def step_fn(carry, _):
        obs, state, rng, done = carry
        rng, action_selection_rng = jax.random.split(rng)
        rng, step_rng = jax.random.split(rng)
        action = model.sample_action(obs, action_selection_rng)
        new_obs, new_state, reward, next_done, _ = env.step(
            step_rng, state, action, env_params
        )
        valid_mask = 1 - done
        new_carry = (new_obs, new_state, rng, jnp.maximum(done, next_done))
        return new_carry, {
            "reward": reward * valid_mask,
            "valid_mask": valid_mask,
            "action": action,
            "obs": obs,
        }

    init_carry = (obs, state, rng, 0.0)
    final_state, trajectories = jax.lax.scan(step_fn, init_carry, length=steps)
    return final_state[0], trajectories


def normalize(val_arr, mask):
    count = jnp.sum(mask) + 1e-8
    mean = jnp.sum(val_arr * mask) / count
    variance = jnp.sum(((val_arr - mean) ** 2) * mask) / count
    std = jnp.sqrt(variance + 1e-8)
    return jnp.divide(jnp.subtract(val_arr, mean), std)


def actor_critic_loss(
    model: ActorCriticNetwork, rewards, action, obs, valid_mask=None, last_obs=None
):
    valid_mask = jnp.array(valid_mask)
    returns = jax.vmap(calculate_returns, in_axes=(0, 0, None, 0), out_axes=0)(
        jnp.array(rewards), model.value(last_obs)[..., 0], 0.99, valid_mask
    )
    returns = jax.vmap(normalize, in_axes=(0, 0), out_axes=0)(returns, valid_mask)
    returns = jax.lax.stop_gradient(returns)
    action = jnp.array(action)
    obs = jnp.stack(obs)
    batches, runs = obs.shape[:2]
    val_arr = model.vmap_value(obs)[..., 0]
    log_logits = nnx.log_softmax(model.vmap_policy(obs))
    # print(log_logits.shape, action.shape, obs.shape)
    action_log_prob = jnp.take_along_axis(
        log_logits, action.reshape(batches, runs, 1), axis=-1
    )[..., 0]

    advantage = returns - val_arr

    advantage = jax.lax.stop_gradient(normalize(advantage, valid_mask))

    count = jnp.sum(valid_mask) + 1e-8

    policy_loss = -jnp.sum(advantage * action_log_prob * valid_mask) / count

    value_loss = jnp.sum(jnp.square(returns - val_arr) * valid_mask) / count

    return value_loss * 0.5 + policy_loss


def actor_critic_update(
    model, optimizer, rewards, action, obs, valid_mask: Any = None, last_obs=None
):
    grad = nnx.grad(actor_critic_loss, argnums=0)(
        model, rewards, action, obs, valid_mask, last_obs
    )

    optimizer.update(model, grad)
