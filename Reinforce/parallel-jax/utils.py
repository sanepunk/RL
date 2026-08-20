from jax import numpy as jnp
import jax
from model import Agent
import orbax.checkpoint as ocp
from flax import nnx
from pathlib import Path

checkpointer = ocp.StandardCheckpointer()


def calculate_returns(rewards, mask, gamma: float = 0.99):
    def step_fn(R, data):
        reward, mask = data
        R = reward + R * mask * gamma
        return R, R

    _, returns = jax.lax.scan(step_fn, 0.0, (rewards, mask), reverse=True)
    return returns


def run_episode(model: Agent, env, env_params, rng, steps: int = 500) -> dict:
    # model.eval()
    rng, rng_reset = jax.random.split(rng)
    obs, state = env.reset(rng_reset, env_params)

    def step_fn(carry, _):
        obs, state, rng, done = carry
        rng, action_selection_rng = jax.random.split(rng)
        rng, step_rng = jax.random.split(rng)
        action = model.select_categorical_action(obs, action_selection_rng)
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
    _, trajectories = jax.lax.scan(step_fn, init_carry, length=steps)
    return trajectories


def save_model(model: nnx.Module, path: str):
    try:
        checkpointer_path = Path(path).absolute()
        checkpointer_path.parent.mkdir(parents=True, exist_ok=True)
        checkpointer_path = ocp.test_utils.erase_and_create_empty(checkpointer_path)

        _, state = nnx.split(model)
        checkpointer.save(checkpointer_path / "state", state)
        return checkpointer_path
    except Exception as e:
        e.add_note(f"Got error while saving model to path: {path}")
        return None


def load_model(model: nnx.Module, path: str):
    try:
        abstract_model = nnx.eval_shape(lambda: model)
        graph, state = nnx.split(abstract_model)
        state = checkpointer.restore(Path(path).absolute() / "state", state)
        return nnx.merge(graph, state)
    except Exception as e:
        e.add_note(f"Got error while loading model from path: {path}")
        return None
