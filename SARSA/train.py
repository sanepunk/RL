import numpy as np
from matplotlib import pyplot as plt
from model import ActionValueNet
import jax
from flax import nnx
from tqdm import tqdm
from utils import get_env, update
import optax

env, env_params = get_env()
agent = ActionValueNet(
    env.observation_space(env_params).shape[0],
    env.action_space(env_params).n,
    nnx.Rngs(44),
)

optax_optim = optax.adam(learning_rate=3e-3)

nnx_optimizer = nnx.Optimizer(agent, optax_optim, wrt=nnx.Param)


def train(model: ActionValueNet, optimizer):
    rng = jax.random.PRNGKey(22)
    greedy_prob = 0.3
    episodic_rewards = []
    with tqdm(range(1000)) as pbar:
        for i in pbar:
            if i > 0 and i % 100 == 0:
                greedy_prob = min(greedy_prob + 0.05, 0.98)

            rng, reset_rng, action_rng = jax.random.split(rng, 3)
            obs, state = env.reset(reset_rng, env_params)
            action = model.epsilon_greedy_strategy(
                obs, epsilon=greedy_prob, rngs=action_rng
            )

            episodic_reward = 0.0
            while True:
                rng, next_ac, step_key = jax.random.split(rng, 3)
                next_obs, next_state, reward, done, _ = env.step(
                    step_key, state, action, env_params
                )
                next_action = model.epsilon_greedy_strategy(
                    next_obs, epsilon=greedy_prob, rngs=next_ac
                )

                update(
                    model,
                    obs,
                    next_obs,
                    reward,
                    action,
                    next_action,
                    done,
                    0.99,
                    optimizer,
                )

                obs, state, action = (
                    next_obs,
                    next_state,
                    next_action,
                )  # carry the action forward — this is the fix
                episodic_reward += reward
                if done:
                    break

            episodic_rewards.append(episodic_reward)
            pbar.set_description(f"{episodic_reward}")
            if episodic_reward > 490:
                break

    plt.plot(np.arange(len(episodic_rewards)), episodic_rewards)
    plt.show()


train(agent, nnx_optimizer)
