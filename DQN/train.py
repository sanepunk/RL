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
target_net = nnx.clone(agent)


optimizer = optax.adam(learning_rate=3e-3)


optimizer = nnx.Optimizer(agent, optimizer, wrt=nnx.Param)


# def train(model: ActionValueNet, optimizer):
#     rng = jax.random.PRNGKey(22)
#     epsilon = 0.3
#     episodic_rewards = []
#     with tqdm(range(1000)) as pbar:
#         for i in pbar:
#             if i > 0 and i % 100 == 0:
#                 epsilon += 0.05
#             rng, reset_rng = jax.random.split(rng)
#             obs, state = env.reset(reset_rng, env_params)
#             episodic_reward = 0.
#             while True:
#                 rng, curr_ac, next_ac, step_key = jax.random.split(rng, 4)
#                 action = model.epsilon_greedy_strategy(obs, epsilon=epsilon, rngs=curr_ac)
#                 next_obs, next_state, reward, done, _ = env.step(step_key, state, action, env_params)
#                 next_action = model.epsilon_greedy_strategy(next_obs, epsilon=epsilon, rngs=next_ac)
#                 update(model, obs, next_obs, reward, action, next_action, done, 0.99, optimizer)
#                 obs, state = next_obs, next_state
#                 episodic_reward += reward
#                 if done:
#                     break
#             episodic_rewards.append(episodic_reward)
#             pbar.set_description(f"{episodic_reward}")
#             if episodic_reward > 490:
#                 break

#     from matplotlib import pyplot as plt
#     plt.plot(np.arange(len(episodic_rewards)), episodic_rewards)
#     plt.show()

# train(agent, optimizer)
#
#


def train(
    model: ActionValueNet, target_network: ActionValueNet, optimizer, sync_time=30
):
    rng = jax.random.PRNGKey(22)
    greedy_prob = 0.3  # prob of acting greedily; rename matches epsilon_greedy_strategy's actual semantics
    episodic_rewards = []
    with tqdm(range(1000)) as pbar:
        for i in pbar:
            if i > 0 and i % 100 == 0:
                greedy_prob = min(
                    greedy_prob + 0.05, 0.98
                )  # cap so there's always a little exploration

            rng, reset_rng = jax.random.split(rng, 2)
            obs, state = env.reset(reset_rng, env_params)

            episodic_reward = 0.0
            while True:
                rng, action_rng, step_key = jax.random.split(rng, 3)
                action = model.epsilon_greedy_strategy(
                    obs, epsilon=greedy_prob, rngs=action_rng
                )
                next_obs, next_state, reward, done, _ = env.step(
                    step_key, state, action, env_params
                )
                # next_action = model.epsilon_greedy_strategy(next_obs, epsilon=greedy_prob, rngs=next_ac)

                update(
                    model,
                    target_net,
                    obs,
                    next_obs,
                    reward,
                    action,
                    done,
                    0.99,
                    optimizer,
                )

                obs, state = (
                    next_obs,
                    next_state,
                )  # carry the action forward — this is the fix
                episodic_reward += reward
                if done:
                    break

            episodic_rewards.append(episodic_reward)
            pbar.set_description(f"{episodic_reward}")
            if episodic_reward > 490:
                break
            if i % sync_time == 0:
                nnx.update(target_net, nnx.state(model))

    plt.plot(np.arange(len(episodic_rewards)), episodic_rewards)
    plt.show()


train(agent, target_net, optimizer)
