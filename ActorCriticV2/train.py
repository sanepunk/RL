from model import ActorCriticNetwork
from gymnax import make
import numpy as np
import jax
import optax
from flax import nnx
from tqdm import tqdm
from utils import actor_critic_update

env, env_params = make("CartPole-v1")
rngs = jax.random.PRNGKey(22)

agent = ActorCriticNetwork(
    obs_dim=env.observation_space(env_params).shape[0],
    action_space=env.action_space(env_params).n,
    rngs=nnx.Rngs(22),
)

optimizer = nnx.Optimizer(
    agent, optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(1e-4)), wrt=nnx.Param
)


def train(model: ActorCriticNetwork, optimizer: nnx.Optimizer):
    rng = jax.random.PRNGKey(0)
    greedy_epsilon = 0.3
    episodic_rewards = []
    with tqdm(range(1000)) as pbar:
        for i in pbar:
            episodic_reward = 0
            if i % 100 == 0:
                greedy_epsilon = min(greedy_epsilon + 0.05, 0.91)
            rng, reset_rng, action_rng = jax.random.split(rng, 3)
            obs, state = env.reset(reset_rng, env_params)
            # action = model.epsilon_greedy_strategy(obs, action_rng, greedy_epsilon)

            while True:
                rng, next_ac, step_key, ac = jax.random.split(rng, 4)
                action = model.epsilon_greedy_strategy(obs, ac, greedy_epsilon)
                next_obs, next_state, reward, done, _ = env.step(
                    step_key, state, action, env_params
                )
                next_action = model.epsilon_greedy_strategy(
                    next_obs, next_ac, greedy_epsilon
                )
                actor_critic_update(
                    model,
                    optimizer,
                    action,
                    obs,
                    next_action,
                    next_obs,
                    reward,
                    done,
                    0.98,
                )
                obs, state = next_obs, next_state
                episodic_reward += reward
                if done:
                    break
            episodic_rewards.append(episodic_reward)
            pbar.set_description(f"{episodic_reward}")
            if episodic_reward >= 490:
                break
    import matplotlib.pyplot as plt

    plt.plot(np.arange(len(episodic_rewards)), episodic_rewards)
    plt.show()


train(agent, optimizer)
