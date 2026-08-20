from model import Agent
import gymnax
from flax import nnx
from train import train
import jax
from matplotlib import pyplot as plt
import os

env, env_params = gymnax.make("CartPole-v1")

val = 9999
ite = 0
# for i in range(2, 3):
agent = Agent(
    env.observation_space(env_params).shape[0],
    env.action_space(env_params).n,
    nnx.Rngs(2),
)
# agent.train()
trained_agent, rewards, episodes = train(
    agent,
    env,
    env_params,
    jax.random.PRNGKey(2),
    learning_rate=1e-2,
    gamma=0.99,
    num_parallel_agents=10,
)
# if len(episodes) <= val:
#     val = len(episodes)
#     ite = i
plt.plot(episodes, rewards)
plt.savefig(os.path.join(os.path.dirname(__file__), "Reward_Over_Time.png"))
plt.show()
# print(f"At Key {ite}, episodes used are {val}")
