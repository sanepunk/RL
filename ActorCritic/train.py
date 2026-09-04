from model import ActorCriticNetwork
from gymnax import make
import numpy as np
import jax
import optax
from flax import nnx
from tqdm import tqdm
from utils import actor_critic_update, run_episode

env, env_params = make("CartPole-v1")
rngs = jax.random.PRNGKey(22)

agent = ActorCriticNetwork(
    obs_dim=env.observation_space(env_params).shape[0],
    action_space=env.action_space(env_params).n,
    rngs=nnx.Rngs(22),
)

optimizer = nnx.Optimizer(
    agent, optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4)), wrt=nnx.Param
)


def train(model: ActorCriticNetwork, optimizer: nnx.Optimizer):
    rng = jax.random.PRNGKey(0)
    greedy_epsilon = 0.3
    episodic_rewards = []
    with tqdm(range(1000)) as pbar:
        for i in pbar:
            arr_obs = []
            arr_action = []
            arr_rewards = []
            episodic_reward = 0
            valid_mask = []
            if i % 100 == 0:
                greedy_epsilon = min(greedy_epsilon + 0.05, 0.91)
            rng, reset_rng, action_rng = jax.random.split(rng, 3)
            obs, state = env.reset(reset_rng, env_params)
            # action = model.epsilon_greedy_strategy(obs, action_rng, greedy_epsilon)

            while True:
                rng, next_ac, step_key, ac = jax.random.split(rng, 4)
                action, logits = model.epsilon_greedy_strategy(obs, ac, greedy_epsilon)
                next_obs, next_state, reward, done, _ = env.step(
                    step_key, state, action, env_params
                )
                arr_action.append(action)
                arr_obs.append(obs)
                arr_rewards.append(reward)
                obs, state = next_obs, next_state
                episodic_reward += reward
                valid_mask.append(1 - done)
                if done:
                    break
            actor_critic_update(
                model,
                optimizer,
                arr_rewards,
                arr_action,
                arr_obs,
                # valid_mask
            )
            episodic_rewards.append(episodic_reward)
            pbar.set_description(f"{episodic_reward}")
            # if episodic_reward >=490:
            #     break
    import matplotlib.pyplot as plt

    plt.plot(np.arange(len(episodic_rewards)), episodic_rewards)
    plt.show()


def train_v2(model: ActorCriticNetwork, optimizer: nnx.Optimizer):
    rng = jax.random.PRNGKey(0)
    episodic_rewards = []
    with tqdm(range(1000)) as pbar:
        for i in pbar:
            rng, episode_run = jax.random.split(rng)
            trajectories = run_episode(model, env, env_params, episode_run)
            actor_critic_update(
                model,
                optimizer,
                trajectories.get("reward"),
                trajectories.get("action"),
                trajectories.get("obs"),
                trajectories.get("valid_mask"),
            )
            episodic_reward = jax.numpy.sum(trajectories.get("reward"))
            episodic_rewards.append(episodic_reward)
            pbar.set_description(f"{episodic_reward}")
            # if episodic_reward >=490:
            #     break
    import matplotlib.pyplot as plt

    plt.plot(np.arange(len(episodic_rewards)), episodic_rewards)
    plt.show()


def train_v3(model: ActorCriticNetwork, optimizer: nnx.Optimizer):
    rng = jax.random.PRNGKey(0)

    episodic_rewards = jax.numpy.zeros(1000)
    vmapped_run_episode = nnx.vmap(
        lambda model, rng: run_episode(model, env, env_params, rng, 500),
        in_axes=(None, 0),
        out_axes=0,
    )

    def train_step(i, carry):
        model, optimizer, rng, episodic_rewards = carry

        rng, episode_run = jax.random.split(rng)

        # trajectories = run_episode(
        #     model,
        #     env,
        #     env_params,
        #     episode_run
        # )
        trajectories = vmapped_run_episode(model, jax.random.split(episode_run, 50))

        actor_critic_update(
            model,
            optimizer,
            trajectories.get("reward"),
            trajectories.get("action"),
            trajectories.get("obs"),
            trajectories.get("valid_mask"),
        )
        # print(trajectories.get("reward").shape)
        # exit(0)

        episodic_reward = jax.numpy.sum(
            jax.numpy.mean(trajectories.get("reward"), axis=1)
        )

        episodic_rewards = episodic_rewards.at[i].set(episodic_reward)

        return model, optimizer, rng, episodic_rewards

    model, optimizer, rng, episodic_rewards = nnx.fori_loop(
        0, 1000, train_step, (model, optimizer, rng, episodic_rewards)
    )

    import matplotlib.pyplot as plt

    plt.plot(np.arange(len(episodic_rewards)), np.asarray(episodic_rewards))
    plt.show()

    return model, optimizer, episodic_rewards


train_v3(agent, optimizer)
