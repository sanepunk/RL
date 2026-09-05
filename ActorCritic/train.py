from model import ActorCriticNetwork
from gymnax import make
import numpy as np
import jax
import optax
from flax import nnx
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
        last_obs, trajectories = vmapped_run_episode(
            model, jax.random.split(episode_run, 50)
        )

        actor_critic_update(
            model,
            optimizer,
            trajectories.get("reward"),
            trajectories.get("action"),
            trajectories.get("obs"),
            trajectories.get("valid_mask"),
            last_obs,
        )
        # print(trajectories.get("reward").shape)
        # exit(0)
        # rewards = jax.numpy.array(trajectories.get("reward"))
        # print("rewards.shape =", rewards.shape)
        # exit()
        episodic_reward = jax.numpy.mean(
            jax.numpy.sum(trajectories.get("reward"), axis=-1)
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
