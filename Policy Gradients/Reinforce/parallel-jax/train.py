from flax.nnx.filterlib import PathIn
from model import Agent
from flax import nnx
from jax import numpy as jnp
from utils import calculate_returns, run_episode
from optimizer import return_optimizer
import jax
import gymnax
from tqdm.auto import tqdm
# env, env_params = gymnax.make("CartPole-v1")

def train(model: Agent, env, env_params, rng, gamma: float = 0.99, episodes: int = 1000, learning_rate: float = 1e-4, num_parallel_agents: int = 10):
    optimizer = return_optimizer(model, 'adam', learning_rate=learning_rate, gradNorm=1.)
    vmap_run_episode = jax.vmap(
        lambda model, key: run_episode(
            model, env, env_params, rng=key, steps=500
        ),
        in_axes=(None, 0)
    )
    
    @nnx.jit
    def update(model: Agent, optimizer, rng_batch):
        # Collect trajectories in eval mode to use running averages
        model.eval()
        trajectories = vmap_run_episode(model, rng_batch)
        returns = jax.vmap(calculate_returns, in_axes=(0, 0, None))(trajectories.get('reward'), trajectories.get("valid_mask"), gamma)
        
        mask_sum = jnp.sum(trajectories.get("valid_mask")) + 1e-8
        mean_return = jnp.sum(returns * trajectories.get("valid_mask")) / mask_sum
        std_return = jnp.sqrt(jnp.sum(((returns - mean_return) ** 2) * trajectories.get("valid_mask")) / mask_sum) + 1e-8
        returns = (returns - mean_return) / std_return
        
        def loss(model: Agent, obs, action):
            batch_size, steps_size = obs.shape[:2]
            # Keep model in eval mode inside gradient computation
            # This prevents BatchNorm from trying to update running stats inside vmap+grad
            logits = nnx.vmap(model)(obs.reshape(-1, obs.shape[-1]))
            # logits = model(obs.reshape(-1, obs.shape[-1]))
            log_softmax_logits = nnx.log_softmax(logits).reshape(batch_size, steps_size, -1)
            logits_along_action = jnp.take_along_axis(log_softmax_logits, action[:, :, None], axis = 2).squeeze(2)
            returns_loss = jnp.sum(logits_along_action * returns * trajectories.get('valid_mask'))
            return -returns_loss / mask_sum# - mask_sum
            
        # Compute gradients only for Param variables (not BatchStat)
        # This allows BatchNorm scale/bias to be trained without trying to update running stats
        model.train()
        diff_state = nnx.DiffState(0, nnx.Param)
        loss, grad = nnx.value_and_grad(loss, argnums=0)(model, trajectories.get('obs'), trajectories.get('action'))
        optimizer.update(model, grad)
        
        avg_reward = jnp.mean(jnp.sum(trajectories.get('reward') * trajectories.get('valid_mask'), axis = 1))
        return loss, avg_reward
    rewards = []
    episodes_list = []
    patience = 0
    with tqdm(range(episodes)) as pbar:
        for episode in pbar:
            rng, batch_key = jax.random.split(rng)
            batch_keys = jax.random.split(batch_key, num=num_parallel_agents)
            loss, avg_reward = update(model, optimizer, batch_keys)
            rewards.append(avg_reward)
            episodes_list.append(episode)
            pbar.set_description(f"Episode: {episode}, reward: {avg_reward:.3f}")
            if avg_reward >= 499:
                if patience <= 0:
                    return model, rewards, episodes_list
                else:
                    patience -= 1
    return model, rewards, episodes_list
            