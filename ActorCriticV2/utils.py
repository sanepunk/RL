from model import ActorCriticNetwork
from flax import nnx
import jax


def critic_loss(
    model: ActorCriticNetwork, action, obs, next_action, next_obs, reward, done, gamma
):
    curr_state_value = model.value(obs)[action]
    next_state_value = model.value(next_obs)[next_action]

    target_value = jax.lax.stop_gradient(reward + gamma * next_state_value * (1 - done))
    return jax.lax.square(
        target_value - curr_state_value
    ), target_value - curr_state_value


def actor_loss(model: ActorCriticNetwork, action, obs, advantage_value):
    policy_value = nnx.log_softmax(model.policy(obs))[action]
    return -jax.lax.stop_gradient(advantage_value) * policy_value


# def actor_critic_update(model: ActorCriticNetwork, optimizer: nnx.Optimizer, action, obs, next_action, next_obs, reward, done, gamma):
#     grad, advantage_value = nnx.grad(critic_loss, argnums=0, has_aux=True)(model, action, obs, next_action, next_obs, reward, done, gamma)
#     optimizer.update(model, grad)
#     # return advantage_value
#     grad = nnx.grad(actor_loss, argnums=0)(model, action, obs, advantage_value)
#     optimizer.update(model, grad)


def actor_critic_loss(
    model: ActorCriticNetwork, action, obs, next_action, next_obs, reward, done, gamma
):
    curr_state_value = model.value(obs)[action]
    next_state_value = model.value(next_obs)[next_action]

    target_value = jax.lax.stop_gradient(reward + gamma * next_state_value * (1 - done))
    critic_loss_val, advantage_value = (
        jax.lax.square(target_value - curr_state_value),
        target_value - curr_state_value,
    )

    probs = nnx.softmax(model.policy(obs))
    log_probs = nnx.log_softmax(model.policy(obs))
    entropy = -jax.numpy.sum(probs * log_probs)
    actor_loss_val = (
        -jax.lax.stop_gradient(advantage_value) * log_probs[action] - 0.01 * entropy
    )

    return critic_loss_val + actor_loss_val


def actor_critic_update(
    model, optimizer, action, obs, next_action, next_obs, reward, done, gamma
):
    grad = nnx.grad(actor_critic_loss, argnums=0)(
        model, action, obs, next_action, next_obs, reward, done, gamma
    )

    optimizer.update(model, grad)
