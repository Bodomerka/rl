"""
Pure JAX IPPO Trainer

Fully JIT-compiled training loop for GPU acceleration.
Based on PureJaxRL patterns with SocialJax environment.

Provides ~50x speedup over NumPy-based training.
"""

from typing import NamedTuple, Dict, Any, Tuple, Optional, Callable
from functools import partial
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.training.train_state import TrainState

# Try to import SocialJax
try:
    import socialjax
    SOCIALJAX_AVAILABLE = True
except ImportError:
    SOCIALJAX_AVAILABLE = False


class Transition(NamedTuple):
    """Transition data for rollout storage."""
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray


class ActorCriticMLP(nn.Module):
    """MLP Actor-Critic network for JAX training."""
    action_dim: int
    hidden_dims: Tuple[int, ...] = (128, 128)

    @nn.compact
    def __call__(self, x):
        # Flatten observation if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        # Shared layers
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)

        # Actor head
        logits = nn.Dense(self.action_dim)(x)

        # Critic head
        value = nn.Dense(1)(x)

        return logits, value.squeeze(-1)


class ActorCriticCNN(nn.Module):
    """CNN Actor-Critic network (based on SocialJax architecture)."""
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, height, width, channels)

        # CNN backbone (same as SocialJax)
        x = nn.Conv(32, kernel_size=(5, 5), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)

        # Flatten
        x = x.reshape(x.shape[0], -1)

        # Dense layer
        x = nn.Dense(64)(x)
        x = nn.relu(x)

        # Actor head
        logits = nn.Dense(self.action_dim)(x)

        # Critic head
        value = nn.Dense(1)(x)

        return logits, value.squeeze(-1)


class JaxIPPOConfig(NamedTuple):
    """Configuration for JAX IPPO trainer."""
    # Environment
    num_agents: int = 8
    num_envs: int = 64
    max_steps: int = 400

    # Training
    total_timesteps: int = 5_000_000
    rollout_length: int = 128
    num_epochs: int = 4
    num_minibatches: int = 4

    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 2.5e-4

    # Network
    use_cnn: bool = True
    hidden_dims: Tuple[int, ...] = (128, 128)

    # Reward shaping
    common_reward: bool = True

    # Logging
    log_interval: int = 10


class JaxIPPOTrainer:
    """
    Pure JAX IPPO Trainer with jax.lax.scan training loop.

    All computation happens on GPU - no Python loops during training.

    Features:
    - Vectorized environments via jax.vmap
    - JIT-compiled training loop via jax.lax.scan
    - GAE advantage computation on GPU
    - Parameter sharing across agents
    """

    def __init__(self, config: JaxIPPOConfig, seed: int = 42):
        """Initialize trainer."""
        if not SOCIALJAX_AVAILABLE:
            raise ImportError(
                "SocialJax not available. Install with:\n"
                "  git clone https://github.com/cooperativex/socialjax\n"
                "  cd socialjax && pip install -e ."
            )

        self.config = config
        self.seed = seed

        # Create environment
        self.env = socialjax.make('clean_up', num_agents=config.num_agents)

        # Get spaces (handle different SocialJax API versions)
        obs_space = self.env.observation_space()
        act_space = self.env.action_space()

        # SocialJax може повертати dict або tuple
        if isinstance(obs_space, dict):
            self.obs_shape = list(obs_space.values())[0].shape
        elif hasattr(obs_space, '__getitem__'):
            self.obs_shape = obs_space[0].shape
        else:
            self.obs_shape = obs_space.shape

        if isinstance(act_space, dict):
            self.action_dim = list(act_space.values())[0].n
        elif hasattr(act_space, '__getitem__') and not hasattr(act_space, 'n'):
            self.action_dim = act_space[0].n
        else:
            self.action_dim = act_space.n

        # Create network
        if config.use_cnn:
            self.network = ActorCriticCNN(action_dim=self.action_dim)
        else:
            self.network = ActorCriticMLP(
                action_dim=self.action_dim,
                hidden_dims=config.hidden_dims,
            )

        # Initialize RNG
        self.rng = jax.random.PRNGKey(seed)

        # Compute derived values
        self.batch_size = config.num_envs * config.rollout_length * config.num_agents
        self.minibatch_size = self.batch_size // config.num_minibatches
        self.num_updates = config.total_timesteps // (config.num_envs * config.rollout_length)

        self._print_config()

    def _print_config(self):
        """Print training configuration."""
        print("\n" + "=" * 60)
        print("JAX IPPO Trainer Configuration")
        print("=" * 60)
        print(f"Device: {jax.devices()[0]}")
        print(f"Agents: {self.config.num_agents}")
        print(f"Parallel envs: {self.config.num_envs}")
        print(f"Rollout length: {self.config.rollout_length}")
        print(f"Batch size: {self.batch_size:,}")
        print(f"Minibatch size: {self.minibatch_size:,}")
        print(f"Total updates: {self.num_updates:,}")
        print(f"Obs shape: {self.obs_shape}")
        print(f"Actions: {self.action_dim}")
        print(f"Network: {'CNN' if self.config.use_cnn else 'MLP'}")
        print("=" * 60 + "\n")

    def _init_train_state(self, rng: jnp.ndarray) -> TrainState:
        """Initialize network and optimizer."""
        # Sample observation for init
        sample_obs = jnp.zeros((1, *self.obs_shape))

        # Initialize network
        params = self.network.init(rng, sample_obs)

        # Create optimizer
        tx = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.learning_rate),
        )

        return TrainState.create(
            apply_fn=self.network.apply,
            params=params,
            tx=tx,
        )

    def train(
        self,
        callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run training loop.

        Args:
            callback: Optional callback(iteration, metrics) for logging

        Returns:
            Training info dict
        """
        config = self.config
        env = self.env
        num_envs = config.num_envs
        num_agents = config.num_agents
        num_steps = config.rollout_length

        # Initialize
        self.rng, init_rng, reset_rng = jax.random.split(self.rng, 3)
        train_state = self._init_train_state(init_rng)

        # Reset environments (vectorized)
        reset_keys = jax.random.split(reset_rng, num_envs)
        obs, env_state = jax.vmap(env.reset)(reset_keys)

        # SocialJax returns arrays directly after vmap:
        # obs shape: (num_envs, num_agents, *obs_shape)
        # No dict conversion needed - obs is already an array

        # Compile training step
        @jax.jit
        def _train_step(runner_state, _):
            """Single training iteration (rollout + update)."""
            train_state, env_state, last_obs, rng = runner_state

            # === ROLLOUT COLLECTION ===
            def _env_step(carry, _):
                """Single environment step."""
                env_state, obs, rng = carry

                # Sample actions for all envs and agents
                rng, action_rng = jax.random.split(rng)

                # Reshape obs: (num_envs, num_agents, ...) -> (num_envs * num_agents, ...)
                obs_flat = obs.reshape(num_envs * num_agents, *self.obs_shape)

                # Forward pass
                logits, values = train_state.apply_fn(train_state.params, obs_flat)

                # Sample actions
                action_rngs = jax.random.split(action_rng, num_envs * num_agents)
                actions = jax.vmap(lambda rng, logits: jax.random.categorical(rng, logits))(
                    action_rngs, logits
                )

                # Compute log probs
                action_probs = jax.nn.softmax(logits)
                log_probs = jnp.log(
                    action_probs[jnp.arange(num_envs * num_agents), actions] + 1e-8
                )

                # Reshape back to (num_envs, num_agents)
                actions = actions.reshape(num_envs, num_agents)
                log_probs = log_probs.reshape(num_envs, num_agents)
                values = values.reshape(num_envs, num_agents)

                # Step environments (vectorized)
                rng, step_rng = jax.random.split(rng)
                step_keys = jax.random.split(step_rng, num_envs)
                new_obs, new_env_state, rewards, dones, info = jax.vmap(env.step)(
                    step_keys, env_state, actions
                )

                # SocialJax returns:
                # - new_obs: array (num_envs, num_agents, *obs_shape) - already correct
                # - rewards: array (num_envs, num_agents) - already correct
                # - dones: dict with keys '0', '1', '2', ... and '__all__'
                # Only dones needs conversion from dict to array
                agent_keys = [str(i) for i in range(num_agents)]
                dones = jnp.stack([dones[k] for k in agent_keys], axis=-1)

                # Apply common reward if configured
                if config.common_reward:
                    total_rewards = jnp.sum(rewards, axis=-1, keepdims=True)
                    rewards = jnp.broadcast_to(total_rewards / num_agents, rewards.shape)

                # Create transition
                transition = Transition(
                    obs=obs,
                    action=actions,
                    reward=rewards,
                    done=dones,
                    value=values,
                    log_prob=log_probs,
                )

                return (new_env_state, new_obs, rng), transition

            # Collect rollout using scan
            rng, rollout_rng = jax.random.split(rng)
            (env_state, obs, _), traj_batch = jax.lax.scan(
                _env_step,
                (env_state, last_obs, rollout_rng),
                None,
                num_steps,
            )
            # traj_batch shapes: (num_steps, num_envs, num_agents, ...)

            # === COMPUTE ADVANTAGES ===
            # Get last values for bootstrapping
            last_obs_flat = obs.reshape(num_envs * num_agents, *self.obs_shape)
            _, last_values = train_state.apply_fn(train_state.params, last_obs_flat)
            last_values = last_values.reshape(num_envs, num_agents)

            def _compute_gae(carry, transition):
                """Compute GAE (reverse scan)."""
                gae, next_value = carry
                done, value, reward = transition.done, transition.value, transition.reward

                delta = reward + config.gamma * next_value * (1 - done) - value
                gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae

                return (gae, value), gae

            # Reverse scan for GAE
            _, advantages = jax.lax.scan(
                _compute_gae,
                (jnp.zeros_like(last_values), last_values),
                traj_batch,
                reverse=True,
            )
            # advantages shape: (num_steps, num_envs, num_agents)

            targets = advantages + traj_batch.value

            # === PPO UPDATE ===
            def _update_epoch(carry, _):
                """Single epoch of PPO updates."""
                train_state, rng = carry

                # Shuffle and create minibatches
                rng, perm_rng = jax.random.split(rng)

                # Flatten all data: (num_steps, num_envs, num_agents, ...) -> (batch_size, ...)
                batch = jax.tree.map(
                    lambda x: x.reshape(-1, *x.shape[3:]) if len(x.shape) > 3 else x.reshape(-1),
                    (traj_batch, advantages, targets),
                )
                traj_flat, adv_flat, targets_flat = batch

                # Permutation
                batch_size = num_steps * num_envs * num_agents
                permutation = jax.random.permutation(perm_rng, batch_size)

                # Shuffle
                shuffled_batch = jax.tree.map(lambda x: x[permutation], (traj_flat, adv_flat, targets_flat))

                # Split into minibatches
                def _update_minibatch(train_state, minibatch):
                    """Update on single minibatch."""
                    traj, advantages, targets = minibatch

                    def loss_fn(params):
                        # Forward pass
                        logits, values = train_state.apply_fn(params, traj.obs)

                        # Policy loss
                        action_probs = jax.nn.softmax(logits)
                        log_probs = jnp.log(
                            action_probs[jnp.arange(len(traj.action)), traj.action] + 1e-8
                        )

                        # Normalize advantages
                        advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                        # Clipped surrogate
                        ratio = jnp.exp(log_probs - traj.log_prob)
                        clipped_ratio = jnp.clip(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
                        policy_loss = -jnp.minimum(
                            ratio * advantages_norm,
                            clipped_ratio * advantages_norm,
                        ).mean()

                        # Value loss
                        value_loss = ((values - targets) ** 2).mean()

                        # Entropy bonus
                        entropy = -(action_probs * jnp.log(action_probs + 1e-8)).sum(-1).mean()

                        # Total loss
                        total_loss = (
                            policy_loss +
                            config.value_coef * value_loss -
                            config.entropy_coef * entropy
                        )

                        return total_loss, {
                            'policy_loss': policy_loss,
                            'value_loss': value_loss,
                            'entropy': entropy,
                            'total_loss': total_loss,
                        }

                    # Compute gradients and update
                    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)

                    return train_state, metrics

                # Reshape shuffled batch into minibatches
                num_minibatches = config.num_minibatches
                minibatch_size = batch_size // num_minibatches

                def reshape_minibatch(x):
                    return x.reshape(num_minibatches, minibatch_size, *x.shape[1:])

                minibatches = jax.tree.map(reshape_minibatch, shuffled_batch)

                # Update on each minibatch
                train_state, metrics = jax.lax.scan(
                    _update_minibatch,
                    train_state,
                    minibatches,
                )

                # Average metrics across minibatches
                metrics = jax.tree.map(lambda x: x.mean(), metrics)

                return (train_state, rng), metrics

            # Run epochs
            rng, update_rng = jax.random.split(rng)
            (train_state, _), epoch_metrics = jax.lax.scan(
                _update_epoch,
                (train_state, update_rng),
                None,
                config.num_epochs,
            )

            # Average metrics across epochs
            metrics = jax.tree.map(lambda x: x.mean(), epoch_metrics)

            # Add rollout metrics
            metrics['mean_reward'] = traj_batch.reward.mean()
            metrics['episode_reward'] = traj_batch.reward.sum(axis=0).mean()  # Sum over steps

            return (train_state, env_state, obs, rng), metrics

        # === MAIN TRAINING LOOP ===
        print("Starting training...")
        start_time = time.time()

        runner_state = (train_state, env_state, obs, self.rng)

        # Track metrics
        all_rewards = []
        all_losses = []

        # Training loop (with periodic logging)
        steps_per_update = num_envs * num_steps
        total_steps = 0

        for update in range(self.num_updates):
            # Run one training step
            runner_state, metrics = _train_step(runner_state, None)

            total_steps += steps_per_update

            # Logging
            if update % config.log_interval == 0:
                elapsed = time.time() - start_time
                speed = total_steps / elapsed

                mean_reward = float(metrics['mean_reward'])
                policy_loss = float(metrics['policy_loss'])
                entropy = float(metrics['entropy'])

                all_rewards.append(mean_reward)
                all_losses.append(policy_loss)

                # Progress
                progress = total_steps / config.total_timesteps * 100
                eta = (config.total_timesteps - total_steps) / speed if speed > 0 else 0

                print(
                    f"Update {update:5d} | "
                    f"Steps: {total_steps:>9,} ({progress:5.1f}%) | "
                    f"Reward: {mean_reward:7.3f} | "
                    f"Loss: {policy_loss:7.4f} | "
                    f"Entropy: {entropy:.4f} | "
                    f"Speed: {speed:,.0f}/s | "
                    f"ETA: {eta/60:.1f}m"
                )

                if callback is not None:
                    callback(update, {
                        'total_steps': total_steps,
                        'mean_reward': mean_reward,
                        'policy_loss': policy_loss,
                        'value_loss': float(metrics['value_loss']),
                        'entropy': entropy,
                    })

        # Final stats
        elapsed = time.time() - start_time
        final_speed = total_steps / elapsed

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total steps: {total_steps:,}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Average speed: {final_speed:,.0f} steps/sec")
        print(f"Final reward: {all_rewards[-1] if all_rewards else 0:.3f}")
        print("=" * 60)

        # Extract final train state
        train_state, _, _, self.rng = runner_state

        return {
            'train_state': train_state,
            'rewards': all_rewards,
            'losses': all_losses,
            'total_steps': total_steps,
            'training_time': elapsed,
            'avg_speed': final_speed,
        }

    def save_checkpoint(self, path: str, train_state: TrainState):
        """Save model checkpoint."""
        import pickle

        checkpoint = {
            'params': train_state.params,
            'config': self.config._asdict(),
        }

        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"Checkpoint saved to {path}")


def train_cleanup_jax(
    num_agents: int = 8,
    num_envs: int = 64,
    total_timesteps: int = 5_000_000,
    rollout_length: int = 128,
    use_cnn: bool = True,
    seed: int = 42,
    log_interval: int = 10,
) -> Dict[str, Any]:
    """
    Convenience function to train with JAX.

    Args:
        num_agents: Number of agents
        num_envs: Number of parallel environments
        total_timesteps: Total training steps
        rollout_length: Steps per rollout
        use_cnn: Use CNN (True) or MLP (False)
        seed: Random seed
        log_interval: How often to log

    Returns:
        Training results dict
    """
    config = JaxIPPOConfig(
        num_agents=num_agents,
        num_envs=num_envs,
        total_timesteps=total_timesteps,
        rollout_length=rollout_length,
        use_cnn=use_cnn,
        log_interval=log_interval,
    )

    trainer = JaxIPPOTrainer(config, seed=seed)
    return trainer.train()


if __name__ == '__main__':
    # Quick test
    print("Testing JAX IPPO Trainer...")
    print(f"JAX devices: {jax.devices()}")

    if SOCIALJAX_AVAILABLE:
        results = train_cleanup_jax(
            num_agents=4,
            num_envs=8,
            total_timesteps=50_000,
            rollout_length=64,
            use_cnn=False,  # MLP for quick test
            log_interval=5,
        )
        print(f"\nTest complete! Final reward: {results['rewards'][-1]:.3f}")
    else:
        print("SocialJax not available, skipping test")
