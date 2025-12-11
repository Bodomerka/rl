"""
Independent PPO Trainer

Implements the IPPO training algorithm for multi-agent environments.
Each agent uses its own actor-critic network (with optional parameter sharing).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Callable, Any, Tuple, List
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import copy
import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from src.algorithms.ippo.network import ActorCritic, create_actor_critic, init_network
from src.algorithms.ippo.buffer import RolloutBuffer
from src.algorithms.ippo.social_influence import SocialInfluenceReward, create_social_influence
from src.algorithms.ippo.future_prediction import CollectiveFuturePrediction, create_future_prediction
from src.utils.logger import TrainingLogger


@dataclass
class IPPOConfig:
    """Configuration for IPPO trainer."""
    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Training
    learning_rate: float = 2.5e-4
    num_epochs: int = 4
    num_minibatches: int = 4
    rollout_length: int = 128
    total_timesteps: int = 10_000_000

    # Parallelization
    num_envs: int = 1  # Number of parallel environments

    # Network
    hidden_dims: tuple = (64, 64)
    use_cnn: bool = True
    use_rnn: bool = False

    # Parameter sharing
    parameter_sharing: bool = True

    # Normalization
    normalize_advantages: bool = True

    # Social Influence (Jaques et al., 2019)
    use_social_influence: bool = False
    influence_weight: float = 0.1  # λ coefficient for influence reward
    moa_learning_rate: float = 1e-3  # Learning rate for Model of Other Agents

    # Collective Future Prediction (intrinsic motivation)
    use_future_prediction: bool = False
    future_prediction_weight: float = 0.5  # Weight of intrinsic reward
    prediction_horizon: int = 50  # How many steps ahead to predict


class IPPOTrainer:
    """
    Independent PPO Trainer for multi-agent environments.

    Features:
    - Optional parameter sharing across agents
    - GAE advantage estimation
    - Clipped surrogate objective
    - Entropy bonus for exploration
    """

    def __init__(
        self,
        env,
        config: IPPOConfig,
        seed: int = 42,
        env_creator: Optional[Callable] = None,
    ):
        """
        Initialize IPPO trainer.

        Args:
            env: Multi-agent environment
            config: Training configuration
            seed: Random seed
            env_creator: Optional function to create new environments for parallelization
        """
        self.env = env
        self.config = config
        self.seed = seed
        self.num_envs = config.num_envs

        self.num_agents = env.num_agents
        self.action_dim = env.action_space_size
        self.obs_shape = env.observation_shape

        # Random keys
        self.rng = jax.random.PRNGKey(seed)
        self.np_rng = np.random.default_rng(seed)

        # Create parallel environments
        if self.num_envs > 1:
            if env_creator is not None:
                self.envs = [env] + [env_creator(seed=seed + i + 1) for i in range(self.num_envs - 1)]
            else:
                # Deep copy environments with different seeds
                self.envs = [env]
                for i in range(self.num_envs - 1):
                    env_copy = copy.deepcopy(env)
                    env_copy.rng = np.random.default_rng(seed + i + 1)
                    self.envs.append(env_copy)
            print(f"Created {self.num_envs} parallel environments")
        else:
            self.envs = [env]

        # Initialize networks and optimizers
        self._init_networks()

        # Initialize buffer (larger to accommodate all envs)
        self.buffer = RolloutBuffer(
            num_agents=self.num_agents,
            rollout_length=config.rollout_length * self.num_envs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        # Initialize social influence module
        self.social_influence = None
        if config.use_social_influence:
            obs_dim = int(np.prod(self.obs_shape))
            self.social_influence = create_social_influence(
                num_agents=self.num_agents,
                num_actions=self.action_dim,
                obs_shape=self.obs_shape,
                influence_weight=config.influence_weight,
                seed=seed,
            )
            print(f"Social Influence enabled with weight λ={config.influence_weight}")

        # Initialize future prediction module
        self.future_prediction = None
        if config.use_future_prediction:
            self.future_prediction = create_future_prediction(
                obs_shape=self.obs_shape,
                num_actions=self.action_dim,
                num_agents=self.num_agents,
                intrinsic_weight=config.future_prediction_weight,
                prediction_horizon=config.prediction_horizon,
                device='cpu',  # Use CPU for compatibility
            )
            print(f"Future Prediction enabled with weight={config.future_prediction_weight}")

        # Metrics tracking
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []

        # Persistent state across rollouts
        self._env_obs = None
        self._env_episode_rewards = None
        self._env_episode_lengths = None

        # Social influence data collection
        self._si_obs_buffer = []
        self._si_actions_buffer = []

    def _init_networks(self):
        """Initialize actor-critic networks."""
        # Create network architecture
        self.network = create_actor_critic(
            action_dim=self.action_dim,
            hidden_dims=self.config.hidden_dims,
            use_cnn=self.config.use_cnn,
            use_rnn=self.config.use_rnn,
        )

        # Sample observation for initialization
        sample_obs = np.zeros((1, *self.obs_shape), dtype=np.float32)

        # Initialize parameters
        self.rng, init_rng = jax.random.split(self.rng)
        params = init_network(self.network, init_rng, sample_obs)

        # Create optimizer
        tx = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.learning_rate),
        )

        if self.config.parameter_sharing:
            # Single set of parameters shared across agents
            self.train_state = train_state.TrainState.create(
                apply_fn=self.network.apply,
                params=params,
                tx=tx,
            )
        else:
            # Separate parameters per agent
            self.train_states = {}
            for agent_id in range(self.num_agents):
                self.rng, init_rng = jax.random.split(self.rng)
                agent_params = init_network(self.network, init_rng, sample_obs)
                self.train_states[agent_id] = train_state.TrainState.create(
                    apply_fn=self.network.apply,
                    params=agent_params,
                    tx=tx,
                )

    def get_action(
        self,
        agent_id: int,
        observation: np.ndarray,
    ) -> Tuple[int, float, float]:
        """
        Get action for a single agent.

        Args:
            agent_id: Agent ID
            observation: Agent's observation

        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: State value estimate
        """
        self.rng, action_rng = jax.random.split(self.rng)

        # Get parameters
        if self.config.parameter_sharing:
            params = self.train_state.params
        else:
            params = self.train_states[agent_id].params

        # Add batch dimension
        obs = observation[None, ...]

        # Forward pass
        logits, value, _ = self.network.apply(params, obs)
        logits = logits[0]
        value = value[0, 0]

        # Sample action
        action_probs = jax.nn.softmax(logits)
        action = jax.random.categorical(action_rng, logits)
        action = int(action)
        log_prob = float(jnp.log(action_probs[action] + 1e-8))

        return action, log_prob, float(value)

    def get_actions(
        self,
        observations: Dict[int, np.ndarray],
    ) -> Tuple[Dict[int, int], Dict[int, float], Dict[int, float]]:
        """
        Get actions for all agents (batched for GPU efficiency).

        Args:
            observations: Dict of observations per agent

        Returns:
            actions: Dict of actions
            log_probs: Dict of log probabilities
            values: Dict of value estimates
        """
        # Batch all observations together for efficient GPU inference
        agent_ids = list(observations.keys())
        obs_batch = np.stack([observations[i] for i in agent_ids], axis=0)

        self.rng, action_rng = jax.random.split(self.rng)

        # Get parameters (with parameter sharing, same for all agents)
        if self.config.parameter_sharing:
            params = self.train_state.params
        else:
            # For non-sharing, fall back to sequential (less common case)
            actions = {}
            log_probs = {}
            values = {}
            for agent_id, obs in observations.items():
                action, log_prob, value = self.get_action(agent_id, obs)
                actions[agent_id] = action
                log_probs[agent_id] = log_prob
                values[agent_id] = value
            return actions, log_probs, values

        # Batched forward pass (GPU-efficient)
        logits, value_preds, _ = self.network.apply(params, obs_batch)
        value_preds = value_preds.squeeze(-1)

        # Sample actions for all agents at once
        action_rngs = jax.random.split(action_rng, len(agent_ids))
        sampled_actions = jax.vmap(jax.random.categorical)(action_rngs, logits)

        # Compute log probs
        action_probs = jax.nn.softmax(logits)
        log_probs_batch = jnp.log(
            action_probs[jnp.arange(len(agent_ids)), sampled_actions] + 1e-8
        )

        # Convert to dicts
        actions = {agent_ids[i]: int(sampled_actions[i]) for i in range(len(agent_ids))}
        log_probs = {agent_ids[i]: float(log_probs_batch[i]) for i in range(len(agent_ids))}
        values = {agent_ids[i]: float(value_preds[i]) for i in range(len(agent_ids))}

        return actions, log_probs, values

    def _env_step(self, env_id: int, env, obs: Dict, actions: Dict):
        """Execute a single environment step (for parallel execution)."""
        next_obs, state, rewards, dones, infos = env.step(actions)
        return env_id, next_obs, state, rewards, dones, infos

    def _compute_influence_rewards(
        self,
        obs: Dict[int, np.ndarray],
        actions: Dict[int, int],
    ) -> Dict[int, float]:
        """
        Compute social influence intrinsic rewards.

        Args:
            obs: Observations for all agents
            actions: Actions taken by all agents

        Returns:
            influence_rewards: Dict of influence rewards per agent
        """
        if self.social_influence is None:
            return {i: 0.0 for i in range(self.num_agents)}

        # Stack observations and actions
        obs_array = np.stack([obs[i].flatten() for i in range(self.num_agents)])
        actions_array = np.array([actions[i] for i in range(self.num_agents)])

        # Collect for MOA training
        self._si_obs_buffer.append(obs_array)
        self._si_actions_buffer.append(actions_array)

        # Compute influence rewards
        moa_params = {i: self.social_influence.moa_states[i].params
                      for i in range(self.num_agents)}

        influence = self.social_influence.compute_influence_reward(
            moa_params, obs_array, actions_array
        )

        return {i: float(influence[i]) for i in range(self.num_agents)}

    def _add_influence_to_rewards(
        self,
        extrinsic_rewards: Dict[int, float],
        influence_rewards: Dict[int, float],
    ) -> Dict[int, float]:
        """Combine extrinsic and influence rewards."""
        return {
            i: extrinsic_rewards[i] + influence_rewards[i]
            for i in range(self.num_agents)
        }

    def _compute_future_prediction_rewards(
        self,
        obs: Dict[int, np.ndarray],
        actions: Dict[int, int],
        collective_reward: float,
    ) -> Dict[int, float]:
        """
        Compute intrinsic rewards from future prediction module.

        Args:
            obs: Observations for all agents
            actions: Actions taken
            collective_reward: Sum of extrinsic rewards for all agents

        Returns:
            intrinsic_rewards: Dict of intrinsic rewards per agent
        """
        if self.future_prediction is None:
            return {i: 0.0 for i in range(self.num_agents)}

        # Add experience to predictor buffer
        self.future_prediction.add_experience(obs, actions, collective_reward)

        # Compute intrinsic reward
        intrinsic_rewards = self.future_prediction.compute_intrinsic_reward(obs, actions)

        return intrinsic_rewards

    def _end_episode_future_prediction(self):
        """Called when episode ends to finalize trajectory for future prediction."""
        if self.future_prediction is not None:
            self.future_prediction.end_trajectory()

    def collect_rollout(self) -> Dict:
        """
        Collect rollout data from environments (parallel if num_envs > 1).

        Returns:
            info: Dictionary with episode statistics
        """
        self.buffer.clear()

        # Clear social influence buffers
        self._si_obs_buffer = []
        self._si_actions_buffer = []

        # Initialize environments only on first call or use persistent state
        if self._env_obs is None:
            env_obs = []
            env_episode_rewards = []
            env_episode_lengths = []
            for env in self.envs:
                obs, _ = env.reset()
                env_obs.append(obs)
                env_episode_rewards.append({i: 0.0 for i in range(self.num_agents)})
                env_episode_lengths.append(0)
            self._env_obs = env_obs
            self._env_episode_rewards = env_episode_rewards
            self._env_episode_lengths = env_episode_lengths

        env_obs = self._env_obs
        env_episode_rewards = self._env_episode_rewards
        env_episode_lengths = self._env_episode_lengths

        # Collect rollouts
        if self.num_envs > 1:
            # Parallel collection with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.num_envs) as executor:
                for step in range(self.config.rollout_length):
                    # Get actions for all environments (batched)
                    all_actions = []
                    all_log_probs = []
                    all_values = []

                    for env_id in range(self.num_envs):
                        actions, log_probs, values = self.get_actions(env_obs[env_id])
                        all_actions.append(actions)
                        all_log_probs.append(log_probs)
                        all_values.append(values)

                    # Submit all env steps in parallel
                    futures = []
                    for env_id, env in enumerate(self.envs):
                        future = executor.submit(
                            self._env_step, env_id, env, env_obs[env_id], all_actions[env_id]
                        )
                        futures.append(future)

                    # Collect results
                    results = [f.result() for f in futures]

                    # Process results and store in buffer
                    for env_id, next_obs, state, rewards, dones, infos in results:
                        # Compute collective reward for future prediction
                        collective_reward = sum(rewards.values())

                        # Compute social influence rewards
                        influence_rewards = self._compute_influence_rewards(
                            env_obs[env_id], all_actions[env_id]
                        )
                        total_rewards = self._add_influence_to_rewards(rewards, influence_rewards)

                        # Compute future prediction intrinsic rewards
                        fp_rewards = self._compute_future_prediction_rewards(
                            env_obs[env_id], all_actions[env_id], collective_reward
                        )
                        total_rewards = self._add_influence_to_rewards(total_rewards, fp_rewards)

                        # Store transitions
                        self.buffer.add_batch(
                            observations=env_obs[env_id],
                            actions=all_actions[env_id],
                            rewards=total_rewards,  # Use total rewards (extrinsic + intrinsic)
                            dones=dones,
                            values=all_values[env_id],
                            log_probs=all_log_probs[env_id],
                        )

                        # Track episode stats (use extrinsic rewards for metrics)
                        for agent_id in range(self.num_agents):
                            env_episode_rewards[env_id][agent_id] += rewards[agent_id]
                        env_episode_lengths[env_id] += 1
                        self.total_steps += 1

                        # Check for episode end
                        if any(dones.values()):
                            self.episode_rewards.append(sum(env_episode_rewards[env_id].values()))
                            self.episode_lengths.append(env_episode_lengths[env_id])

                            # End trajectory for future prediction
                            self._end_episode_future_prediction()

                            # Reset this environment
                            env_obs[env_id], _ = self.envs[env_id].reset()
                            env_episode_rewards[env_id] = {i: 0.0 for i in range(self.num_agents)}
                            env_episode_lengths[env_id] = 0
                        else:
                            env_obs[env_id] = next_obs
        else:
            # Single environment (original behavior)
            for step in range(self.config.rollout_length):
                actions, log_probs, values = self.get_actions(env_obs[0])
                next_obs, state, rewards, dones, infos = self.env.step(actions)

                # Compute collective reward for future prediction
                collective_reward = sum(rewards.values())

                # Compute social influence rewards
                influence_rewards = self._compute_influence_rewards(env_obs[0], actions)
                total_rewards = self._add_influence_to_rewards(rewards, influence_rewards)

                # Compute future prediction intrinsic rewards
                fp_rewards = self._compute_future_prediction_rewards(
                    env_obs[0], actions, collective_reward
                )
                total_rewards = self._add_influence_to_rewards(total_rewards, fp_rewards)

                self.buffer.add_batch(
                    observations=env_obs[0],
                    actions=actions,
                    rewards=total_rewards,  # Use total rewards (extrinsic + intrinsic)
                    dones=dones,
                    values=values,
                    log_probs=log_probs,
                )

                # Track episode stats (use extrinsic rewards for metrics)
                for agent_id in range(self.num_agents):
                    env_episode_rewards[0][agent_id] += rewards[agent_id]
                env_episode_lengths[0] += 1
                self.total_steps += 1

                if any(dones.values()):
                    self.episode_rewards.append(sum(env_episode_rewards[0].values()))
                    self.episode_lengths.append(env_episode_lengths[0])

                    # End trajectory for future prediction
                    self._end_episode_future_prediction()

                    env_obs[0], _ = self.env.reset()
                    env_episode_rewards[0] = {i: 0.0 for i in range(self.num_agents)}
                    env_episode_lengths[0] = 0
                else:
                    env_obs[0] = next_obs

        # Compute last values for bootstrapping (use average across envs)
        all_last_values = []
        for env_id in range(self.num_envs):
            _, _, last_values = self.get_actions(env_obs[env_id])
            all_last_values.append(last_values)

        # Average last values across environments
        avg_last_values = {
            agent_id: np.mean([lv[agent_id] for lv in all_last_values])
            for agent_id in range(self.num_agents)
        }
        self.buffer.compute_advantages(avg_last_values)

        # Aggregate cleaning rate from all environments
        avg_cleaning_rate = np.mean([env.get_cleaning_rate() for env in self.envs])

        return {
            'episode_rewards': self.episode_rewards[-10:] if self.episode_rewards else [],
            'episode_lengths': self.episode_lengths[-10:] if self.episode_lengths else [],
            'cleaning_rate': avg_cleaning_rate,
        }

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update on collected data.

        Returns:
            metrics: Dictionary of training metrics
        """
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0,
        }
        num_updates = 0

        for epoch in range(self.config.num_epochs):
            for agent_id in range(self.num_agents):
                # Get data for this agent
                obs, actions, old_log_probs, advantages, returns = \
                    self.buffer.get_all_data(agent_id)

                # Normalize advantages
                if self.config.normalize_advantages:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Convert to JAX arrays
                obs = jnp.array(obs)
                actions = jnp.array(actions)
                old_log_probs = jnp.array(old_log_probs)
                advantages = jnp.array(advantages)
                returns = jnp.array(returns)

                # Get training state
                if self.config.parameter_sharing:
                    state = self.train_state
                else:
                    state = self.train_states[agent_id]

                # Compute loss and update
                state, loss_info = self._update_step(
                    state, obs, actions, old_log_probs, advantages, returns
                )

                # Store updated state
                if self.config.parameter_sharing:
                    self.train_state = state
                else:
                    self.train_states[agent_id] = state

                # Accumulate metrics
                for key in metrics:
                    if key in loss_info:
                        metrics[key] += loss_info[key]
                num_updates += 1

        # Average metrics
        for key in metrics:
            metrics[key] /= max(num_updates, 1)

        # Update MOA (Model of Other Agents) if using social influence
        if self.social_influence is not None and len(self._si_obs_buffer) > 0:
            # Stack collected data
            obs_batch = jnp.array(np.stack(self._si_obs_buffer))  # (T, num_agents, obs_dim)
            actions_batch = jnp.array(np.stack(self._si_actions_buffer))  # (T, num_agents)

            # Update MOA models
            moa_losses = self.social_influence.update_moa(obs_batch, actions_batch)

            # Add MOA loss to metrics (average across agents)
            avg_moa_loss = np.mean([v for k, v in moa_losses.items()])
            metrics['moa_loss'] = avg_moa_loss

        # Update future prediction model
        if self.future_prediction is not None:
            fp_metrics = self.future_prediction.update_predictor(num_updates=5)
            metrics['predictor_loss'] = fp_metrics['predictor_loss']

        return metrics

    def _create_update_fn(self):
        """Create JIT-compiled update function."""
        clip_epsilon = self.config.clip_epsilon
        value_loss_coef = self.config.value_loss_coef
        entropy_coef = self.config.entropy_coef
        network = self.network

        @jax.jit
        def update_step(
            state: train_state.TrainState,
            obs: jnp.ndarray,
            actions: jnp.ndarray,
            old_log_probs: jnp.ndarray,
            advantages: jnp.ndarray,
            returns: jnp.ndarray,
        ):
            def loss_fn(params):
                # Forward pass
                logits, values, _ = network.apply(params, obs)
                values = values.squeeze(-1)

                # Compute action probabilities
                action_probs = jax.nn.softmax(logits)
                log_probs = jnp.log(action_probs[jnp.arange(len(actions)), actions] + 1e-8)

                # Policy loss (clipped surrogate)
                ratio = jnp.exp(log_probs - old_log_probs)
                clipped_ratio = jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                policy_loss = -jnp.minimum(ratio * advantages, clipped_ratio * advantages).mean()

                # Value loss
                value_loss = ((values - returns) ** 2).mean()

                # Entropy bonus
                entropy = -(action_probs * jnp.log(action_probs + 1e-8)).sum(axis=-1).mean()

                # Total loss
                total_loss = (
                    policy_loss +
                    value_loss_coef * value_loss -
                    entropy_coef * entropy
                )

                return total_loss, {
                    'policy_loss': policy_loss,
                    'value_loss': value_loss,
                    'entropy': entropy,
                    'total_loss': total_loss,
                }

            # Compute gradients
            (loss, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

            # Apply gradients
            state = state.apply_gradients(grads=grads)

            return state, loss_info

        return update_step

    def _update_step(
        self,
        state: train_state.TrainState,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        old_log_probs: jnp.ndarray,
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
    ) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """Single PPO update step (JIT-compiled)."""
        # Lazy initialization of JIT-compiled function
        if not hasattr(self, '_jit_update_fn'):
            self._jit_update_fn = self._create_update_fn()

        state, loss_info = self._jit_update_fn(
            state, obs, actions, old_log_probs, advantages, returns
        )

        return state, {k: float(v) for k, v in loss_info.items()}

    def train(
        self,
        total_timesteps: Optional[int] = None,
        callback: Optional[Callable] = None,
        log_interval: int = 10,
    ) -> Dict:
        """
        Main training loop.

        Args:
            total_timesteps: Override total timesteps from config
            callback: Optional callback function(iteration, metrics)
            log_interval: How often to log progress

        Returns:
            training_info: Dictionary with training statistics
        """
        if total_timesteps is None:
            total_timesteps = self.config.total_timesteps

        # Account for parallel environments in iteration count
        # Note: steps are counted per environment step, not per agent
        steps_per_iteration = self.config.rollout_length * self.num_envs
        num_iterations = total_timesteps // steps_per_iteration

        training_info = {
            'iterations': [],
            'episode_rewards': [],
            'cleaning_rates': [],
            'losses': [],
        }

        # Initialize rich logger
        logger = TrainingLogger(
            total_timesteps=total_timesteps,
            num_agents=self.num_agents,
            log_interval=log_interval,
            num_envs=self.num_envs,
            rollout_length=self.config.rollout_length,
        )

        # Track best cleaning rate for milestones
        best_cleaning_rate = 0.0
        milestones = {0.1: False, 0.2: False, 0.3: False, 0.5: False}

        for iteration in range(num_iterations):
            # Collect experience
            rollout_info = self.collect_rollout()

            # Update policy
            update_metrics = self.update()

            # Get average reward
            avg_reward = np.mean(rollout_info['episode_rewards']) if rollout_info['episode_rewards'] else 0
            cleaning_rate = rollout_info['cleaning_rate']

            # Log with rich logger
            logger.log(
                iteration=iteration,
                steps=self.total_steps,
                reward=avg_reward,
                cleaning_rate=cleaning_rate,
                policy_loss=update_metrics['policy_loss'],
                value_loss=update_metrics['value_loss'],
                entropy=update_metrics['entropy'],
            )

            # Check for milestones
            if cleaning_rate > best_cleaning_rate:
                best_cleaning_rate = cleaning_rate
                for threshold, achieved in milestones.items():
                    if not achieved and cleaning_rate >= threshold:
                        milestones[threshold] = True
                        logger.print_milestone(
                            f"Cleaning rate reached {threshold*100:.0f}%!"
                        )

            # Track history
            training_info['iterations'].append(iteration)
            training_info['episode_rewards'].append(avg_reward)
            training_info['cleaning_rates'].append(cleaning_rate)
            training_info['losses'].append(update_metrics['total_loss'])

            # Callback
            if callback is not None:
                callback(iteration, {**rollout_info, **update_metrics})

        # Print final summary
        logger.print_summary()
        return training_info

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        import pickle

        checkpoint = {
            'config': self.config,
            'total_steps': self.total_steps,
        }

        if self.config.parameter_sharing:
            checkpoint['params'] = self.train_state.params
        else:
            checkpoint['params'] = {
                agent_id: state.params
                for agent_id, state in self.train_states.items()
            }

        # Save MOA parameters if using social influence
        if self.social_influence is not None:
            checkpoint['moa_params'] = self.social_influence.get_moa_params()

        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        import pickle

        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.total_steps = checkpoint['total_steps']

        if self.config.parameter_sharing:
            self.train_state = self.train_state.replace(params=checkpoint['params'])
        else:
            for agent_id, params in checkpoint['params'].items():
                self.train_states[agent_id] = self.train_states[agent_id].replace(params=params)

        # Load MOA parameters if available
        if self.social_influence is not None and 'moa_params' in checkpoint:
            self.social_influence.set_moa_params(checkpoint['moa_params'])

        print(f"Checkpoint loaded from {path}")

    def get_policy(self, agent_id: int = 0):
        """Get policy function for inference."""
        if self.config.parameter_sharing:
            params = self.train_state.params
        else:
            params = self.train_states[agent_id].params

        def policy(obs, rng):
            obs = obs[None, ...]
            logits, _, _ = self.network.apply(params, obs)
            action = jax.random.categorical(rng, logits[0])
            return int(action)

        return policy
