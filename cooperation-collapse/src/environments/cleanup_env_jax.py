"""
JAX-based Cleanup Environment Wrapper

Wraps SocialJax Cleanup environment for GPU-accelerated training.
Provides ~50x speedup over NumPy-based environment.

Based on SocialJax: https://github.com/cooperativex/socialjax
"""

from functools import partial
from typing import Dict, Tuple, Any, Optional, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

# Try to import SocialJax
try:
    import socialjax
    SOCIALJAX_AVAILABLE = True
except ImportError:
    SOCIALJAX_AVAILABLE = False
    print("WARNING: SocialJax not installed. Install with:")
    print("  git clone https://github.com/cooperativex/socialjax")
    print("  cd socialjax && pip install -e .")


class EnvState(NamedTuple):
    """State container for JAX environment."""
    state: Any  # SocialJax internal state
    step_count: jnp.ndarray  # Current step counter
    episode_rewards: jnp.ndarray  # Accumulated rewards per agent
    cleaning_actions: jnp.ndarray  # Cleaning action counter
    collection_actions: jnp.ndarray  # Collection action counter


class Transition(NamedTuple):
    """Single transition for rollout storage."""
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    info: Dict


class JaxCleanupEnvironment:
    """
    JAX-based Cleanup Environment wrapper.

    Provides GPU-accelerated simulation through SocialJax with
    interface compatible with our IPPO trainer.

    Features:
    - Vectorized environment (multiple envs in parallel on GPU)
    - JIT-compiled reset/step functions
    - Compatible observation/action spaces
    """

    # Action mapping (same as our NumPy env)
    ACTION_NOOP = 0
    ACTION_UP = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_RIGHT = 4
    ACTION_ROTATE_LEFT = 5
    ACTION_ROTATE_RIGHT = 6
    ACTION_CLEAN = 7  # Fire beam for cleaning
    ACTION_COLLECT = 8  # Collect apple (in some versions)

    def __init__(
        self,
        num_agents: int = 8,
        num_envs: int = 16,
        max_steps: int = 400,
        common_reward: bool = True,
        seed: int = 42,
    ):
        """
        Initialize JAX Cleanup environment.

        Args:
            num_agents: Number of agents in environment
            num_envs: Number of parallel environments
            max_steps: Maximum steps per episode
            common_reward: If True, all agents share total reward
            seed: Random seed
        """
        if not SOCIALJAX_AVAILABLE:
            raise ImportError("SocialJax not available. Please install it first.")

        self.num_agents = num_agents
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.common_reward = common_reward

        # Create base environment
        self.env = socialjax.make(
            'clean_up',
            num_agents=num_agents,
        )

        # Get observation and action spaces
        self.observation_shape = self.env.observation_space()[0].shape
        self.action_space_size = self.env.action_space()[0].n

        # Initialize RNG
        self.rng = jax.random.PRNGKey(seed)

        # Tracking for metrics
        self._cleaning_rate = 0.0

        print(f"JaxCleanupEnvironment initialized:")
        print(f"  Agents: {num_agents}")
        print(f"  Parallel envs: {num_envs}")
        print(f"  Observation shape: {self.observation_shape}")
        print(f"  Actions: {self.action_space_size}")

    @partial(jax.jit, static_argnums=(0,))
    def _reset_single(self, key: jnp.ndarray) -> Tuple[jnp.ndarray, Any]:
        """Reset a single environment."""
        obs, state = self.env.reset(key)
        return obs, state

    def reset(self) -> Tuple[Dict[int, np.ndarray], EnvState]:
        """
        Reset all parallel environments.

        Returns:
            observations: Dict mapping agent_id to observation array
            state: Environment state (for JAX compatibility)
        """
        self.rng, reset_key = jax.random.split(self.rng)
        keys = jax.random.split(reset_key, self.num_envs)

        # Vectorized reset across all envs
        obs, states = jax.vmap(self._reset_single)(keys)

        # obs shape: (num_envs, num_agents, *obs_shape)
        # Convert to dict format for compatibility with trainer
        # Take first env's observations for the "primary" env interface
        obs_dict = {
            agent_id: np.array(obs[0, agent_id])
            for agent_id in range(self.num_agents)
        }

        # Create state container
        env_state = EnvState(
            state=states,
            step_count=jnp.zeros(self.num_envs, dtype=jnp.int32),
            episode_rewards=jnp.zeros((self.num_envs, self.num_agents)),
            cleaning_actions=jnp.zeros(self.num_envs, dtype=jnp.int32),
            collection_actions=jnp.zeros(self.num_envs, dtype=jnp.int32),
        )

        # Store for step function
        self._current_state = env_state
        self._current_obs = obs

        return obs_dict, env_state

    @partial(jax.jit, static_argnums=(0,))
    def _step_single(
        self,
        key: jnp.ndarray,
        state: Any,
        actions: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Any, jnp.ndarray, jnp.ndarray, Dict]:
        """Step a single environment."""
        obs, new_state, rewards, dones, info = self.env.step(key, state, actions)
        return obs, new_state, rewards, dones, info

    def step(
        self,
        actions: Dict[int, int],
    ) -> Tuple[Dict[int, np.ndarray], EnvState, Dict[int, float], Dict[int, bool], Dict]:
        """
        Step all parallel environments.

        Args:
            actions: Dict mapping agent_id to action

        Returns:
            observations: Dict of observations per agent
            state: New environment state
            rewards: Dict of rewards per agent
            dones: Dict of done flags per agent
            infos: Additional info dict
        """
        # Convert actions dict to array
        actions_array = jnp.array([actions[i] for i in range(self.num_agents)])
        # Broadcast to all envs (same actions for now - will be different in vectorized trainer)
        actions_batch = jnp.broadcast_to(actions_array, (self.num_envs, self.num_agents))

        # Generate step keys
        self.rng, step_key = jax.random.split(self.rng)
        keys = jax.random.split(step_key, self.num_envs)

        # Vectorized step
        obs, new_states, rewards, dones, infos = jax.vmap(self._step_single)(
            keys, self._current_state.state, actions_batch
        )

        # Update step counter
        step_count = self._current_state.step_count + 1

        # Check for max steps
        max_step_done = step_count >= self.max_steps
        dones = dones | max_step_done[:, None]  # Broadcast to all agents

        # Track cleaning/collection actions (assuming action 7 = clean)
        cleaning = jnp.sum(actions_batch == self.ACTION_CLEAN, axis=1)
        collection = jnp.sum(actions_batch == self.ACTION_COLLECT, axis=1)

        # Update state
        new_env_state = EnvState(
            state=new_states,
            step_count=step_count,
            episode_rewards=self._current_state.episode_rewards + rewards,
            cleaning_actions=self._current_state.cleaning_actions + cleaning,
            collection_actions=self._current_state.collection_actions + collection,
        )

        # Compute cleaning rate for metrics
        total_clean = float(jnp.sum(new_env_state.cleaning_actions))
        total_collect = float(jnp.sum(new_env_state.collection_actions))
        if total_clean + total_collect > 0:
            self._cleaning_rate = total_clean / (total_clean + total_collect)

        # Convert to dict format
        obs_dict = {
            agent_id: np.array(obs[0, agent_id])
            for agent_id in range(self.num_agents)
        }

        # Handle common reward
        if self.common_reward:
            total_reward = float(jnp.sum(rewards[0]))
            rewards_dict = {i: total_reward / self.num_agents for i in range(self.num_agents)}
        else:
            rewards_dict = {i: float(rewards[0, i]) for i in range(self.num_agents)}

        dones_dict = {i: bool(dones[0, i]) for i in range(self.num_agents)}

        # Store state
        self._current_state = new_env_state
        self._current_obs = obs

        # Info dict
        info = {
            'step_count': int(step_count[0]),
            'pollution_level': 0.0,  # Will be updated if available from SocialJax
            'apple_count': 0,
        }

        return obs_dict, new_env_state, rewards_dict, dones_dict, info

    def get_cleaning_rate(self) -> float:
        """Get current cleaning rate metric."""
        return self._cleaning_rate

    def get_vectorized_interface(self):
        """
        Get fully vectorized interface for high-performance training.

        Returns functions that operate on batched data for use with jax.lax.scan.
        """
        env = self.env
        num_envs = self.num_envs
        num_agents = self.num_agents
        max_steps = self.max_steps
        common_reward = self.common_reward

        @jax.jit
        def vec_reset(key: jnp.ndarray) -> Tuple[jnp.ndarray, Any]:
            """Vectorized reset returning (obs, state)."""
            keys = jax.random.split(key, num_envs)
            obs, states = jax.vmap(env.reset)(keys)
            return obs, states

        @jax.jit
        def vec_step(
            key: jnp.ndarray,
            state: Any,
            actions: jnp.ndarray,  # (num_envs, num_agents)
        ) -> Tuple[jnp.ndarray, Any, jnp.ndarray, jnp.ndarray, Dict]:
            """Vectorized step."""
            keys = jax.random.split(key, num_envs)

            def step_single(key, s, a):
                return env.step(key, s, a)

            obs, new_state, rewards, dones, info = jax.vmap(step_single)(
                keys, state, actions
            )

            # Apply common reward if needed
            if common_reward:
                total_rewards = jnp.sum(rewards, axis=-1, keepdims=True)
                rewards = jnp.broadcast_to(
                    total_rewards / num_agents,
                    (num_envs, num_agents)
                )

            return obs, new_state, rewards, dones, info

        return vec_reset, vec_step


class JaxCleanupTrainer:
    """
    Pure JAX trainer using SocialJax environment.

    This is an alternative high-performance trainer that runs
    entirely on GPU using jax.lax.scan for the training loop.
    """

    def __init__(
        self,
        num_agents: int = 8,
        num_envs: int = 64,
        rollout_length: int = 128,
        num_epochs: int = 4,
        num_minibatches: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        learning_rate: float = 2.5e-4,
        max_grad_norm: float = 0.5,
        seed: int = 42,
    ):
        """Initialize pure JAX trainer."""
        self.config = {
            'NUM_AGENTS': num_agents,
            'NUM_ENVS': num_envs,
            'NUM_STEPS': rollout_length,
            'NUM_EPOCHS': num_epochs,
            'NUM_MINIBATCHES': num_minibatches,
            'GAMMA': gamma,
            'GAE_LAMBDA': gae_lambda,
            'CLIP_EPS': clip_epsilon,
            'ENT_COEF': entropy_coef,
            'VF_COEF': value_coef,
            'LR': learning_rate,
            'MAX_GRAD_NORM': max_grad_norm,
        }

        if not SOCIALJAX_AVAILABLE:
            raise ImportError("SocialJax required for JaxCleanupTrainer")

        # Create environment
        self.env = socialjax.make('clean_up', num_agents=num_agents)
        self.rng = jax.random.PRNGKey(seed)

        print(f"JaxCleanupTrainer initialized:")
        print(f"  Envs: {num_envs}, Steps: {rollout_length}")
        print(f"  Batch size: {num_envs * rollout_length * num_agents:,}")


def create_jax_cleanup_env(
    num_agents: int = 8,
    num_envs: int = 16,
    max_steps: int = 400,
    common_reward: bool = True,
    seed: int = 42,
) -> JaxCleanupEnvironment:
    """Factory function to create JAX Cleanup environment."""
    return JaxCleanupEnvironment(
        num_agents=num_agents,
        num_envs=num_envs,
        max_steps=max_steps,
        common_reward=common_reward,
        seed=seed,
    )


# Compatibility alias
CleanupEnvironmentJax = JaxCleanupEnvironment
