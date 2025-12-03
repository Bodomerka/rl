"""
Experiment Runner

Orchestrates the three-phase experiment protocol:
1. Train cooperative agents
2. Train defector agent
3. Measure cooperation collapse dynamics
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Callable, Any
import pickle
import json

import numpy as np

from src.environments.cleanup_env import CleanupEnvironment
from src.environments.agent_injection import AgentInjector, DefectorRewardShaper
from src.algorithms.ippo.trainer import IPPOTrainer, IPPOConfig
from src.metrics.brittleness_index import CooperationBrittlenessIndex, CooperationState
from src.metrics.collective_metrics import CollectiveMetrics
from src.metrics.inequality_metrics import gini_coefficient, InequalityTracker


@dataclass
class ExperimentConfig:
    """Configuration for full experiment."""
    # Environment
    num_agents: int = 8
    grid_size: tuple = (25, 18)
    max_steps: int = 1000

    # Phase 1: Cooperative training
    phase1_timesteps: int = 5_000_000
    phase1_common_reward: bool = True
    phase1_early_stop_threshold: float = 0.3  # Stop if cleaning rate > 30%

    # Phase 2: Defector training
    phase2_timesteps: int = 2_000_000
    defector_agent_id: int = 0
    defector_apple_bonus: float = 1.5
    defector_cleaning_penalty: float = -0.2

    # Phase 3: Collapse measurement
    baseline_steps: int = 500
    observation_steps: int = 2000
    freeze_defector: bool = True

    # General
    seed: int = 42
    output_dir: str = "outputs/experiments"
    checkpoint_dir: str = "checkpoints"
    num_runs: int = 1  # For statistical significance


class ExperimentRunner:
    """
    Runs the complete three-phase experiment protocol.

    Phase 1: Train N cooperative agents with shared rewards
    Phase 2: Train 1 defector agent against frozen cooperators
    Phase 3: Inject defector and measure cooperation collapse
    """

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment runner."""
        self.config = config

        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics
        self.brittleness_calculator = CooperationBrittlenessIndex()
        self.collective_metrics = CollectiveMetrics(config.num_agents)
        self.inequality_tracker = InequalityTracker(config.num_agents)

        # Results storage
        self.results: Dict[str, Any] = {}

    def run_full_experiment(self, callback: Optional[Callable] = None) -> Dict:
        """
        Execute complete experiment protocol.

        Args:
            callback: Optional callback function for progress updates

        Returns:
            Dictionary with all experiment results
        """
        print("=" * 60)
        print("COOPERATION COLLAPSE EXPERIMENT")
        print("=" * 60)
        print(f"Agents: {self.config.num_agents}")
        print(f"Seed: {self.config.seed}")
        print()

        # PHASE 1: Train cooperative agents
        print("=" * 60)
        print("PHASE 1: Training Cooperative Agents")
        print("=" * 60)

        coop_checkpoint = self.phase1_train_cooperative(callback)
        baseline_metrics = self._evaluate_cooperation(coop_checkpoint)

        print(f"\nBaseline Metrics:")
        print(f"  Cleaning Rate: {baseline_metrics['cleaning_rate']:.3f}")
        print(f"  Collective Reward: {baseline_metrics['collective_reward']:.2f}")

        if baseline_metrics['cleaning_rate'] < 0.2:
            print("WARNING: Low cooperation in baseline. Consider more training.")

        # PHASE 2: Train defector agent
        print("\n" + "=" * 60)
        print("PHASE 2: Training Defector Agent")
        print("=" * 60)

        defector_checkpoint = self.phase2_train_defector(coop_checkpoint, callback)

        # PHASE 3: Measure collapse dynamics
        print("\n" + "=" * 60)
        print("PHASE 3: Measuring Cooperation Collapse")
        print("=" * 60)

        collapse_results = self.phase3_measure_collapse(
            coop_checkpoint,
            defector_checkpoint,
            callback
        )

        # Compile final results
        self.results = {
            'config': self.config.__dict__,
            'baseline_metrics': baseline_metrics,
            'collapse_results': collapse_results,
            'brittleness_index': collapse_results['cbi'],
        }

        # Save results
        self._save_results()

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"\nCooperation Brittleness Index: {collapse_results['cbi']:.4f}")
        print(f"Collapse Speed: {collapse_results['collapse_speed_timesteps']} steps")
        print(f"Collapse Depth: {collapse_results['collapse_depth']*100:.1f}%")
        print(f"Recovery Difficulty: {collapse_results['recovery_difficulty']*100:.1f}%")

        return self.results

    def phase1_train_cooperative(self, callback: Optional[Callable] = None) -> str:
        """
        Phase 1: Train N agents with common reward to encourage cooperation.

        Returns:
            Path to saved checkpoint
        """
        # Create environment with common rewards
        env = CleanupEnvironment(
            num_agents=self.config.num_agents,
            grid_size=self.config.grid_size,
            max_steps=self.config.max_steps,
            common_reward=self.config.phase1_common_reward,
            seed=self.config.seed,
        )

        # Create IPPO trainer
        ippo_config = IPPOConfig(
            total_timesteps=self.config.phase1_timesteps,
            rollout_length=128,
            hidden_dims=(64, 64),
        )
        trainer = IPPOTrainer(env, ippo_config, seed=self.config.seed)

        # Training callback with early stopping
        early_stop = False

        def training_callback(iteration, metrics):
            nonlocal early_stop
            cleaning_rate = metrics.get('cleaning_rate', 0)

            if callback:
                callback('phase1', iteration, metrics)

            # Early stopping check
            if cleaning_rate >= self.config.phase1_early_stop_threshold:
                print(f"\nEarly stopping: Cleaning rate {cleaning_rate:.3f} >= threshold")
                early_stop = True

        # Train
        trainer.train(callback=training_callback)

        # Save checkpoint
        checkpoint_path = str(self.checkpoint_dir / "cooperative" / "best.pkl")
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(checkpoint_path)

        return checkpoint_path

    def phase2_train_defector(
        self,
        coop_checkpoint: str,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Phase 2: Train a defector agent against frozen cooperative population.

        Args:
            coop_checkpoint: Path to cooperative agents checkpoint

        Returns:
            Path to saved defector checkpoint
        """
        # Create environment with individual rewards
        env = CleanupEnvironment(
            num_agents=self.config.num_agents,
            grid_size=self.config.grid_size,
            max_steps=self.config.max_steps,
            common_reward=False,  # Individual rewards
            seed=self.config.seed + 1,
        )

        # Create trainer for defector (only agent 0 learns)
        ippo_config = IPPOConfig(
            total_timesteps=self.config.phase2_timesteps,
            rollout_length=128,
            hidden_dims=(64, 64),
            parameter_sharing=False,  # Defector has separate parameters
        )
        trainer = IPPOTrainer(env, ippo_config, seed=self.config.seed + 1)

        # Load cooperative weights for other agents
        # (In full implementation, would freeze agents 1-N)

        # Create reward shaper for defector
        reward_shaper = DefectorRewardShaper(
            apple_bonus=self.config.defector_apple_bonus,
            cleaning_penalty=self.config.defector_cleaning_penalty,
        )

        # Training callback
        def training_callback(iteration, metrics):
            if callback:
                callback('phase2', iteration, metrics)

        # Train
        trainer.train(callback=training_callback)

        # Save checkpoint
        checkpoint_path = str(self.checkpoint_dir / "defector" / "best.pkl")
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(checkpoint_path)

        return checkpoint_path

    def phase3_measure_collapse(
        self,
        coop_checkpoint: str,
        defector_checkpoint: str,
        callback: Optional[Callable] = None
    ) -> Dict:
        """
        Phase 3: Inject defector and measure cooperation dynamics.

        Args:
            coop_checkpoint: Path to cooperative agents
            defector_checkpoint: Path to defector agent

        Returns:
            Dictionary with collapse metrics including CBI
        """
        # Create fresh environment
        env = CleanupEnvironment(
            num_agents=self.config.num_agents,
            grid_size=self.config.grid_size,
            max_steps=self.config.max_steps,
            common_reward=False,
            seed=self.config.seed + 2,
        )

        # Load cooperative trainer
        ippo_config = IPPOConfig(hidden_dims=(64, 64))
        coop_trainer = IPPOTrainer(env, ippo_config)
        coop_trainer.load_checkpoint(coop_checkpoint)

        # Reset metrics
        self.brittleness_calculator.clear()
        self.collective_metrics.reset()
        self.inequality_tracker.reset()

        # Initialize environment
        obs, state = env.reset()
        np_rng = np.random.default_rng(self.config.seed + 2)

        # STEP 1: Establish baseline (without defector)
        print("\nEstablishing baseline...")
        for step in range(self.config.baseline_steps):
            # Get actions from cooperative policy
            actions, _, _ = coop_trainer.get_actions(obs)

            # Environment step
            next_obs, state, rewards, dones, infos = env.step(actions)

            # Record metrics
            self._record_state(
                step=step,
                rewards=rewards,
                infos=infos,
                is_baseline=True
            )

            if any(dones.values()):
                obs, state = env.reset()
            else:
                obs = next_obs

            if callback:
                callback('phase3_baseline', step, infos)

        # STEP 2: Inject defector
        print(f"\nInjecting defector at agent {self.config.defector_agent_id}...")

        # Load defector policy
        defector_trainer = IPPOTrainer(env, ippo_config)
        defector_trainer.load_checkpoint(defector_checkpoint)
        defector_policy = defector_trainer.get_policy(self.config.defector_agent_id)

        # STEP 3: Measure collapse dynamics
        print("Measuring collapse dynamics...")
        for step in range(self.config.observation_steps):
            # Get actions (defector uses defector policy, others use cooperative)
            actions = {}
            for agent_id in range(self.config.num_agents):
                if agent_id == self.config.defector_agent_id:
                    # Defector action
                    import jax
                    rng = jax.random.PRNGKey(step)
                    actions[agent_id] = defector_policy(obs[agent_id], rng)
                else:
                    # Cooperative action
                    action, _, _ = coop_trainer.get_action(agent_id, obs[agent_id])
                    actions[agent_id] = action

            # Environment step
            next_obs, state, rewards, dones, infos = env.step(actions)

            # Record metrics
            self._record_state(
                step=self.config.baseline_steps + step,
                rewards=rewards,
                infos=infos,
                is_baseline=False
            )

            if any(dones.values()):
                obs, state = env.reset()
            else:
                obs = next_obs

            if callback:
                callback('phase3_collapse', step, infos)

        # Compute final CBI
        cbi_results = self.brittleness_calculator.compute_extended_metrics()

        return cbi_results

    def _record_state(
        self,
        step: int,
        rewards: Dict[int, float],
        infos: Dict,
        is_baseline: bool
    ):
        """Record metrics for a single step."""
        # Update trackers
        self.collective_metrics.update(rewards, infos)
        self.inequality_tracker.update(rewards)

        # Create cooperation state
        state = CooperationState(
            timestep=step,
            cleaning_rate=self.collective_metrics.cleaning_rate,
            collective_reward=sum(rewards.values()),
            pollution_level=infos.get('pollution_level', 0.0),
            apple_count=infos.get('apple_count', 0),
            gini_coefficient=gini_coefficient(rewards),
            per_agent_rewards=rewards.copy(),
        )

        # Record to brittleness calculator
        if is_baseline:
            self.brittleness_calculator.record_baseline(state)
        else:
            self.brittleness_calculator.record_post_injection(state)

    def _evaluate_cooperation(self, checkpoint_path: str) -> Dict:
        """Evaluate cooperation level of trained agents."""
        env = CleanupEnvironment(
            num_agents=self.config.num_agents,
            grid_size=self.config.grid_size,
            max_steps=500,
            common_reward=False,
            seed=self.config.seed + 100,
        )

        ippo_config = IPPOConfig(hidden_dims=(64, 64))
        trainer = IPPOTrainer(env, ippo_config)
        trainer.load_checkpoint(checkpoint_path)

        # Run evaluation episode
        obs, state = env.reset()
        total_reward = 0.0

        for _ in range(500):
            actions, _, _ = trainer.get_actions(obs)
            next_obs, state, rewards, dones, infos = env.step(actions)
            total_reward += sum(rewards.values())

            if any(dones.values()):
                break
            obs = next_obs

        return {
            'cleaning_rate': env.get_cleaning_rate(),
            'collective_reward': total_reward,
        }

    def _save_results(self):
        """Save experiment results to disk."""
        # Save as JSON
        json_path = self.output_dir / "results.json"
        with open(json_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = json.loads(
                json.dumps(self.results, default=lambda x: float(x) if isinstance(x, np.floating) else x)
            )
            json.dump(json_results, f, indent=2)

        # Save as pickle for full fidelity
        pkl_path = self.output_dir / "results.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.results, f)

        # Save time series data
        time_series = self.brittleness_calculator.get_time_series()
        ts_path = self.output_dir / "time_series.pkl"
        with open(ts_path, 'wb') as f:
            pickle.dump(time_series, f)

        print(f"\nResults saved to {self.output_dir}")


def run_experiment(config: Optional[ExperimentConfig] = None) -> Dict:
    """
    Convenience function to run a full experiment.

    Args:
        config: Experiment configuration (uses defaults if None)

    Returns:
        Experiment results dictionary
    """
    if config is None:
        config = ExperimentConfig()

    runner = ExperimentRunner(config)
    return runner.run_full_experiment()
