#!/usr/bin/env python3
"""
Train Cooperative Agents (Phase 1)

Trains N agents with shared rewards to encourage cooperation.

Supports two modes:
- NumPy environment (default): ~30 steps/s, full feature support
- JAX environment (--jax): ~10,000-50,000 steps/s, requires SocialJax
"""

import argparse
import sys
from pathlib import Path

# Show JAX device info early
import jax
print(f"\n{'='*60}")
print(f"JAX Backend: {jax.default_backend().upper()}")
print(f"JAX Devices: {jax.devices()}")
print(f"{'='*60}\n")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check SocialJax availability
try:
    import socialjax
    SOCIALJAX_AVAILABLE = True
    print("SocialJax: Available (use --jax for 50x speedup)")
except ImportError:
    SOCIALJAX_AVAILABLE = False
    print("SocialJax: Not installed (--jax unavailable)")


def main():
    parser = argparse.ArgumentParser(description="Train cooperative agents")
    parser.add_argument("--num-agents", type=int, default=8, help="Number of agents")
    parser.add_argument("--timesteps", type=int, default=5_000_000, help="Total timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="checkpoints/cooperative/best.pkl",
                        help="Output checkpoint path")
    parser.add_argument("--log-interval", type=int, default=1, help="Log every N iterations")
    parser.add_argument("--plots-dir", type=str, default="outputs/plots/cooperative",
                        help="Directory to save plots")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting")
    parser.add_argument("--rollout-length", type=int, default=512,
                        help="Rollout length (larger = faster on GPU)")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    # Social Influence (Jaques et al., 2019)
    parser.add_argument("--social-influence", action="store_true",
                        help="Enable Social Influence intrinsic motivation")
    parser.add_argument("--influence-weight", type=float, default=0.1,
                        help="Weight (λ) for influence reward (default: 0.1)")
    # Future Prediction (intrinsic motivation for collective future)
    parser.add_argument("--future-prediction", action="store_true",
                        help="Enable Collective Future Prediction intrinsic motivation")
    parser.add_argument("--future-weight", type=float, default=0.5,
                        help="Weight for future prediction reward (default: 0.5)")
    # JAX mode (50x faster)
    parser.add_argument("--jax", action="store_true",
                        help="Use JAX-based training (requires SocialJax, ~50x faster)")
    args = parser.parse_args()

    # === JAX MODE ===
    if args.jax:
        if not SOCIALJAX_AVAILABLE:
            print("ERROR: --jax requires SocialJax. Install with:")
            print("  git clone https://github.com/cooperativex/socialjax")
            print("  cd socialjax && pip install -e .")
            sys.exit(1)

        from src.algorithms.ippo.jax_trainer import JaxIPPOTrainer, JaxIPPOConfig

        print("Using JAX-based training (SocialJax environment)")

        config = JaxIPPOConfig(
            num_agents=args.num_agents,
            num_envs=args.num_envs * 16,  # JAX can handle more envs
            total_timesteps=args.timesteps,
            rollout_length=args.rollout_length,
            log_interval=args.log_interval,
        )

        trainer = JaxIPPOTrainer(config, seed=args.seed)
        results = trainer.train()

        # Save checkpoint
        output_path = Path(args.output.replace('.pkl', '_jax.pkl'))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(output_path), results['train_state'])

        print(f"\nCheckpoint saved to: {output_path}")
        return

    # === NUMPY MODE (original) ===
    print("Using NumPy-based training (original environment)")
    print("Note: For 50x faster training, use --jax flag (requires SocialJax)")
    print()

    from src.environments.cleanup_env import CleanupEnvironment
    from src.algorithms.ippo.trainer import IPPOTrainer, IPPOConfig
    from src.visualization.training_plots import LivePlotter

    # Create environment with shared rewards
    env = CleanupEnvironment(
        num_agents=args.num_agents,
        common_reward=True,  # Shared rewards for cooperation
        seed=args.seed,
    )

    # Configure trainer (optimized for GPU)
    config = IPPOConfig(
        total_timesteps=args.timesteps,
        rollout_length=args.rollout_length,  # Larger = better GPU utilization
        num_envs=args.num_envs,  # Parallel environments
        learning_rate=2.5e-4,
        num_epochs=4,
        hidden_dims=(128, 128),  # Larger MLP to compensate for no CNN
        parameter_sharing=True,
        use_cnn=False,  # Use MLP instead of CNN (cuDNN compatibility)
        # Social Influence (Jaques et al., 2019)
        use_social_influence=args.social_influence,
        influence_weight=args.influence_weight,
        # Collective Future Prediction
        use_future_prediction=args.future_prediction,
        future_prediction_weight=args.future_weight,
    )

    if args.social_influence:
        print(f"Social Influence ENABLED with λ={args.influence_weight}")

    if args.future_prediction:
        print(f"Future Prediction ENABLED with weight={args.future_weight}")

    # Environment creator for parallel envs
    def env_creator(seed):
        return CleanupEnvironment(
            num_agents=args.num_agents,
            common_reward=True,
            seed=seed,
        )

    # Create trainer
    trainer = IPPOTrainer(env, config, seed=args.seed, env_creator=env_creator)

    # Setup visualization
    plotter = None
    if not args.no_plots:
        plotter = LivePlotter(
            update_interval=args.log_interval,
            save_dir=args.plots_dir,
        )

        def training_callback(iteration, metrics):
            """Callback to update visualization during training."""
            plotter.update(
                timestep=trainer.total_steps,
                mean_reward=metrics.get('episode_rewards', [0])[-1] if metrics.get('episode_rewards') else 0,
                cleaning_rate=metrics.get('cleaning_rate', 0),
                policy_loss=metrics.get('policy_loss', 0),
                value_loss=metrics.get('value_loss', 0),
                entropy=metrics.get('entropy', 0),
            )
    else:
        training_callback = None

    # Train
    training_info = trainer.train(
        log_interval=args.log_interval,
        callback=training_callback,
    )

    # Save checkpoint
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(output_path))

    # Save plots
    if plotter:
        plotter.save_plots(prefix="cooperative")

    # Print final info
    print(f"Checkpoint saved to: {output_path}")
    if not args.no_plots:
        print(f"Plots saved to: {args.plots_dir}")


if __name__ == "__main__":
    main()
