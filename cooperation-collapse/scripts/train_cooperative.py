#!/usr/bin/env python3
"""
Train Cooperative Agents (Phase 1)

Trains N agents with shared rewards to encourage cooperation.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environments.cleanup_env import CleanupEnvironment
from src.algorithms.ippo.trainer import IPPOTrainer, IPPOConfig
from src.visualization.training_plots import LivePlotter


def main():
    parser = argparse.ArgumentParser(description="Train cooperative agents")
    parser.add_argument("--num-agents", type=int, default=8, help="Number of agents")
    parser.add_argument("--timesteps", type=int, default=5_000_000, help="Total timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="checkpoints/cooperative/best.pkl",
                        help="Output checkpoint path")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N iterations")
    parser.add_argument("--plots-dir", type=str, default="outputs/plots/cooperative",
                        help="Directory to save plots")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting")
    parser.add_argument("--rollout-length", type=int, default=512,
                        help="Rollout length (larger = faster on GPU)")
    args = parser.parse_args()

    print("=" * 60)
    print("COOPERATIVE TRAINING")
    print("=" * 60)
    print(f"Agents: {args.num_agents}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Seed: {args.seed}")
    print()

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
        learning_rate=2.5e-4,
        num_epochs=4,
        hidden_dims=(64, 64),
        parameter_sharing=True,
    )

    print(f"Rollout length: {args.rollout_length}")
    print()

    # Create trainer
    trainer = IPPOTrainer(env, config, seed=args.seed)

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
        summary = plotter.get_summary()
        print(f"\nMax cleaning rate achieved: {summary['cleaning_rate']['max']:.3f}")
        print(f"Mean cleaning rate: {summary['cleaning_rate']['mean']:.3f}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final cleaning rate: {training_info['cleaning_rates'][-1]:.3f}")
    print(f"Checkpoint saved to: {output_path}")
    if not args.no_plots:
        print(f"Plots saved to: {args.plots_dir}")


if __name__ == "__main__":
    main()
