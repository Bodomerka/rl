#!/usr/bin/env python3
"""
Train Cooperative Agents with JAX (GPU-accelerated)

Uses SocialJax environment and pure JAX training loop for ~50x speedup.
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
    print("SocialJax: Available")
except ImportError:
    SOCIALJAX_AVAILABLE = False
    print("SocialJax: NOT AVAILABLE")
    print("\nTo install SocialJax:")
    print("  git clone https://github.com/cooperativex/socialjax")
    print("  cd socialjax && pip install -e .")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Train cooperative agents with JAX (GPU-accelerated)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environment
    parser.add_argument("--num-agents", type=int, default=8,
                        help="Number of agents")
    parser.add_argument("--num-envs", type=int, default=64,
                        help="Number of parallel environments (more = faster)")

    # Training
    parser.add_argument("--timesteps", type=int, default=5_000_000,
                        help="Total timesteps")
    parser.add_argument("--rollout-length", type=int, default=128,
                        help="Rollout length per update")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Network
    parser.add_argument("--use-cnn", action="store_true", default=True,
                        help="Use CNN network (default)")
    parser.add_argument("--use-mlp", action="store_true",
                        help="Use MLP network instead of CNN")

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=2.5e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip-epsilon", type=float, default=0.2,
                        help="PPO clip epsilon")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--num-epochs", type=int, default=4,
                        help="PPO update epochs")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="Number of minibatches")

    # Output
    parser.add_argument("--output", type=str, default="checkpoints/cooperative_jax/best.pkl",
                        help="Output checkpoint path")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N iterations")

    args = parser.parse_args()

    if not SOCIALJAX_AVAILABLE:
        print("ERROR: SocialJax is required for JAX training.")
        print("Install it first, then re-run this script.")
        sys.exit(1)

    # Import trainer
    from src.algorithms.ippo.jax_trainer import JaxIPPOTrainer, JaxIPPOConfig

    # Determine network type
    use_cnn = not args.use_mlp

    # Create config
    config = JaxIPPOConfig(
        num_agents=args.num_agents,
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        rollout_length=args.rollout_length,
        num_epochs=args.num_epochs,
        num_minibatches=args.num_minibatches,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        learning_rate=args.lr,
        use_cnn=use_cnn,
        log_interval=args.log_interval,
    )

    # Print expected speedup
    batch_size = args.num_envs * args.rollout_length * args.num_agents
    print(f"\nExpected performance:")
    print(f"  Batch size: {batch_size:,}")
    print(f"  Expected speed: ~10,000-50,000 steps/s (vs ~30 steps/s with NumPy)")
    print(f"  Expected time for {args.timesteps:,} steps: ~{args.timesteps / 30000 / 60:.0f}-{args.timesteps / 10000 / 60:.0f} minutes")
    print()

    # Create trainer
    trainer = JaxIPPOTrainer(config, seed=args.seed)

    # Train
    results = trainer.train()

    # Save checkpoint
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(output_path), results['train_state'])

    # Print final stats
    print(f"\nCheckpoint saved to: {output_path}")
    print(f"Final speed: {results['avg_speed']:,.0f} steps/s")


if __name__ == "__main__":
    main()
