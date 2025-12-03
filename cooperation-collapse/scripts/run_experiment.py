#!/usr/bin/env python3
"""
Run Full Experiment

Executes the complete three-phase experiment:
1. Train cooperative agents
2. Train defector agent
3. Measure cooperation collapse
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.experiment_runner import ExperimentRunner, ExperimentConfig


def main():
    parser = argparse.ArgumentParser(description="Run cooperation collapse experiment")

    # Environment settings
    parser.add_argument("--num-agents", type=int, default=8, help="Number of agents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Phase 1 settings
    parser.add_argument("--phase1-timesteps", type=int, default=5_000_000,
                        help="Phase 1 training timesteps")
    parser.add_argument("--early-stop-threshold", type=float, default=0.3,
                        help="Cleaning rate threshold for early stopping")

    # Phase 2 settings
    parser.add_argument("--phase2-timesteps", type=int, default=2_000_000,
                        help="Phase 2 training timesteps")
    parser.add_argument("--defector-id", type=int, default=0,
                        help="Which agent becomes defector")
    parser.add_argument("--apple-bonus", type=float, default=1.5,
                        help="Defector apple reward multiplier")
    parser.add_argument("--cleaning-penalty", type=float, default=-0.2,
                        help="Defector penalty for cleaning")

    # Phase 3 settings
    parser.add_argument("--baseline-steps", type=int, default=500,
                        help="Steps to establish baseline")
    parser.add_argument("--observation-steps", type=int, default=2000,
                        help="Steps to observe collapse")

    # Output settings
    parser.add_argument("--output-dir", type=str, default="outputs/experiments",
                        help="Output directory")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of runs for statistical significance")

    args = parser.parse_args()

    # Create config
    config = ExperimentConfig(
        num_agents=args.num_agents,
        seed=args.seed,
        phase1_timesteps=args.phase1_timesteps,
        phase1_early_stop_threshold=args.early_stop_threshold,
        phase2_timesteps=args.phase2_timesteps,
        defector_agent_id=args.defector_id,
        defector_apple_bonus=args.apple_bonus,
        defector_cleaning_penalty=args.cleaning_penalty,
        baseline_steps=args.baseline_steps,
        observation_steps=args.observation_steps,
        output_dir=args.output_dir,
        num_runs=args.num_runs,
    )

    # Run experiment
    runner = ExperimentRunner(config)
    results = runner.run_full_experiment()

    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Cooperation Brittleness Index (CBI): {results['brittleness_index']:.4f}")
    print(f"Baseline Cleaning Rate: {results['baseline_metrics']['cleaning_rate']:.3f}")
    print(f"Collapse Speed: {results['collapse_results']['collapse_speed_timesteps']} steps")
    print(f"Collapse Depth: {results['collapse_results']['collapse_depth']*100:.1f}%")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
