"""
Render video of trained agents in Cleanup environment.
"""

import pickle
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import jax
import jax.numpy as jnp

try:
    import imageio
except ImportError:
    print("Installing imageio...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio[ffmpeg]"])
    import imageio

from src.environments.cleanup_env import CleanupEnvironment
from src.algorithms.ippo.network import ActorCritic


def load_checkpoint(path: str):
    """Load trained model checkpoint."""
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


def render_episode(
    env: CleanupEnvironment,
    network: ActorCritic,
    params: dict,
    rng: jax.random.PRNGKey,
    max_steps: int = 500,
    deterministic: bool = False,
) -> list:
    """
    Render one episode and return frames.

    Args:
        env: Cleanup environment
        network: Policy network
        params: Network parameters
        rng: Random key
        max_steps: Maximum steps to render
        deterministic: If True, use argmax instead of sampling

    Returns:
        List of RGB frames
    """
    frames = []
    obs_dict, state = env.reset()

    for step in range(max_steps):
        # Render current state
        frame = env.render()
        frames.append(frame)

        # Get actions for all agents
        actions = {}
        for agent_id in range(env.num_agents):
            obs = obs_dict[agent_id]
            obs_batch = obs[None, ...]  # Add batch dimension

            rng, action_rng = jax.random.split(rng)
            logits, _, _ = network.apply(params, obs_batch)

            if deterministic:
                action = int(jnp.argmax(logits[0]))
            else:
                action = int(jax.random.categorical(action_rng, logits[0]))

            actions[agent_id] = action

        # Step environment
        obs_dict, state, rewards, dones, infos = env.step(actions)

        # Check if episode done
        if all(dones.values()):
            break

    # Add final frame
    frames.append(env.render())

    return frames


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Render video of trained agents")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/cooperative/best.pkl",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/videos/episode.mp4",
        help="Output video path"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Maximum steps to render"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (argmax)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint['config']
    params = checkpoint['params']

    print(f"Checkpoint loaded. Total training steps: {checkpoint['total_steps']}")

    # Create environment (default 8 agents as in training)
    env = CleanupEnvironment(
        num_agents=8,
        grid_size=(25, 18),
        max_steps=args.steps,
        seed=args.seed,
    )

    # Create network with same config as training
    network = ActorCritic(
        action_dim=env.action_space_size,
        hidden_dims=getattr(config, 'hidden_dims', (64, 64)),
        use_cnn=getattr(config, 'use_cnn', True),
        use_rnn=getattr(config, 'use_rnn', False),
    )

    # Render episode
    print(f"Rendering episode (max {args.steps} steps)...")
    rng = jax.random.PRNGKey(args.seed)

    frames = render_episode(
        env=env,
        network=network,
        params=params,
        rng=rng,
        max_steps=args.steps,
        deterministic=args.deterministic,
    )

    print(f"Captured {len(frames)} frames")

    # Save video
    print(f"Saving video to {args.output}...")

    if args.output.endswith('.gif'):
        imageio.mimsave(args.output, frames, fps=args.fps)
    else:
        imageio.mimsave(args.output, frames, fps=args.fps, codec='libx264')

    print(f"Video saved! Duration: {len(frames) / args.fps:.1f}s")

    # Print episode stats
    print(f"\nEpisode stats:")
    print(f"  Cleaning rate: {env.get_cleaning_rate():.1%}")
    print(f"  Cleaning actions: {env.episode_cleaning_actions}")
    print(f"  Collection actions: {env.episode_collection_actions}")


if __name__ == "__main__":
    main()
