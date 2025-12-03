"""Training visualization utilities."""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class TrainingMetrics:
    """Container for training metrics history."""

    iterations: List[int] = field(default_factory=list)
    timesteps: List[int] = field(default_factory=list)
    mean_rewards: List[float] = field(default_factory=list)
    cleaning_rates: List[float] = field(default_factory=list)
    apple_rates: List[float] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)

    def add(
        self,
        iteration: int,
        timestep: int,
        mean_reward: float,
        cleaning_rate: float,
        apple_rate: float = 0.0,
        policy_loss: float = 0.0,
        value_loss: float = 0.0,
        entropy: float = 0.0,
    ):
        """Add metrics for one iteration."""
        self.iterations.append(iteration)
        self.timesteps.append(timestep)
        self.mean_rewards.append(mean_reward)
        self.cleaning_rates.append(cleaning_rate)
        self.apple_rates.append(apple_rate)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)

    def save(self, path: str):
        """Save metrics to JSON."""
        data = {
            "iterations": self.iterations,
            "timesteps": self.timesteps,
            "mean_rewards": self.mean_rewards,
            "cleaning_rates": self.cleaning_rates,
            "apple_rates": self.apple_rates,
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses,
            "entropies": self.entropies,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingMetrics":
        """Load metrics from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        metrics = cls()
        metrics.iterations = data["iterations"]
        metrics.timesteps = data["timesteps"]
        metrics.mean_rewards = data["mean_rewards"]
        metrics.cleaning_rates = data["cleaning_rates"]
        metrics.apple_rates = data.get("apple_rates", [])
        metrics.policy_losses = data.get("policy_losses", [])
        metrics.value_losses = data.get("value_losses", [])
        metrics.entropies = data.get("entropies", [])
        return metrics


class TrainingVisualizer:
    """Static visualization of training metrics."""

    def __init__(self, metrics: TrainingMetrics):
        self.metrics = metrics

    def plot_rewards(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        window: int = 10,
    ):
        """Plot reward curve with smoothing."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        timesteps = np.array(self.metrics.timesteps)
        rewards = np.array(self.metrics.mean_rewards)

        # Raw data (transparent)
        ax.plot(timesteps, rewards, alpha=0.3, color="blue", label="Raw")

        # Smoothed data
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
            smoothed_steps = timesteps[window-1:]
            ax.plot(smoothed_steps, smoothed, color="blue", linewidth=2,
                   label=f"Smoothed (window={window})")

        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Training Reward Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)
        return fig

    def plot_cleaning_rate(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        window: int = 10,
    ):
        """Plot cleaning rate over time."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        timesteps = np.array(self.metrics.timesteps)
        rates = np.array(self.metrics.cleaning_rates)

        # Raw data
        ax.plot(timesteps, rates, alpha=0.3, color="green", label="Raw")

        # Smoothed
        if len(rates) >= window:
            smoothed = np.convolve(rates, np.ones(window)/window, mode="valid")
            smoothed_steps = timesteps[window-1:]
            ax.plot(smoothed_steps, smoothed, color="green", linewidth=2,
                   label=f"Smoothed (window={window})")

        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Cleaning Rate")
        ax.set_title("Cleaning Rate During Training")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)
        return fig

    def plot_losses(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 4),
    ):
        """Plot policy and value losses."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        timesteps = np.array(self.metrics.timesteps)

        # Policy loss
        if self.metrics.policy_losses:
            axes[0].plot(timesteps, self.metrics.policy_losses, color="red")
            axes[0].set_title("Policy Loss")
            axes[0].set_xlabel("Timesteps")
            axes[0].grid(True, alpha=0.3)

        # Value loss
        if self.metrics.value_losses:
            axes[1].plot(timesteps, self.metrics.value_losses, color="orange")
            axes[1].set_title("Value Loss")
            axes[1].set_xlabel("Timesteps")
            axes[1].grid(True, alpha=0.3)

        # Entropy
        if self.metrics.entropies:
            axes[2].plot(timesteps, self.metrics.entropies, color="purple")
            axes[2].set_title("Entropy")
            axes[2].set_xlabel("Timesteps")
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)
        return fig

    def plot_dashboard(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
        window: int = 10,
    ):
        """Create comprehensive training dashboard."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize)

        # Create grid
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        timesteps = np.array(self.metrics.timesteps)

        # 1. Rewards (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        rewards = np.array(self.metrics.mean_rewards)
        ax1.plot(timesteps, rewards, alpha=0.3, color="blue")
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
            ax1.plot(timesteps[window-1:], smoothed, color="blue", linewidth=2)
        ax1.set_title("Mean Reward", fontweight="bold")
        ax1.set_xlabel("Timesteps")
        ax1.grid(True, alpha=0.3)

        # 2. Cleaning Rate (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        rates = np.array(self.metrics.cleaning_rates)
        ax2.plot(timesteps, rates, alpha=0.3, color="green")
        if len(rates) >= window:
            smoothed = np.convolve(rates, np.ones(window)/window, mode="valid")
            ax2.plot(timesteps[window-1:], smoothed, color="green", linewidth=2)
        ax2.set_title("Cleaning Rate", fontweight="bold")
        ax2.set_xlabel("Timesteps")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # 3. Apple Collection Rate (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        if self.metrics.apple_rates:
            apples = np.array(self.metrics.apple_rates)
            ax3.plot(timesteps, apples, alpha=0.3, color="red")
            if len(apples) >= window:
                smoothed = np.convolve(apples, np.ones(window)/window, mode="valid")
                ax3.plot(timesteps[window-1:], smoothed, color="red", linewidth=2)
        ax3.set_title("Apple Collection Rate", fontweight="bold")
        ax3.set_xlabel("Timesteps")
        ax3.grid(True, alpha=0.3)

        # 4. Policy & Value Loss (middle right)
        ax4 = fig.add_subplot(gs[1, 1])
        if self.metrics.policy_losses:
            ax4.plot(timesteps, self.metrics.policy_losses, label="Policy", color="red", alpha=0.7)
        if self.metrics.value_losses:
            ax4_twin = ax4.twinx()
            ax4_twin.plot(timesteps, self.metrics.value_losses, label="Value", color="orange", alpha=0.7)
            ax4_twin.set_ylabel("Value Loss", color="orange")
        ax4.set_title("Training Losses", fontweight="bold")
        ax4.set_xlabel("Timesteps")
        ax4.set_ylabel("Policy Loss", color="red")
        ax4.grid(True, alpha=0.3)

        # 5. Entropy (bottom, spanning both columns)
        ax5 = fig.add_subplot(gs[2, :])
        if self.metrics.entropies:
            ax5.plot(timesteps, self.metrics.entropies, color="purple", alpha=0.7)
            ax5.fill_between(timesteps, self.metrics.entropies, alpha=0.2, color="purple")
        ax5.set_title("Policy Entropy (Exploration)", fontweight="bold")
        ax5.set_xlabel("Timesteps")
        ax5.grid(True, alpha=0.3)

        plt.suptitle("Training Progress Dashboard", fontsize=14, fontweight="bold", y=1.02)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)
        return fig


class LivePlotter:
    """Real-time plotting during training (terminal-based)."""

    def __init__(
        self,
        update_interval: int = 10,
        save_dir: Optional[str] = None,
    ):
        self.update_interval = update_interval
        self.save_dir = Path(save_dir) if save_dir else None
        self.metrics = TrainingMetrics()
        self._iteration = 0

    def update(
        self,
        timestep: int,
        mean_reward: float,
        cleaning_rate: float,
        apple_rate: float = 0.0,
        policy_loss: float = 0.0,
        value_loss: float = 0.0,
        entropy: float = 0.0,
    ):
        """Update with new metrics and print progress bar."""
        self._iteration += 1

        self.metrics.add(
            iteration=self._iteration,
            timestep=timestep,
            mean_reward=mean_reward,
            cleaning_rate=cleaning_rate,
            apple_rate=apple_rate,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
        )

        # Print ASCII progress bar
        if self._iteration % self.update_interval == 0:
            self._print_progress(timestep, mean_reward, cleaning_rate)

    def _print_progress(self, timestep: int, reward: float, cleaning: float):
        """Print ASCII visualization."""
        # Cleaning rate bar
        bar_width = 30
        filled = int(cleaning * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Color codes for terminal
        if cleaning >= 0.5:
            color = "\033[92m"  # Green
        elif cleaning >= 0.2:
            color = "\033[93m"  # Yellow
        else:
            color = "\033[91m"  # Red
        reset = "\033[0m"

        print(f"Step {timestep:>8,} | Reward: {reward:>7.2f} | "
              f"Cleaning: {color}[{bar}]{reset} {cleaning:.1%}")

    def save_plots(self, prefix: str = "training"):
        """Save all plots to disk."""
        if self.save_dir is None:
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)

        visualizer = TrainingVisualizer(self.metrics)

        # Save individual plots
        visualizer.plot_rewards(str(self.save_dir / f"{prefix}_rewards.png"))
        visualizer.plot_cleaning_rate(str(self.save_dir / f"{prefix}_cleaning.png"))
        visualizer.plot_losses(str(self.save_dir / f"{prefix}_losses.png"))
        visualizer.plot_dashboard(str(self.save_dir / f"{prefix}_dashboard.png"))

        # Save metrics JSON
        self.metrics.save(str(self.save_dir / f"{prefix}_metrics.json"))

        print(f"\nPlots saved to: {self.save_dir}")

    def get_summary(self) -> Dict:
        """Get training summary statistics."""
        if not self.metrics.mean_rewards:
            return {}

        return {
            "total_iterations": self._iteration,
            "final_timestep": self.metrics.timesteps[-1] if self.metrics.timesteps else 0,
            "mean_reward": {
                "final": self.metrics.mean_rewards[-1],
                "max": max(self.metrics.mean_rewards),
                "mean": np.mean(self.metrics.mean_rewards),
            },
            "cleaning_rate": {
                "final": self.metrics.cleaning_rates[-1],
                "max": max(self.metrics.cleaning_rates),
                "mean": np.mean(self.metrics.cleaning_rates),
            },
        }


def plot_collapse_comparison(
    baseline_metrics: TrainingMetrics,
    collapse_metrics: TrainingMetrics,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
):
    """Compare baseline vs collapse metrics for CBI visualization."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Cleaning rates comparison
    ax1 = axes[0]
    ax1.plot(baseline_metrics.timesteps, baseline_metrics.cleaning_rates,
             color="green", label="Baseline (cooperative)", linewidth=2)
    ax1.plot(collapse_metrics.timesteps, collapse_metrics.cleaning_rates,
             color="red", label="With defector", linewidth=2)
    ax1.axhline(y=np.mean(baseline_metrics.cleaning_rates[-10:]),
                color="green", linestyle="--", alpha=0.5)
    ax1.set_title("Cleaning Rate: Baseline vs Collapse")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Cleaning Rate")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Rewards comparison
    ax2 = axes[1]
    ax2.plot(baseline_metrics.timesteps, baseline_metrics.mean_rewards,
             color="blue", label="Baseline", linewidth=2)
    ax2.plot(collapse_metrics.timesteps, collapse_metrics.mean_rewards,
             color="orange", label="With defector", linewidth=2)
    ax2.set_title("Mean Reward: Baseline vs Collapse")
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Mean Reward")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig
