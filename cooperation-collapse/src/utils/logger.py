"""
Training Logger with Rich Formatting

Beautiful console output for IPPO training with progress bars,
colored metrics, and real-time statistics.
"""

import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import sys


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    # Background
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"
    BG_RED = "\033[41m"


def colored(text: str, color: str) -> str:
    """Wrap text with color code."""
    return f"{color}{text}{Colors.RESET}"


def progress_bar(current: int, total: int, width: int = 30,
                 fill_char: str = "█", empty_char: str = "░") -> str:
    """Create a progress bar string."""
    progress = current / total if total > 0 else 0
    filled = int(width * progress)
    bar = fill_char * filled + empty_char * (width - filled)
    percent = progress * 100
    return f"[{bar}] {percent:5.1f}%"


def format_number(n: float, precision: int = 2) -> str:
    """Format number with K/M suffixes."""
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.{precision}f}M"
    elif abs(n) >= 1_000:
        return f"{n/1_000:.{precision}f}K"
    else:
        return f"{n:.{precision}f}"


def format_time(seconds: float) -> str:
    """Format seconds to human readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours:.0f}h {mins:.0f}m"


@dataclass
class TrainingMetrics:
    """Accumulator for training metrics."""
    rewards: List[float] = field(default_factory=list)
    cleaning_rates: List[float] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)
    pollution_levels: List[float] = field(default_factory=list)
    apple_counts: List[float] = field(default_factory=list)

    def add(self, **kwargs):
        """Add metrics from a training step."""
        if 'reward' in kwargs:
            self.rewards.append(kwargs['reward'])
        if 'cleaning_rate' in kwargs:
            self.cleaning_rates.append(kwargs['cleaning_rate'])
        if 'policy_loss' in kwargs:
            self.policy_losses.append(kwargs['policy_loss'])
        if 'value_loss' in kwargs:
            self.value_losses.append(kwargs['value_loss'])
        if 'entropy' in kwargs:
            self.entropies.append(kwargs['entropy'])
        if 'pollution_level' in kwargs:
            self.pollution_levels.append(kwargs['pollution_level'])
        if 'apple_count' in kwargs:
            self.apple_counts.append(kwargs['apple_count'])

    def get_recent_avg(self, metric: str, window: int = 100) -> float:
        """Get recent average of a metric."""
        data = getattr(self, metric, [])
        if not data:
            return 0.0
        recent = data[-window:]
        return sum(recent) / len(recent)

    def get_best(self, metric: str) -> float:
        """Get best value of a metric."""
        data = getattr(self, metric, [])
        if not data:
            return 0.0
        if metric in ['policy_losses', 'value_losses', 'pollution_levels']:
            return min(data)  # Lower is better
        return max(data)  # Higher is better


class TrainingLogger:
    """
    Rich training logger with progress visualization.

    Features:
    - Progress bar with ETA
    - Colored metric display
    - Rolling averages
    - Speed tracking
    - Best metric tracking
    """

    def __init__(
        self,
        total_timesteps: int,
        num_agents: int = 8,
        log_interval: int = 10,
        show_header: bool = True,
    ):
        self.total_timesteps = total_timesteps
        self.num_agents = num_agents
        self.log_interval = log_interval

        self.metrics = TrainingMetrics()
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_steps = 0

        self.iteration = 0
        self.current_steps = 0

        if show_header:
            self._print_header()

    def _print_header(self):
        """Print training header."""
        print()
        print(colored("╔" + "═" * 70 + "╗", Colors.CYAN))
        print(colored("║", Colors.CYAN) + colored("  IPPO TRAINING - Cleanup Environment  ".center(70), Colors.BOLD + Colors.WHITE) + colored("║", Colors.CYAN))
        print(colored("╠" + "═" * 70 + "╣", Colors.CYAN))
        print(colored("║", Colors.CYAN) + f"  Target: {format_number(self.total_timesteps)} steps | Agents: {self.num_agents}".ljust(70) + colored("║", Colors.CYAN))
        print(colored("╚" + "═" * 70 + "╝", Colors.CYAN))
        print()

        # Column headers
        header = (
            f"{'Progress':<35} │ "
            f"{'Reward':>8} │ "
            f"{'Clean%':>7} │ "
            f"{'Loss':>8} │ "
            f"{'Speed':>10}"
        )
        print(colored("─" * 80, Colors.DIM))
        print(colored(header, Colors.BOLD))
        print(colored("─" * 80, Colors.DIM))

    def _get_speed(self) -> float:
        """Calculate steps per second."""
        now = time.time()
        elapsed = now - self.last_log_time
        steps_done = self.current_steps - self.last_log_steps

        if elapsed > 0:
            return steps_done / elapsed
        return 0.0

    def _get_eta(self) -> str:
        """Estimate time remaining."""
        elapsed = time.time() - self.start_time
        if self.current_steps > 0:
            rate = self.current_steps / elapsed
            remaining_steps = self.total_timesteps - self.current_steps
            if rate > 0:
                eta_seconds = remaining_steps / rate
                return format_time(eta_seconds)
        return "???"

    def _color_for_cleaning_rate(self, rate: float) -> str:
        """Get color based on cleaning rate."""
        if rate >= 0.3:
            return Colors.GREEN
        elif rate >= 0.15:
            return Colors.YELLOW
        else:
            return Colors.RED

    def _color_for_reward(self, reward: float) -> str:
        """Get color based on reward."""
        if reward >= 50:
            return Colors.GREEN
        elif reward >= 20:
            return Colors.YELLOW
        else:
            return Colors.WHITE

    def log(
        self,
        iteration: int,
        steps: int,
        reward: float,
        cleaning_rate: float,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        pollution_level: float = 0.0,
        apple_count: int = 0,
        extra_info: Optional[Dict] = None,
    ):
        """Log training progress."""
        self.iteration = iteration
        self.current_steps = steps

        # Store metrics
        self.metrics.add(
            reward=reward,
            cleaning_rate=cleaning_rate,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            pollution_level=pollution_level,
            apple_count=apple_count,
        )

        # Only print at intervals
        if iteration % self.log_interval != 0:
            return

        # Calculate speed
        speed = self._get_speed()
        self.last_log_time = time.time()
        self.last_log_steps = steps

        # Progress bar
        bar = progress_bar(steps, self.total_timesteps, width=20)
        eta = self._get_eta()
        progress_str = f"{bar} ETA: {eta}"

        # Format metrics with colors
        reward_color = self._color_for_reward(reward)
        clean_color = self._color_for_cleaning_rate(cleaning_rate)

        reward_str = colored(f"{reward:8.1f}", reward_color)
        clean_str = colored(f"{cleaning_rate*100:6.1f}%", clean_color)
        loss_str = f"{policy_loss:8.4f}"
        speed_str = f"{format_number(speed, 0)}/s"

        # Build line
        line = (
            f"{progress_str:<35} │ "
            f"{reward_str} │ "
            f"{clean_str} │ "
            f"{loss_str} │ "
            f"{speed_str:>10}"
        )

        print(line)

        # Print detailed stats every 50 iterations
        if iteration > 0 and iteration % 50 == 0:
            self._print_detailed_stats()

    def _print_detailed_stats(self):
        """Print detailed statistics."""
        print(colored("─" * 80, Colors.DIM))

        # Recent averages
        avg_reward = self.metrics.get_recent_avg('rewards', 50)
        avg_clean = self.metrics.get_recent_avg('cleaning_rates', 50)
        best_clean = self.metrics.get_best('cleaning_rates')
        avg_entropy = self.metrics.get_recent_avg('entropies', 50)

        stats = (
            f"  📊 "
            f"Avg(50): {colored(f'{avg_reward:.1f}', Colors.CYAN)} reward, "
            f"{colored(f'{avg_clean*100:.1f}%', Colors.CYAN)} clean │ "
            f"Best clean: {colored(f'{best_clean*100:.1f}%', Colors.GREEN)} │ "
            f"Entropy: {avg_entropy:.3f}"
        )
        print(stats)
        print(colored("─" * 80, Colors.DIM))

    def log_episode_end(
        self,
        episode: int,
        total_reward: float,
        cleaning_actions: int,
        collection_actions: int,
        final_pollution: float,
        final_apples: int,
    ):
        """Log end of episode summary (optional detailed logging)."""
        clean_ratio = cleaning_actions / max(cleaning_actions + collection_actions, 1)

        print(colored(f"  Episode {episode}: ", Colors.DIM) +
              f"R={total_reward:.1f} | "
              f"Clean={cleaning_actions} ({clean_ratio*100:.0f}%) | "
              f"Collect={collection_actions} | "
              f"Poll={final_pollution:.2f}")

    def print_summary(self):
        """Print final training summary."""
        elapsed = time.time() - self.start_time

        print()
        print(colored("╔" + "═" * 70 + "╗", Colors.GREEN))
        print(colored("║", Colors.GREEN) + colored("  TRAINING COMPLETE  ".center(70), Colors.BOLD + Colors.WHITE) + colored("║", Colors.GREEN))
        print(colored("╠" + "═" * 70 + "╣", Colors.GREEN))

        # Final stats
        final_reward = self.metrics.rewards[-1] if self.metrics.rewards else 0
        final_clean = self.metrics.cleaning_rates[-1] if self.metrics.cleaning_rates else 0
        best_clean = self.metrics.get_best('cleaning_rates')
        avg_clean = self.metrics.get_recent_avg('cleaning_rates', 100)
        avg_speed = self.current_steps / elapsed if elapsed > 0 else 0

        lines = [
            f"  Total Steps: {format_number(self.current_steps)}",
            f"  Total Time:  {format_time(elapsed)}",
            f"  Avg Speed:   {format_number(avg_speed)}/s",
            "",
            f"  Final Reward:       {final_reward:.1f}",
            f"  Final Cleaning:     {final_clean*100:.1f}%",
            f"  Best Cleaning:      {best_clean*100:.1f}%",
            f"  Avg Cleaning (100): {avg_clean*100:.1f}%",
        ]

        for line in lines:
            print(colored("║", Colors.GREEN) + line.ljust(70) + colored("║", Colors.GREEN))

        print(colored("╚" + "═" * 70 + "╝", Colors.GREEN))
        print()

    def print_milestone(self, message: str):
        """Print a milestone message."""
        print()
        print(colored(f"  🎯 {message}", Colors.YELLOW + Colors.BOLD))
        print()


def create_logger(
    total_timesteps: int,
    num_agents: int = 8,
    log_interval: int = 10,
) -> TrainingLogger:
    """Create a training logger instance."""
    return TrainingLogger(
        total_timesteps=total_timesteps,
        num_agents=num_agents,
        log_interval=log_interval,
    )
