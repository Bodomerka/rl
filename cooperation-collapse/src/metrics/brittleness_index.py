"""
Cooperation Brittleness Index (CBI)

A novel metric for measuring how quickly and severely cooperation collapses
when a defector is introduced into a cooperative multi-agent system.

CBI = (Collapse_Speed × 0.3 + Collapse_Depth × 0.4 + Recovery_Difficulty × 0.3) / Baseline_Stability

Components:
- Collapse_Speed: How quickly cooperation metrics decline (time to 50% drop)
- Collapse_Depth: Maximum deviation from cooperative baseline
- Recovery_Difficulty: Whether system can recover (0-1 scale)
- Baseline_Stability: Variance in cooperation before defector introduction
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class CooperationState:
    """Snapshot of cooperation metrics at a timestep."""
    timestep: int
    cleaning_rate: float
    collective_reward: float
    pollution_level: float
    apple_count: int
    gini_coefficient: float
    per_agent_rewards: Dict[int, float] = field(default_factory=dict)


class CooperationBrittlenessIndex:
    """
    Computes the Cooperation Brittleness Index (CBI) from experiment data.

    Usage:
        cbi = CooperationBrittlenessIndex()

        # Record baseline (before defector)
        for state in baseline_states:
            cbi.record_baseline(state)

        # Record post-injection
        for state in post_injection_states:
            cbi.record_post_injection(state)

        # Compute CBI
        results = cbi.compute()
    """

    def __init__(
        self,
        collapse_threshold: float = 0.5,
        baseline_window: int = 100,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize CBI calculator.

        Args:
            collapse_threshold: Threshold for "collapse" (e.g., 0.5 = 50% drop)
            baseline_window: Number of steps to use for baseline calculation
            weights: Custom weights for CBI components
        """
        self.collapse_threshold = collapse_threshold
        self.baseline_window = baseline_window

        # Default weights (sum to 1.0)
        self.weights = weights or {
            'collapse_speed': 0.3,
            'collapse_depth': 0.4,
            'recovery_difficulty': 0.3,
        }

        # Storage
        self.baseline_states: List[CooperationState] = []
        self.post_injection_states: List[CooperationState] = []
        self.injection_timestep: Optional[int] = None

    def record_baseline(self, state: CooperationState):
        """Record cooperation state before defector injection."""
        self.baseline_states.append(state)

    def record_post_injection(self, state: CooperationState):
        """Record cooperation state after defector injection."""
        if self.injection_timestep is None:
            self.injection_timestep = state.timestep
        self.post_injection_states.append(state)

    def clear(self):
        """Clear all recorded data."""
        self.baseline_states = []
        self.post_injection_states = []
        self.injection_timestep = None

    def compute_baseline_metrics(self) -> Dict[str, float]:
        """
        Compute baseline cooperation metrics.

        Returns:
            Dictionary with mean and std of key metrics
        """
        if len(self.baseline_states) < 10:
            raise ValueError("Insufficient baseline data (need at least 10 steps)")

        # Use last N steps for baseline
        window = min(self.baseline_window, len(self.baseline_states))
        recent = self.baseline_states[-window:]

        cleaning_rates = [s.cleaning_rate for s in recent]
        collective_rewards = [s.collective_reward for s in recent]
        pollution_levels = [s.pollution_level for s in recent]

        return {
            'mean_cleaning_rate': np.mean(cleaning_rates),
            'std_cleaning_rate': np.std(cleaning_rates),
            'mean_collective_reward': np.mean(collective_rewards),
            'std_collective_reward': np.std(collective_rewards),
            'mean_pollution': np.mean(pollution_levels),
            'std_pollution': np.std(pollution_levels),
        }

    def compute_collapse_speed(self, baseline: Dict[str, float]) -> Tuple[float, int]:
        """
        Compute how quickly cooperation collapses.

        Args:
            baseline: Baseline metrics dictionary

        Returns:
            normalized_speed: 0 (slow/no collapse) to 1 (instant collapse)
            timesteps: Actual timesteps to reach threshold
        """
        if not self.post_injection_states:
            return 0.0, 0

        target = baseline['mean_cleaning_rate'] * self.collapse_threshold

        # Find first timestep below threshold
        for i, state in enumerate(self.post_injection_states):
            if state.cleaning_rate <= target:
                # Normalize: faster collapse = higher value
                # Assume max expected collapse time of 500 timesteps
                max_collapse_time = 500
                normalized_speed = 1 - (i / max_collapse_time)
                return max(0.0, min(1.0, normalized_speed)), i + 1

        # Never collapsed
        return 0.0, len(self.post_injection_states)

    def compute_collapse_depth(self, baseline: Dict[str, float]) -> float:
        """
        Compute maximum depth of cooperation collapse.

        Returns:
            depth: 0 (no collapse) to 1 (complete collapse)
        """
        if not self.post_injection_states:
            return 0.0

        min_cleaning = min(s.cleaning_rate for s in self.post_injection_states)
        baseline_cleaning = baseline['mean_cleaning_rate']

        if baseline_cleaning <= 0:
            return 0.0

        # Depth is the percentage drop
        depth = 1 - (min_cleaning / baseline_cleaning)
        return max(0.0, min(1.0, depth))

    def compute_recovery_difficulty(self, baseline: Dict[str, float]) -> float:
        """
        Compute whether/how much the system recovers.

        Returns:
            difficulty: 0 (full recovery) to 1 (no recovery)
        """
        if len(self.post_injection_states) < 50:
            return 1.0  # Not enough data to assess recovery

        # Look at last 50 steps
        final_states = self.post_injection_states[-50:]
        final_cleaning = np.mean([s.cleaning_rate for s in final_states])
        baseline_cleaning = baseline['mean_cleaning_rate']

        if baseline_cleaning <= 0:
            return 1.0

        recovery_ratio = final_cleaning / baseline_cleaning
        difficulty = 1 - min(recovery_ratio, 1.0)
        return max(0.0, min(1.0, difficulty))

    def compute_baseline_stability(self, baseline: Dict[str, float]) -> float:
        """
        Compute how stable cooperation was before injection.

        Returns:
            stability: Coefficient of variation (lower = more stable)
        """
        cv = baseline['std_cleaning_rate'] / (baseline['mean_cleaning_rate'] + 1e-8)
        # Return minimum of 0.01 to avoid division issues
        return max(cv, 0.01)

    def compute(self) -> Dict[str, float]:
        """
        Compute the full Cooperation Brittleness Index.

        Returns:
            Dictionary containing:
            - cbi: The main Brittleness Index (higher = more brittle)
            - collapse_speed: Normalized collapse speed
            - collapse_speed_timesteps: Actual timesteps to collapse
            - collapse_depth: Maximum drop magnitude
            - recovery_difficulty: Ability to recover
            - baseline_stability: Pre-injection stability
        """
        # Compute baseline metrics
        baseline = self.compute_baseline_metrics()

        # Compute components
        collapse_speed, collapse_timesteps = self.compute_collapse_speed(baseline)
        collapse_depth = self.compute_collapse_depth(baseline)
        recovery_difficulty = self.compute_recovery_difficulty(baseline)
        baseline_stability = self.compute_baseline_stability(baseline)

        # Compute CBI
        cbi = (
            self.weights['collapse_speed'] * collapse_speed +
            self.weights['collapse_depth'] * collapse_depth +
            self.weights['recovery_difficulty'] * recovery_difficulty
        ) / (baseline_stability + 0.1)

        return {
            'cbi': cbi,
            'collapse_speed': collapse_speed,
            'collapse_speed_timesteps': collapse_timesteps,
            'collapse_depth': collapse_depth,
            'recovery_difficulty': recovery_difficulty,
            'baseline_stability': baseline_stability,
            'baseline_cleaning_rate': baseline['mean_cleaning_rate'],
            'baseline_collective_reward': baseline['mean_collective_reward'],
        }

    def compute_extended_metrics(self) -> Dict[str, float]:
        """
        Compute additional analysis metrics beyond CBI.

        Returns:
            Extended metrics dictionary
        """
        base_metrics = self.compute()

        # Add cascade effect analysis
        cascade_effect = self._compute_cascade_effect()
        contagion_rate = self._compute_contagion_rate()
        equilibrium_shift = self._compute_equilibrium_shift()

        return {
            **base_metrics,
            'cascade_effect': cascade_effect,
            'contagion_rate': contagion_rate,
            'equilibrium_shift': equilibrium_shift,
        }

    def _compute_cascade_effect(self) -> float:
        """
        Measure if cooperative agents start defecting (cascade).

        Returns:
            cascade: 0 (no cascade) to 1 (complete cascade)
        """
        if len(self.post_injection_states) < 2:
            return 0.0

        # Compare first and last cleaning rates
        early_cleaning = np.mean([s.cleaning_rate for s in self.post_injection_states[:10]])
        late_cleaning = np.mean([s.cleaning_rate for s in self.post_injection_states[-10:]])

        if early_cleaning <= 0:
            return 0.0

        # Cascade = reduction in cleaning beyond initial drop
        reduction = (early_cleaning - late_cleaning) / early_cleaning
        return max(0.0, min(1.0, reduction))

    def _compute_contagion_rate(self) -> float:
        """
        Measure how quickly defection spreads.

        Returns:
            rate: Rate of cleaning decline per timestep
        """
        if len(self.post_injection_states) < 10:
            return 0.0

        cleaning_rates = [s.cleaning_rate for s in self.post_injection_states]

        # Fit linear trend
        x = np.arange(len(cleaning_rates))
        slope, _ = np.polyfit(x, cleaning_rates, 1)

        # Negative slope = spreading defection
        return max(0.0, -slope)

    def _compute_equilibrium_shift(self) -> float:
        """
        Measure permanent shift in equilibrium behavior.

        Returns:
            shift: 0 (same equilibrium) to 1 (completely different)
        """
        if len(self.post_injection_states) < 50 or len(self.baseline_states) < 50:
            return 0.0

        baseline = self.compute_baseline_metrics()
        final_cleaning = np.mean([s.cleaning_rate for s in self.post_injection_states[-50:]])

        baseline_cleaning = baseline['mean_cleaning_rate']
        if baseline_cleaning <= 0:
            return 0.0

        shift = abs(final_cleaning - baseline_cleaning) / baseline_cleaning
        return min(1.0, shift)

    def get_time_series(self) -> Dict[str, List]:
        """
        Get time series data for visualization.

        Returns:
            Dictionary with lists for each metric over time
        """
        all_states = self.baseline_states + self.post_injection_states

        return {
            'timesteps': [s.timestep for s in all_states],
            'cleaning_rates': [s.cleaning_rate for s in all_states],
            'collective_rewards': [s.collective_reward for s in all_states],
            'pollution_levels': [s.pollution_level for s in all_states],
            'gini_coefficients': [s.gini_coefficient for s in all_states],
            'injection_timestep': self.injection_timestep,
        }


# Convenience type for external imports
from typing import Tuple
