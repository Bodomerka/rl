"""
Inequality Metrics for Multi-Agent Systems

Implements various measures of inequality in reward distribution,
including Gini coefficient and related metrics.
"""

from typing import Dict, List, Union
import numpy as np


def gini_coefficient(values: Union[List[float], np.ndarray, Dict[int, float]]) -> float:
    """
    Compute Gini coefficient for a distribution of values.

    The Gini coefficient measures inequality on a scale from 0 to 1:
    - 0 = perfect equality (everyone has the same)
    - 1 = perfect inequality (one entity has everything)

    Args:
        values: Array-like of values or dict mapping agent_id to value

    Returns:
        gini: Gini coefficient between 0 and 1
    """
    # Convert to numpy array
    if isinstance(values, dict):
        arr = np.array(list(values.values()))
    else:
        arr = np.array(values)

    # Handle edge cases
    if len(arr) == 0:
        return 0.0

    if len(arr) == 1:
        return 0.0

    # Ensure positive values (shift if needed)
    if np.any(arr < 0):
        arr = arr - arr.min() + 1e-8

    # Avoid division by zero
    if np.sum(arr) == 0:
        return 0.0

    # Sort values
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)

    # Compute Gini using the standard formula
    # G = (2 * sum(i * x_i) - (n + 1) * sum(x_i)) / (n * sum(x_i))
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_arr) - (n + 1) * np.sum(sorted_arr)) / (n * np.sum(sorted_arr))

    return float(gini)


def hoover_index(values: Union[List[float], np.ndarray, Dict[int, float]]) -> float:
    """
    Compute Hoover index (Robin Hood index).

    The Hoover index represents the proportion of total income that would
    have to be redistributed to achieve perfect equality.

    Args:
        values: Array-like of values or dict mapping agent_id to value

    Returns:
        hoover: Hoover index between 0 and 1
    """
    if isinstance(values, dict):
        arr = np.array(list(values.values()))
    else:
        arr = np.array(values)

    if len(arr) == 0 or np.sum(arr) == 0:
        return 0.0

    mean_value = np.mean(arr)
    return float(np.sum(np.abs(arr - mean_value)) / (2 * np.sum(arr)))


def palma_ratio(values: Union[List[float], np.ndarray, Dict[int, float]]) -> float:
    """
    Compute Palma ratio.

    The Palma ratio is the ratio of the richest 10% to the poorest 40%.
    Higher values indicate more inequality.

    Args:
        values: Array-like of values or dict mapping agent_id to value

    Returns:
        palma: Palma ratio (can be > 1)
    """
    if isinstance(values, dict):
        arr = np.array(list(values.values()))
    else:
        arr = np.array(values)

    if len(arr) < 3:
        return 1.0

    sorted_arr = np.sort(arr)
    n = len(sorted_arr)

    # Bottom 40%
    bottom_40_idx = int(n * 0.4)
    bottom_40 = sorted_arr[:max(bottom_40_idx, 1)]

    # Top 10%
    top_10_idx = int(n * 0.9)
    top_10 = sorted_arr[top_10_idx:]

    bottom_sum = np.sum(bottom_40)
    top_sum = np.sum(top_10)

    if bottom_sum == 0:
        return float('inf') if top_sum > 0 else 1.0

    return float(top_sum / bottom_sum)


def theil_index(values: Union[List[float], np.ndarray, Dict[int, float]]) -> float:
    """
    Compute Theil index (Generalized Entropy index with alpha=1).

    The Theil index is sensitive to changes at the upper end of the distribution.

    Args:
        values: Array-like of values or dict mapping agent_id to value

    Returns:
        theil: Theil index (>= 0, with 0 being perfect equality)
    """
    if isinstance(values, dict):
        arr = np.array(list(values.values()))
    else:
        arr = np.array(values)

    if len(arr) == 0:
        return 0.0

    # Ensure positive values
    arr = np.maximum(arr, 1e-8)

    mean_value = np.mean(arr)
    if mean_value == 0:
        return 0.0

    # Theil T index
    n = len(arr)
    ratios = arr / mean_value
    theil = np.sum(ratios * np.log(ratios + 1e-8)) / n

    return float(max(0.0, theil))


def reward_spread(values: Union[List[float], np.ndarray, Dict[int, float]]) -> Dict[str, float]:
    """
    Compute various statistics about reward spread.

    Args:
        values: Array-like of values or dict mapping agent_id to value

    Returns:
        Dictionary with min, max, mean, std, range, cv (coefficient of variation)
    """
    if isinstance(values, dict):
        arr = np.array(list(values.values()))
    else:
        arr = np.array(values)

    if len(arr) == 0:
        return {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'range': 0.0,
            'cv': 0.0,
        }

    mean_val = np.mean(arr)
    std_val = np.std(arr)

    return {
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'mean': float(mean_val),
        'std': float(std_val),
        'range': float(np.max(arr) - np.min(arr)),
        'cv': float(std_val / (mean_val + 1e-8)),  # Coefficient of variation
    }


class InequalityTracker:
    """
    Track inequality metrics over time.
    """

    def __init__(self, num_agents: int):
        """Initialize tracker."""
        self.num_agents = num_agents
        self.gini_history: List[float] = []
        self.hoover_history: List[float] = []
        self.cumulative_rewards: Dict[int, float] = {i: 0.0 for i in range(num_agents)}

    def update(self, rewards: Dict[int, float]):
        """Update with new reward data."""
        # Update cumulative rewards
        for agent_id, reward in rewards.items():
            self.cumulative_rewards[agent_id] += reward

        # Compute current inequality
        self.gini_history.append(gini_coefficient(rewards))
        self.hoover_history.append(hoover_index(rewards))

    def get_current_gini(self) -> float:
        """Get current Gini coefficient of cumulative rewards."""
        return gini_coefficient(self.cumulative_rewards)

    def get_mean_gini(self) -> float:
        """Get mean Gini over episode."""
        if not self.gini_history:
            return 0.0
        return float(np.mean(self.gini_history))

    def get_summary(self) -> Dict[str, float]:
        """Get summary of inequality metrics."""
        return {
            'gini_current': self.get_current_gini(),
            'gini_mean': self.get_mean_gini(),
            'gini_final': self.gini_history[-1] if self.gini_history else 0.0,
            'hoover_mean': float(np.mean(self.hoover_history)) if self.hoover_history else 0.0,
            'palma_current': palma_ratio(self.cumulative_rewards),
            'theil_current': theil_index(self.cumulative_rewards),
            **reward_spread(self.cumulative_rewards),
        }

    def reset(self):
        """Reset tracker."""
        self.gini_history = []
        self.hoover_history = []
        self.cumulative_rewards = {i: 0.0 for i in range(self.num_agents)}
