"""
Collective Metrics for Multi-Agent Systems

Metrics for measuring overall system performance, cooperation levels,
and resource sustainability.
"""

from typing import Dict, List, Any
import numpy as np


class CollectiveMetrics:
    """
    Computes collective metrics for multi-agent environments.

    Tracks:
    - Total rewards
    - Cooperation rates
    - Resource sustainability
    - Pollution dynamics
    """

    def __init__(self, num_agents: int):
        """
        Initialize metrics tracker.

        Args:
            num_agents: Number of agents in the system
        """
        self.num_agents = num_agents
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.episode_rewards: Dict[int, float] = {i: 0.0 for i in range(self.num_agents)}
        self.cleaning_actions = 0
        self.collection_actions = 0
        self.total_actions = 0
        self.pollution_history: List[float] = []
        self.apple_history: List[int] = []
        self.reward_history: List[float] = []

    def update(
        self,
        rewards: Dict[int, float],
        infos: Dict[str, Any],
    ):
        """
        Update metrics with step data.

        Args:
            rewards: Rewards per agent
            infos: Info dictionary from environment
        """
        # Accumulate rewards
        for agent_id, reward in rewards.items():
            self.episode_rewards[agent_id] += reward

        # Track actions
        self.cleaning_actions += infos.get('cleaning_actions', 0)
        self.collection_actions += infos.get('collection_actions', 0)
        self.total_actions += self.num_agents

        # Track environment state
        self.pollution_history.append(infos.get('pollution_level', 0.0))
        self.apple_history.append(infos.get('apple_count', 0))
        self.reward_history.append(sum(rewards.values()))

    @property
    def collective_reward(self) -> float:
        """Sum of all agent rewards."""
        return sum(self.episode_rewards.values())

    @property
    def mean_reward(self) -> float:
        """Mean reward per agent."""
        return self.collective_reward / max(self.num_agents, 1)

    @property
    def cleaning_rate(self) -> float:
        """Fraction of productive actions that are cleaning."""
        productive = self.cleaning_actions + self.collection_actions
        if productive == 0:
            return 0.0
        return self.cleaning_actions / productive

    @property
    def cooperation_ratio(self) -> float:
        """Ratio of cleaning to total actions."""
        if self.total_actions == 0:
            return 0.0
        return self.cleaning_actions / self.total_actions

    @property
    def mean_pollution(self) -> float:
        """Average pollution level over episode."""
        if not self.pollution_history:
            return 0.0
        return np.mean(self.pollution_history)

    @property
    def final_pollution(self) -> float:
        """Final pollution level."""
        if not self.pollution_history:
            return 0.0
        return self.pollution_history[-1]

    def resource_sustainability(self, window: int = 100) -> float:
        """
        Measure if apple production is sustainable.

        Args:
            window: Window size for measurement

        Returns:
            sustainability: Ratio approaching 1.0 if sustainable
        """
        if len(self.apple_history) < window:
            return 1.0

        # Compare early vs late apple counts
        early = np.mean(self.apple_history[:window//2])
        late = np.mean(self.apple_history[-window//2:])

        if early <= 0:
            return 1.0 if late > 0 else 0.0

        return min(late / early, 1.0)

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics."""
        return {
            'collective_reward': self.collective_reward,
            'mean_reward': self.mean_reward,
            'cleaning_rate': self.cleaning_rate,
            'cooperation_ratio': self.cooperation_ratio,
            'mean_pollution': self.mean_pollution,
            'final_pollution': self.final_pollution,
            'resource_sustainability': self.resource_sustainability(),
            'total_cleaning_actions': self.cleaning_actions,
            'total_collection_actions': self.collection_actions,
        }


def compute_cleaning_rate(infos_history: List[Dict]) -> float:
    """
    Compute cleaning rate from history of info dicts.

    Args:
        infos_history: List of info dicts from environment steps

    Returns:
        cleaning_rate: Fraction of cleaning actions
    """
    total_cleaning = sum(info.get('cleaning_actions', 0) for info in infos_history)
    total_collection = sum(info.get('collection_actions', 0) for info in infos_history)

    total = total_cleaning + total_collection
    if total == 0:
        return 0.0

    return total_cleaning / total


def compute_collective_reward(rewards_history: List[Dict[int, float]]) -> float:
    """
    Compute total collective reward from history.

    Args:
        rewards_history: List of reward dicts per step

    Returns:
        total_reward: Sum of all rewards
    """
    return sum(sum(rewards.values()) for rewards in rewards_history)
