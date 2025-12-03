"""
Rollout Buffer for PPO

Stores experience data and computes advantages using GAE.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Transition:
    """Single transition data."""
    observation: np.ndarray
    action: int
    reward: float
    done: bool
    value: float
    log_prob: float
    hidden_state: Optional[np.ndarray] = None


@dataclass
class RolloutBuffer:
    """
    Buffer for storing rollout data during training.

    Stores transitions for all agents and computes GAE advantages.
    """
    num_agents: int
    rollout_length: int
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Storage
    observations: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    actions: Dict[int, List[int]] = field(default_factory=dict)
    rewards: Dict[int, List[float]] = field(default_factory=dict)
    dones: Dict[int, List[bool]] = field(default_factory=dict)
    values: Dict[int, List[float]] = field(default_factory=dict)
    log_probs: Dict[int, List[float]] = field(default_factory=dict)
    hidden_states: Dict[int, List[Optional[np.ndarray]]] = field(default_factory=dict)

    # Computed
    advantages: Dict[int, np.ndarray] = field(default_factory=dict)
    returns: Dict[int, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize storage for all agents."""
        self.clear()

    def clear(self):
        """Clear all stored data."""
        for agent_id in range(self.num_agents):
            self.observations[agent_id] = []
            self.actions[agent_id] = []
            self.rewards[agent_id] = []
            self.dones[agent_id] = []
            self.values[agent_id] = []
            self.log_probs[agent_id] = []
            self.hidden_states[agent_id] = []
            self.advantages[agent_id] = np.array([])
            self.returns[agent_id] = np.array([])

    def add(
        self,
        agent_id: int,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
        hidden_state: Optional[np.ndarray] = None,
    ):
        """Add a transition for an agent."""
        self.observations[agent_id].append(observation)
        self.actions[agent_id].append(action)
        self.rewards[agent_id].append(reward)
        self.dones[agent_id].append(done)
        self.values[agent_id].append(value)
        self.log_probs[agent_id].append(log_prob)
        self.hidden_states[agent_id].append(hidden_state)

    def add_batch(
        self,
        observations: Dict[int, np.ndarray],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        dones: Dict[int, bool],
        values: Dict[int, float],
        log_probs: Dict[int, float],
        hidden_states: Optional[Dict[int, np.ndarray]] = None,
    ):
        """Add transitions for all agents at once."""
        for agent_id in range(self.num_agents):
            self.add(
                agent_id=agent_id,
                observation=observations[agent_id],
                action=actions[agent_id],
                reward=rewards[agent_id],
                done=dones[agent_id],
                value=values[agent_id],
                log_prob=log_probs[agent_id],
                hidden_state=hidden_states[agent_id] if hidden_states else None,
            )

    def compute_advantages(self, last_values: Dict[int, float]):
        """
        Compute GAE advantages for all agents.

        Args:
            last_values: Value estimates for the last state (for bootstrapping)
        """
        for agent_id in range(self.num_agents):
            rewards = np.array(self.rewards[agent_id])
            values = np.array(self.values[agent_id])
            dones = np.array(self.dones[agent_id])

            advantages = np.zeros_like(rewards)
            last_gae = 0

            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = last_values[agent_id]
                    next_non_terminal = 1.0 - float(dones[t])
                else:
                    next_value = values[t + 1]
                    next_non_terminal = 1.0 - float(dones[t])

                delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae

            self.advantages[agent_id] = advantages
            self.returns[agent_id] = advantages + values

    def get_batches(
        self,
        agent_id: int,
        num_minibatches: int = 4,
        shuffle: bool = True,
    ):
        """
        Generate minibatches for training.

        Args:
            agent_id: Which agent's data to use
            num_minibatches: Number of minibatches to create
            shuffle: Whether to shuffle data

        Yields:
            Batches of (observations, actions, old_log_probs, advantages, returns)
        """
        size = len(self.observations[agent_id])
        batch_size = size // num_minibatches

        indices = np.arange(size)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            batch_indices = indices[start:end]

            yield (
                np.array([self.observations[agent_id][i] for i in batch_indices]),
                np.array([self.actions[agent_id][i] for i in batch_indices]),
                np.array([self.log_probs[agent_id][i] for i in batch_indices]),
                self.advantages[agent_id][batch_indices],
                self.returns[agent_id][batch_indices],
            )

    def get_all_data(self, agent_id: int):
        """Get all data for an agent as arrays."""
        return (
            np.array(self.observations[agent_id]),
            np.array(self.actions[agent_id]),
            np.array(self.log_probs[agent_id]),
            self.advantages[agent_id],
            self.returns[agent_id],
        )

    @property
    def is_full(self) -> bool:
        """Check if buffer has collected enough data."""
        if not self.observations:
            return False
        return len(self.observations[0]) >= self.rollout_length

    def __len__(self) -> int:
        """Return number of transitions stored."""
        if not self.observations:
            return 0
        return len(self.observations[0])
