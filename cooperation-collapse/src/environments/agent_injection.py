"""
Agent Injection Module

Handles injection of defector agents into cooperative populations.
Supports hot-swapping agent policies and tracking behavior changes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
import numpy as np


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    agent_id: int
    policy: Any  # Policy network or callable
    is_defector: bool = False
    is_frozen: bool = False
    reward_shaper: Optional[Callable] = None


class DefectorRewardShaper:
    """
    Shapes rewards to encourage selfish/defecting behavior.
    Used during defector training phase.
    """

    def __init__(
        self,
        apple_bonus: float = 1.5,
        cleaning_penalty: float = -0.2,
        proximity_bonus: float = 0.05,
        blocking_bonus: float = 0.0,
    ):
        """
        Initialize reward shaper.

        Args:
            apple_bonus: Multiplier for apple collection reward
            cleaning_penalty: Penalty for cleaning actions
            proximity_bonus: Bonus for being near apples
            blocking_bonus: Bonus for blocking other agents
        """
        self.apple_bonus = apple_bonus
        self.cleaning_penalty = cleaning_penalty
        self.proximity_bonus = proximity_bonus
        self.blocking_bonus = blocking_bonus

    def shape_reward(
        self,
        base_reward: float,
        action: int,
        agent_position: tuple,
        apple_positions: List[tuple],
        other_agent_positions: List[tuple],
    ) -> float:
        """
        Compute shaped reward for defector training.

        Args:
            base_reward: Original environment reward
            action: Action taken (6 = clean)
            agent_position: Current agent position
            apple_positions: List of apple positions
            other_agent_positions: List of other agent positions

        Returns:
            Shaped reward value
        """
        shaped_reward = base_reward * self.apple_bonus

        # Penalize cleaning actions
        CLEAN_ACTION = 6
        if action == CLEAN_ACTION:
            shaped_reward += self.cleaning_penalty

        # Reward proximity to apples
        if apple_positions and self.proximity_bonus > 0:
            min_dist = min(
                abs(agent_position[0] - ap[0]) + abs(agent_position[1] - ap[1])
                for ap in apple_positions
            )
            shaped_reward += self.proximity_bonus / (min_dist + 1)

        # Reward blocking other agents (optional)
        if other_agent_positions and self.blocking_bonus > 0:
            for other_pos in other_agent_positions:
                if self._is_blocking(agent_position, other_pos, apple_positions):
                    shaped_reward += self.blocking_bonus

        return shaped_reward

    def _is_blocking(
        self,
        agent_pos: tuple,
        other_pos: tuple,
        apple_positions: List[tuple]
    ) -> bool:
        """Check if agent is blocking another from apples."""
        # Simple heuristic: agent is between other and nearest apple
        if not apple_positions:
            return False

        # Find nearest apple to other agent
        nearest_apple = min(
            apple_positions,
            key=lambda ap: abs(other_pos[0] - ap[0]) + abs(other_pos[1] - ap[1])
        )

        # Check if agent is on the path
        min_y = min(other_pos[0], nearest_apple[0])
        max_y = max(other_pos[0], nearest_apple[0])
        min_x = min(other_pos[1], nearest_apple[1])
        max_x = max(other_pos[1], nearest_apple[1])

        return (min_y <= agent_pos[0] <= max_y and
                min_x <= agent_pos[1] <= max_x)


class AgentInjector:
    """
    Manages agent population and enables defector injection.

    Supports:
    - Hot-swapping agent policies mid-episode
    - Freezing/unfreezing agent learning
    - Tracking pre/post injection metrics
    """

    def __init__(self, num_agents: int):
        """
        Initialize agent injector.

        Args:
            num_agents: Total number of agents
        """
        self.num_agents = num_agents
        self.agents: Dict[int, AgentConfig] = {}
        self.injection_timestep: Optional[int] = None
        self.injection_history: List[Dict] = []

    def register_agent(
        self,
        agent_id: int,
        policy: Any,
        is_defector: bool = False,
        is_frozen: bool = False,
        reward_shaper: Optional[Callable] = None,
    ):
        """Register an agent with its policy."""
        self.agents[agent_id] = AgentConfig(
            agent_id=agent_id,
            policy=policy,
            is_defector=is_defector,
            is_frozen=is_frozen,
            reward_shaper=reward_shaper,
        )

    def register_cooperative_population(self, policy: Any, agent_ids: List[int]):
        """Register cooperative policy for multiple agents."""
        for agent_id in agent_ids:
            self.register_agent(
                agent_id=agent_id,
                policy=policy,
                is_defector=False,
                is_frozen=False,
            )

    def inject_defector(
        self,
        defector_policy: Any,
        target_agent_id: int,
        current_timestep: int,
        freeze_defector: bool = True,
        reward_shaper: Optional[DefectorRewardShaper] = None,
    ):
        """
        Replace a cooperative agent with a defector.

        Args:
            defector_policy: Pre-trained defector policy
            target_agent_id: Which agent to replace
            current_timestep: Current environment timestep
            freeze_defector: If True, defector doesn't learn
            reward_shaper: Optional reward shaping for defector
        """
        # Record injection event
        self.injection_timestep = current_timestep
        self.injection_history.append({
            'timestep': current_timestep,
            'target_agent': target_agent_id,
            'previous_policy': 'cooperative',
        })

        # Replace agent
        self.agents[target_agent_id] = AgentConfig(
            agent_id=target_agent_id,
            policy=defector_policy,
            is_defector=True,
            is_frozen=freeze_defector,
            reward_shaper=reward_shaper,
        )

    def remove_defector(self, agent_id: int, cooperative_policy: Any):
        """Remove defector and restore cooperative behavior."""
        if agent_id in self.agents and self.agents[agent_id].is_defector:
            self.agents[agent_id] = AgentConfig(
                agent_id=agent_id,
                policy=cooperative_policy,
                is_defector=False,
                is_frozen=False,
            )

    def freeze_agent(self, agent_id: int):
        """Freeze agent learning."""
        if agent_id in self.agents:
            self.agents[agent_id].is_frozen = True

    def unfreeze_agent(self, agent_id: int):
        """Unfreeze agent learning."""
        if agent_id in self.agents:
            self.agents[agent_id].is_frozen = False

    def freeze_all_except(self, agent_ids: List[int]):
        """Freeze all agents except specified ones."""
        for agent_id in self.agents:
            self.agents[agent_id].is_frozen = agent_id not in agent_ids

    def get_actions(
        self,
        observations: Dict[int, np.ndarray],
        rng: np.random.Generator,
    ) -> Dict[int, int]:
        """
        Get actions from all agents.

        Args:
            observations: Dict mapping agent_id to observation
            rng: Random number generator

        Returns:
            Dict mapping agent_id to action
        """
        actions = {}
        for agent_id, obs in observations.items():
            if agent_id in self.agents:
                policy = self.agents[agent_id].policy
                # Assume policy has act() method or is callable
                if hasattr(policy, 'act'):
                    actions[agent_id] = policy.act(obs, rng)
                elif callable(policy):
                    actions[agent_id] = policy(obs, rng)
                else:
                    # Random action fallback
                    actions[agent_id] = rng.integers(0, 9)
            else:
                actions[agent_id] = rng.integers(0, 9)
        return actions

    def get_shaped_rewards(
        self,
        rewards: Dict[int, float],
        actions: Dict[int, int],
        env_state: Any,
    ) -> Dict[int, float]:
        """
        Apply reward shaping for defector agents.

        Args:
            rewards: Original rewards from environment
            actions: Actions taken by each agent
            env_state: Current environment state

        Returns:
            Shaped rewards
        """
        shaped_rewards = rewards.copy()

        for agent_id, config in self.agents.items():
            if config.reward_shaper is not None:
                # Extract needed info from env_state
                agent_pos = env_state.agents[agent_id].position
                apple_positions = [
                    (y, x) for y in range(env_state.grid.shape[0])
                    for x in range(env_state.grid.shape[1])
                    if env_state.grid[y, x] == 5  # APPLE
                ]
                other_positions = [
                    s.position for aid, s in env_state.agents.items()
                    if aid != agent_id
                ]

                shaped_rewards[agent_id] = config.reward_shaper.shape_reward(
                    base_reward=rewards[agent_id],
                    action=actions[agent_id],
                    agent_position=agent_pos,
                    apple_positions=apple_positions,
                    other_agent_positions=other_positions,
                )

        return shaped_rewards

    def get_defector_ids(self) -> List[int]:
        """Get list of defector agent IDs."""
        return [aid for aid, config in self.agents.items() if config.is_defector]

    def get_frozen_ids(self) -> List[int]:
        """Get list of frozen agent IDs."""
        return [aid for aid, config in self.agents.items() if config.is_frozen]

    def get_learnable_ids(self) -> List[int]:
        """Get list of agent IDs that can learn."""
        return [aid for aid, config in self.agents.items() if not config.is_frozen]

    @property
    def has_defector(self) -> bool:
        """Check if any defector is present."""
        return any(config.is_defector for config in self.agents.values())

    @property
    def defector_ratio(self) -> float:
        """Get ratio of defectors to total agents."""
        num_defectors = sum(1 for c in self.agents.values() if c.is_defector)
        return num_defectors / max(len(self.agents), 1)
