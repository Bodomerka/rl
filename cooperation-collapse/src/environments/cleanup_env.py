"""
Cleanup Environment Wrapper

A grid-world environment implementing the Cleanup social dilemma:
- Agents can collect apples (+1 reward) or clean the river (0 reward)
- If river is polluted, apples don't regrow
- Cooperation dilemma: everyone benefits from cleaning, but individuals prefer collecting

This is a simplified implementation for research purposes.
For production, integrate with SocialJax when available.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class AgentState:
    """State of a single agent."""
    position: Tuple[int, int]
    orientation: int  # 0=up, 1=right, 2=down, 3=left


@dataclass
class EnvState:
    """Complete environment state."""
    grid: np.ndarray  # 2D grid with cell types
    agents: Dict[int, AgentState]
    pollution_level: float  # 0.0 to 1.0
    apple_count: int
    step_count: int


class CleanupEnvironment:
    """
    Cleanup Environment for Sequential Social Dilemmas research.

    Grid cell types:
        0: Empty
        1: Wall
        2: River (clean)
        3: River (polluted)
        4: Apple spawn zone
        5: Apple

    Actions:
        0-3: Move (up, right, down, left)
        4: Turn left
        5: Turn right
        6: Clean (removes pollution in front)
        7: Collect (picks apple in front)
        8: Noop
    """

    # Cell types
    EMPTY = 0
    WALL = 1
    RIVER_CLEAN = 2
    RIVER_POLLUTED = 3
    APPLE_ZONE = 4
    APPLE = 5

    # Actions
    MOVE_UP = 0
    MOVE_RIGHT = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    TURN_LEFT = 4
    TURN_RIGHT = 5
    CLEAN = 6
    COLLECT = 7
    NOOP = 8

    # Direction vectors (dy, dx)
    DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(
        self,
        num_agents: int = 8,
        grid_size: Tuple[int, int] = (25, 18),
        max_steps: int = 1000,
        apple_respawn_prob: float = 0.05,
        pollution_spawn_prob: float = 0.5,
        common_reward: bool = False,
        seed: int = 42,
    ):
        """
        Initialize Cleanup environment.

        Args:
            num_agents: Number of agents in the environment
            grid_size: (height, width) of the grid
            apple_respawn_prob: Probability of apple respawning each step
            pollution_spawn_prob: Probability of pollution appearing
            common_reward: If True, rewards are shared equally among agents
            seed: Random seed
        """
        self.num_agents = num_agents
        self.grid_height, self.grid_width = grid_size
        self.max_steps = max_steps
        self.apple_respawn_prob = apple_respawn_prob
        self.pollution_spawn_prob = pollution_spawn_prob
        self.common_reward = common_reward

        self.rng = np.random.default_rng(seed)
        self.state: Optional[EnvState] = None

        # Observation settings
        self.obs_radius = 5  # 11x11 observation window
        self.obs_size = 2 * self.obs_radius + 1

        # Track metrics
        self.episode_cleaning_actions = 0
        self.episode_collection_actions = 0

        # Initialize grid layout
        self._init_grid_layout()

    def _init_grid_layout(self):
        """Create the base grid layout with walls, river, and apple zones."""
        self.base_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)

        # Add walls around edges
        self.base_grid[0, :] = self.WALL
        self.base_grid[-1, :] = self.WALL
        self.base_grid[:, 0] = self.WALL
        self.base_grid[:, -1] = self.WALL

        # River in the middle (horizontal strip)
        river_y = self.grid_height // 2
        self.base_grid[river_y-1:river_y+2, 1:-1] = self.RIVER_CLEAN
        self.river_positions = [
            (y, x)
            for y in range(river_y-1, river_y+2)
            for x in range(1, self.grid_width-1)
        ]

        # Apple spawn zones (top and bottom)
        for y in range(2, river_y-2):
            for x in range(2, self.grid_width-2):
                self.base_grid[y, x] = self.APPLE_ZONE

        for y in range(river_y+3, self.grid_height-2):
            for x in range(2, self.grid_width-2):
                self.base_grid[y, x] = self.APPLE_ZONE

        self.apple_zone_positions = list(zip(*np.where(self.base_grid == self.APPLE_ZONE)))

    def reset(self) -> Tuple[Dict[int, np.ndarray], EnvState]:
        """
        Reset environment to initial state.

        Returns:
            observations: Dict mapping agent_id to observation array
            state: Environment state
        """
        # Create fresh grid
        grid = self.base_grid.copy()

        # Spawn initial apples
        num_initial_apples = len(self.apple_zone_positions) // 4
        apple_positions = self.rng.choice(
            len(self.apple_zone_positions),
            size=num_initial_apples,
            replace=False
        )
        for idx in apple_positions:
            pos = self.apple_zone_positions[idx]
            grid[pos] = self.APPLE

        # Spawn agents
        agents = {}
        available_positions = [
            (y, x) for y, x in self.apple_zone_positions
            if grid[y, x] != self.APPLE
        ]
        agent_positions = self.rng.choice(
            len(available_positions),
            size=self.num_agents,
            replace=False
        )
        for agent_id, pos_idx in enumerate(agent_positions):
            pos = available_positions[pos_idx]
            agents[agent_id] = AgentState(
                position=pos,
                orientation=self.rng.integers(0, 4)
            )

        self.state = EnvState(
            grid=grid,
            agents=agents,
            pollution_level=0.0,
            apple_count=num_initial_apples,
            step_count=0
        )

        # Reset metrics
        self.episode_cleaning_actions = 0
        self.episode_collection_actions = 0

        observations = self._get_observations()
        return observations, self.state

    def step(
        self,
        actions: Dict[int, int]
    ) -> Tuple[Dict[int, np.ndarray], EnvState, Dict[int, float], Dict[int, bool], Dict]:
        """
        Execute one environment step.

        Args:
            actions: Dict mapping agent_id to action

        Returns:
            observations: New observations for each agent
            state: New environment state
            rewards: Rewards for each agent
            dones: Done flags for each agent
            infos: Additional information
        """
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        rewards = {agent_id: 0.0 for agent_id in range(self.num_agents)}
        infos = {
            'cleaning_actions': 0,
            'collection_actions': 0,
            'apples_spawned': 0,
            'pollution_added': 0,
        }

        # Process actions for each agent
        for agent_id, action in actions.items():
            reward = self._process_action(agent_id, action, infos)
            rewards[agent_id] = reward

        # Update environment dynamics
        self._update_pollution()
        self._update_apples(infos)

        # Update step count
        self.state.step_count += 1

        # Check termination
        done = self.state.step_count >= self.max_steps
        dones = {agent_id: done for agent_id in range(self.num_agents)}

        # Apply common reward if enabled
        if self.common_reward:
            total_reward = sum(rewards.values())
            shared_reward = total_reward / self.num_agents
            rewards = {agent_id: shared_reward for agent_id in range(self.num_agents)}

        # Add metrics to info
        infos['pollution_level'] = self.state.pollution_level
        infos['apple_count'] = self.state.apple_count
        infos['step'] = self.state.step_count

        observations = self._get_observations()
        return observations, self.state, rewards, dones, infos

    def _process_action(self, agent_id: int, action: int, infos: Dict) -> float:
        """Process action for a single agent. Returns reward."""
        agent = self.state.agents[agent_id]
        reward = 0.0

        if action in [self.MOVE_UP, self.MOVE_RIGHT, self.MOVE_DOWN, self.MOVE_LEFT]:
            # Move action
            dy, dx = self.DIRECTIONS[action]
            new_y = agent.position[0] + dy
            new_x = agent.position[1] + dx

            # Check if move is valid
            if self._is_valid_position(new_y, new_x):
                agent.position = (new_y, new_x)

        elif action == self.TURN_LEFT:
            agent.orientation = (agent.orientation - 1) % 4

        elif action == self.TURN_RIGHT:
            agent.orientation = (agent.orientation + 1) % 4

        elif action == self.CLEAN:
            # Clean pollution in front of agent
            front_pos = self._get_front_position(agent)
            if front_pos and self._is_river_polluted(front_pos):
                self.state.grid[front_pos] = self.RIVER_CLEAN
                self.state.pollution_level = max(0, self.state.pollution_level - 0.1)
                infos['cleaning_actions'] += 1
                self.episode_cleaning_actions += 1

        elif action == self.COLLECT:
            # Collect apple in front or at current position
            front_pos = self._get_front_position(agent)
            current_pos = agent.position

            for pos in [front_pos, current_pos]:
                if pos and self.state.grid[pos] == self.APPLE:
                    self.state.grid[pos] = self.APPLE_ZONE
                    self.state.apple_count -= 1
                    reward = 1.0
                    infos['collection_actions'] += 1
                    self.episode_collection_actions += 1
                    break

        # NOOP does nothing

        return reward

    def _get_front_position(self, agent: AgentState) -> Optional[Tuple[int, int]]:
        """Get position in front of agent."""
        dy, dx = self.DIRECTIONS[agent.orientation]
        front_y = agent.position[0] + dy
        front_x = agent.position[1] + dx

        if 0 <= front_y < self.grid_height and 0 <= front_x < self.grid_width:
            return (front_y, front_x)
        return None

    def _is_valid_position(self, y: int, x: int) -> bool:
        """Check if position is valid for movement."""
        if not (0 <= y < self.grid_height and 0 <= x < self.grid_width):
            return False
        cell = self.state.grid[y, x]
        # Can't walk through walls or into river
        return cell not in [self.WALL, self.RIVER_CLEAN, self.RIVER_POLLUTED]

    def _is_river_polluted(self, pos: Tuple[int, int]) -> bool:
        """Check if position is polluted river."""
        return self.state.grid[pos] == self.RIVER_POLLUTED

    def _update_pollution(self):
        """Add pollution to river with some probability."""
        if self.rng.random() < self.pollution_spawn_prob:
            # Find clean river cell and pollute it
            clean_river = [
                pos for pos in self.river_positions
                if self.state.grid[pos] == self.RIVER_CLEAN
            ]
            if clean_river:
                pos = clean_river[self.rng.integers(len(clean_river))]
                self.state.grid[pos] = self.RIVER_POLLUTED
                self.state.pollution_level = min(1.0, self.state.pollution_level + 0.1)

    def _update_apples(self, infos: Dict):
        """Respawn apples based on pollution level."""
        # Apple respawn probability decreases with pollution
        effective_prob = self.apple_respawn_prob * (1 - self.state.pollution_level)

        for pos in self.apple_zone_positions:
            if self.state.grid[pos] == self.APPLE_ZONE:
                if self.rng.random() < effective_prob:
                    self.state.grid[pos] = self.APPLE
                    self.state.apple_count += 1
                    infos['apples_spawned'] += 1

    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for all agents."""
        observations = {}
        for agent_id in range(self.num_agents):
            observations[agent_id] = self._get_agent_observation(agent_id)
        return observations

    def _get_agent_observation(self, agent_id: int) -> np.ndarray:
        """
        Get observation for a single agent.

        Returns:
            Observation array of shape (obs_size, obs_size, num_channels)
            Channels: [walls, river_clean, river_polluted, apples, agents, self]
        """
        agent = self.state.agents[agent_id]
        y, x = agent.position

        # Create observation tensor
        num_channels = 6
        obs = np.zeros((self.obs_size, self.obs_size, num_channels), dtype=np.float32)

        # Extract local view
        for dy in range(-self.obs_radius, self.obs_radius + 1):
            for dx in range(-self.obs_radius, self.obs_radius + 1):
                gy, gx = y + dy, x + dx
                obs_y, obs_x = dy + self.obs_radius, dx + self.obs_radius

                if 0 <= gy < self.grid_height and 0 <= gx < self.grid_width:
                    cell = self.state.grid[gy, gx]

                    if cell == self.WALL:
                        obs[obs_y, obs_x, 0] = 1.0
                    elif cell == self.RIVER_CLEAN:
                        obs[obs_y, obs_x, 1] = 1.0
                    elif cell == self.RIVER_POLLUTED:
                        obs[obs_y, obs_x, 2] = 1.0
                    elif cell == self.APPLE:
                        obs[obs_y, obs_x, 3] = 1.0

                    # Check for other agents
                    for other_id, other_agent in self.state.agents.items():
                        if other_agent.position == (gy, gx):
                            if other_id == agent_id:
                                obs[obs_y, obs_x, 5] = 1.0  # Self
                            else:
                                obs[obs_y, obs_x, 4] = 1.0  # Other agents
                else:
                    # Out of bounds = wall
                    obs[obs_y, obs_x, 0] = 1.0

        return obs

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Shape of observation array."""
        return (self.obs_size, self.obs_size, 6)

    @property
    def action_space_size(self) -> int:
        """Number of possible actions."""
        return 9

    def get_cleaning_rate(self) -> float:
        """Get ratio of cleaning actions to total productive actions."""
        total = self.episode_cleaning_actions + self.episode_collection_actions
        if total == 0:
            return 0.0
        return self.episode_cleaning_actions / total

    def render(self) -> np.ndarray:
        """Render environment as RGB image."""
        # Color mapping
        colors = {
            self.EMPTY: [200, 200, 200],      # Light gray
            self.WALL: [50, 50, 50],           # Dark gray
            self.RIVER_CLEAN: [50, 150, 255],  # Blue
            self.RIVER_POLLUTED: [139, 90, 43], # Brown
            self.APPLE_ZONE: [180, 230, 180],  # Light green
            self.APPLE: [50, 200, 50],         # Green
        }

        # Create base image
        img = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell = self.state.grid[y, x]
                img[y, x] = colors.get(cell, [0, 0, 0])

        # Draw agents
        agent_color = [150, 50, 200]  # Purple
        for agent_id, agent in self.state.agents.items():
            y, x = agent.position
            img[y, x] = agent_color

        # Scale up for visibility
        scale = 20
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

        return img
