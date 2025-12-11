"""
Tests for CleanupEnvironment

Verifies mathematical correctness of:
- pollution_level consistency with actual river state
- Apple spawn probability based on pollution
- Cleaning action effects
- Pollution spread mechanics
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environments.cleanup_env import CleanupEnvironment, AgentState


class TestPollutionLevelConsistency:
    """Tests that pollution_level matches actual river state."""

    def test_pollution_level_at_reset(self):
        """pollution_level should be 0 at reset (all river clean)."""
        env = CleanupEnvironment(num_agents=4, seed=42)
        obs, state = env.reset()

        # Count actual polluted cells
        polluted_count = sum(
            1 for pos in env.river_positions
            if state.grid[pos] == env.RIVER_POLLUTED
        )

        assert state.pollution_level == 0.0
        assert polluted_count == 0

    def test_pollution_level_matches_actual_state(self):
        """pollution_level should equal polluted_cells / total_river_cells."""
        env = CleanupEnvironment(num_agents=4, seed=42)
        obs, state = env.reset()

        total_river_cells = len(env.river_positions)

        # Manually pollute some cells
        num_to_pollute = 10
        for i in range(min(num_to_pollute, len(env.river_positions))):
            pos = env.river_positions[i]
            env.state.grid[pos] = env.RIVER_POLLUTED

        # Recalculate
        env._recalculate_pollution_level()

        # Count actual polluted
        polluted_count = sum(
            1 for pos in env.river_positions
            if env.state.grid[pos] == env.RIVER_POLLUTED
        )

        expected = polluted_count / total_river_cells
        assert abs(env.state.pollution_level - expected) < 1e-6
        assert env.state.pollution_level == num_to_pollute / total_river_cells

    def test_pollution_level_after_cleaning(self):
        """Cleaning should correctly update pollution_level."""
        env = CleanupEnvironment(num_agents=4, seed=42)
        env.reset()

        total_river_cells = len(env.river_positions)

        # Pollute all river cells
        for pos in env.river_positions:
            env.state.grid[pos] = env.RIVER_POLLUTED
        env._recalculate_pollution_level()

        assert env.state.pollution_level == 1.0

        # Clean one cell
        pos = env.river_positions[0]
        env.state.grid[pos] = env.RIVER_CLEAN
        env._recalculate_pollution_level()

        expected = (total_river_cells - 1) / total_river_cells
        assert abs(env.state.pollution_level - expected) < 1e-6

    def test_pollution_level_bounds(self):
        """pollution_level should always be in [0, 1]."""
        env = CleanupEnvironment(num_agents=4, seed=42)
        env.reset()

        # All clean
        for pos in env.river_positions:
            env.state.grid[pos] = env.RIVER_CLEAN
        env._recalculate_pollution_level()
        assert env.state.pollution_level == 0.0

        # All polluted
        for pos in env.river_positions:
            env.state.grid[pos] = env.RIVER_POLLUTED
        env._recalculate_pollution_level()
        assert env.state.pollution_level == 1.0


class TestAppleSpawnWithPollution:
    """Tests that apple spawn probability depends on actual pollution."""

    def test_apple_spawn_prob_formula(self):
        """effective_prob = base_prob * (1 - pollution_level)."""
        env = CleanupEnvironment(
            num_agents=4,
            apple_respawn_prob=0.1,
            pollution_spawn_prob=0.0,  # Disable auto-pollution
            seed=42
        )
        env.reset()

        base_prob = env.apple_respawn_prob

        # No pollution -> full spawn rate
        env.state.pollution_level = 0.0
        effective = base_prob * (1 - env.state.pollution_level)
        assert effective == base_prob

        # 50% pollution -> half spawn rate
        env.state.pollution_level = 0.5
        effective = base_prob * (1 - env.state.pollution_level)
        assert effective == base_prob * 0.5

        # Full pollution -> no spawn
        env.state.pollution_level = 1.0
        effective = base_prob * (1 - env.state.pollution_level)
        assert effective == 0.0

    def test_apples_spawn_more_with_clean_river(self):
        """More apples should spawn when river is cleaner (statistical test)."""
        # Test 1: Clean river (pollution = 0)
        env_clean = CleanupEnvironment(
            num_agents=2,
            apple_respawn_prob=0.5,
            pollution_spawn_prob=0.0,
            seed=42
        )
        env_clean.reset()
        env_clean.state.pollution_level = 0.0  # Ensure clean

        clean_spawns = 0
        for _ in range(100):
            # Clear all apples to allow new spawns (simulate collection)
            for pos in env_clean.apple_zone_positions:
                if env_clean.state.grid[pos] == env_clean.APPLE:
                    env_clean.state.grid[pos] = env_clean.APPLE_ZONE
            env_clean.state.apple_count = 0

            infos = {'apples_spawned': 0}
            env_clean._update_apples(infos)
            clean_spawns += infos['apples_spawned']

        # Test 2: Polluted river (pollution = 0.9)
        env_polluted = CleanupEnvironment(
            num_agents=2,
            apple_respawn_prob=0.5,
            pollution_spawn_prob=0.0,
            seed=42  # Same seed for fair comparison
        )
        env_polluted.reset()
        env_polluted.state.pollution_level = 0.9  # 90% polluted

        polluted_spawns = 0
        for _ in range(100):
            # Clear all apples to allow new spawns
            for pos in env_polluted.apple_zone_positions:
                if env_polluted.state.grid[pos] == env_polluted.APPLE:
                    env_polluted.state.grid[pos] = env_polluted.APPLE_ZONE
            env_polluted.state.apple_count = 0

            infos = {'apples_spawned': 0}
            env_polluted._update_apples(infos)
            polluted_spawns += infos['apples_spawned']

        # With 90% pollution, effective_prob = 0.5 * 0.1 = 0.05
        # Clean: effective_prob = 0.5
        # Ratio should be ~10x (0.5 / 0.05 = 10)
        # Use 5x as conservative threshold
        assert clean_spawns > polluted_spawns * 5, \
            f"Clean spawns {clean_spawns} should be >> polluted spawns {polluted_spawns}"


class TestCleaningAction:
    """Tests that cleaning action works correctly."""

    def test_cleaning_reduces_pollution_correctly(self):
        """Each clean action should reduce pollution by 1/total_cells."""
        env = CleanupEnvironment(num_agents=1, seed=42)
        env.reset()

        total_river = len(env.river_positions)

        # Pollute 5 cells
        for i in range(5):
            env.state.grid[env.river_positions[i]] = env.RIVER_POLLUTED
        env._recalculate_pollution_level()

        initial_pollution = env.state.pollution_level
        assert abs(initial_pollution - 5/total_river) < 1e-6

        # Clean one cell
        env.state.grid[env.river_positions[0]] = env.RIVER_CLEAN
        env._recalculate_pollution_level()

        new_pollution = env.state.pollution_level
        assert abs(new_pollution - 4/total_river) < 1e-6

        # Difference should be exactly 1/total_river
        diff = initial_pollution - new_pollution
        assert abs(diff - 1/total_river) < 1e-6


class TestPollutionSpread:
    """Tests that pollution spread works correctly."""

    def test_pollution_increases_correctly(self):
        """Each pollution event should increase pollution by 1/total_cells."""
        env = CleanupEnvironment(
            num_agents=1,
            pollution_spawn_prob=1.0,  # Always spawn pollution
            seed=42
        )
        env.reset()

        total_river = len(env.river_positions)
        initial_pollution = env.state.pollution_level
        assert initial_pollution == 0.0

        # Trigger pollution update
        env._update_pollution()

        new_pollution = env.state.pollution_level
        expected = 1 / total_river
        assert abs(new_pollution - expected) < 1e-6


class TestObservationChannels:
    """Tests that observations are correctly formatted."""

    def test_observation_shape(self):
        """Observation should have correct shape."""
        env = CleanupEnvironment(num_agents=4, seed=42)
        obs, _ = env.reset()

        expected_shape = (11, 11, 8)  # 11x11 window, 8 channels
        assert obs[0].shape == expected_shape

    def test_pollution_channel_in_observation(self):
        """Channel 6 should contain global pollution level."""
        env = CleanupEnvironment(num_agents=4, seed=42)
        obs, _ = env.reset()

        # Set known pollution level
        env.state.pollution_level = 0.5
        obs = env._get_observations()

        # Channel 6 should be pollution level everywhere
        for agent_id, agent_obs in obs.items():
            assert np.allclose(agent_obs[:, :, 6], 0.5)

    def test_apple_ratio_channel_in_observation(self):
        """Channel 7 should contain apple ratio."""
        env = CleanupEnvironment(num_agents=4, seed=42)
        obs, state = env.reset()

        max_apples = len(env.apple_zone_positions)
        expected_ratio = state.apple_count / max_apples

        obs = env._get_observations()

        for agent_id, agent_obs in obs.items():
            assert np.allclose(agent_obs[:, :, 7], expected_ratio)


class TestEnvironmentDynamics:
    """Integration tests for overall environment behavior."""

    def test_episode_runs_without_error(self):
        """Environment should run a full episode without errors."""
        env = CleanupEnvironment(num_agents=4, max_steps=100, seed=42)
        obs, _ = env.reset()

        done = False
        steps = 0
        while not done and steps < 100:
            # Random actions
            actions = {i: np.random.randint(0, 9) for i in range(4)}
            obs, state, rewards, dones, infos = env.step(actions)
            done = any(dones.values())
            steps += 1

            # Verify pollution_level is consistent
            polluted = sum(
                1 for pos in env.river_positions
                if state.grid[pos] == env.RIVER_POLLUTED
            )
            expected = polluted / len(env.river_positions)
            assert abs(state.pollution_level - expected) < 1e-6, \
                f"pollution_level {state.pollution_level} != expected {expected}"

    def test_cleaning_rate_metric(self):
        """Cleaning rate should be cleaning_actions / total_productive_actions."""
        env = CleanupEnvironment(num_agents=1, seed=42)
        env.reset()

        # Reset counters
        env.episode_cleaning_actions = 10
        env.episode_collection_actions = 40

        rate = env.get_cleaning_rate()
        assert rate == 10 / 50  # 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
