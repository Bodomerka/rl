"""
Collective Future Prediction for Cooperative RL

Agents learn to predict future collective reward and get intrinsic
motivation for actions that improve the group's future.

Key insight: If an agent learns that "cleaning leads to higher future
collective reward", they will be motivated to clean even without
direct individual reward.

Based on ideas from:
- Opponent Shaping (Foerster et al.)
- Cooperative Inverse RL
- Intrinsic Motivation via Future Prediction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque


class FuturePredictorNetwork(nn.Module):
    """
    Predicts future collective reward given current state and action.

    Input: (observation, action) for a single agent
    Output: predicted discounted collective reward over next N steps

    The key is that this network sees the GLOBAL state (including
    pollution level), so it can learn that cleaning → better future.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        num_actions: int,
        hidden_dim: int = 128,
        num_agents: int = 8,
    ):
        super().__init__()

        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.num_agents = num_agents

        # Flatten observation
        obs_dim = np.prod(obs_shape)

        # Action embedding
        self.action_embedding = nn.Embedding(num_actions, 16)

        # Main network
        # Input: flattened obs + action embedding + global features
        input_dim = obs_dim + 16

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Predict single scalar: future reward
        )

    def forward(
        self,
        obs: torch.Tensor,      # (batch, *obs_shape)
        action: torch.Tensor,   # (batch,) int
    ) -> torch.Tensor:
        """
        Predict future collective reward.

        Returns:
            predicted_future: (batch,) predicted discounted collective reward
        """
        batch_size = obs.shape[0]

        # Flatten observation
        obs_flat = obs.view(batch_size, -1)

        # Embed action
        action_emb = self.action_embedding(action)  # (batch, 16)

        # Concatenate
        x = torch.cat([obs_flat, action_emb], dim=-1)

        # Predict
        predicted = self.network(x).squeeze(-1)

        return predicted

    def predict_all_actions(
        self,
        obs: torch.Tensor,  # (batch, *obs_shape)
    ) -> torch.Tensor:
        """
        Predict future reward for ALL possible actions.

        Returns:
            predictions: (batch, num_actions) predicted future for each action
        """
        batch_size = obs.shape[0]
        predictions = []

        for action_idx in range(self.num_actions):
            actions = torch.full((batch_size,), action_idx, dtype=torch.long, device=obs.device)
            pred = self.forward(obs, actions)
            predictions.append(pred)

        return torch.stack(predictions, dim=-1)  # (batch, num_actions)


class CollectiveFuturePrediction:
    """
    Intrinsic motivation based on predicting collective future reward.

    How it works:
    1. Predictor learns: P(obs, action) → future collective reward
    2. For each action, compute: intrinsic_reward = P(obs, action) - baseline
    3. Baseline can be: average over all actions, or value function

    Why this helps cooperation:
    - Predictor sees global state including pollution level
    - It learns: low pollution → more apples → higher future reward
    - Cleaning actions lead to lower pollution → higher predicted future
    - So cleaning gets positive intrinsic reward!
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        num_actions: int = 9,
        num_agents: int = 8,
        prediction_horizon: int = 50,  # How far ahead to predict
        discount: float = 0.99,
        intrinsic_weight: float = 0.5,  # Weight of intrinsic vs extrinsic
        learning_rate: float = 1e-3,
        buffer_size: int = 10000,
        batch_size: int = 256,
        device: str = 'cpu',
    ):
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.prediction_horizon = prediction_horizon
        self.discount = discount
        self.intrinsic_weight = intrinsic_weight
        self.batch_size = batch_size
        self.device = device

        # Create predictor network
        self.predictor = FuturePredictorNetwork(
            obs_shape=obs_shape,
            num_actions=num_actions,
            num_agents=num_agents,
        ).to(device)

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=learning_rate)

        # Experience buffer for training predictor
        # Stores: (obs, action, discounted_future_reward)
        self.buffer = deque(maxlen=buffer_size)

        # Rolling statistics for normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

        # Trajectory buffer for computing returns
        self._current_trajectory: List[Dict] = []

    def compute_intrinsic_reward(
        self,
        obs: Dict[int, np.ndarray],
        actions: Dict[int, int],
    ) -> Dict[int, float]:
        """
        Compute intrinsic reward for each agent based on future prediction.

        The intrinsic reward is: P(obs, my_action) - E[P(obs, a)]
        This rewards actions that lead to BETTER than average future.

        Args:
            obs: Observations for all agents
            actions: Actions taken by all agents

        Returns:
            intrinsic_rewards: Dict of intrinsic rewards per agent
        """
        intrinsic_rewards = {}

        self.predictor.eval()
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                # Get this agent's observation
                agent_obs = torch.tensor(
                    obs[agent_id], dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                # Predict future for ALL actions
                all_predictions = self.predictor.predict_all_actions(agent_obs)  # (1, num_actions)

                # Get prediction for actual action taken
                action = actions[agent_id]
                actual_prediction = all_predictions[0, action].item()

                # Baseline: average prediction across all actions
                baseline = all_predictions[0].mean().item()

                # Intrinsic reward: how much better is this action than average?
                intrinsic = self.intrinsic_weight * (actual_prediction - baseline)

                # Normalize
                intrinsic = intrinsic / (self.reward_std + 1e-8)

                intrinsic_rewards[agent_id] = intrinsic

        return intrinsic_rewards

    def add_experience(
        self,
        obs: Dict[int, np.ndarray],
        actions: Dict[int, int],
        collective_reward: float,
    ):
        """
        Add a step to the current trajectory.

        Args:
            obs: Observations for all agents
            actions: Actions taken
            collective_reward: Sum of rewards for all agents this step
        """
        self._current_trajectory.append({
            'obs': {k: v.copy() for k, v in obs.items()},
            'actions': actions.copy(),
            'collective_reward': collective_reward,
        })

    def end_trajectory(self):
        """
        Called when episode ends. Computes discounted returns and adds to buffer.
        """
        if len(self._current_trajectory) == 0:
            return

        # Compute discounted future rewards for each step
        trajectory = self._current_trajectory
        n = len(trajectory)

        # Compute returns from each timestep
        future_rewards = np.zeros(n)
        running_return = 0.0

        for t in reversed(range(n)):
            running_return = trajectory[t]['collective_reward'] + self.discount * running_return
            future_rewards[t] = running_return

        # Update running statistics
        for r in future_rewards:
            self.reward_count += 1
            delta = r - self.reward_mean
            self.reward_mean += delta / self.reward_count
            delta2 = r - self.reward_mean
            # Welford's online algorithm for variance
            if self.reward_count > 1:
                self.reward_std = np.sqrt(
                    ((self.reward_count - 2) * self.reward_std**2 + delta * delta2)
                    / (self.reward_count - 1)
                )

        # Add to buffer
        for t in range(n):
            step = trajectory[t]
            for agent_id in range(self.num_agents):
                self.buffer.append({
                    'obs': step['obs'][agent_id],
                    'action': step['actions'][agent_id],
                    'future_reward': future_rewards[t],
                })

        # Clear trajectory
        self._current_trajectory = []

    def update_predictor(self, num_updates: int = 10) -> Dict[str, float]:
        """
        Train the predictor network on collected experience.

        Returns:
            metrics: Training metrics
        """
        if len(self.buffer) < self.batch_size:
            return {'predictor_loss': 0.0}

        self.predictor.train()
        total_loss = 0.0

        for _ in range(num_updates):
            # Sample batch
            indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]

            # Prepare tensors
            obs = torch.tensor(
                np.stack([b['obs'] for b in batch]),
                dtype=torch.float32, device=self.device
            )
            actions = torch.tensor(
                [b['action'] for b in batch],
                dtype=torch.long, device=self.device
            )
            targets = torch.tensor(
                [b['future_reward'] for b in batch],
                dtype=torch.float32, device=self.device
            )

            # Normalize targets
            targets = (targets - self.reward_mean) / (self.reward_std + 1e-8)

            # Forward pass
            predictions = self.predictor(obs, actions)

            # MSE loss
            loss = nn.functional.mse_loss(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return {'predictor_loss': total_loss / num_updates}

    def get_action_values(
        self,
        obs: np.ndarray,  # Single agent observation
    ) -> np.ndarray:
        """
        Get predicted future value for each action (for debugging/analysis).

        Returns:
            values: (num_actions,) predicted future for each action
        """
        self.predictor.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            values = self.predictor.predict_all_actions(obs_tensor)
            return values[0].cpu().numpy()


def create_future_prediction(
    obs_shape: Tuple[int, ...],
    num_actions: int = 9,
    num_agents: int = 8,
    intrinsic_weight: float = 0.5,
    prediction_horizon: int = 50,
    device: str = 'cpu',
) -> CollectiveFuturePrediction:
    """
    Factory function to create CollectiveFuturePrediction module.

    Args:
        obs_shape: Shape of single agent observation
        num_actions: Number of possible actions
        num_agents: Number of agents
        intrinsic_weight: Weight of intrinsic reward (0.5 = equal to extrinsic)
        prediction_horizon: How many steps ahead to consider
        device: 'cpu' or 'cuda'

    Returns:
        Configured CollectiveFuturePrediction instance
    """
    return CollectiveFuturePrediction(
        obs_shape=obs_shape,
        num_actions=num_actions,
        num_agents=num_agents,
        intrinsic_weight=intrinsic_weight,
        prediction_horizon=prediction_horizon,
        device=device,
    )


# Quick test
if __name__ == '__main__':
    print("Testing Collective Future Prediction...")

    # Create module
    cfp = create_future_prediction(
        obs_shape=(11, 11, 8),
        num_actions=9,
        num_agents=8,
        intrinsic_weight=0.5,
    )

    # Simulate some experience
    for episode in range(5):
        for step in range(100):
            obs = {i: np.random.randn(11, 11, 8).astype(np.float32) for i in range(8)}
            actions = {i: np.random.randint(0, 9) for i in range(8)}
            collective_reward = np.random.randn() * 10

            cfp.add_experience(obs, actions, collective_reward)

        cfp.end_trajectory()

    # Train predictor
    metrics = cfp.update_predictor(num_updates=20)
    print(f"Predictor loss: {metrics['predictor_loss']:.4f}")

    # Test intrinsic reward
    obs = {i: np.random.randn(11, 11, 8).astype(np.float32) for i in range(8)}
    actions = {i: np.random.randint(0, 9) for i in range(8)}
    intrinsic = cfp.compute_intrinsic_reward(obs, actions)
    print(f"Intrinsic rewards: {intrinsic}")

    print("\nTest passed!")
