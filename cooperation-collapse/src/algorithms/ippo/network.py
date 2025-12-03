"""
Neural Network Architectures for IPPO/MAPPO

Implements Actor-Critic networks with support for:
- MLP and CNN encoders
- Optional RNN layers
- Parameter sharing across agents
"""

from typing import Sequence, Tuple, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn


class CNNEncoder(nn.Module):
    """CNN encoder for grid-world observations."""
    features: Sequence[int] = (32, 64, 64)

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')
        return x.reshape((x.shape[0], -1))  # Flatten


class MLPEncoder(nn.Module):
    """MLP encoder for flattened observations."""
    hidden_dims: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        return x


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Features:
    - Shared encoder for actor and critic
    - Separate output heads
    - Optional RNN layer for temporal dependencies
    """
    action_dim: int
    hidden_dims: Sequence[int] = (64, 64)
    use_cnn: bool = True
    use_rnn: bool = False
    rnn_hidden_size: int = 64

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        hidden_state: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Forward pass.

        Args:
            obs: Observation tensor [batch, ...]
            hidden_state: RNN hidden state (if use_rnn=True)
            deterministic: If True, return argmax action

        Returns:
            action_logits: Logits for action distribution [batch, action_dim]
            value: State value estimate [batch, 1]
            new_hidden_state: Updated RNN state (or None)
        """
        batch_size = obs.shape[0]

        # Encode observation
        if self.use_cnn and len(obs.shape) > 2:
            x = CNNEncoder()(obs)
        else:
            x = MLPEncoder(hidden_dims=self.hidden_dims)(obs)

        # Optional RNN
        new_hidden_state = None
        if self.use_rnn:
            if hidden_state is None:
                hidden_state = jnp.zeros((batch_size, self.rnn_hidden_size))

            rnn_cell = nn.GRUCell(features=self.rnn_hidden_size)
            new_hidden_state, x = rnn_cell(hidden_state, x)

        # Shared representation
        shared = nn.Dense(self.hidden_dims[-1])(x)
        shared = nn.relu(shared)

        # Actor head
        actor_hidden = nn.Dense(self.hidden_dims[-1])(shared)
        actor_hidden = nn.relu(actor_hidden)
        action_logits = nn.Dense(self.action_dim)(actor_hidden)

        # Critic head
        critic_hidden = nn.Dense(self.hidden_dims[-1])(shared)
        critic_hidden = nn.relu(critic_hidden)
        value = nn.Dense(1)(critic_hidden)

        return action_logits, value, new_hidden_state


class CentralizedCritic(nn.Module):
    """
    Centralized critic for MAPPO.
    Takes concatenated observations (world state) as input.
    """
    hidden_dims: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, world_state: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            world_state: Concatenated agent observations [batch, state_dim]

        Returns:
            value: State value estimate [batch, 1]
        """
        x = world_state.reshape((world_state.shape[0], -1))

        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)

        value = nn.Dense(1)(x)
        return value


def create_actor_critic(
    action_dim: int,
    hidden_dims: Sequence[int] = (64, 64),
    use_cnn: bool = True,
    use_rnn: bool = False,
    rnn_hidden_size: int = 64,
) -> ActorCritic:
    """Factory function to create ActorCritic network."""
    return ActorCritic(
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        use_cnn=use_cnn,
        use_rnn=use_rnn,
        rnn_hidden_size=rnn_hidden_size,
    )


def init_network(
    network: nn.Module,
    rng: jax.random.PRNGKey,
    sample_obs: jnp.ndarray,
) -> dict:
    """Initialize network parameters."""
    # Add batch dimension if needed
    if len(sample_obs.shape) == 3:
        sample_obs = sample_obs[None, ...]

    return network.init(rng, sample_obs)
