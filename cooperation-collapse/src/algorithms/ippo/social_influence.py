"""
Social Influence as Intrinsic Motivation for Multi-Agent RL

Based on Jaques et al. (2019): "Social Influence as Intrinsic Motivation
for Multi-Agent Deep Reinforcement Learning"

Key idea: Reward agents for having causal influence over other agents' actions.
Influence = how much agent i's action changes the predicted actions of agent j.

This promotes coordination and cooperation in social dilemmas.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from typing import Tuple, Dict, Any, Sequence
from functools import partial


class ModelOfOtherAgents(nn.Module):
    """
    Model of Other Agents (MOA) - predicts other agents' actions.

    Two modes:
    1. Marginal: P(a_j | s) - predict without knowing agent i's action
    2. Conditional: P(a_j | s, a_i) - predict given agent i's action

    Influence = KL(Conditional || Marginal)
    """
    hidden_dims: Sequence[int] = (64, 64)
    num_actions: int = 9
    num_agents: int = 8

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,           # (batch, obs_dim) - flattened observation
        agent_action: jnp.ndarray,  # (batch,) - agent i's action (optional, -1 for marginal)
        training: bool = True
    ) -> jnp.ndarray:
        """
        Predict action distribution for all other agents.

        Returns:
            logits: (batch, num_agents-1, num_actions) - action logits for each other agent
        """
        x = obs

        # Encode observation
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f'obs_encoder_{i}')(x)
            x = nn.relu(x)

        # Conditionally include agent's action (one-hot encoded)
        # If agent_action >= 0, include it; otherwise marginal prediction
        action_embedding = jnp.zeros((obs.shape[0], self.num_actions))

        # One-hot encode action where valid
        valid_mask = agent_action >= 0
        action_one_hot = jax.nn.one_hot(
            jnp.maximum(agent_action, 0),  # Avoid negative indexing
            self.num_actions
        )
        action_embedding = jnp.where(
            valid_mask[:, None],
            action_one_hot,
            jnp.zeros_like(action_one_hot)
        )

        # Concatenate observation encoding with action embedding
        x = jnp.concatenate([x, action_embedding], axis=-1)

        # Predict other agents' actions
        x = nn.Dense(self.hidden_dims[-1], name='combined_encoder')(x)
        x = nn.relu(x)

        # Output head for each other agent
        # Shape: (batch, (num_agents-1) * num_actions)
        logits = nn.Dense(
            (self.num_agents - 1) * self.num_actions,
            name='action_predictor'
        )(x)

        # Reshape to (batch, num_agents-1, num_actions)
        logits = logits.reshape(-1, self.num_agents - 1, self.num_actions)

        return logits


class SocialInfluenceReward:
    """
    Computes social influence intrinsic reward.

    Influence of agent i on agent j:
    I(i→j) = KL(P(a_j|s,a_i) || P(a_j|s))

    Total influence reward for agent i:
    R_influence = sum_j I(i→j)
    """

    def __init__(
        self,
        num_agents: int = 8,
        num_actions: int = 9,
        obs_dim: int = 968,  # 11*11*8 flattened
        hidden_dims: Sequence[int] = (64, 64),
        learning_rate: float = 1e-3,
        influence_weight: float = 0.1,  # λ coefficient for influence reward
        seed: int = 42,
    ):
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.obs_dim = obs_dim
        self.influence_weight = influence_weight

        # Create MOA for each agent
        self.moa = ModelOfOtherAgents(
            hidden_dims=hidden_dims,
            num_actions=num_actions,
            num_agents=num_agents,
        )

        # Initialize parameters for each agent's MOA
        self.rng = jax.random.PRNGKey(seed)
        self.moa_states = {}

        for agent_id in range(num_agents):
            self.rng, init_rng = jax.random.split(self.rng)
            dummy_obs = jnp.zeros((1, obs_dim))
            dummy_action = jnp.zeros((1,), dtype=jnp.int32)

            params = self.moa.init(init_rng, dummy_obs, dummy_action)

            self.moa_states[agent_id] = TrainState.create(
                apply_fn=self.moa.apply,
                params=params,
                tx=optax.adam(learning_rate),
            )

    @partial(jax.jit, static_argnums=(0,))
    def compute_influence_reward(
        self,
        moa_params: Dict,
        obs: jnp.ndarray,      # (num_agents, obs_dim)
        actions: jnp.ndarray,  # (num_agents,)
    ) -> jnp.ndarray:
        """
        Compute influence reward for each agent.

        Args:
            moa_params: Parameters for all MOA models
            obs: Observations for all agents
            actions: Actions taken by all agents

        Returns:
            influence_rewards: (num_agents,) influence reward for each agent
        """
        influence_rewards = jnp.zeros(self.num_agents)

        for agent_id in range(self.num_agents):
            agent_obs = obs[agent_id:agent_id+1]  # (1, obs_dim)
            agent_action = actions[agent_id:agent_id+1]  # (1,)

            # Get marginal prediction P(a_j | s)
            marginal_action = jnp.array([-1])  # Signal for marginal prediction
            marginal_logits = self.moa.apply(
                moa_params[agent_id],
                agent_obs,
                marginal_action,
                training=False
            )  # (1, num_agents-1, num_actions)

            # Get conditional prediction P(a_j | s, a_i)
            conditional_logits = self.moa.apply(
                moa_params[agent_id],
                agent_obs,
                agent_action,
                training=False
            )  # (1, num_agents-1, num_actions)

            # Convert to probabilities
            marginal_probs = jax.nn.softmax(marginal_logits, axis=-1)
            conditional_probs = jax.nn.softmax(conditional_logits, axis=-1)

            # Compute KL divergence: KL(conditional || marginal)
            # KL = sum(p * log(p/q))
            kl_div = jnp.sum(
                conditional_probs * (
                    jnp.log(conditional_probs + 1e-8) -
                    jnp.log(marginal_probs + 1e-8)
                ),
                axis=-1  # Sum over actions
            )  # (1, num_agents-1)

            # Total influence = sum over all other agents
            total_influence = jnp.sum(kl_div)

            influence_rewards = influence_rewards.at[agent_id].set(total_influence)

        return self.influence_weight * influence_rewards

    def compute_influence_batch(
        self,
        obs_batch: jnp.ndarray,      # (batch, num_agents, obs_dim)
        actions_batch: jnp.ndarray,  # (batch, num_agents)
    ) -> jnp.ndarray:
        """Compute influence rewards for a batch of transitions."""
        moa_params = {i: self.moa_states[i].params for i in range(self.num_agents)}

        # Vectorize over batch
        influence_rewards = jax.vmap(
            lambda o, a: self.compute_influence_reward(moa_params, o, a)
        )(obs_batch, actions_batch)

        return influence_rewards  # (batch, num_agents)

    @partial(jax.jit, static_argnums=(0,))
    def _update_moa_single(
        self,
        state: TrainState,
        obs: jnp.ndarray,           # (batch, obs_dim)
        agent_actions: jnp.ndarray, # (batch,)
        other_actions: jnp.ndarray, # (batch, num_agents-1)
    ) -> Tuple[TrainState, float]:
        """Update single agent's MOA."""

        def loss_fn(params):
            # Predict other agents' actions given state and own action
            logits = state.apply_fn(
                params, obs, agent_actions, training=True
            )  # (batch, num_agents-1, num_actions)

            # Cross-entropy loss for each other agent
            one_hot_targets = jax.nn.one_hot(other_actions, self.num_actions)
            loss = -jnp.sum(one_hot_targets * jax.nn.log_softmax(logits, axis=-1))
            loss = loss / (obs.shape[0] * (self.num_agents - 1))

            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)

        return state, loss

    def update_moa(
        self,
        obs_batch: jnp.ndarray,      # (batch, num_agents, obs_dim)
        actions_batch: jnp.ndarray,  # (batch, num_agents)
    ) -> Dict[str, float]:
        """
        Update all MOA models using observed transitions.

        Args:
            obs_batch: Observations for all agents
            actions_batch: Actions taken by all agents

        Returns:
            losses: Dictionary of MOA losses per agent
        """
        losses = {}

        for agent_id in range(self.num_agents):
            # Get this agent's observations and actions
            agent_obs = obs_batch[:, agent_id]  # (batch, obs_dim)
            agent_actions = actions_batch[:, agent_id]  # (batch,)

            # Get other agents' actions (excluding this agent)
            other_mask = jnp.arange(self.num_agents) != agent_id
            other_actions = actions_batch[:, other_mask]  # (batch, num_agents-1)

            # Update MOA
            self.moa_states[agent_id], loss = self._update_moa_single(
                self.moa_states[agent_id],
                agent_obs,
                agent_actions,
                other_actions,
            )

            losses[f'moa_loss_{agent_id}'] = float(loss)

        return losses

    def get_moa_params(self) -> Dict:
        """Get all MOA parameters for checkpointing."""
        return {i: self.moa_states[i].params for i in range(self.num_agents)}

    def set_moa_params(self, params: Dict):
        """Set MOA parameters from checkpoint."""
        for i, p in params.items():
            self.moa_states[i] = self.moa_states[i].replace(params=p)


def create_social_influence(
    num_agents: int = 8,
    num_actions: int = 9,
    obs_shape: Tuple[int, ...] = (11, 11, 8),
    influence_weight: float = 0.1,
    seed: int = 42,
) -> SocialInfluenceReward:
    """Factory function to create SocialInfluenceReward."""
    obs_dim = int(jnp.prod(jnp.array(obs_shape)))

    return SocialInfluenceReward(
        num_agents=num_agents,
        num_actions=num_actions,
        obs_dim=obs_dim,
        influence_weight=influence_weight,
        seed=seed,
    )
