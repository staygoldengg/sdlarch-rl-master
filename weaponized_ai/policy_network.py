# policy_network.py
"""
Neural network policy heads for Striker — The Enlightened.

EnlightenedPolicyNetwork
  Monolithic Actor-Critic with shared encoder. Accepts a flat RAM state vector
  and outputs a Categorical action distribution + scalar value estimate V(s).

FactorizedMultiDiscreteActorHead
  Decomposes the action space into three independent Categorical heads
  (move_x, move_y, action) so the agent can learn movement and attack
  decisions with separated gradient flows.

TemporalConditioningEncoder
  Projects raw RAM state + real-world time-delta dt into a fused latent
  context vector for use with any downstream policy head.
"""

import time  # noqa: F401 — kept for downstream callers that import from this module
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ─────────────────────────────────────────────────────────────────────────────
# Monolithic Actor-Critic
# ─────────────────────────────────────────────────────────────────────────────

class EnlightenedPolicyNetwork(nn.Module):
    def __init__(self, ram_state_dim: int = 64, num_actions: int = 14):
        super().__init__()

        # Hyper-dense feature encoder to extract fighting game interactions
        self.shared_encoder = nn.Sequential(
            nn.Linear(ram_state_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),  # Mish prevents dead neurons during intense training state transitions
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Mish(),
        )

        # Actor Head — outputs policy action log-probabilities
        self.actor = nn.Linear(256, num_actions)

        # Critic Head — outputs baseline value estimate V(s)
        self.critic = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch_size, ram_state_dim) pulled from the C++ SHM bridge.

        Returns:
            (Categorical distribution, value tensor of shape (batch_size, 1))
        """
        features = self.shared_encoder(x)
        logits = self.actor(features)
        value = self.critic(features)
        # Categorical distribution over input combinations
        # (Jump, Attack, NLight, Recovery, Dodge, …)
        return Categorical(logits=logits), value


# ─────────────────────────────────────────────────────────────────────────────
# Factorised Multi-Discrete Action Head
# ─────────────────────────────────────────────────────────────────────────────

class FactorizedMultiDiscreteActorHead(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()

        # Head 1: Horizontal movement  (0=Neutral, 1=Left, 2=Right)
        self.move_x_head = nn.Linear(latent_dim, 3)

        # Head 2: Vertical modifiers   (0=Neutral, 1=Up, 2=Down)
        self.move_y_head = nn.Linear(latent_dim, 3)

        # Head 3: Action execution     (0=Neutral, 1=Light, 2=Heavy, 3=Dodge, 4=Jump)
        self.action_head = nn.Linear(latent_dim, 5)

    def forward(self, latent_features: torch.Tensor) -> tuple[list[Categorical], torch.Tensor]:
        """
        Returns:
            distributions: [dist_x, dist_y, dist_action]
            joint_entropy:  scalar tensor — sum of per-head entropies
        """
        dist_x   = Categorical(logits=self.move_x_head(latent_features))
        dist_y   = Categorical(logits=self.move_y_head(latent_features))
        dist_act = Categorical(logits=self.action_head(latent_features))

        joint_entropy = dist_x.entropy() + dist_y.entropy() + dist_act.entropy()
        return [dist_x, dist_y, dist_act], joint_entropy

    def sample_to_macro(self, distributions: list[Categorical]) -> dict[str, int]:
        """
        Samples from each head and returns a macro dict compatible with
        ActionTranslationEngine.execute_macro_dict().

        Returns:
            {
                "move_x":  0 (None)  | 1 (Left) | 2 (Right),
                "move_y":  0 (None)  | 1 (Up)   | 2 (Down),
                "action":  0 (None)  | 1 (Light) | 2 (Heavy) | 3 (Dodge) | 4 (Jump),
            }
        """
        return {
            "move_x": int(distributions[0].sample().item()),
            "move_y": int(distributions[1].sample().item()),
            "action": int(distributions[2].sample().item()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Conditioning Encoder
# ─────────────────────────────────────────────────────────────────────────────

class TemporalConditioningEncoder(nn.Module):
    """
    Fuses raw RAM state features with real-world time delta into a single
    latent vector.  Used when precise loop timing matters for credit assignment.
    """

    def __init__(self, raw_ram_dim: int = 64, latent_dim: int = 256):
        super().__init__()

        # State projector that expands raw RAM attributes
        self.state_projector = nn.Linear(raw_ram_dim, latent_dim - 16)

        # Temporal positional mapping network (processes time delta signatures)
        self.temporal_projector = nn.Sequential(
            nn.Linear(1, 16),
            nn.Mish(),
        )

        self.combined_encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Mish(),
        )

    def forward(self, raw_state: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_state: (batch_size, raw_ram_dim) — state features from SHM.
            dt:        (batch_size, 1)            — real-world time delta in seconds.

        Returns:
            Fused context tensor of shape (batch_size, latent_dim).
        """
        s_feat = self.state_projector(raw_state)
        t_feat = self.temporal_projector(dt)
        fused_context = torch.cat([s_feat, t_feat], dim=-1)
        return self.combined_encoder(fused_context)
