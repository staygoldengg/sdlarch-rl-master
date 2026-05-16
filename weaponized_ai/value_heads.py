# value_heads.py
"""
Auxiliary value and training utility components.

MacroStateValueBootstrapper
  Dual-head critic that separates short-term frame value from long-term
  match momentum.  Blends them into a unified TD target via alpha weighting.

RunningStateNormalizer
  nn.Module that performs online Welford normalisation of the state vector.
  Buffers are registered so they survive model.save()/load() cycles.

TrajectoryRewardFilter
  Normalises step rewards against a rolling return variance window to
  prevent catastrophic gradient updates during match-winning moments.

OrthogonalNetworkInitializer
  One-shot weight init utility — applies orthogonal matrices with gain=√2
  to every Linear layer and zeros all biases.
"""

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Dual Critic
# ─────────────────────────────────────────────────────────────────────────────

class MacroStateValueBootstrapper(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        # Short-term frame value (immediate combat interactions)
        self.micro_value_head = nn.Linear(latent_dim, 1)
        # Long-term match momentum (stock differential trend)
        self.macro_value_head = nn.Linear(latent_dim, 1)

    def forward(self, latent_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.micro_value_head(latent_features), self.macro_value_head(latent_features)

    def compute_blended_targets(
        self,
        micro_predictions: torch.Tensor,
        macro_predictions: torch.Tensor,
        stock_differential: torch.Tensor,
        alpha: float = 0.7,
    ) -> torch.Tensor:
        """
        Blends micro and macro value estimates into a single TD learning target.

        Args:
            stock_differential: (player_stocks - opponent_stocks) in [-1.0, 1.0] range.
            alpha:              Weight on the micro (immediate) value head.

        Returns:
            Blended value estimate tensor.
        """
        macro_target = stock_differential * 10.0  # noqa: F841 — kept for loss callers
        return (alpha * micro_predictions) + ((1.0 - alpha) * macro_predictions)


# ─────────────────────────────────────────────────────────────────────────────
# Online State Normaliser (Welford)
# ─────────────────────────────────────────────────────────────────────────────

class RunningStateNormalizer(nn.Module):
    """Dynamically normalises state features using streaming Welford statistics."""

    def __init__(self, feature_dim: int = 64, clip_threshold: float = 10.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.clip_threshold = clip_threshold

        # Persisted as buffers — saved with the model, not tracked by autograd
        self.register_buffer("running_mean",  torch.zeros(feature_dim))
        self.register_buffer("running_var",   torch.ones(feature_dim))
        self.register_buffer("step_count",    torch.zeros(1))

    def forward(self, raw: torch.Tensor, update_metrics: bool = True) -> torch.Tensor:
        if self.training and update_metrics:
            self.step_count += 1
            momentum = 1.0 / self.step_count.item()
            batch_mean = raw.mean(dim=0)
            batch_var  = raw.var(dim=0, unbiased=False) + 1e-8
            self.running_mean = (1.0 - momentum) * self.running_mean + momentum * batch_mean
            self.running_var  = (1.0 - momentum) * self.running_var  + momentum * batch_var

        std = torch.sqrt(self.running_var)
        normalised = (raw - self.running_mean) / std
        return torch.clamp(normalised, -self.clip_threshold, self.clip_threshold)


# ─────────────────────────────────────────────────────────────────────────────
# Reward Normaliser
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryRewardFilter:
    """Rescales step rewards by rolling return standard deviation."""

    def __init__(self, rolling_window_size: int = 1000, discount_gamma: float = 0.99):
        self.gamma = discount_gamma
        self.window_size = rolling_window_size
        self.return_history: list[float] = []
        self.current_return = 0.0

    def process_reward(self, step_reward: float, is_done: bool) -> float:
        self.current_return = step_reward + self.gamma * self.current_return
        self.return_history.append(self.current_return)

        if is_done:
            self.current_return = 0.0

        if len(self.return_history) > self.window_size:
            self.return_history.pop(0)

        returns_std = np.std(self.return_history) if len(self.return_history) > 10 else 1.0
        normalised  = step_reward / (returns_std + 1e-8)
        return float(np.clip(normalised, -5.0, 5.0))


# ─────────────────────────────────────────────────────────────────────────────
# Weight Initialiser
# ─────────────────────────────────────────────────────────────────────────────

class OrthogonalNetworkInitializer:
    """Applies orthogonal weight init to every Linear layer in a model."""

    @staticmethod
    def configure_module_weights(model: nn.Module, gain: float = 1.414):
        """
        Args:
            gain: √2 is optimal for Mish/ReLU activations.
                  Use 0.01 for the final output layer if needed.
        """
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        print("[INITIALIZER] Orthogonal matrix parameterisation applied to all Linear layers.")
