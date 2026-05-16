# ppo_loss.py
"""
Covariate-Shift-Protected PPO Loss anchored to a frozen human reference policy.

CovariateShiftProtectedPPOLoss
  Standard clipped PPO objective augmented with an adaptive KL-divergence
  penalty term that keeps the live policy close to a frozen pretrained
  reference network (similar to RLHF / DPO anchoring).

  The KL beta coefficient self-adjusts via a PID-style rule:
    - If KL > 1.5 × target  → increase beta (tighten the leash)
    - If KL < target / 1.5  → decrease beta (loosen the leash)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CovariateShiftProtectedPPOLoss:
    def __init__(self, kl_target: float = 0.01, kl_beta: float = 0.5):
        self.kl_target = kl_target
        self.kl_beta = kl_beta  # Dynamic adaptive scaling coefficient

    def compute_loss(
        self,
        policy_network: nn.Module,
        ref_network: nn.Module,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the full PPO + KL anchoring loss.

        Args:
            policy_network: Live network being optimised.
            ref_network:    Frozen pretrained reference network (no gradient).
            states:         (batch, state_dim) state tensor.
            actions:        (batch,) action indices.
            old_log_probs:  (batch,) log π_old(a|s) from the rollout.
            advantages:     (batch,) normalised advantage estimates.

        Returns:
            Scalar loss tensor.
        """
        # 1. Forward pass on live network
        action_dist, _value_pred = policy_network(states)
        new_log_probs = action_dist.log_prob(actions)

        # 2. Forward pass on frozen reference network (zero gradient overhead)
        with torch.no_grad():
            ref_dist, _ = ref_network(states)

        # 3. Standard PPO Clipped Objective
        ratios = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
        ppo_loss = -torch.min(surr1, surr2).mean()

        # 4. KL divergence: KL(ref || live)
        # F.kl_div(log_input, target) computes target*(log(target) - log_input)
        live_probs = action_dist.probs
        ref_probs = ref_dist.probs
        kl_div = F.kl_div(
            torch.log(live_probs + 1e-8),
            ref_probs,
            reduction="batchmean",
        )

        # 5. Adaptive beta tuning (PID-style regulation)
        if kl_div.item() > self.kl_target * 1.5:
            self.kl_beta *= 1.1
        elif kl_div.item() < self.kl_target / 1.5:
            self.kl_beta *= 0.9

        return ppo_loss + (self.kl_beta * kl_div)
