# advantage_engine.py
"""
Eligibility-Traced Generalised Advantage Estimation (ET-GAE).

EligibilityTracedAdvantageEngine extends standard GAE with a per-step
eligibility weight that differentiates intentional attack actions from
passive/idle frames.

  - Active attack frames   → eligibility weight = 1.0 (full credit)
  - Passive / idle frames  → eligibility weight = trace_decay (attenuated)

This prevents the agent from earning inflated advantage credit for being in
the right place at the right time and instead focuses credit on deliberate
offensive/defensive decisions.
"""

import numpy as np
import torch


class EligibilityTracedAdvantageEngine:
    def __init__(
        self,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        trace_decay: float = 0.85,
    ):
        self.gamma = gamma
        self.lmbda = lmbda
        self.trace_decay = trace_decay

    def compute_precise_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        action_execution_flags: np.ndarray,
    ) -> torch.Tensor:
        """
        Computes an eligibility-traced advantage vector.

        Args:
            rewards:                (T,) float32 shaped rewards.
            values:                 (T,) float32 value estimates V(s_t).
            dones:                  (T,) float32 terminal flags (1.0 = done).
            action_execution_flags: (T,) binary mask — 1 if an offensive or
                                    defensive attack action was taken, 0 if idle.

        Returns:
            Normalised advantage tensor of shape (T,) on CUDA.
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T - 1)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * non_terminal - values[t]

            # Active attack → preserve full credit; idle → decay
            eligibility_weight = 1.0 if action_execution_flags[t] == 1 else self.trace_decay

            gae = delta + self.gamma * self.lmbda * non_terminal * gae * eligibility_weight
            advantages[t] = gae

        # Normalise over batch dimensions to stabilise gradient updates
        normalised = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return torch.from_numpy(normalised).float().cuda()
