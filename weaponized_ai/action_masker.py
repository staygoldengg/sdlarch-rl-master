# action_masker.py
"""
StateDependentActionMasker — filters mechanically invalid actions before
they reach the softmax layer.

Reads raw float values from the SHM bridge state vector and generates
per-head logit masks:
  - mask value 1.0 → action is legal
  - mask value 0.0 → action is illegal (logit set to -1e9 before softmax)

Rules implemented:
  1. Jump masked when air-jump count is exhausted (state_vector[12])
  2. (Extensible) weapon-status-dependent signature masking (state_vector[15])
"""

import numpy as np
import torch


class StateDependentActionMasker:
    # State vector index map (matches BrawlhallaStateBuffer + feature_vector layout)
    IDX_AIR_JUMPS_REMAINING = 12
    IDX_IS_UNARMED          = 15

    def generate_masks(self, state_vector: np.ndarray) -> dict[str, torch.Tensor]:
        """
        Returns per-head boolean masks as float CUDA tensors.

        Heads:
          mask_x      → [Neutral, Left, Right]  (dim 3)
          mask_y      → [Neutral, Up, Down]      (dim 3)
          mask_action → [Neutral, Light, Heavy, Dodge, Jump]  (dim 5)
        """
        mask_x      = np.ones(3, dtype=np.float32)
        mask_y      = np.ones(3, dtype=np.float32)
        mask_action = np.ones(5, dtype=np.float32)

        jumps_left = float(state_vector[self.IDX_AIR_JUMPS_REMAINING])
        is_unarmed = float(state_vector[self.IDX_IS_UNARMED])

        # Rule 1: Prevent air-jump when movement budget is exhausted
        if jumps_left <= 0.0:
            mask_action[4] = 0.0  # block Jump (index 4)

        # Rule 2: Weapon-status modifiers (extend as needed)
        if is_unarmed > 0.5:
            # Currently a no-op — placeholder for signature-specific masking
            pass

        return {
            "mask_x":      torch.from_numpy(mask_x).cuda(),
            "mask_y":      torch.from_numpy(mask_y).cuda(),
            "mask_action": torch.from_numpy(mask_action).cuda(),
        }

    @staticmethod
    def apply_mask_to_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Zeros the probability of blocked actions by subtracting 1e9 from their logits
        before softmax.

        Args:
            logits: Raw policy logits (batch, n_actions).
            mask:   Float mask (n_actions,) — 0.0 blocks, 1.0 allows.
        """
        penalty = (1.0 - mask) * -1e9
        return logits + penalty
