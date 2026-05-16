# entropy_tuner.py
"""
AdaptiveEntropyTuner — self-calibrating exploration regulariser.

Tracks the live policy's entropy relative to the theoretical maximum for the
current action space and adjusts the entropy coefficient (beta) to keep
exploration within a configurable target band.

  - beta increases when entropy is too low  (repetitive / greedy actions)
  - beta decreases when entropy is too high (noisy / random actions)
"""

import torch


class AdaptiveEntropyTuner:
    def __init__(
        self,
        target_entropy_ratio: float = 0.25,
        adaptation_speed: float = 0.05,
        beta_min: float = 0.001,
        beta_max: float = 0.1,
    ):
        self.target_entropy_ratio = target_entropy_ratio
        self.adaptation_speed = adaptation_speed
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.current_entropy_beta = 0.01  # Starting coefficient

    def tune_and_regularize(
        self,
        joint_entropy: torch.Tensor,
        max_possible_entropy: float,
    ) -> tuple[torch.Tensor, float]:
        """
        Computes the entropy loss component and updates beta for the next step.

        Args:
            joint_entropy:       Mean joint entropy across the multi-discrete heads.
            max_possible_entropy: Theoretical maximum — sum of log(n) for each head.

        Returns:
            (entropy_loss, updated_beta)
            entropy_loss is negative (gradient ascent → maximise entropy).
        """
        mean_entropy_val = joint_entropy.mean().item()
        normalised = mean_entropy_val / (max_possible_entropy + 1e-8)

        # PID-style proportional adjustment
        entropy_error = self.target_entropy_ratio - normalised
        self.current_entropy_beta += self.adaptation_speed * entropy_error
        self.current_entropy_beta = max(
            self.beta_min, min(self.beta_max, self.current_entropy_beta)
        )

        # Negate because we maximise entropy during gradient descent
        entropy_loss = -self.current_entropy_beta * joint_entropy.mean()
        return entropy_loss, self.current_entropy_beta
