# reward_shaper.py
"""
Potential-Based Reward Shaping (PBRS) for Brawlhalla.

AlignedRewardShaper computes a composite shaped reward that:
  - Penalises edge/offstage positioning via a spatial potential function
  - Adds a PBRS delta each step (V(s) - V(s')) — shaping-theorem compliant
  - Rewards damage dealt, punishes stock loss, bonuses combo connections
"""

import numpy as np
from typing import Dict, Any


class AlignedRewardShaper:
    def __init__(self, stage_center_x: float = 0.0, blastzone_y_limit: float = -1200.0):
        self.stage_center_x = stage_center_x
        self.blastzone_y_limit = blastzone_y_limit

        # Tracks prior step potentials to compute formal PBRS changes: V(s) - V(s')
        self.prev_potential: float | None = None

    def calculate_step_reward(self, state_dict: Dict[str, Any], raw_env_reward: float) -> float:
        """
        Calculates potential-based reward adjustments.
        Expected keys in state_dict:
          'player_x', 'player_y', 'opponent_x', 'opponent_y',
          'player_damage', 'opponent_damage', 'player_stocks', 'opponent_stocks'
        Optional event keys:
          'event_player_died', 'event_combo_connected'
        """
        # 1. Base Game State Extraction
        p_x = state_dict['player_x']
        p_y = state_dict['player_y']

        # 2. Formulate Potential Function V(s) for Spatial Safety Alignment
        # Measures quadratic distance degradation from neutral safety on stage
        dist_from_center = np.sqrt(
            (p_x - self.stage_center_x) ** 2 + (p_y if p_y < 0 else 0) ** 2
        )

        # Vectorised potential assignment: high score is safe, low score is edge danger
        spatial_potential = -0.005 * dist_from_center

        # Proximity alert scaling if agent drops below stage lips towards hard blastzone limits
        if p_y < -500:
            danger_delta = abs(p_y - self.blastzone_y_limit)
            spatial_potential -= (1000.0 / (danger_delta + 1.0)) * 0.05

        # 3. Compute Formal Potential-Based Delta
        if self.prev_potential is None:
            self.prev_potential = spatial_potential

        pbrs_delta = spatial_potential - self.prev_potential
        self.prev_potential = spatial_potential

        # 4. Integrate Non-Linear Punishment/Reward Balancing (Anti-Suicide Shaping)
        # Prevents agent from trading its life for cheap damage
        damage_dealt_reward = max(0.0, raw_env_reward) * 1.5

        stock_loss_penalty = 0.0
        if state_dict.get('event_player_died', False):
            # Heavy penalty to completely negate suicidal trades
            stock_loss_penalty = -25.0

        combo_accel_bonus = 0.0
        if state_dict.get('event_combo_connected', False):
            # Reinforces high learning rate values targeted by Whisper pretraining scripts
            combo_accel_bonus = 2.0

        # 5. Composite Unified Reward Synthesis
        total_shaped_reward = (
            damage_dealt_reward
            + pbrs_delta
            + stock_loss_penalty
            + combo_accel_bonus
        )

        return float(total_shaped_reward)

    def reset(self):
        """Call at the start of each episode to clear potential history."""
        self.prev_potential = None
