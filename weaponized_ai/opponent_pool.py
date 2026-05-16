# opponent_pool.py
"""
Self-play opponent pool with dynamic matchmaking sampling.

OpponentPoolReservoir manages historical policy snapshots and in-game
bot profiles to provide a curriculum of opponents during RL training.

Sampling strategy:
  - 30 % chance: load a historical self-play weight snapshot
  - 70 % chance: select a baseline in-engine bot difficulty profile
"""

import random
import glob
import os
import torch


class OpponentPoolReservoir:
    def __init__(self, weights_directory: str = "brain/snapshots/"):
        self.weights_directory = weights_directory
        os.makedirs(weights_directory, exist_ok=True)
        self.active_profiles = ["aggressive_bot", "passive_bot", "balanced_bot"]
        self.snapshot_interval_steps = 50_000

    def register_snapshot(self, model: torch.nn.Module, total_steps: int):
        """Saves a frozen copy of the current policy at snapshot_interval_steps boundaries."""
        # Guard against total_steps == 0 (Python's modulo returns 0 for any divisor)
        # which would snapshot an untrained network at loop initialisation.
        if total_steps > 0 and total_steps % self.snapshot_interval_steps == 0:
            snapshot_path = os.path.join(
                self.weights_directory, f"striker_v_{total_steps}.pt"
            )
            torch.save(model.state_dict(), snapshot_path)
            print(f"[METAGAME] Logged snapshot profile: {snapshot_path}")

    def sample_matchmaking_target(self) -> dict:
        """
        Returns a matchmaking target descriptor.

        Return format:
          {"type": "self_play_clone",      "source": "<path>.pt"}
          {"type": "native_engine_profile", "source": "<bot_name>"}
        """
        historical = glob.glob(os.path.join(self.weights_directory, "*.pt"))

        if historical and random.random() < 0.30:
            return {"type": "self_play_clone", "source": random.choice(historical)}

        return {"type": "native_engine_profile", "source": random.choice(self.active_profiles)}
