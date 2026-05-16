# unified_pipeline.py
"""
UnifiedAcceleratedPipeline — end-to-end training orchestrator.

Stage 1 (execute_behavioral_cloning_bootstrap):
  Pre-trains the policy on a compiled human demonstration corpus using
  supervised cross-entropy loss across all factorised action heads.
  Saves a frozen copy as the reference anchor for Stage 2.

Stage 2 (run_online_reinforcement_step):
  Runs a standard PPO policy gradient step on live rollout batches.
  A KL penalty against the frozen reference policy prevents the network
  from unlearning core human combat mechanics during online exploration.
"""

import copy
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from weaponized_ai.hardware_driver import MutexHardwareDriver


class UnifiedAcceleratedPipeline:
    def __init__(
        self,
        policy_network: nn.Module,
        ram_shm_bridge: Any,
        lr: float = 3e-4,
    ):
        self.policy    = policy_network.cuda()
        self.shm       = ram_shm_bridge
        self.driver    = MutexHardwareDriver()
        self.optimizer = optim.AdamW(
            self.policy.parameters(), lr=lr, weight_decay=1e-4
        )
        # Frozen reference policy (set after Stage 1 pretraining)
        self.reference_policy: nn.Module | None = None

    # ── Stage 1: Behavioural Cloning ──────────────────────────────────────────

    def execute_behavioral_cloning_bootstrap(
        self,
        corpus_path: str,
        epochs: int = 10,
        batch_size: int = 128,
    ):
        """
        Pre-trains the policy on human demonstrations.

        Corpus npz format expected:
            states:  float32 (N, state_dim)
            actions: int64   (N, 3)   — [move_x, move_y, action]
        """
        print(f"[STAGE 1] Loading demonstration corpus: {corpus_path}")
        if not os.path.exists(corpus_path):
            print("[WARN] Corpus not found — skipping to online RL.")
            return

        data    = np.load(corpus_path)
        states  = torch.from_numpy(data["states"]).float().cuda()
        actions = torch.from_numpy(data["actions"]).long().cuda()

        loader    = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(states, actions),
            batch_size=batch_size, shuffle=True,
        )
        criterion = nn.CrossEntropyLoss()
        self.policy.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_s, batch_a in loader:
                self.optimizer.zero_grad()
                dists, _ = self.policy(batch_s)
                loss = (
                    criterion(dists[0].logits, batch_a[:, 0])
                    + criterion(dists[1].logits, batch_a[:, 1])
                    + criterion(dists[2].logits, batch_a[:, 2])
                )
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(
                f"  Epoch {epoch + 1}/{epochs} — "
                f"mean loss: {total_loss / len(loader):.4f}"
            )

        # Deep-copy and freeze as the human-prior reference
        self.reference_policy = copy.deepcopy(self.policy)
        self.reference_policy.eval()
        for p in self.reference_policy.parameters():
            p.requires_grad_(False)
        print("[STAGE 1] Pretraining complete. Human policy priors frozen.")

    # ── Stage 2: Online PPO with KL tether ───────────────────────────────────

    def run_online_reinforcement_step(self, rollouts_batch: dict) -> float:
        """
        Updates the live policy on a rollout batch.

        rollouts_batch keys:
            states        float32 (B, state_dim)
            actions       int64   (B, 3)
            old_log_probs float32 (B,)
            advantages    float32 (B,)
        """
        self.policy.train()

        states        = torch.from_numpy(rollouts_batch["states"]).float().cuda()
        actions       = torch.from_numpy(rollouts_batch["actions"]).long().cuda()
        old_log_probs = torch.from_numpy(rollouts_batch["old_log_probs"]).float().cuda()
        advantages    = torch.from_numpy(rollouts_batch["advantages"]).float().cuda()

        self.optimizer.zero_grad()

        dists, _values = self.policy(states)
        new_log_probs = (
            dists[0].log_prob(actions[:, 0])
            + dists[1].log_prob(actions[:, 1])
            + dists[2].log_prob(actions[:, 2])
        )

        # PPO clipped objective
        ratios = torch.exp(new_log_probs - old_log_probs)
        ppo_loss = -torch.min(
            ratios * advantages,
            torch.clamp(ratios, 0.8, 1.2) * advantages,
        ).mean()

        # KL penalty against frozen reference policy
        kl_penalty = torch.tensor(0.0, device="cuda")
        if self.reference_policy is not None:
            with torch.no_grad():
                ref_dists, _ = self.reference_policy(states)
            for i in range(3):
                kl_penalty += torch.distributions.kl_divergence(
                    dists[i], ref_dists[i]
                ).mean()

        total_loss = ppo_loss + 0.02 * kl_penalty
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        return total_loss.item()
