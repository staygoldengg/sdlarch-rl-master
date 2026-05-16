# rl_agent.py
"""
Reinforcement learning agent — PPO with:
  - Generalized Advantage Estimation (GAE, λ=0.95)
  - Entropy bonus for exploration
  - Gradient clipping
  - Multiple PPO epochs per batch
  - Live training statistics
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional

OBS_DIM = 18   # pos(2), vel(2), airborne, damage, stocks, opp_pos(2), opp_vel(2), opp_damage, opp_stocks, dist, rel_x, rel_y, armed
ACT_DIM = 16   # NLight,SLight,DLight,NHeavy,SHeavy,DHeavy,Nair,Sair,Dair,Jump,DJ,Dodge,Dash,WT,Pickup,NSig


def _mlp(in_dim: int, out_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM):
        super().__init__()
        self.net = _mlp(obs_dim, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def act(self, obs: torch.Tensor) -> Tuple[int, torch.Tensor]:
        dist = torch.distributions.Categorical(logits=self.forward(obs))
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, obs: torch.Tensor, acts: torch.Tensor):
        dist = torch.distributions.Categorical(logits=self.forward(obs))
        return dist.log_prob(acts), dist.entropy()


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.net = _mlp(obs_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RLAgent:
    """
    PPO agent with GAE, entropy bonus, gradient clipping, multi-epoch updates.
    """
    MODEL_PATH = "weaponized_ai/model.pt"

    # Hyperparameters
    GAMMA       = 0.99
    GAE_LAMBDA  = 0.95
    CLIP_EPS    = 0.2
    ENTROPY_C   = 0.01    # entropy bonus coefficient
    VALUE_C     = 0.5     # value loss coefficient
    MAX_GRAD    = 0.5     # gradient clipping
    PPO_EPOCHS  = 4       # update passes per batch
    LR          = 3e-4

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.policy = PolicyNet(obs_dim, act_dim)
        self.value  = ValueNet(obs_dim)
        # Shared optimizer over both networks
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=self.LR, eps=1e-5
        )
        self.buffer: List[dict] = []
        # Cumulative training stats
        self.total_steps   = 0
        self.total_updates = 0
        self.episode_count = 0
        self.episode_rewards: List[float] = []   # reward per finished episode
        self._ep_reward = 0.0

    def select_action(self, obs: List[float]) -> Tuple[int, float]:
        t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action, log_prob = self.policy.act(t)
            value = self.value(t).item()
        return action, log_prob.item()

    def store(self, obs: List[float], action: int, reward: float,
              log_prob: float, done: bool):
        self._ep_reward += reward
        if done:
            self.episode_rewards.append(self._ep_reward)
            self.episode_count += 1
            self._ep_reward = 0.0
        self.buffer.append(dict(obs=obs, action=action, reward=reward,
                                log_prob=log_prob, done=done))
        self.total_steps += 1

    # ── GAE computation ───────────────────────────────────────────────────────
    def _compute_gae(self, obs_t: torch.Tensor, rewards, dones) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            values = self.value(obs_t).numpy()
        T = len(rewards)
        advantages = [0.0] * T
        last_adv = 0.0
        for t in reversed(range(T)):
            next_val  = values[t + 1] if t + 1 < T else 0.0
            delta     = rewards[t] + self.GAMMA * next_val * (1 - dones[t]) - values[t]
            last_adv  = delta + self.GAMMA * self.GAE_LAMBDA * (1 - dones[t]) * last_adv
            advantages[t] = last_adv
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        ret_t = adv_t + torch.tensor(values[:T], dtype=torch.float32)
        return adv_t, ret_t

    def train_step(self) -> dict:
        if len(self.buffer) < 8:
            return {"loss_policy": 0.0, "loss_value": 0.0, "entropy": 0.0}

        obs_t    = torch.tensor([s['obs']      for s in self.buffer], dtype=torch.float32)
        acts_t   = torch.tensor([s['action']   for s in self.buffer], dtype=torch.long)
        old_lp_t = torch.tensor([s['log_prob'] for s in self.buffer], dtype=torch.float32)
        rewards  = [s['reward'] for s in self.buffer]
        dones    = [float(s['done']) for s in self.buffer]

        adv_t, ret_t = self._compute_gae(obs_t, rewards, dones)
        # Normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_lp = total_lv = total_ent = 0.0
        for _ in range(self.PPO_EPOCHS):
            new_lp, entropy = self.policy.evaluate(obs_t, acts_t)
            vals = self.value(obs_t)

            ratio  = (new_lp - old_lp_t.detach()).exp()
            surr1  = ratio * adv_t
            surr2  = torch.clamp(ratio, 1 - self.CLIP_EPS, 1 + self.CLIP_EPS) * adv_t
            loss_p = -torch.min(surr1, surr2).mean()
            loss_v = self.VALUE_C * nn.functional.mse_loss(vals, ret_t)
            loss_e = -self.ENTROPY_C * entropy.mean()
            loss   = loss_p + loss_v + loss_e

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value.parameters()),
                self.MAX_GRAD
            )
            self.optimizer.step()

            total_lp  += loss_p.item()
            total_lv  += loss_v.item()
            total_ent += entropy.mean().item()

        self.buffer.clear()
        self.total_updates += 1
        n = self.PPO_EPOCHS
        return {
            "loss_policy": total_lp / n,
            "loss_value":  total_lv / n,
            "entropy":     total_ent / n,
        }

    def stats(self) -> dict:
        recent = self.episode_rewards[-20:] if self.episode_rewards else []
        return {
            "total_steps":   self.total_steps,
            "total_updates": self.total_updates,
            "episode_count": self.episode_count,
            "mean_ep_reward": round(sum(recent) / len(recent), 3) if recent else 0.0,
            "best_ep_reward": round(max(self.episode_rewards), 3) if self.episode_rewards else 0.0,
        }

    def save(self, path: Optional[str] = None):
        path = path or self.MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy':          self.policy.state_dict(),
            'value':           self.value.state_dict(),
            'optimizer':       self.optimizer.state_dict(),
            'total_steps':     self.total_steps,
            'total_updates':   self.total_updates,
            'episode_count':   self.episode_count,
            # Persist rolling reward history so stats survive restarts
            'episode_rewards': self.episode_rewards[-500:],
        }, path)

    def load(self, path: Optional[str] = None) -> bool:
        path = path or self.MODEL_PATH
        if not os.path.exists(path):
            return False
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            self.policy.load_state_dict(ckpt['policy'])
            self.value.load_state_dict(ckpt['value'])
            if 'optimizer' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            self.total_steps     = ckpt.get('total_steps',     0)
            self.total_updates   = ckpt.get('total_updates',   0)
            self.episode_count   = ckpt.get('episode_count',   0)
            self.episode_rewards = ckpt.get('episode_rewards', [])
            return True
        except (RuntimeError, KeyError):
            # Architecture mismatch (e.g. old checkpoint) — start fresh
            return False


# Module-level singleton
_agent: Optional[RLAgent] = None

def get_agent() -> RLAgent:
    global _agent
    if _agent is None:
        _agent = RLAgent()
        _agent.load()
    return _agent


# ══════════════════════════════════════════════════════════════════════════════
#  BTR — Beyond The Rainbow (adapted from VIPTankz/Wii-RL, ICML 2025)
#  ─────────────────────────────────────────────────────────────────────────────
#  Key ideas ported from BTR.py:
#    1. FactorizedNoisyLinear  — structured exploration without ε-greedy
#    2. Dueling architecture   — separate value + advantage streams
#    3. N-step returns         — credit assignment across multiple frames
#    4. Prioritized Replay     — SumTree replay buffer, focus on hard samples
#    5. Double-DQN target      — decouple selection from evaluation
#
#  This is an independent DQN-family agent that can be used alongside or
#  instead of the PPO RLAgent above.
# ══════════════════════════════════════════════════════════════════════════════

import math
import random
from collections import deque

# ── Factorized Noisy Linear ────────────────────────────────────────────────

class FactorizedNoisyLinear(nn.Module):
    """Factorized Gaussian noise layer (NoisyNets, Fortunato et al. 2017)."""

    def __init__(self, in_features: int, out_features: int, sigma_0: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sigma_0      = sigma_0

        scale = 1.0 / math.sqrt(in_features)
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features).uniform_(-scale, scale))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), sigma_0 * scale))
        self.bias_mu      = nn.Parameter(torch.empty(out_features).uniform_(-scale, scale))
        self.bias_sigma   = nn.Parameter(torch.full((out_features,), sigma_0 * scale))

        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon",   torch.zeros(out_features))
        self._noise_on = True
        self.reset_noise()

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    @torch.no_grad()
    def reset_noise(self):
        eps_in  = self._f(torch.randn(self.in_features))
        eps_out = self._f(torch.randn(self.out_features))
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def disable_noise(self):
        self._noise_on = False
        self.weight_epsilon.zero_()
        self.bias_epsilon.zero_()

    def enable_noise(self):
        self._noise_on = True
        self.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_mu + self.weight_sigma * self.weight_epsilon
        b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        return torch.nn.functional.linear(x, w, b)


# ── Noisy Dueling DQN Network ─────────────────────────────────────────────

def _noisy_mlp(in_dim: int, out_dim: int, hidden: int,
               noisy: bool = True) -> nn.Sequential:
    Lin = FactorizedNoisyLinear if noisy else nn.Linear
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        Lin(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        Lin(hidden, out_dim),
    )


class BTRNet(nn.Module):
    """
    Dueling Noisy DQN network.
      trunk → shared representation
      value_head → V(s)
      adv_head   → A(s,a)
      Q(s,a) = V(s) + A(s,a) - mean(A)
    """

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.value_head = _noisy_mlp(hidden, 1,       hidden // 2, noisy=True)
        self.adv_head   = _noisy_mlp(hidden, act_dim, hidden // 2, noisy=True)

    def forward(self, x: torch.Tensor, advantages_only: bool = False) -> torch.Tensor:
        h   = self.trunk(x)
        adv = self.adv_head(h)
        if advantages_only:
            return adv
        val = self.value_head(h)
        return val + (adv - adv.mean(dim=-1, keepdim=True))

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.reset_noise()


# ── Prioritized Replay Buffer (SumTree) ───────────────────────────────────

class _SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = torch.zeros(2 * capacity)
        self.data: list = [None] * capacity
        self.ptr      = 0
        self.size     = 0

    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def update(self, idx: int, priority: float):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def add(self, priority: float, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _retrieve(self, idx: int, s: float) -> int:
        left, right = 2 * idx + 1, 2 * idx + 2
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def sample(self, s: float) -> Tuple[int, float, object]:
        idx      = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]

    @property
    def total(self) -> float:
        return float(self.tree[0])


class PrioritizedReplayBuffer:
    """SumTree-backed PER with importance-sampling weights."""

    def __init__(self, capacity: int = 100_000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 100_000):
        self.tree        = _SumTree(capacity)
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_frames = beta_frames
        self.frame       = 0
        self.eps         = 1e-5
        self.max_prio    = 1.0

    @property
    def beta(self) -> float:
        return min(1.0, self.beta_start + self.frame / self.beta_frames * (1.0 - self.beta_start))

    def push(self, obs, action, reward, next_obs, done):
        self.tree.add(self.max_prio ** self.alpha,
                      (obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        self.frame += 1
        indices, priorities, samples = [], [], []
        seg = self.tree.total / batch_size
        for i in range(batch_size):
            s   = random.uniform(seg * i, seg * (i + 1))
            idx, prio, data = self.tree.sample(s)
            indices.append(idx)
            priorities.append(prio)
            samples.append(data)

        probs   = torch.tensor(priorities) / self.tree.total
        weights = (self.tree.size * probs).pow(-self.beta)
        weights = (weights / weights.max()).float()

        obs_t    = torch.tensor([s[0] for s in samples], dtype=torch.float32)
        acts_t   = torch.tensor([s[1] for s in samples], dtype=torch.long)
        rews_t   = torch.tensor([s[2] for s in samples], dtype=torch.float32)
        nobs_t   = torch.tensor([s[3] for s in samples], dtype=torch.float32)
        dones_t  = torch.tensor([s[4] for s in samples], dtype=torch.float32)
        return obs_t, acts_t, rews_t, nobs_t, dones_t, weights, indices

    def update_priorities(self, indices: list, td_errors: torch.Tensor):
        for idx, err in zip(indices, td_errors.abs().detach().cpu()):
            prio = (float(err) + self.eps) ** self.alpha
            self.tree.update(idx, prio)
            self.max_prio = max(self.max_prio, prio)

    def __len__(self) -> int:
        return self.tree.size


# ── N-step Return Buffer ──────────────────────────────────────────────────

class NStepBuffer:
    def __init__(self, n: int, gamma: float):
        self.n     = n
        self.gamma = gamma
        self._buf: deque = deque()

    def push(self, obs, action, reward, next_obs, done) -> Optional[tuple]:
        self._buf.append((obs, action, reward, next_obs, done))
        if len(self._buf) < self.n:
            return None
        R = sum(self.gamma ** i * self._buf[i][2] for i in range(self.n))
        first = self._buf[0]
        last  = self._buf[-1]
        self._buf.popleft()
        return first[0], first[1], R, last[3], last[4]

    def flush(self) -> list[tuple]:
        results = []
        while self._buf:
            n   = len(self._buf)
            R   = sum(self.gamma ** i * self._buf[i][2] for i in range(n))
            first, last = self._buf[0], self._buf[-1]
            results.append((first[0], first[1], R, last[3], last[4]))
            self._buf.popleft()
        return results


# ── BTR Agent ─────────────────────────────────────────────────────────────

class BTRAgent:
    """
    Beyond The Rainbow agent — Noisy Dueling Double-DQN with PER + n-step.

    Hyper-parameters match the original BTR paper's MLP configuration
    (adapted for Brawlhalla's 18-dim state space rather than pixels).
    """

    MODEL_PATH    = "weaponized_ai/btr_model.pt"
    LR            = 6.25e-5
    GAMMA         = 0.99
    N_STEP        = 3
    BATCH_SIZE    = 256
    REPLAY_CAP    = 100_000
    MIN_REPLAY    = 1_000
    UPDATE_FREQ   = 4       # train every N steps
    TARGET_SYNC   = 1_000   # hard target sync every N steps
    GRAD_CLIP     = 10.0

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM):
        self.obs_dim  = obs_dim
        self.act_dim  = act_dim

        self.online = BTRNet(obs_dim, act_dim)
        self.target = BTRNet(obs_dim, act_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(self.online.parameters(), lr=self.LR, eps=1.5e-4)
        self.replay    = PrioritizedReplayBuffer(self.REPLAY_CAP)
        self.nstep     = NStepBuffer(self.N_STEP, self.GAMMA)

        self.total_steps   = 0
        self.total_updates = 0
        self.episode_count = 0
        self.episode_rewards: List[float] = []
        self._ep_reward    = 0.0

    # ── Action selection ─────────────────────────────────────────────────────

    def select_action(self, obs: List[float]) -> int:
        """Noisy nets handle exploration — no ε-greedy needed."""
        t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        self.online.reset_noise()
        with torch.no_grad():
            q = self.online(t, advantages_only=True)
        return int(q.argmax(dim=-1).item())

    # ── Experience storage ────────────────────────────────────────────────────

    def store(self, obs: List[float], action: int, reward: float,
              next_obs: List[float], done: bool):
        self._ep_reward += reward
        if done:
            self.episode_rewards.append(self._ep_reward)
            self.episode_count += 1
            self._ep_reward = 0.0
            for t in self.nstep.flush():
                self.replay.push(*t)

        transition = self.nstep.push(obs, action, reward, next_obs, done)
        if transition is not None:
            self.replay.push(*transition)

        self.total_steps += 1
        if (self.total_steps % self.UPDATE_FREQ == 0
                and len(self.replay) >= self.MIN_REPLAY):
            self.train_step()

    # ── Training ─────────────────────────────────────────────────────────────

    def train_step(self) -> dict:
        if len(self.replay) < self.MIN_REPLAY:
            return {}

        obs, acts, rews, nobs, dones, is_weights, tree_idx = \
            self.replay.sample(self.BATCH_SIZE)

        self.online.reset_noise()
        self.target.reset_noise()

        # Double-DQN: online selects, target evaluates
        with torch.no_grad():
            next_actions = self.online(nobs).argmax(dim=-1)
            next_q       = self.target(nobs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            gamma_n      = self.GAMMA ** self.N_STEP
            targets      = rews + gamma_n * next_q * (1 - dones)

        current_q = self.online(obs).gather(1, acts.unsqueeze(1)).squeeze(1)
        td_errors  = targets - current_q
        loss       = (is_weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.GRAD_CLIP)
        self.optimizer.step()

        self.replay.update_priorities(tree_idx, td_errors)

        # Periodic hard target sync
        if self.total_updates % self.TARGET_SYNC == 0:
            self.target.load_state_dict(self.online.state_dict())

        self.total_updates += 1
        return {"loss": loss.item(), "mean_td": float(td_errors.abs().mean())}

    # ── BC pre-training (from corpus) ────────────────────────────────────────

    def pretrain_bc(self, obs_arr, acts_arr, weights_arr, epochs: int = 3) -> float:
        """
        Supervised behavioural cloning from replay corpus.
        Treat expert actions as supervised labels on the advantage head.
        """
        obs_t  = torch.tensor(obs_arr,     dtype=torch.float32)
        acts_t = torch.tensor(acts_arr,    dtype=torch.long)
        wgts_t = torch.tensor(weights_arr, dtype=torch.float32)
        N      = len(obs_t)
        total_loss = 0.0
        for _ in range(epochs):
            idx = torch.randperm(N)
            for start in range(0, N, self.BATCH_SIZE):
                b   = idx[start: start + self.BATCH_SIZE]
                logits = self.online(obs_t[b], advantages_only=True)
                loss   = (wgts_t[b] * nn.functional.cross_entropy(
                    logits, acts_t[b], reduction="none")).mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.online.parameters(), self.GRAD_CLIP)
                self.optimizer.step()
                total_loss += loss.item()
        return total_loss

    # ── Stats / persistence ───────────────────────────────────────────────────

    def stats(self) -> dict:
        recent = self.episode_rewards[-20:]
        return {
            "agent":         "BTR",
            "total_steps":   self.total_steps,
            "total_updates": self.total_updates,
            "episode_count": self.episode_count,
            "replay_size":   len(self.replay),
            "mean_ep_reward": round(sum(recent) / len(recent), 3) if recent else 0.0,
            "best_ep_reward": round(max(self.episode_rewards), 3) if self.episode_rewards else 0.0,
        }

    def save(self, path: Optional[str] = None):
        path = path or self.MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "online":          self.online.state_dict(),
            "target":          self.target.state_dict(),
            "optimizer":       self.optimizer.state_dict(),
            "total_steps":     self.total_steps,
            "total_updates":   self.total_updates,
            "episode_count":   self.episode_count,
            "episode_rewards": self.episode_rewards[-500:],
        }, path)

    def load(self, path: Optional[str] = None) -> bool:
        path = path or self.MODEL_PATH
        if not os.path.exists(path):
            return False
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.online.load_state_dict(ckpt["online"])
            self.target.load_state_dict(ckpt["target"])
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            self.total_steps     = ckpt.get("total_steps",     0)
            self.total_updates   = ckpt.get("total_updates",   0)
            self.episode_count   = ckpt.get("episode_count",   0)
            self.episode_rewards = ckpt.get("episode_rewards", [])
            return True
        except (RuntimeError, KeyError):
            return False


# Module-level BTR singleton
_btr_agent: Optional[BTRAgent] = None

def get_btr_agent() -> BTRAgent:
    global _btr_agent
    if _btr_agent is None:
        _btr_agent = BTRAgent()
        _btr_agent.load()
    return _btr_agent

