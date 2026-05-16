# Striker — The Enlightened

A fully autonomous AI combat training suite for Brawlhalla. The system reads raw game state directly from the process's memory, injects hardware-level keystrokes through the Win32 API, learns from its own mistakes via Proximal Policy Optimization, absorbs human expert knowledge from YouTube videos and real match replays, and packages everything into a native Windows desktop application built on React and Tauri.

The game is never modified. No DLL injection, no memory patches, no screenshot OCR. All sensing is done through `ReadProcessMemory` and a C++ shared-memory bridge, and all actions are hardware-indistinguishable from a real keyboard.

---

## How It Works — End to End

Training happens in three sequenced stages that feed into each other:

### Stage 1 — Behavioural Cloning Bootstrap

Before the AI plays a single game, it is pre-trained on human demonstrations to avoid starting from random noise:

1. **Replay digestion** — `replay_engine.py` decompresses 500+ real Brawlhalla `.replay` files, simulates the physics frame-by-frame against known stage bounds, and converts each frame into `(obs, action, reward)` transitions stored in `brain/corpus.npz`.
2. **Video learning** — `video_learner.py` downloads Brawlhalla YouTube videos via `yt-dlp`, extracts frames at ~4 fps, estimates player positions via color detection, transcribes commentary with Whisper to identify high-value moments (combos, punishes, edgeguards), and generates additional BC transitions weighted 3× for expert moments.
3. **Pretraining** — `UnifiedAcceleratedPipeline.execute_behavioral_cloning_bootstrap()` runs supervised cross-entropy over all three factorised action heads (move_x / move_y / attack) until the policy can roughly reproduce human action distributions.
4. A **frozen copy** of the BC-trained weights is saved as the reference anchor for Stage 2. This prevents the RL agent from catastrophically forgetting fighting-game fundamentals while it explores.

### Stage 2 — Online Reinforcement Learning

With a seeded policy, the system enters live self-play:

1. **State reading** — The C++ `SharedMemoryBridge` DLL (loaded by the game-side writer) publishes a `BrawlhallaStateBuffer` (236 bytes, `#pragma pack(1)`) into a Windows named shared-memory segment every game frame. `TelemetryWatchdogMemoryReader` reads and validates these bytes (sentinel checksum, coordinate bounds, velocity delta, stock overflow) before feeding them to the policy.
2. **60 Hz inference** — `HighFidelityTrainingLoop` pulls the latest state vector, converts it to a CUDA tensor, runs one forward pass through `EnlightenedPolicyNetwork`, samples from the three factorised `Categorical` distributions, and dispatches the chosen key-state map.
3. **Hardware injection** — `FrameDeterministicDispatcher` runs a dedicated nanosecond spin-lock thread. The inference cycle stages an action map; the dispatcher thread consumes the latest map at the exact next frame boundary using `timeBeginPeriod(1)` for 1 ms OS timer resolution. Inputs are diff-based: only keys that changed state are transmitted via `SendInput` with physical scan codes.
4. **Reward shaping** — `AlignedRewardShaper` computes a composite signal: damage dealt (×1.5), a PBRS spatial-safety potential (`γ·Φ(s') - Φ(s)`) that penalises edgeguarding yourself, a –25 stock-loss penalty to suppress suicidal trades, and a +2 combo-connection bonus.
5. **Advantage estimation** — `EligibilityTracedAdvantageEngine` computes GAE with per-step eligibility weights: attack frames keep full credit (weight=1.0), idle frames are attenuated (weight=0.85) so the policy does not learn to claim credit for being lucky.
6. **PPO update** — `CovariateShiftProtectedPPOLoss` computes the clipped surrogate objective plus a forward-KL penalty `KL(live‖ref)` against the frozen BC anchor. The KL beta coefficient self-adjusts every update to keep the live policy from drifting while still allowing improvement.
7. **Exploration** — `AdaptiveEntropyTuner` monitors the joint entropy across all three action heads and nudges the entropy coefficient to maintain a configurable target ratio (default 25% of max), preventing the policy from collapsing to a single deterministic action.
8. **Action masking** — `StateDependentActionMasker` reads air-jump count and weapon status from the state vector and sets logits for mechanically impossible actions to −1e9 before sampling, so the network never wastes gradient on illegal moves.
9. **Opponent curriculum** — `OpponentPoolReservoir` mixes 30% self-play clones (frozen historical weight snapshots) with 70% native-engine bot profiles to provide a structured difficulty curriculum.

### Stage 3 — Continuous Improvement

The system stays running between matches:

- `PersistentStorageEngine` writes user settings to `%APPDATA%\StrikerEnlightened\user_settings.json`.
- `BrainStore` accumulates the knowledge base, BC corpus, and video registry across sessions so nothing is re-learned from scratch after a restart.
- `ReplayDesyncValidator` verifies that the internal physics simulation stays aligned with ground-truth binary events (KO count and frame drift checks) and flags desync before it can corrupt the training signal.
- `AutonomousSessionManager` wraps every training context as a Python context manager: on clean exit **or** crash, `emergency_flush()` is called unconditionally so the keyboard always returns to normal.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Quick Start](#quick-start)
3. [Build & Distribution](#build--distribution)
4. [AI Engine Subsystems](#ai-engine-subsystems)
   - [Policy Networks](#policy-networks)
   - [PPO Agent](#ppo-agent)
   - [BTR Agent](#btr-agent)
   - [PPO Loss (Covariate-Shift Protected)](#ppo-loss-covariate-shift-protected)
   - [Adaptive Entropy Tuner](#adaptive-entropy-tuner)
   - [Eligibility-Traced Advantage Engine](#eligibility-traced-advantage-engine)
   - [Reward Shaper (PBRS)](#reward-shaper-pbrs)
   - [Value Heads & Normalization](#value-heads--normalization)
   - [Action Masker](#action-masker)
   - [Unified Accelerated Pipeline (BC → PPO)](#unified-accelerated-pipeline-bc--ppo)
5. [State Reading](#state-reading)
   - [Shared Memory Bridge (C++)](#shared-memory-bridge-c)
   - [Telemetry Watchdog Reader](#telemetry-watchdog-reader)
   - [Dynamic Memory Reader (pymem)](#dynamic-memory-reader-pymem)
   - [Brawlhalla Memory Reader (AoB scan)](#brawlhalla-memory-reader-aob-scan)
6. [Vision Pipeline](#vision-pipeline)
   - [High-Speed Visual Engine (120 Hz DXGI)](#high-speed-visual-engine-120-hz-dxgi)
   - [Zero-Copy Vision Engine](#zero-copy-vision-engine)
   - [Video Feature Anchor Extractor](#video-feature-anchor-extractor)
   - [Semantic Kinematic Extractor (YOLO)](#semantic-kinematic-extractor-yolo)
7. [Input Injection](#input-injection)
   - [Frame Deterministic Dispatcher](#frame-deterministic-dispatcher)
   - [Mutex Hardware Driver](#mutex-hardware-driver)
   - [Input Controller (VK-code layer)](#input-controller-vk-code-layer)
8. [Training Loops](#training-loops)
   - [HighFidelityTrainingLoop (60 Hz SHM)](#highfidelitytrainingloop-60-hz-shm)
   - [TrainingLoop (8 Hz Vision)](#trainingloop-8-hz-vision)
9. [Data Ingestion](#data-ingestion)
   - [Replay Engine](#replay-engine)
   - [Video Learner (YouTube + Whisper)](#video-learner-youtube--whisper)
10. [Session Infrastructure](#session-infrastructure)
    - [Opponent Pool Reservoir](#opponent-pool-reservoir)
    - [Replay Desync Validator](#replay-desync-validator)
    - [Autonomous Session Manager](#autonomous-session-manager)
    - [Persistent Storage Engine](#persistent-storage-engine)
    - [Brain Store](#brain-store)
    - [Process Utilities](#process-utilities)
11. [Strategy Engine](#strategy-engine)
12. [Frontend (React + Tauri)](#frontend-react--tauri)
13. [Runner Scripts](#runner-scripts)
14. [API Reference](#api-reference)
15. [What Was Improved](#what-was-improved)
16. [Configuration Reference](#configuration-reference)
17. [Project Structure](#project-structure)
18. [Requirements](#requirements)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Tauri Desktop App  (localhost:1420)                │
│         React 18 + TypeScript + Vite  ↔  WeaponizedAPI.ts          │
└────────────────────────────┬────────────────────────────────────────┘
                             │ REST/JSON  http://127.0.0.1:8000
┌────────────────────────────▼────────────────────────────────────────┐
│                  FastAPI Backend  (api_server.py)                   │
│  /policy  /rl  /btr  /replay  /memory  /brain  /video  /strategy   │
│  /input   /obs  /health                                             │
└──┬─────┬──────────┬────────────┬─────────────┬──────────┬──────────┘
   │     │          │            │             │          │
   ▼     ▼          ▼            ▼             ▼          ▼
 PPO   BTR       Replay       Memory        Brain      Video
Agent  Agent     Engine       Reader        Store      Learner
                 (binary      (AoB scan     (NPZ +     (yt-dlp +
                  parser)      or SHM)       JSON)      Whisper)
   │     │          │
   ▼     ▼          ▼
  EnlightenedPolicyNetwork  /  FactorizedMultiDiscreteActorHead
  (Mish activations, 256-unit shared encoder, actor + critic)
   │
   ▼
 HighFidelityTrainingLoop  (60 Hz SHM path)
 TrainingLoop              (8 Hz vision path)
   │
   ├── TelemetryWatchdogMemoryReader  (validates SHM bytes)
   ├── AlignedRewardShaper            (PBRS + anti-suicide)
   ├── EligibilityTracedAdvantageEngine (GAE + eligibility weights)
   ├── CovariateShiftProtectedPPOLoss  (PPO + KL(live‖ref))
   ├── AdaptiveEntropyTuner           (self-calibrating entropy β)
   ├── StateDependentActionMasker     (air-jump / weapon masking)
   └── FrameDeterministicDispatcher   (nanosecond spin-lock injection)

C++ Layer (shm_bridge.cpp / DLL):
  Brawlhalla.exe  ──►  BrawlhallaStateBuffer (236 bytes, pack(1))
                          Sentinel: 0xDEADC0DEFA57FEED
                          feature_vector[48]  ──►  Python SHM reader
```

---

## Quick Start

### Prerequisites

- Python 3.10+ with `.venv` at project root
- Node.js 18+ and Rust toolchain (for Tauri/React frontend)
- NVIDIA GPU with CUDA (required for 60 Hz inference path)
- Brawlhalla installed via Steam on Windows 10/11

### 1. Start the AI backend

```powershell
.\.venv\Scripts\activate
python server_entry.py
```

Or use the convenience launcher:

```powershell
.\start_server.ps1
```

The server starts at `http://127.0.0.1:8000`. The PowerShell script auto-detects `.venv\Scripts\python.exe` and routes uvicorn stderr to stdout so INFO logs are not misreported as errors.

### 2. Launch the desktop UI

```powershell
npm install
npm run tauri dev
```

### 3. Verify health

```
GET http://127.0.0.1:8000/health
-> {"status":"ok","obs_dim":18,"act_dim":16}
```

### 4. Run the background priority service (optional)

```powershell
python striker_service.py
```

Elevates the process to `ABOVE_NORMAL` scheduling priority and holds the service alive. Uncomment the `HighFidelityTrainingLoop` block inside to attach a live training loop.

### 5. Run the live agent directly (optional — requires YOLO model)

```powershell
python live_agent_node.py
```

Wires the full `ZeroCopyVisionEngine → SemanticKinematicExtractor → policy → FrameDeterministicDispatcher` pipeline. Requires `models/brawlhalla_yolo_nano.pt` and `pip install ultralytics dxcam`.

---

## Build & Distribution

The project ships as a single Windows installer built by `build_installer.ps1`:

```powershell
.\build_installer.ps1
```

**What it does:**

1. **PyInstaller** (`striker_server.spec`) bundles the entire Python backend — FastAPI, PyTorch, weaponized_ai — into `dist/striker-server/striker-server.exe` (onedir, not onefile, for faster cold start).
2. **`server_entry.py`** is the PyInstaller entry point. It bootstraps `sys._MEIPASS` path resolution, sets `STRIKER_DATA_DIR` in the environment so `BrainStore` writes to `%APPDATA%\StrikerEnlightened` instead of inside the frozen bundle, and starts uvicorn.
3. **`npm run tauri build`** compiles the React frontend and links it with the Rust Tauri shell. Tauri's `resources` config copies `dist/striker-server/**/*` into the installer.
4. The Tauri `lib.rs` calls `auto_start_server()` at startup: it locates `striker-server.exe` inside the resource directory (or falls back to `src-tauri/target/debug/` for dev builds), spawns it as a hidden child process, and registers a `stop_ai_server` command that terminates it on app exit.

Output: `src-tauri/target/release/bundle/nsis/Striker The Enlightened_x.y.z_x64-setup.exe`

**Skip flags:**

```powershell
.\build_installer.ps1 -SkipPyInstaller   # only rebuild Tauri
.\build_installer.ps1 -SkipTauri         # only rebuild Python bundle
```

---

## AI Engine Subsystems

### Policy Networks

**File:** `weaponized_ai/policy_network.py`

Three network classes, composable for different use cases:

**`EnlightenedPolicyNetwork`** — monolithic actor-critic for the SHM training path.

```
Input: (batch, 64)  ← raw RAM state vector from BrawlhallaStateBuffer

Shared encoder:
  Linear(64 → 256) → LayerNorm(256) → Mish
  Linear(256 → 256) → LayerNorm(256) → Mish

Actor:  Linear(256 → num_actions)  →  Categorical distribution
Critic: Linear(256 → 1)            →  scalar V(s)
```

Mish (`f(x) = x·tanh(softplus(x))`) is used instead of ReLU because it is smooth and non-zero for negative inputs, preventing dead neurons during rapid action-space exploration. LayerNorm (not BatchNorm) is used because the policy runs on single-observation rollouts where batch statistics are meaningless.

**`FactorizedMultiDiscreteActorHead`** — decomposes the action space into three independent heads with separated gradient flows:

| Head | Choices | Meaning |
|------|---------|---------|
| `move_x_head` | 0=Neutral, 1=Left, 2=Right | Horizontal direction |
| `move_y_head` | 0=Neutral, 1=Up, 2=Down | Vertical modifier |
| `action_head` | 0=Neutral, 1=Light, 2=Heavy, 3=Dodge, 4=Jump | Attack or movement action |

The joint log-probability is the **sum** of the three independent head log-probs. The joint entropy (returned alongside the distributions) drives `AdaptiveEntropyTuner`. `sample_to_macro()` converts samples to a dict compatible with `ActionTranslationEngine`.

**`TemporalConditioningEncoder`** — fuses raw RAM state with a real-world dt signal into a single latent vector, enabling the policy to reason about input lag and frame timing.

---

### PPO Agent

**File:** `weaponized_ai/rl_agent.py`

The primary online RL agent. Implements Proximal Policy Optimization with the following properties:

**Observation space (18 floats):**

| Index | Field |
|-------|-------|
| 0–1 | P1 XY position (normalised by stage half-width) |
| 2–3 | P1 XY velocity (estimated by frame-diff) |
| 4 | P1 airborne flag |
| 5 | P1 damage % |
| 6 | P1 stocks remaining |
| 7–8 | P2 XY position |
| 9–10 | P2 XY velocity |
| 11 | P2 damage % |
| 12 | P2 stocks remaining |
| 13 | Inter-player distance |
| 14–15 | Relative XY (P2 − P1) |
| 16 | P1 armed flag |
| 17 | Reserved |

**Action space (16 discrete):**

`NLight, SLight, DLight, NHeavy, SHeavy, DHeavy, Nair, Sair, Dair, Jump, DoubleJump, Dodge, DashRight, DashLeft, WeaponPickup, NSig`

**PPO training pass:**

1. Collect `N` steps into the experience buffer.
2. Compute GAE advantages (λ=0.95) with bootstrapped value baseline.
3. Normalise advantages to zero-mean unit-variance per mini-batch.
4. Run 4 gradient passes per collected batch with clipped surrogate objective (ε=0.2).
5. Entropy bonus (coefficient=0.01) prevents early collapse.
6. Shared AdamW optimizer (lr=3e-4, ε=1e-5), gradient norm clipped at 0.5.

**Hyperparameters:**

```python
GAMMA=0.99, GAE_LAMBDA=0.95, CLIP_EPS=0.2
ENTROPY_C=0.01, VALUE_C=0.5, MAX_GRAD=0.5
PPO_EPOCHS=4, LR=3e-4
```

---

### BTR Agent

**File:** `weaponized_ai/rl_agent.py`

Secondary offline/pretrain agent implementing **Beyond The Rainbow** (ICML 2025) — Noisy Dueling Double DQN with Prioritized Experience Replay and n-step returns.

**BTRNet architecture:**

```
Shared encoder:
  Linear(obs → 256) → LayerNorm → ReLU × 2

Dueling heads (NoisyLinear replaces ε-greedy):
  Value:     NoisyLinear(256→128) → ReLU → NoisyLinear(128→1)
  Advantage: NoisyLinear(256→128) → ReLU → NoisyLinear(128→act_dim)

Q(s,a) = V(s) + A(s,a) − mean(A(s,·))
```

`NoisyLinear` uses learned per-weight Gaussian noise — the network learns *when* to explore rather than following a fixed ε schedule.

**PrioritizedReplayBuffer:** Sampling probability ∝ `|δ|^0.6`. Importance-sampling weights (β annealed 0.4→1.0) correct for the introduced bias. Capacity 100,000.

**NStepBuffer:** Accumulates n=3 transitions computing `r + γr' + γ²r''` before writing to PER. Reduces variance versus 1-step TD.

**Double DQN target:**
```
y = r + γ · Q_target(s', argmax_a Q_online(s', a))
```
Target network soft-updated (τ=0.005) every step.

---

### PPO Loss (Covariate-Shift Protected)

**File:** `weaponized_ai/ppo_loss.py`

`CovariateShiftProtectedPPOLoss` adds a **forward-KL anchor penalty** to the standard PPO objective:

```
L = L_PPO_clip + β · KL(π_live ‖ π_ref)
```

`KL(live ‖ ref)` penalises the live policy for placing probability mass in regions the reference (BC-pretrained) policy considers impossible. This is the **correct anchoring direction** (equivalent to InstructGPT / DPO anchoring) — it prevents catastrophic forgetting of human fundamentals while allowing improvement.

> **Fix applied:** The original code computed `KL(ref ‖ live)` (reverse KL), which is mode-seeking and would allow the live policy to ignore entire regions of the human prior. Corrected to forward KL by swapping `F.kl_div` arguments.

The beta coefficient self-adjusts every update:
- KL > 1.5 × target → `β = min(5.0, β × 1.1)` (tighten the leash)
- KL < target / 1.5 → `β = max(0.05, β × 0.9)` (loosen the leash)

> **Fix applied:** Added `min/max` clamping to `[0.05, 5.0]` to prevent runaway beta growth that would freeze the live policy or a collapsed beta that would let the live policy drift without penalty.

---

### Adaptive Entropy Tuner

**File:** `weaponized_ai/entropy_tuner.py`

`AdaptiveEntropyTuner` monitors the joint entropy across all three factorised action heads and adjusts the entropy coefficient `β_entropy` to maintain a target exploration ratio (default 25% of the theoretical maximum entropy).

```
β_new = β_old + adaptation_speed × (target_ratio − observed_ratio)
β     ∈ [0.001, 0.1]
```

The entropy loss component returned is `−β · H_joint`, where the negative sign turns the entropy maximisation objective into a standard gradient descent term.

Without this, PPO with a fixed entropy coefficient either under-explores (coefficient too small) or wastes gradient budget on randomness (coefficient too large). The adaptive version self-calibrates across training phases where the optimal exploration level changes.

---

### Eligibility-Traced Advantage Engine

**File:** `weaponized_ai/advantage_engine.py`

`EligibilityTracedAdvantageEngine` extends standard GAE with a per-step eligibility weight that prevents idle frames from claiming credit for attacks that arrived several steps later:

```
gae_t = δ_t + γ · λ · (1 − done_t) · gae_{t+1} · w_t

where w_t = 1.0  if action was an attack/dodge (flag=1)
      w_t = 0.85  if action was idle movement (flag=0)
```

The output is normalised to zero-mean unit-variance over the batch and pushed to CUDA before returning.

---

### Reward Shaper (PBRS)

**File:** `weaponized_ai/reward_shaper.py`

`AlignedRewardShaper` assembles a composite step reward from four components:

| Component | Formula | Purpose |
|-----------|---------|---------|
| Damage dealt | `max(0, raw) × 1.5` | Reinforce offensive success |
| PBRS spatial potential | `γ·Φ(s') − Φ(s)` | Penalise edgeguarding yourself |
| Stock loss penalty | `−25.0` on death event | Suppress suicidal trades |
| Combo bonus | `+2.0` on combo event | Match Whisper BC weight signal |

The potential function `Φ(s)` measures quadratic distance from stage centre, with an additional `1000/(danger_delta + 1)` term that spikes when the player drops below y = −500 toward the hard blastzone.

> **Fix applied:** The PBRS formula was missing the γ discount factor — it computed `Φ(s') − Φ(s)` instead of `γ·Φ(s') − Φ(s)`. Without gamma, the potential delta is non-zero even at terminal states where s' is the reset initial state, introducing a small systematic bias that inflated reward for returning to starting positions. Fixed.

---

### Value Heads & Normalization

**File:** `weaponized_ai/value_heads.py`

**`MacroStateValueBootstrapper`** — dual critic that separates frame-level value from match-momentum:

```
micro_value_head: Linear(256 → 1)  ← frame-level combat interactions
macro_value_head: Linear(256 → 1)  ← stock differential trend

blended_target = α·V_micro + (1−α)·V_macro   (α=0.7 by default)
macro_target = stock_differential × 10.0
```

The dual heads allow the network to learn fast within-stock tactics (micro) separately from slow match-win strategy (macro), with the blending ratio tunable at call time.

**`RunningStateNormalizer`** — online Welford normalisation as an `nn.Module`. Running mean and variance are `register_buffer` entries so they survive `torch.save()` / `torch.load()` cycles. Clips to `±10σ` to suppress outlier observations from corrupted SHM reads.

**`TrajectoryRewardFilter`** — normalises raw step rewards against a rolling 1000-step return standard deviation, clipping to `[−5.0, 5.0]`. This prevents match-winning KO bonuses from producing catastrophically large gradient updates that destabilise the current policy.

**`OrthogonalNetworkInitializer`** — applies `nn.init.orthogonal_(weight, gain=√2)` to every `Linear` layer in any model, with biases zeroed. Orthogonal initialisation preserves gradient norms through deep networks and is especially effective with Mish activations.

---

### Action Masker

**File:** `weaponized_ai/action_masker.py`

`StateDependentActionMasker` prevents the policy from sampling mechanically impossible actions by setting their logits to −1e9 before the softmax layer.

Rules:
- `state_vector[12]` (air jumps remaining) ≤ 0 → mask Jump head index 4
- `state_vector[15]` (is_unarmed) → extensible placeholder for weapon-specific signature masking

This means the network never receives gradient signal for "jump when out of jumps" — a common early training failure mode that wastes capacity learning an impossible correction.

---

### Unified Accelerated Pipeline (BC → PPO)

**File:** `weaponized_ai/unified_pipeline.py`

`UnifiedAcceleratedPipeline` manages the full two-stage training lifecycle in a single object:

**Stage 1 — `execute_behavioral_cloning_bootstrap(corpus_path, epochs, batch_size)`:**
- Loads `(states, actions)` from a `.npz` corpus
- Runs supervised cross-entropy over all three factorised heads
- Calls `copy.deepcopy(self.policy)` on the trained weights and freezes the copy as `self.reference_policy`

> **Note:** Uses `copy.deepcopy` — not `torch.deepcopy` (which does not exist). This was a known issue and is correctly implemented here.

**Stage 2 — `run_online_reinforcement_step(rollouts_batch)`:**
- Standard PPO clipped objective (ε=0.8/1.2)
- Forward KL penalty against the frozen reference: `KL(live‖ref)` for each of the three heads
- `clip_grad_norm_(0.5)` before each AdamW step
- Returns the scalar total loss for logging

---

## State Reading

### Shared Memory Bridge (C++)

**File:** `tools/shared_mem/shm_bridge.cpp`

The C++ side of the zero-copy state pipeline. Compiled as a DLL and loaded by a game-side writer process. Key details:

**Buffer layout (`#pragma pack(1)` — no padding, 236 bytes total):**

| Offset | Type | Field |
|--------|------|-------|
| 0 | `uint64` | `alignment_checksum` — sentinel value `0xDEADC0DEFA57FEED` |
| 8 | `uint32` | `current_frame` |
| 12 | `float` | `player_x` |
| 16 | `float` | `player_y` |
| 20 | `float` | `opponent_x` |
| 24 | `float` | `opponent_y` |
| 28 | `float` | `player_damage` |
| 32 | `float` | `opponent_damage` |
| 36 | `uint32` | `player_stocks` |
| 40 | `uint32` | `opponent_stocks` |
| 44 | `float[48]` | `feature_vector` — 192 bytes of extended state |

**C exports:** `CreateBridge()`, `WriteBridgeState(buffer)`, `DestroyBridge()`

The Python `TelemetryWatchdogMemoryReader` reads the same named segment and validates the sentinel checksum before consuming any fields.

---

### Telemetry Watchdog Reader

**File:** `weaponized_ai/watchdog_reader.py`

`TelemetryWatchdogMemoryReader` validates every SHM frame before it reaches the policy network. Validation pipeline:

1. **Minimum buffer size** — rejects buffers shorter than 236 bytes
2. **Sentinel checksum** — first 8 bytes must equal `0xDEADC0DEFA57FEED`; rejects pre-bridge-init reads
3. **Coordinate bounds** — `|x| > 5000` or `|y| > 5000` triggers fallback (blastzone teleportation / corrupt pointer)
4. **Velocity delta** — frame-to-frame Euclidean distance > `max_expected_velocity` (default 150 px/frame) triggers fallback (map reload / pointer drift)
5. **Stock overflow** — player stocks > 3 triggers fallback (pointer jumped to garbage)

On any failure the **last valid state is returned** so the loop continues without a NaN.

> **Fix applied:** The sentinel checksum was present in `shm_bridge.cpp` but never verified on the Python side. Any garbage bytes or a pre-init read would pass straight to the policy network. The checksum validation was added as the first gate in the validation chain.

---

### Dynamic Memory Reader (pymem)

**File:** `weaponized_ai/dynamic_memory_reader.py`

`DynamicBrawlhallaReader` is an alternative state source that reads directly from Brawlhalla's heap via `pymem` pointer chains. Useful when the SHM bridge DLL is not loaded.

`follow_pointer_chain(base_address, offsets)` walks multi-level pointer chains (handles the common pattern where game objects are heap-allocated at runtime). `read_live_state(p1_base_offset, p1_chain)` reads P1 x/y at +0x10/+0x14 and damage at +0x20 into a 64-float state vector matching the SHM bridge layout.

Requires: `pip install pymem` — raises `ImportError` with install instruction if missing.

---

### Brawlhalla Memory Reader (AoB scan)

**File:** `weaponized_ai/brawlhalla_memory.py`

The original state reader. Uses `ReadProcessMemory` with Array-of-Bytes signature scanning for cases where pointer offsets are not known:

1. `OpenToolhelp32Snapshot` → find `Brawlhalla.exe` PID
2. `OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION)`
3. `VirtualQueryEx` enumerates committed/readable regions
4. Sliding-window AoB scan with `None` wildcards
5. Matched addresses cached to `_addr_cache.json` so scan only runs once per session

Reads: `p1_damage`, `p2_damage`, `p1_stocks`, `p2_stocks`, `p1_x`, `p1_y`, `p2_x`, `p2_y`. Degrades gracefully per field rather than crashing on any partial failure.

---

## Vision Pipeline

### High-Speed Visual Engine (120 Hz DXGI)

**File:** `weaponized_ai/visual_engine.py`

`HighSpeedVisualEngine` captures the game window at 120 Hz using DXCAM's Desktop Duplication API (GPU → GPU, no CPU round-trip):

- Runs a dedicated background thread that polls `camera.get_latest_frame()` and downsizes to the policy input dimensions (default 256×256) via bilinear interpolation
- `get_gpu_tensor()` returns the latest frame as a `(1, 3, H, W)` float32 CUDA tensor normalised to `[0, 1]`, or a zero tensor if no frame has arrived yet
- Thread-safe via a `threading.Lock`

---

### Zero-Copy Vision Engine

**File:** `weaponized_ai/cuda_vision.py`

`ZeroCopyVisionEngine` is the higher-level capture interface used by `live_agent_node.py`. Key differences from `HighSpeedVisualEngine`:

- Configurable `region_box` (default full 1920×1080) for windowed capture
- Runs at 60 Hz target (matches inference rate)
- `capture_next_nn_input()` returns `(1, 3, H, W)` NCHW float32 on CUDA, or `None` if no frame
- `shutdown()` calls `camera.stop()` for clean resource release

---

### Video Feature Anchor Extractor

**File:** `weaponized_ai/video_feature_extractor.py`

`VideoFeatureAnchorExtractor` converts raw BGR frames to a 64-dimensional feature vector without neural network inference, for use in latency-constrained paths:

| Indices | Source | Content |
|---------|--------|---------|
| 0–7 | Optical flow (Farneback) | Mean absolute X and Y flow per screen quadrant (Q1–Q4 × 2 axes) |
| 8 | Threshold on top-right UI zone | Damage readout pixel density |
| 9–63 | Reserved / zeros | Future structural features |

Maintains `prev_gray` across calls for inter-frame optical flow. `reset()` clears it at match start to prevent cross-match flow artifacts.

---

### Semantic Kinematic Extractor (YOLO)

**File:** `weaponized_ai/semantic_extractor.py`

`SemanticKinematicExtractor` runs a custom-trained YOLO nano model to detect Brawlhalla entities and return a 9-dimensional coordinate vector:

| Indices | Class | Content |
|---------|-------|---------|
| 0–3 | Class 0 (Player 1) | xywh bounding box |
| 4–7 | Class 1 (Player 2) | xywh bounding box |
| 8 | Class 2 (Weapons) | Count of weapon entities |

Accepts the same `(1, 3, H, W)` CUDA float32 tensor produced by `ZeroCopyVisionEngine`. Requires `pip install ultralytics` and `models/brawlhalla_yolo_nano.pt`.

---

## Input Injection

### Frame Deterministic Dispatcher

**File:** `weaponized_ai/hardware_driver.py`

`FrameDeterministicDispatcher` is the primary production input driver. Architecture:

- **Main thread** calls `stage_action_map(dict[str, bool])` to enqueue the desired key state (maxlen=2 deque, always consumes the freshest)
- **Worker thread** runs a nanosecond spin-lock loop, calls `_diff_and_execute()` at each frame boundary
- Uses `timeBeginPeriod(1)` / `timeEndPeriod(1)` to request 1 ms OS timer resolution for sub-millisecond accuracy
- `emergency_release()` (aliased as `global_flush()`) releases every held scan code — called on shutdown, crash, and zero-state guard

**Conflict resolution in `_diff_and_execute()`:**

```
LEFT + RIGHT pressed simultaneously → both released
UP + DOWN pressed simultaneously   → UP released, DOWN kept (fast-fall priority)
```

All events are issued as `KEYEVENTF_SCANCODE` (physical keyboard scan codes, not VK codes) making them hardware-indistinguishable.

---

### Mutex Hardware Driver

**File:** `weaponized_ai/hardware_driver.py`

`MutexHardwareDriver` is a simpler synchronous alternative. `update_input_matrix(target_states)` applies the same conflict resolution and diff-based injection under a `threading.Lock`. Use this when deterministic frame timing is not required (menus, setup, BC corpus collection).

---

### Input Controller (VK-code layer)

**File:** `weaponized_ai/input_controller.py`

The original VK-code macro engine used by `TrainingLoop`:

- `tap(vk, hold_s)` — press + hold + release with 16 ms default (one game frame at 60 fps)
- `execute_macro(name)` — runs named combo sequences from `MACROS` in a background thread
- `ActionTranslationEngine` — higher-level wrapper that maps factorised action dicts to hardware sequences
- `WindowsHardwareController` — scan-code variant matching the `hardware_driver.py` interface
- `emergency_flush()` — releases all tracked VK keys unconditionally

---

## Training Loops

### HighFidelityTrainingLoop (60 Hz SHM)

**File:** `weaponized_ai/training_loop.py`

The production training loop. Per-frame pipeline:

```
1. shm_reader.read_latest_state()          → (state_vector, tracking_id)
2. TelemetryWatchdogMemoryReader checks     → None check, zero-state guard
3. torch.from_numpy(state_vector).cuda()   → state_tensor (batch=1)
4. agent.policy(state_tensor)              → (distribution, value)
5. distribution.sample()                   → action
6. io_executor.submit(controller.inject)   → async fire-and-forget
7. perf_counter_ns spin-wait               → precision frame alignment
```

**Zero-state guard** (step 2): If `np.sum(np.abs(state_vector)) == 0.0`, the frame is a loading/corruption artifact. `controller.global_flush()` is called to release all keys and the loop sleeps 100 ms before the next attempt.

Uses a dedicated `ThreadPoolExecutor(max_workers=1)` for input injection so the inference cycle is never blocked waiting for `SendInput` to return.

---

### TrainingLoop (8 Hz Vision)

**File:** `weaponized_ai/training_loop.py`

The fallback training loop for systems without the SHM bridge. Runs at ~8 Hz:

1. `mss` screen capture or OBS virtual camera (auto-detects Brawlhalla window title)
2. `game_state_reader.read_state(frame)` → 18-float obs via color detection + template matching
3. Frame-diff velocity estimation (`_inject_velocity`)
4. `RLAgent.select_action(obs)` + `execute_macro()`
5. `compute_reward()` from consecutive state diffs
6. PPO store + train every 64 steps + checkpoint every 512 steps

Episode boundaries trigger when either player's stock count changes.

---

## Data Ingestion

### Replay Engine

**File:** `weaponized_ai/replay_engine.py`

Parses real Brawlhalla `.replay` files and extracts `(obs, action, reward, next_obs, done)` transitions for BC pretraining. Tested against 504 real v10.06 replays with zero parse failures.

**Binary format:**

Replay files are **zlib-deflate compressed** (`0x78 0xDA` magic). After `zlib.decompress()`:

| Offset | Content |
|--------|---------|
| `0x00` | Format magic `6B 10 DF 35` |
| `0x04` | Per-match seed |
| `0x09` | Mode flags |
| `0x0C` | Player metadata (character IDs) |
| `~0x30+` | Per-frame input stream (1 byte per player per frame) |

**Input byte bit layout:**

```
Bit 7: Up/Jump  Bit 6: Right   Bit 5: Left    Bit 4: Down
Bit 3: Light    Bit 2: Heavy   Bit 1: Dodge   Bit 0: Weapon
```

**Stage bounds:** 23 stages mapped to `(half_width, ground_y)` tuples for accurate position normalisation and physics simulation.

**Processing pipeline:** `discover()` → `parse_meta()` → `process_replay()`. Each replay is simulated frame-by-frame with `PlayerPhysics`. Damage delta + KO bonus drives the heuristic reward signal.

---

### Video Learner (YouTube + Whisper)

**File:** `weaponized_ai/video_learner.py`

Downloads Brawlhalla videos and generates BC transitions:

| Stage | Tool | Output |
|-------|------|--------|
| DOWNLOAD | yt-dlp (best ≤ 1080p) | `.mp4` temp file |
| EXTRACT | OpenCV @ ~4 fps | 18-float obs vectors |
| INFER | Optical flow + position delta | Action labels + confidence weights |
| FILTER | Discard menu/death frames | Active-game transitions only |
| TRANSCRIBE | openai-whisper (optional) | 200+ vocab term → action index map |
| STORE | brain_store | `corpus.npz` + `knowledge.json` update |

Whisper transcription triples the weight (`×3`) of transitions within 2 seconds of a quality term (combo, punish, edgeguard, etc.), biasing BC training toward expert moments.

All heavy work runs in a background thread. `GET /video/status` streams live progress.

---

## Session Infrastructure

### Opponent Pool Reservoir

**File:** `weaponized_ai/opponent_pool.py`

`OpponentPoolReservoir` maintains a curriculum of training opponents:

- **30%** of matchups: load a historical self-play weight snapshot from `brain/snapshots/*.pt`
- **70%** of matchups: use a native engine bot profile (`aggressive_bot`, `passive_bot`, `balanced_bot`)

`register_snapshot(model, total_steps)` saves a frozen copy every 50,000 steps.

> **Fix applied:** The snapshot condition used `total_steps % interval == 0`, which evaluates to True at step 0 in Python (0 is divisible by anything). This would save an untrained, random-weight network as the first snapshot, polluting self-play with a worst-case opponent. Fixed with `total_steps > 0` guard.

---

### Replay Desync Validator

**File:** `weaponized_ai/replay_validator.py`

`ReplayDesyncValidator` verifies that the internal physics simulation stays synchronized with ground-truth binary events after each match. Checks:

1. KO count matches between simulated trajectory and binary truth events
2. Per-KO frame drift ≤ `max_allowed_drift_frames` (default 4)

If either check fails, the caller should invalidate the simulation state and resync. This prevents corrupted advantage estimates from training the policy on phantom rewards.

---

### Autonomous Session Manager

**File:** `weaponized_ai/session_manager.py`

`AutonomousSessionManager` is a Python context manager wrapping the full training session lifecycle:

```python
with AutonomousSessionManager(translator, watchdog) as session:
    run_high_fidelity_loop(...)
# translator.emergency_flush() called unconditionally here
```

On clean exit **and** on any exception, `emergency_flush()` is called before the exception propagates. This guarantees the keyboard is always returned to normal state even if an exception is raised mid-combo.

---

### Persistent Storage Engine

**File:** `weaponized_ai/config_manager.py`

`PersistentStorageEngine` stores user settings in `%APPDATA%\StrikerEnlightened\user_settings.json`. Default settings:

| Key | Default | Description |
|-----|---------|-------------|
| `target_fps` | 60 | Training loop tick rate |
| `exploration_entropy_ratio` | 0.25 | Target entropy fraction for `AdaptiveEntropyTuner` |
| `kl_divergence_tether` | 0.01 | KL target for `CovariateShiftProtectedPPOLoss` |
| `auto_start_on_game_launch` | `true` | Auto-start loop when Brawlhalla is detected |
| `emergency_panic_key` | `"ESCAPE"` | Key that triggers `emergency_flush()` |

`load_settings()` merges from disk into defaults (missing keys fall back). `save_settings(new_settings)` merges and flushes to disk atomically.

---

### Brain Store

**File:** `weaponized_ai/brain_store.py`

Persistent storage for all accumulated AI knowledge. All files in `weaponized_ai/brain/` (gitignored):

| File | Format | Content |
|------|--------|---------|
| `knowledge.json` | JSON dict | Brawlhalla term → occurrence count from Whisper transcripts |
| `corpus.npz` | NumPy compressed | BC transitions: `obs, acts, rwds, next_obs, dones, weights` (capped at 50,000) |
| `registry.json` | JSON list | Per-video metadata (URL, title, date, transitions, Whisper used) |

`BRAIN_DIR` reads from the `STRIKER_DATA_DIR` environment variable (set by `server_entry.py` when frozen by PyInstaller) so data always writes to `%APPDATA%` regardless of install location.

---

### Process Utilities

**File:** `weaponized_ai/process_utils.py`

**`HeadlessExecutionHost.spawn_silent_submodule(script_path, args)`** — launches Python subprocesses with `CREATE_NO_WINDOW=0x08000000` so no console window appears on the taskbar during background module execution.

**`HighPriorityExecutionShield.claim_cpu_dominance()`** — elevates the current process to `ABOVE_NORMAL` priority via `psutil`. Deliberately avoids `REALTIME_PRIORITY_CLASS` which can starve OS input handling. No-ops gracefully on non-Windows or if psutil is not installed.

---

## Strategy Engine

**File:** `weaponized_ai/strategy_engine.py`

Real-time physics prediction and strategy ranking used by the UI's Strategy tab:

**Physics primitives:**
- `predict_landing(state, gravity=980)` — projectile-motion integrator using Brawlhalla's actual gravity constant (980 px/s²), returns `(landing_x, landing_y, time_s)`
- `projectile_lead(shooter, target, proj_speed)` — intercept solver for projectile aiming at a moving target

**Strategy ranker:**

`rank_strategies(p1, p2)` scores 12+ named strategies against the current game state based on damage differential, stock differential, distance, airborne status, buffs, and weapon possession. Returns strategies sorted by priority score; the top result drives suggested moves in the UI.

---

## Frontend (React + Tauri)

Built with **React 18 + TypeScript + Vite** bundled as a native Windows app by **Tauri 2**. The Tauri `lib.rs` Rust shell handles:

- `auto_start_server()` — spawns `striker-server.exe` as a hidden child process at app launch
- `start_ai_server` / `stop_ai_server` — Tauri commands to control the Python backend lifecycle
- `execute_macro(name)` — proxies to `POST http://127.0.0.1:8000/input/macro`

### WeaponizedAPI.ts

**File:** `src/WeaponizedAPI.ts`

Typed HTTP client. Every endpoint has a corresponding async function with a concrete return type. Key interfaces:

- `ReplayFileMeta` — `{ path, name, size_kb, parse_ok, frame_count, stage, game_version }`
- `MemoryState` — 18-float obs + named fields for damage/stocks/positions
- `BTRStats` — steps, mean Q-value, loss, buffer size, epsilon
- `BrainInfo` — knowledge term count, corpus size, registry entries, disk usage

### FightTab.tsx Panels

**File:** `src/FightTab.tsx`

Six collapsible control panels:

1. **Replay Digestion** — list and ingest `.replay` files; "Ingest All" batch-processes all 504 replays
2. **Live Memory Reader** — polls `/memory/state` at 10 Hz; P1/P2 damage bars (green/yellow/red by tier), stock hearts, XY, weapon status
3. **BTR Agent** — step count, mean Q-value, TD loss, buffer fill %; pretrain / save / load buttons; single-obs inference
4. **Brain Storage** — knowledge term count, corpus transitions, video registry; manual save / reload / clear
5. **YouTube Learner** — URL input, live progress bar (DOWNLOAD → EXTRACT → INFER → FILTER → TRANSCRIBE → STORE), transitions counter, pretrain trigger
6. **RL Stats** — PPO total steps, episodes, mean episode reward, policy/value/entropy losses; observation normaliser calibration

---

## Runner Scripts

### striker_service.py

Root-level background service entry point. Bootstraps `PersistentStorageEngine` and `HighPriorityExecutionShield`, then holds an idle loop. Uncomment the `HighFidelityTrainingLoop` block to attach a live training session:

```python
python striker_service.py
```

### live_agent_node.py

Root-level live inference entry point. Wires the full pipeline:

```
ZeroCopyVisionEngine
  → SemanticKinematicExtractor (YOLO)
    → zero-state guard
      → policy_network inference
        → FrameDeterministicDispatcher
```

Runs at 1 ms poll interval. `emergency_release()` + `vision.shutdown()` are called in the `finally` block on any exit path.

```python
from weaponized_ai.policy_network import EnlightenedPolicyNetwork
policy = EnlightenedPolicyNetwork(...)
run_agent_live_loop(policy)
```

---

## API Reference

### Health

```
GET /health
→ { status, obs_dim, act_dim }
```

### Policy (PPO)

```
POST /policy/infer    { obs: float[18] }
→ { action, log_prob, logits: float[16], value }

POST /rl/store        { obs, action, reward, log_prob, done }
POST /rl/train        → { loss_policy, loss_value, entropy, updates }
POST /rl/save         → { saved: true }
POST /rl/load         → { loaded: bool }
GET  /rl/stats        → { total_steps, episode_count, mean_reward, last_loss_* }
GET  /rl/buffer_size  → { size }
```

### BTR Agent

```
POST /btr/action   { obs: float[18] }  → { action, q_value }
POST /btr/store    { obs, action, reward, next_obs, done }
GET  /btr/stats    → { steps, mean_q, loss, buffer_size, epsilon }
POST /btr/pretrain → { transitions_used, loss }
POST /btr/save     → { saved: true }
POST /btr/load     → { loaded: bool }
```

### Replay

```
GET  /replay/scan       → ReplayFileMeta[]
POST /replay/ingest     { path }  → { transitions, stage, game_version }
POST /replay/ingest_all → { files, transitions, errors }
```

### Memory

```
GET  /memory/info   → { attached, pid, cached_addrs }
GET  /memory/state  → MemoryState (18-float obs + named fields)
POST /memory/rescan → { scanned, addrs_found }
```

### Brain

```
GET  /brain/info   → { knowledge_terms, corpus_size, registry_size }
POST /brain/save   → { saved: true }
POST /brain/reload → { reloaded: true }
POST /brain/clear  → { cleared: true }
```

### Video

```
POST /video/ingest   { url }
GET  /video/status   → { stage, frames, transitions, progress_pct, eta_s }
POST /video/pretrain → { transitions_used, loss }
```

### Input

```
POST /input/tap    { vk?, key?, hold_s? }
POST /input/macro  { name }
GET  /input/macros → { macros: string[] }
```

### Strategy

```
POST /strategy/rank            { p1, p2 } → { strategies[], top }
POST /strategy/predict_landing { state }  → { landing_x, landing_y, time_s }
POST /strategy/projectile_lead { shooter, target, proj_speed } → { angle_deg, intercept_x, intercept_y }
```

---

## What Was Improved

These bugs were identified during development and corrected. All fixes are documented here for reproducibility.

| File | Issue | Fix |
|------|-------|-----|
| `ppo_loss.py` | KL divergence direction was `KL(ref‖live)` (reverse KL — mode-seeking). RLHF anchoring requires `KL(live‖ref)` (forward KL) to prevent the live policy from assigning mass where the reference has none. | Swapped `F.kl_div` arguments: `log_input=log(ref)`, `target=live` |
| `ppo_loss.py` | Adaptive beta had no bounds — `kl_beta *= 1.1` every update could grow unboundedly or collapse toward 0 with no floor. | Clamped beta to `[0.05, 5.0]` |
| `reward_shaper.py` | PBRS formula was `Φ(s') − Φ(s)` — missing the γ discount factor. Per the PBRS theorem, the correct expression is `γ·Φ(s') − Φ(s)`; without γ the potential is non-zero at terminal boundaries, introducing a systematic bias toward start positions. | Added `_GAMMA = 0.99` multiplier to the new potential term |
| `opponent_pool.py` | `total_steps % interval == 0` evaluates to True at step 0 in Python (0 mod N = 0 for any N). This caused an untrained random-weight network to be saved as the first self-play clone, flooding the opponent curriculum with the worst possible opponent. | Added `total_steps > 0` guard |
| `watchdog_reader.py` | The SHM sentinel checksum (`0xDEADC0DEFA57FEED`) was written by `shm_bridge.cpp` but never verified on the Python side. Pre-init reads and garbage bytes passed silently to the policy network. | Added sentinel validation as the first check in `validate_and_parse()` |
| `training_loop.py` | No guard against all-zero state vectors from loading screens or SHM startup frames — the policy would infer on corrupted input. | Added zero-state safety lock: if `np.sum(np.abs(state)) == 0`, call `global_flush()` and sleep 100 ms |

---

## Configuration Reference

| Setting | File | Default | Description |
|---------|------|---------|-------------|
| Replay directory | `replay_engine.py` | `C:\Users\carli\BrawlhallaReplays` | Primary path; falls back to `%APPDATA%\Brawlhalla\replays` |
| Max BC corpus | `brain_store.py` | 50,000 | Maximum transitions retained on disk |
| PPO learning rate | `rl_agent.py` | 3e-4 | AdamW lr for policy + value networks |
| PPO epochs | `rl_agent.py` | 4 | Gradient passes per collected batch |
| PPO clip ε | `rl_agent.py` | 0.2 | Surrogate objective clip ratio |
| BTR buffer | `rl_agent.py` | 100,000 | PER replay buffer capacity |
| KL target | `ppo_loss.py` | 0.01 | Target KL divergence from reference policy |
| KL beta range | `ppo_loss.py` | [0.05, 5.0] | Adaptive KL coefficient bounds |
| Entropy target | `entropy_tuner.py` | 0.25 | Target entropy fraction (25% of max) |
| Entropy β range | `entropy_tuner.py` | [0.001, 0.1] | Adaptive entropy coefficient bounds |
| Target FPS | `config_manager.py` | 60 | `HighFidelityTrainingLoop` tick rate |
| Snapshot interval | `opponent_pool.py` | 50,000 steps | Self-play clone save frequency |
| SHM buffer size | `watchdog_reader.py` | 236 bytes | Minimum valid `BrawlhallaStateBuffer` size |
| Max velocity delta | `watchdog_reader.py` | 150 px/frame | Watchdog drift threshold |
| Coordinate bounds | `watchdog_reader.py` | ±5000 | OOB detection limit |
| Vision capture FPS | `visual_engine.py` | 120 Hz | DXGI capture rate |
| Flow target | `cuda_vision.py` | 60 Hz | `ZeroCopyVisionEngine` target rate |
| Train every | `training_loop.py` | 64 steps | PPO update interval (8 Hz loop) |
| Save every | `training_loop.py` | 512 steps | Checkpoint interval |
| Server port | `start_server.ps1` | 8000 | FastAPI uvicorn port |

---

## Project Structure

```
Striker-The-Enlightened/
├── start_server.ps1              ← One-click server launcher
├── server_entry.py               ← PyInstaller entry point (sets STRIKER_DATA_DIR)
├── striker_server.spec           ← PyInstaller onedir bundle spec
├── build_installer.ps1           ← Full build pipeline (PyInstaller + Tauri)
├── striker_service.py            ← Background priority service runner
├── live_agent_node.py            ← Live inference entry point (Vision→YOLO→Policy→Input)
│
├── weaponized_ai/
│   ├── api_server.py             ← FastAPI: all REST endpoints
│   ├── rl_agent.py               ← PPO + BTR agents (PyTorch)
│   ├── policy_network.py         ← EnlightenedPolicyNetwork, FactorizedMultiDiscreteActorHead
│   ├── ppo_loss.py               ← CovariateShiftProtectedPPOLoss (KL-anchored PPO)
│   ├── reward_shaper.py          ← AlignedRewardShaper (PBRS + anti-suicide)
│   ├── advantage_engine.py       ← EligibilityTracedAdvantageEngine (ET-GAE)
│   ├── entropy_tuner.py          ← AdaptiveEntropyTuner
│   ├── action_masker.py          ← StateDependentActionMasker
│   ├── value_heads.py            ← MacroStateValueBootstrapper + RunningStateNormalizer
│   │                                + TrajectoryRewardFilter + OrthogonalNetworkInitializer
│   ├── unified_pipeline.py       ← UnifiedAcceleratedPipeline (BC pretraining + PPO)
│   ├── training_loop.py          ← TrainingLoop (8 Hz) + HighFidelityTrainingLoop (60 Hz SHM)
│   ├── replay_engine.py          ← .replay binary parser + BC extractor (504 replays)
│   ├── brawlhalla_memory.py      ← ReadProcessMemory + AoB signature scan
│   ├── watchdog_reader.py        ← TelemetryWatchdogMemoryReader (SHM validator)
│   ├── dynamic_memory_reader.py  ← DynamicBrawlhallaReader (pymem pointer chains)
│   ├── brain_store.py            ← knowledge.json / corpus.npz / registry.json
│   ├── video_learner.py          ← YouTube → BC transitions (yt-dlp + Whisper)
│   ├── visual_engine.py          ← HighSpeedVisualEngine (120 Hz DXGI)
│   ├── cuda_vision.py            ← ZeroCopyVisionEngine (60 Hz, region-aware)
│   ├── video_feature_extractor.py← VideoFeatureAnchorExtractor (optical flow + UI)
│   ├── semantic_extractor.py     ← SemanticKinematicExtractor (YOLO nano entity tracker)
│   ├── hardware_driver.py        ← MutexHardwareDriver + FrameDeterministicDispatcher
│   ├── input_controller.py       ← VK-code macro engine + ActionTranslationEngine
│   ├── strategy_engine.py        ← Physics prediction + strategy ranking
│   ├── opponent_pool.py          ← OpponentPoolReservoir (self-play curriculum)
│   ├── replay_validator.py       ← ReplayDesyncValidator (KO frame drift check)
│   ├── session_manager.py        ← AutonomousSessionManager (context manager)
│   ├── config_manager.py         ← PersistentStorageEngine (AppData JSON settings)
│   ├── process_utils.py          ← HeadlessExecutionHost + HighPriorityExecutionShield
│   ├── game_state_reader.py      ← Frame → 18-float obs (mss / OBS vision path)
│   ├── obs_capture.py            ← Screen capture helpers
│   ├── obs_manager.py            ← OBS WebSocket integration
│   └── __init__.py
│
├── src/                          ← React + TypeScript frontend
│   ├── WeaponizedAPI.ts          ← Typed HTTP client for all endpoints
│   ├── FightTab.tsx              ← Main AI control UI (6 panels)
│   ├── App.tsx / main.tsx        ← Tauri app entry points
│   ├── CaptureTab.tsx            ← Screen capture configuration
│   ├── IntelTab.tsx              ← Game intel + opponent profiling
│   ├── SimulationTab.tsx         ← Offline simulation runner
│   ├── StrategyTab.tsx           ← Strategy engine visualiser
│   ├── BackendBridge.tsx         ← Tauri ↔ backend health monitor
│   ├── LatencyMonitor.tsx        ← API round-trip latency display
│   └── spatialSafety.ts          ← Blastzone boundary safety checks
│
├── src-tauri/
│   ├── src/lib.rs                ← Tauri commands + auto_start_server logic
│   ├── src/main.rs               ← Tauri entry point
│   └── tauri.conf.json           ← productName, identifier, resources bundle
│
└── tools/
    ├── smoke_test_replay.py      ← End-to-end replay validation (504 files)
    ├── shared_mem/
    │   ├── shm_bridge.cpp        ← C++ SHM DLL (BrawlhallaStateBuffer + sentinel)
    │   ├── shm_writer.cpp        ← Game-side buffer writer
    │   ├── shm_reader.cpp        ← Standalone debug reader
    │   └── shm_client.ts         ← Node.js FFI SHM reader
    └── input_replay/
        ├── replay.cpp            ← Win32 input record + replay tool
        └── input_log.txt         ← Recorded input sequence
```

---

## Requirements

**Python backend:**

```
fastapi>=0.100
uvicorn[standard]
torch>=2.0            # CUDA build recommended for 60 Hz inference
numpy
pydantic>=2.0
yt-dlp
opencv-python
openai-whisper        # optional — enables audio transcription weighting
dxcam                 # optional — required for HighSpeedVisualEngine / ZeroCopyVisionEngine
ultralytics           # optional — required for SemanticKinematicExtractor (YOLO)
pymem                 # optional — required for DynamicBrawlhallaReader
psutil                # optional — required for HighPriorityExecutionShield
```

**Node / Tauri:**

```
node >= 18
npm >= 9
@tauri-apps/cli  (bundled via package.json devDependencies)
vite + react + typescript
```

**System:**

- Windows 10 / 11 (required for `ReadProcessMemory`, `SendInput`, `DXGI Desktop Duplication`, named shared memory)
- NVIDIA GPU with CUDA (required for 60 Hz `HighFidelityTrainingLoop`; 8 Hz `TrainingLoop` runs CPU-only)
- Rust toolchain (required only for `npm run tauri build`)
- Brawlhalla installed and accessible by path

**File:** `weaponized_ai/api_server.py`

The central FastAPI application. Every AI subsystem is exposed as a REST endpoint. CORS is open (`allow_origins=["*"]`) so the Tauri WebView can reach it on localhost without restriction.

**Endpoint groups:**

| Prefix | Purpose |
|--------|---------|
| `/health` | Liveness check + obs/act dimensions |
| `/policy/*` | PPO policy inference (single forward pass) |
| `/rl/*` | PPO experience buffer + training + save/load |
| `/btr/*` | BTR agent action, store, pretrain, stats |
| `/replay/*` | Scan replay directory, ingest single or all replays |
| `/memory/*` | Live Brawlhalla process-memory state |
| `/brain/*` | Knowledge base info, manual save, reload, clear |
| `/video/*` | YouTube ingest, status polling, pretrain trigger |
| `/input/*` | Key tap by VK code or name, macro execution |
| `/strategy/*` | Predict landing, projectile lead, rank strategies |
| `/obs/*` | Calibrate observation normalizer |

All request/response schemas are Pydantic `BaseModel` classes co-located with their endpoint handler, so the auto-generated OpenAPI docs at `http://127.0.0.1:8000/docs` are always accurate.

---

### RL Agent (PPO)

**File:** `weaponized_ai/rl_agent.py`

The primary online RL agent. Implements Proximal Policy Optimization with the following enhancements:

**Networks:**

Both `PolicyNet` and `ValueNet` share the same three-layer MLP backbone:

```
Linear(obs_dim -> 256) -> LayerNorm(256) -> ReLU
Linear(256 -> 256)      -> LayerNorm(256) -> ReLU
Linear(256 -> out_dim)
```

`LayerNorm` after each linear layer keeps activations stable during rapid on-policy learning where the data distribution shifts every update.

**Observation space (18 floats):**

| Index | Meaning |
|-------|---------|
| 0-1 | P1 XY position (normalized by stage half-width) |
| 2-3 | P1 XY velocity |
| 4 | P1 airborne flag |
| 5 | P1 damage % |
| 6 | P1 stocks remaining |
| 7-8 | P2 XY position |
| 9-10 | P2 XY velocity |
| 11 | P2 damage % |
| 12 | P2 stocks remaining |
| 13 | Inter-player distance |
| 14-15 | Relative XY (P2 - P1) |
| 16 | P1 armed (has weapon) |
| 17 | Reserved / extra flag |

**Action space (16 discrete):**

`NLight, SLight, DLight, NHeavy, SHeavy, DHeavy, Nair, Sair, Dair, Jump, DoubleJump, Dodge, DashRight, DashLeft, WeaponPickup, NSig`

**PPO training (GAE + clipping + entropy):**

1. Collect `N` steps in the experience buffer.
2. Compute Generalized Advantage Estimates (lambda=0.95) with a bootstrapped value baseline.
3. Normalize advantages to zero-mean, unit-variance per batch.
4. Run `PPO_EPOCHS=4` gradient passes per collected batch.
5. Clipped surrogate objective (epsilon=0.2) prevents destructively large policy updates.
6. Entropy bonus (coefficient=0.01) keeps the policy from collapsing to a single deterministic action during early training.
7. Shared Adam optimizer (lr=3e-4, eps=1e-5) over both networks.
8. Gradient norm clipping at 0.5 prevents exploding gradients during spiky reward signals.

**Hyperparameters:**

```python
GAMMA      = 0.99   # discount
GAE_LAMBDA = 0.95   # GAE smoothing
CLIP_EPS   = 0.2    # PPO clip ratio
ENTROPY_C  = 0.01   # entropy coefficient
VALUE_C    = 0.5    # value loss weight
MAX_GRAD   = 0.5    # gradient clip norm
PPO_EPOCHS = 4      # update passes per batch
LR         = 3e-4
```

Model persists to `weaponized_ai/model.pt` on `/rl/save` and is auto-reloaded on `/rl/load`.

---

### BTR Agent

**File:** `weaponized_ai/rl_agent.py` (same file, lower section)

The secondary offline/pretrain agent. Implements **Beyond The Rainbow** (ICML 2025), a Noisy Dueling Double DQN with Prioritized Experience Replay and n-step returns.

**BTRNet architecture:**

```
Shared encoder:
  Linear(obs_dim -> 256) -> LayerNorm -> ReLU
  Linear(256 -> 256)      -> LayerNorm -> ReLU

Dueling heads (both use NoisyLinear):
  Value stream:     NoisyLinear(256 -> 128) -> ReLU -> NoisyLinear(128 -> 1)
  Advantage stream: NoisyLinear(256 -> 128) -> ReLU -> NoisyLinear(128 -> act_dim)

Q(s,a) = V(s) + A(s,a) - mean(A(s,.))
```

`NoisyLinear` replaces epsilon-greedy exploration with learned per-weight Gaussian noise — the network learns *when* to explore rather than always exploring uniformly.

**PrioritizedReplayBuffer:**

Transitions are stored with TD-error priorities. Sampling probability is proportional to `|delta|^alpha` (alpha=0.6). Importance-sampling weights (beta annealed 0.4->1.0) correct for the introduced bias. Buffer capacity: 100,000 transitions.

**NStepBuffer:**

Accumulates n=3 consecutive transitions before writing to the PER buffer, computing the discounted n-step return `r + gamma*r' + gamma^2*r''` and bootstrapping from the state n steps ahead. This reduces variance in the TD target compared to 1-step returns.

**Double DQN target:**

```
y = r + gamma * Q_target(s', argmax_a Q_online(s', a))
```

The online network selects the greedy action; the target network evaluates it. Target network is soft-updated (tau=0.005) every step.

Model persists to `weaponized_ai/btr_model.pt`.

---

### Replay Engine

**File:** `weaponized_ai/replay_engine.py`

Parses real Brawlhalla `.replay` files and converts them into `(obs, action, reward, next_obs, done)` transitions for Behavioural Cloning pre-training.

**Binary format (discovered by hex analysis of 504 v10.06 replays):**

Brawlhalla replay files are **zlib-deflate compressed** (raw bytes begin with `0x78 0xDA`). After `zlib.decompress()`:

| Offset | Size | Content |
|--------|------|---------|
| `0x00` | 4 | Format magic: `6B 10 DF 35` (constant across all files) |
| `0x04` | 5 | Per-match seed / identifier |
| `0x09` | 3 | Mode flags (game mode encoding) |
| `0x0C` | ~36 | Player metadata block (character IDs — proprietary encoding) |
| `~0x30` | N*2 | Per-frame input stream (1 byte per player per frame) |

**Input byte bit layout:**

```
Bit 7: Up / Jump
Bit 6: Right
Bit 5: Left
Bit 4: Down
Bit 3: Light Attack
Bit 2: Heavy Attack
Bit 1: Dodge
Bit 0: Weapon Pickup
```

**Stage-aware physics (`_STAGE_BOUNDS`):**

23 stages are mapped to `(half_width, ground_y)` tuples used to normalize position observations and simulate wall/blastzone boundaries. `PlayerPhysics` takes `stage_half_w` as a constructor parameter so each replay uses the correct normalization for its stage.

**Metadata from filename:**

Character name data in the binary is proprietary. Stage name and game version are extracted from the filename pattern:

```
[10.06] WesternAirTemple (3).replay
 ^^^^^  ^^^^^^^^^^^^^^^^^
version  stage_name
```

**Processing pipeline:**

1. `discover()` — scans `C:\Users\carli\BrawlhallaReplays` (or APPDATA fallback) for all `.replay` files.
2. `parse_meta(path)` — decompresses, validates magic, estimates frame count from file size, reads stage/version from filename.
3. `process_replay(path, max_transitions)` — simulates P1/P2 physics frame-by-frame, computes obs vectors, maps input bits to action indices, assigns a heuristic reward (damage delta + KO bonus), yields `Transition` named-tuples.

Tested against 504 real v10.06 Brawlhalla replays with zero parse failures.

---

### Brawlhalla Memory Reader

**File:** `weaponized_ai/brawlhalla_memory.py`

Reads live game state directly from Brawlhalla's process memory using the Windows `ReadProcessMemory` API — no OCR, no screen capture, native game speed.

**Attach flow:**

1. `OpenToolhelp32Snapshot(TH32CS_SNAPPROCESS)` enumerates all running processes.
2. Finds the PID of `Brawlhalla.exe`.
3. `OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION)` obtains a handle.

**AoB signature scan:**

Because memory addresses change with every game update and every launch, fixed addresses are not used. Instead, each AoB signature is a pattern of known bytes with `None` wildcards. The scanner:

1. Calls `VirtualQueryEx` to enumerate all committed, readable memory regions.
2. Reads each region with `ReadProcessMemory`.
3. Searches for the byte pattern using sliding-window matching.
4. Resolves the final data address by following pointer chains from the match offset.

Matched addresses are cached to `weaponized_ai/_addr_cache.json` so the scan only runs once per game session.

**Signatures:**

| Field | What it reads |
|-------|--------------|
| `p1_damage` | P1 current damage % (float) |
| `p2_damage` | P2 current damage % (float) |
| `p1_stocks` | P1 stocks remaining (int) |
| `p2_stocks` | P2 stocks remaining (int) |
| `p1_x`, `p1_y` | P1 XY position (float, float) |
| `p2_x`, `p2_y` | P2 XY position (float, float) |

If any signature fails to match, that field falls back to its default value and the system continues. The reader degrades gracefully to partial state rather than crashing.

**Output:** Same 18-float observation format as the PPO agent expects. The `/memory/state` endpoint polls at up to 10 Hz.

---

### Brain Store

**File:** `weaponized_ai/brain_store.py`

Persistent storage layer for all learned knowledge. All files live in `weaponized_ai/brain/` (excluded from git via `.gitignore`).

**Files:**

| File | Format | Content |
|------|--------|---------|
| `knowledge.json` | JSON dict | Brawlhalla term -> occurrence count. Accumulated from Whisper transcripts of every ingested video. Grows across sessions. |
| `corpus.npz` | NumPy compressed | BC transition arrays: `obs`, `acts`, `rwds`, `next_obs`, `dones`, `weights`. Capped at 50,000 entries; oldest transitions are replaced when full. |
| `registry.json` | JSON list | Metadata for every ingested video: URL, title, date, transition count, whether Whisper was available. Prevents re-ingesting the same video. |

**Auto-save triggers:**

- After every successful video ingest: `knowledge.json` + `registry.json`
- After every BC pretrain pass: `corpus.npz` + `knowledge.json`
- `POST /brain/save`: full flush of all three files

**Bootstrap on server start:**

`bootstrap_learner(learner)` is called by `get_learner()` in `video_learner.py`. It pre-populates the `VideoLearner`'s `knowledge_base` dict and `corpus` list from disk so the AI resumes exactly where it left off after a server restart — no re-ingestion required.

---

### Video Learner

**File:** `weaponized_ai/video_learner.py`

Downloads Brawlhalla YouTube videos, extracts game-state observations frame by frame, transcribes audio commentary with Whisper, and generates Behavioural Cloning training data.

**Pipeline stages:**

1. **DOWNLOAD** — `yt-dlp` pulls the best stream up to 1080p into a temp `.mp4`. Quality tiers: 1080p -> 720p -> best available.

2. **EXTRACT** — OpenCV reads frames at ~4 fps (every 15th frame at 60fps source). Each frame is passed to `game_state_reader` which uses color-range detection and template matching to read damage numbers, stock icons, and approximate player positions — producing an 18-float obs vector.

3. **INFER** — Optical flow between consecutive frames estimates velocity vectors. These, combined with position changes and damage deltas, drive a heuristic action labeler that maps each frame transition to one of the 16 action indices. High-confidence frames (large damage spike, clear trajectory change) receive higher BC weights.

4. **FILTER** — Frames where the game is not active (menu screens, loading, death freeze frames, both players at 0%) are discarded. Only frames where both players are alive and damage is visible are kept.

5. **TRANSCRIBE (optional)** — If `openai-whisper` is installed, the video audio is transcribed. Words matching `_BRAWLHALLA_VOCAB` (200+ Brawlhalla-specific terms) are mapped to action indices or added to `knowledge.json`. Transitions within 2 seconds of a quality term (combo, punish, edgeguard, etc.) receive 3x weight in BC training.

6. **STORE** — Transitions are written to the in-memory corpus and auto-saved to `brain/corpus.npz` via `brain_store`.

7. **PRETRAIN** — `POST /video/pretrain` runs supervised cross-entropy on `(obs -> action)` pairs using the BC corpus. This seeds the PPO policy with human-expert demonstrations before self-play RL begins, dramatically accelerating early training.

All heavy work runs in a background thread. `GET /video/status` streams live progress (stage, frames processed, transitions stored, ETA).

---

### Strategy Engine

**File:** `weaponized_ai/strategy_engine.py`

Real-time physics prediction and strategy ranking. Used by the UI's Strategy tab and callable from the API.

**Physics primitives:**

- `Vec2` — 2D vector with `dist_to`, `normalized`, `add`, `sub`, `scale` operations.
- `PlayerState` — full player snapshot: position, velocity, airborne flag, invulnerability, active move, stocks, damage, buff state.
- `predict_landing(state, gravity=980)` — projectile-motion integrator. Given a player's current position and velocity, returns the XY coordinates and time-to-land when they next touch the ground. Uses Brawlhalla's actual gravity constant (980 px/s^2).
- `projectile_lead(shooter, target, proj_speed)` — intercept calculation. Solves for the angle a projectile must be fired so it arrives at the target's predicted position, accounting for target velocity.

**Strategy ranker:**

`rank_strategies(p1, p2)` evaluates a library of 12+ named strategies (`ALL_STRATEGIES`) against the current game state and returns them sorted by computed priority score. Factors include: damage differential, stock differential, inter-player distance, airborne status, active buff, and weapon possession. The top-ranked strategy and its suggested move list are surfaced in the `/strategy/rank` endpoint and the UI's Strategy panel.

---

### Input Controller

**File:** `weaponized_ai/input_controller.py`

Injects keyboard inputs into Brawlhalla using the Win32 `SendInput` API. All key events go through the OS input subsystem — no DirectInput hooks, no memory patches.

**Key components:**

- `_sendinput_key(vk, down)` — constructs a `KEYBDINPUT` structure and calls `SendInput`. Thread-safe via `_input_lock`.
- `tap(vk, hold_s=0.016)` — press + hold for `hold_s` seconds + release. Default 16ms = one game frame at 60fps.
- `tap_name(key_name)` — maps string names (`'A'`, `'SPACE'`, `'LEFT'`) to virtual key codes via `VK_MAP`.
- `combo(steps)` — executes a sequence of `(vk, hold_ms, delay_after_ms)` tuples. Used for multi-input combos.
- `macro_thread(steps)` — same as `combo` but runs in a background thread so the API endpoint returns immediately.
- `execute_macro(name)` — looks up a named macro from `MACROS` dict and runs it in a thread. Macros encode full Brawlhalla combo sequences with precise frame-window timings.

**VK_MAP defaults** (remappable in config):

```python
'N': 0x4E,  'M': 0x4D,  'SPACE': 0x20,
'A': 0x41,  'D': 0x44,  'W': 0x57,  'S': 0x53,
'E': 0x45,  'F': 0x46,
'LEFT': 0x25, 'RIGHT': 0x27, 'UP': 0x26, 'DOWN': 0x28
```

---

### Training Loop

**File:** `weaponized_ai/training_loop.py`

The autonomous game-play driver. Runs in a background thread and executes the full RL loop without human interaction.

**Per-tick pipeline (~8 ticks/second):**

1. **Capture** — grabs the current game frame via `mss` screen capture or OBS virtual camera (auto-detects the Brawlhalla window by title).
2. **Read state** — `game_state_reader` converts the frame to an 18-float obs. Velocity is estimated by differencing consecutive position readings.
3. **Infer** — `RLAgent.select_action(obs)` runs one forward pass through the policy network and samples an action from the resulting categorical distribution.
4. **Execute** — `execute_macro(ACTION_MACROS[action])` injects the corresponding keypresses.
5. **Reward** — computed from: damage dealt to opponent (+), damage taken (-), KO bonus (+5.0), being KO'd (-5.0), weapon pickup (+0.2).
6. **Store** — `agent.store(obs, action, reward, log_prob, done)` appends to the experience buffer.
7. **Train** — every `TRAIN_EVERY=64` steps, `agent.train_step()` runs the full PPO update.
8. **Save** — every `SAVE_EVERY=512` steps, model checkpoint is written to disk.

**Action index -> macro mapping:**

```python
ACTION_MACROS = [
    "nlight", "slight", "dlight",
    "nheavy", "sheavy", "dheavy",
    "jump",   "jump",   "jump",
    "jump",   "jump",
    "dodge",  "dash_right", "dash_left",
    "nlight", "nheavy",
]
```

The loop is started/stopped via `POST /rl/start` and `POST /rl/stop`. Live tick stats (step count, last reward, episode count, mean reward) are readable via `GET /rl/stats`.

---

## Frontend (React + Tauri)

The desktop app is built with **React 18 + TypeScript + Vite** bundled into a native Windows application by **Tauri 2**. The WebView communicates with the FastAPI backend over localhost HTTP — no IPC bridge is required because all AI logic runs in the Python process.

### WeaponizedAPI.ts

**File:** `src/WeaponizedAPI.ts`

Typed API client layer. Every backend endpoint has a corresponding async function:

```typescript
scanReplays(): Promise<ReplayFileMeta[]>
ingestReplay(path: string): Promise<ReplayIngestResult>
getMemoryState(): Promise<MemoryState>
btrAction(obs: number[]): Promise<BTRActionResult>
ingestVideo(url: string): Promise<VideoIngestResult>
getBrainInfo(): Promise<BrainInfo>
```

**Key interfaces:**

- `ReplayFileMeta` — `{ path, name, size_kb, parse_ok, frame_count, stage, game_version, level_id, game_mode, characters }`
- `MemoryState` — 18-float obs vector plus named fields for damage/stocks/positions
- `BTRStats` — steps, mean Q-value, loss, buffer size, epsilon
- `BrainInfo` — knowledge term count, corpus size, registry entry count, disk usage

All fetch calls use `http://127.0.0.1:8000` as the base URL and return typed results with error handling.

### FightTab.tsx Panels

**File:** `src/FightTab.tsx`

The main combat AI control panel. Contains six collapsible sections:

**1. Replay Digestion**
Lists all `.replay` files from the replay directory. Each file shows:
`{stage_name} - v{game_version} ({size_kb}KB, ~{duration}s)`
Buttons: Ingest selected file -> sends BC transitions to the BTR/PPO agents. Ingest All -> batch-processes all 504 replays.

**2. Live Memory Reader**
Polls `/memory/state` at 10Hz when enabled. Displays:
- P1/P2 damage bars (color-coded: green < 60%, yellow < 120%, red >= 120%)
- P1/P2 stock heart icons
- XY positions
- Armed status

**3. BTR Agent**
Controls the Beyond-The-Rainbow agent:
- Stats grid: step count, mean Q-value, TD loss, buffer fill %
- Pretrain button: runs BC pass on the corpus
- Save/Load model buttons
- Action inference: paste an obs array, get back the greedy action

**4. Brain Storage**
Knowledge base management:
- Shows term count, corpus size (transitions), registry size (videos ingested)
- Manual Save, Reload from disk, Clear all buttons

**5. YouTube Learner**
Video ingestion panel:
- URL input + Ingest button
- Live progress bar with stage label (DOWNLOAD / EXTRACT / INFER / FILTER / TRANSCRIBE / STORE)
- Transitions generated counter
- Pretrain trigger once ingest completes

**6. RL Stats**
PPO agent live metrics:
- Total steps, episodes, mean episode reward
- Last policy loss, value loss, entropy
- Calibrate observation normalizer button

---

## Tools

### smoke_test_replay.py

**File:** `tools/smoke_test_replay.py`

End-to-end validation script. Verifies:
1. All `weaponized_ai` modules import without errors.
2. `REPLAY_DIR` exists and contains `.replay` files.
3. Filename regex `_FNAME_RE` correctly parses edge-case filenames.
4. `_STAGE_BOUNDS` contains at least 10 stages, all with valid `(half_width, ground_y)` tuples.
5. `PlayerPhysics(stage_half_w=...)` constructs and produces a valid obs vector.
6. `discover()` returns all 504 replay file paths.
7. `parse_meta()` successfully reads metadata from 5 random replays.
8. `process_replay()` extracts at least one transition from each of those 5 replays.

Run with: `.venv\Scripts\python.exe tools/smoke_test_replay.py`

### Shared Memory Bridge

**Files:** `tools/shared_mem/shm_reader.cpp`, `shm_writer.cpp`, `shm_client.ts`

A Windows named shared memory segment for zero-copy IPC between a C++ game-state producer and the TypeScript/Node layer:

- `shm_writer.cpp` — maps a `GameState` struct into shared memory and writes updated values each game tick.
- `shm_reader.cpp` — standalone reader/monitor for debugging.
- `shm_client.ts` — Node.js FFI client that reads the same shared memory region and exposes the state as a typed JavaScript object, used as an alternative to the HTTP memory endpoint for latency-sensitive code paths.

### Input Replay (C++)

**File:** `tools/input_replay/replay.cpp`

Records and replays raw Win32 input events from `input_log.txt`. Format: `timestamp_ms vk_code down_flag`. Used for:
- Debugging macro timing by recording a human playing and replaying at precise frame intervals.
- Stress-testing the input controller under rapid-fire input sequences.
- Generating ground-truth input logs from human play sessions for BC label verification.

---

## API Reference

### Health

```
GET /health
-> { status: "ok", obs_dim: 18, act_dim: 16 }
```

### Policy (PPO)

```
POST /policy/infer    { obs: float[18] }
-> { action: int, log_prob: float, logits: float[16], value: float }

POST /rl/store        { obs, action, reward, log_prob, done }
POST /rl/train        -> { loss_policy, loss_value, entropy, updates }
POST /rl/save         -> { saved: true }
POST /rl/load         -> { loaded: bool }
GET  /rl/stats        -> { total_steps, total_updates, episode_count, mean_reward, last_loss_* }
GET  /rl/buffer_size  -> { size: int }
```

### BTR Agent

```
POST /btr/action      { obs: float[18] }   -> { action: int, q_value: float }
POST /btr/store       { obs, action, reward, next_obs, done }
GET  /btr/stats       -> { steps, mean_q, loss, buffer_size, epsilon }
POST /btr/pretrain    -> { transitions_used: int, loss: float }
POST /btr/save        -> { saved: true }
POST /btr/load        -> { loaded: bool }
```

### Replay

```
GET  /replay/scan         -> ReplayFileMeta[]
POST /replay/ingest       { path: string }  -> { transitions: int, stage, game_version }
POST /replay/ingest_all   -> { files: int, transitions: int, errors: int }
```

### Memory

```
GET  /memory/info         -> { attached: bool, pid: int, cached_addrs: int }
GET  /memory/state        -> MemoryState (18-float obs + named fields)
POST /memory/rescan       -> { scanned: bool, addrs_found: int }
```

### Brain

```
GET  /brain/info          -> { knowledge_terms: int, corpus_size: int, registry_size: int }
POST /brain/save          -> { saved: true }
POST /brain/reload        -> { reloaded: true }
POST /brain/clear         -> { cleared: true }
```

### Video

```
POST /video/ingest        { url: string }
GET  /video/status        -> { stage, frames, transitions, progress_pct, eta_s }
POST /video/pretrain      -> { transitions_used: int, loss: float }
```

### Input

```
POST /input/tap           { vk?: int, key?: string, hold_s?: float }
POST /input/macro         { name: string }
GET  /input/macros        -> { macros: string[] }
```

### Strategy

```
POST /strategy/rank             { p1: PlayerState, p2: PlayerState }
-> { strategies: Strategy[], top: Strategy }

POST /strategy/predict_landing  { state: PlayerState }
-> { landing_x, landing_y, time_s }

POST /strategy/projectile_lead  { shooter, target, proj_speed }
-> { angle_deg, intercept_x, intercept_y }
```

---

## Configuration

| Setting | File | Default | Description |
|---------|------|---------|-------------|
| Replay directory | `replay_engine.py` | `C:\Users\carli\BrawlhallaReplays` | Primary replay path; falls back to `%APPDATA%\Brawlhalla\replays` |
| Max corpus size | `brain_store.py` | 50,000 | Maximum BC transitions kept on disk |
| PPO learning rate | `rl_agent.py` | 3e-4 | Adam LR for both policy and value networks |
| PPO epochs | `rl_agent.py` | 4 | Gradient passes per collected batch |
| BTR buffer size | `rl_agent.py` | 100,000 | PER replay buffer capacity |
| Training tick rate | `training_loop.py` | 0.12s | Seconds between game ticks (~8/sec) |
| Train frequency | `training_loop.py` | Every 64 steps | PPO update interval |
| Save frequency | `training_loop.py` | Every 512 steps | Model checkpoint interval |
| Server port | `start_server.ps1` | 8000 | FastAPI uvicorn port |

---

## Project Structure

```
Striker-The-Enlightened/
+-- start_server.ps1              # One-click server launcher
+-- weaponized_ai/
|   +-- api_server.py             # FastAPI: all REST endpoints
|   +-- rl_agent.py               # PPO + BTR agents (PyTorch)
|   +-- replay_engine.py          # .replay binary parser + BC extractor
|   +-- brawlhalla_memory.py      # ReadProcessMemory + AoB scan
|   +-- brain_store.py            # knowledge.json / corpus.npz / registry.json
|   +-- video_learner.py          # YouTube -> BC transitions (yt-dlp + Whisper)
|   +-- strategy_engine.py        # Physics prediction + strategy ranking
|   +-- input_controller.py       # Win32 SendInput macro engine
|   +-- training_loop.py          # Autonomous RL game-play driver
|   +-- game_state_reader.py      # Frame -> obs vector (mss / OBS)
|   +-- obs_capture.py            # Screen capture helpers
|   +-- obs_manager.py            # OBS WebSocket integration
|   +-- __init__.py
+-- src/                          # React + TypeScript frontend
|   +-- WeaponizedAPI.ts          # Typed HTTP client for all endpoints
|   +-- FightTab.tsx              # Main AI control UI (6 panels)
|   +-- App.tsx / main.tsx        # Tauri app entry points
|   +-- CaptureTab.tsx            # Screen capture configuration
|   +-- IntelTab.tsx              # Game intel + opponent profiling
|   +-- SimulationTab.tsx         # Offline simulation runner
|   +-- StrategyTab.tsx           # Strategy engine visualizer
|   +-- BackendBridge.tsx         # Tauri <-> backend health monitor
|   +-- LatencyMonitor.tsx        # API round-trip latency display
|   +-- spatialSafety.ts          # Blastzone boundary safety checks
+-- src-tauri/                    # Tauri native shell
|   +-- src/main.rs               # Tauri entry point
|   +-- tauri.conf.json           # Window/app configuration
+-- tools/
|   +-- smoke_test_replay.py      # Full end-to-end replay validation
|   +-- shared_mem/               # C++ shared memory IPC bridge
|   |   +-- shm_writer.cpp
|   |   +-- shm_reader.cpp
|   |   +-- shm_client.ts
|   +-- input_replay/             # C++ input record/replay tool
|       +-- replay.cpp
+-- public/                       # Static assets
```

---

## Requirements

**Python backend:**

```
fastapi>=0.100
uvicorn[standard]
torch>=2.0
numpy
pydantic>=2.0
yt-dlp
opencv-python
openai-whisper      # optional -- enables audio transcription
```

**Node / Tauri frontend:**

```
node >= 18
npm  >= 9
@tauri-apps/cli    # bundled via package.json
vite + react + typescript
```

**System:**

- Windows 10/11 (required for ReadProcessMemory and SendInput)
- Brawlhalla installed and accessible by path
- `.venv` Python virtual environment at project root
