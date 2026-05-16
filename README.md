# Striker — The Enlightened

An autonomous AI training suite for Brawlhalla built on a full reinforcement-learning stack: PPO + Beyond-The-Rainbow agents, direct process-memory reading, binary replay digestion, YouTube knowledge ingestion, real-time strategy analysis, and a React/Tauri desktop interface — all wired together through a FastAPI backend.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Quick Start](#quick-start)
3. [Backend Components](#backend-components)
   - [API Server](#api-server)
   - [RL Agent (PPO)](#rl-agent-ppo)
   - [BTR Agent](#btr-agent)
   - [Replay Engine](#replay-engine)
   - [Brawlhalla Memory Reader](#brawlhalla-memory-reader)
   - [Brain Store](#brain-store)
   - [Video Learner](#video-learner)
   - [Strategy Engine](#strategy-engine)
   - [Input Controller](#input-controller)
   - [Training Loop](#training-loop)
4. [Frontend (React + Tauri)](#frontend-react--tauri)
   - [WeaponizedAPI.ts](#weaponizedapits)
   - [FightTab.tsx Panels](#fighttabtsx-panels)
5. [Tools](#tools)
   - [smoke_test_replay.py](#smoke_test_replaypy)
   - [Shared Memory Bridge](#shared-memory-bridge)
   - [Input Replay (C++)](#input-replay-c)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Project Structure](#project-structure)
9. [Requirements](#requirements)

---

## Architecture Overview

```
+-------------------------------------------------------------+
|                   Tauri Desktop App (port 1420)             |
|   React + TypeScript UI <-> WeaponizedAPI.ts HTTP client    |
+------------------------+------------------------------------+
                         | REST/JSON  http://127.0.0.1:8000
+------------------------v------------------------------------+
|                FastAPI Backend (api_server.py)              |
|  /policy  /rl  /btr  /replay  /memory  /brain  /video      |
|  /input   /strategy  /obs  /health                         |
+--+--------+----------+----------+----------+---------------+
   |        |          |          |          |
   v        v          v          v          v
PPO      BTR        Replay     Memory     Brain
Agent   Agent       Engine     Reader     Store
(rl_    (rl_       (replay_   (brawlhalla (brain_
agent)  agent)     engine)    _memory)    store)
                                |          |
                                v          v
                          Brawlhalla   YouTube
                          .exe RAM     Videos
                          (live)     (yt-dlp +
                                      Whisper)
```

The backend is a single FastAPI process that coordinates all AI subsystems. The Tauri app ships the React UI as a native Windows desktop window and communicates with the backend over localhost HTTP. The game itself is never patched — all data is read via Windows `ReadProcessMemory` and all inputs are injected via `SendInput`.

---

## Quick Start

### Prerequisites

- Python 3.10+ with `.venv` at the project root
- Node.js 18+ (for the Tauri/React frontend)
- Rust toolchain (for Tauri native build)
- Brawlhalla installed via Steam

### 1. Start the AI backend

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

---

## Backend Components

### API Server

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
