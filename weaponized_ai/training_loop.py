# training_loop.py
"""
Autonomous RL training loop for Brawlhalla.

Pipeline every tick:
  1. Grab frame via OBS/mss (auto-detect Brawlhalla window)
  2. Read game state (damage, positions, weapons, KO) from frame
  3. Estimate velocity via frame-diff of positions
  4. Ask PPO policy for action
  5. Execute action via input controller
  6. Compute rich reward (damage dealt/taken, KO, weapon pickup)
  7. Store experience; every TRAIN_EVERY steps -> PPO update + save
"""

import threading
import time
import logging
from typing import Optional, List

from weaponized_ai.rl_agent import get_agent
from weaponized_ai.input_controller import execute_macro

log = logging.getLogger("training_loop")

# ── Action index -> macro name ─────────────────────────────────────────────────
ACTION_MACROS = [
    "nlight", "slight", "dlight",
    "nheavy", "sheavy", "dheavy",
    "jump",   "jump",   "jump",
    "jump",   "jump",
    "dodge",  "dash_right", "dash_left",
    "nlight", "nheavy",
]


class TrainingLoop:
    TICK_S      = 0.12    # ~8 ticks/sec — faster but still readable
    TRAIN_EVERY = 64      # bigger batch for GAE
    SAVE_EVERY  = 512

    def __init__(
        self,
        capture_mode: str = "mss",
        capture_region: Optional[dict] = None,
        obs_camera_index: int = 1,
        tick_s: float = TICK_S,
        auto_find_window: bool = True,
    ):
        self.capture_mode      = capture_mode
        self.capture_region    = capture_region
        self.obs_camera_idx    = obs_camera_index
        self.tick_s            = tick_s
        self.auto_find_window  = auto_find_window

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Live status
        self.running     = False
        self.step_count  = 0
        self.last_reward = 0.0
        self.last_action = ""
        self.last_loss_p = 0.0
        self.last_loss_v = 0.0
        self.last_entropy = 0.0
        self.last_state: Optional[dict] = None
        self.errors: List[str] = []
        # Episode stats
        self.episode_count     = 0
        self.current_ep_reward = 0.0
        self.best_ep_reward    = float("-inf")
        self.mean_ep_reward    = 0.0
        self._ep_reward_history: List[float] = []

    def start(self):
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="TrainingLoop")
        self._thread.start()
        self.running = True
        log.info("Training loop started.")

    def stop(self):
        self._stop_event.set()
        self.running = False
        log.info("Training loop stopping…")

    def status(self) -> dict:
        s = self.last_state or {}
        agent_stats = get_agent().stats()
        return {
            "running":     self.running,
            "step_count":  self.step_count,
            "last_reward": round(self.last_reward, 3),
            "last_action": self.last_action,
            "loss_policy": round(self.last_loss_p, 5),
            "loss_value":  round(self.last_loss_v, 5),
            "entropy":     round(self.last_entropy, 4),
            # Episode stats
            "episode_count":     self.episode_count,
            "current_ep_reward": round(self.current_ep_reward, 2),
            "best_ep_reward":    round(self.best_ep_reward, 2) if self.best_ep_reward != float("-inf") else 0.0,
            "mean_ep_reward":    round(self.mean_ep_reward, 2),
            # Agent lifetime stats
            "total_updates":     agent_stats["total_updates"],
            # Game state
            "last_p1_damage":      s.get("p1_damage", 0),
            "last_p2_damage":      s.get("p2_damage", 0),
            "last_p1_stocks":      s.get("p1_stocks", 3),
            "last_p2_stocks":      s.get("p2_stocks", 3),
            "last_ko_flash":       s.get("ko_flash", False),
            "last_p1_damage_tier": s.get("p1_damage_tier", "white"),
            "last_p2_damage_tier": s.get("p2_damage_tier", "white"),
            "last_p1_weapon":      s.get("p1_weapon", "none"),
            "last_p2_weapon":      s.get("p2_weapon", "none"),
            "last_stage_pickups":  s.get("stage_pickups", 0),
            "errors": self.errors[-5:],
        }

    def _inject_velocity(self, curr: dict, prev: Optional[dict], tick_s: float) -> List[float]:
        """
        Replace the placeholder zeros in obs[2,3,9,10] with estimated velocities
        computed from frame-diff of screen positions, scaled to game units.
        """
        obs = list(curr["obs"])
        if prev is None:
            return obs
        try:
            p1_prev = prev["p1_pos"]
            p2_prev = prev["p2_pos"]
            p1_curr = curr["p1_pos"]
            p2_curr = curr["p2_pos"]
            # pixels/tick -> approximate game units (rough factor 600 for 1920px wide)
            scale = 600.0 / tick_s
            obs[2] = (p1_curr[0] - p1_prev[0]) * scale
            obs[3] = (p1_curr[1] - p1_prev[1]) * scale
            obs[9] = (p2_curr[0] - p2_prev[0]) * scale
            obs[10]= (p2_curr[1] - p2_prev[1]) * scale
        except Exception:
            pass
        return obs

    def _run(self):
        from weaponized_ai.obs_capture import ScreenCapture
        from weaponized_ai.game_state_reader import read_state, compute_reward
        from weaponized_ai.obs_manager import find_brawlhalla_window

        agent = get_agent()

        # Auto-detect Brawlhalla window region
        region = self.capture_region
        if self.auto_find_window and region is None:
            found = find_brawlhalla_window()
            if found:
                region = found
                log.info(f"Brawlhalla window found: {region}")
            else:
                log.info("Brawlhalla window not found, using full screen.")

        try:
            cap = ScreenCapture(
                mode=self.capture_mode,
                region=region,
                obs_camera_index=self.obs_camera_idx,
            )
        except Exception as e:
            self.errors.append(f"Capture init failed: {e}")
            self.running = False
            return

        prev_state: Optional[dict] = None

        try:
            while not self._stop_event.is_set():
                tick_start = time.perf_counter()

                try:
                    frame = cap.grab()
                    curr_state = read_state(frame)
                    self.last_state = curr_state

                    # Inject frame-diff velocity into obs vector
                    obs = self._inject_velocity(curr_state, prev_state, self.tick_s)

                    reward = 0.0
                    done   = False
                    if prev_state is not None:
                        reward = compute_reward(prev_state, curr_state)
                        done = (
                            curr_state["p1_stocks"] != prev_state["p1_stocks"] or
                            curr_state["p2_stocks"] != prev_state["p2_stocks"]
                        )

                    action, log_prob = agent.select_action(obs)
                    macro_name = ACTION_MACROS[action % len(ACTION_MACROS)]
                    self.last_action = macro_name

                    execute_macro(macro_name)

                    if prev_state is not None:
                        agent.store(obs, action, reward, log_prob, done)
                        self.last_reward = reward
                        self.current_ep_reward += reward
                        self.step_count += 1

                        if done:
                            self._ep_reward_history.append(self.current_ep_reward)
                            if self.current_ep_reward > self.best_ep_reward:
                                self.best_ep_reward = self.current_ep_reward
                            recent = self._ep_reward_history[-20:]
                            self.mean_ep_reward = sum(recent) / len(recent)
                            log.info(f"[ep {self.episode_count}] reward={self.current_ep_reward:.1f} "
                                     f"mean={self.mean_ep_reward:.1f}")
                            self.episode_count += 1
                            self.current_ep_reward = 0.0

                    prev_state = curr_state

                    if self.step_count > 0 and self.step_count % self.TRAIN_EVERY == 0:
                        result = agent.train_step()
                        self.last_loss_p  = result["loss_policy"]
                        self.last_loss_v  = result["loss_value"]
                        self.last_entropy = result.get("entropy", 0.0)
                        log.info(f"[step {self.step_count}] "
                                 f"policy={self.last_loss_p:.4f} "
                                 f"value={self.last_loss_v:.4f} "
                                 f"entropy={self.last_entropy:.4f}")

                    if self.step_count > 0 and self.step_count % self.SAVE_EVERY == 0:
                        agent.save()
                        log.info(f"[step {self.step_count}] Model saved.")

                except Exception as e:
                    self.errors.append(str(e))
                    log.warning(f"Tick error: {e}")

                elapsed = time.perf_counter() - tick_start
                time.sleep(max(0.0, self.tick_s - elapsed))

        finally:
            cap.close()
            self.running = False
            log.info("Training loop stopped.")


# ── Module-level singleton ────────────────────────────────────────────────────
_loop: Optional[TrainingLoop] = None

def get_loop() -> TrainingLoop:
    global _loop
    if _loop is None:
        _loop = TrainingLoop()
    return _loop



# ─────────────────────────────────────────────────────────────────────────────
# HighFidelityTrainingLoop — Native 60 Hz precision loop
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402


class HighFidelityTrainingLoop:
    """
    High-throughput RL execution loop targeting a strict 60 Hz tick rate.

    Pipeline every frame:
      1. Zero-copy state fetch from the C++ Shared Memory bridge
      2. GPU tensor conversion + policy inference (torch.inference_mode)
      3. Asynchronous input injection via a dedicated ThreadPoolExecutor
      4. Precision sleep to maintain exact frame deadline alignment

    Args:
        agent:      Policy object exposing .policy(tensor) -> (dist, value).
        shm_reader: Object exposing .read_latest_state() -> (ndarray|None, id).
        controller: Object exposing .inject_inputs(action).
        target_fps: Desired loop rate (default 60).
    """

    def __init__(self, agent, shm_reader, controller, target_fps: int = 60):
        self.agent = agent
        self.shm_reader = shm_reader
        self.controller = controller
        self.frame_time = 1.0 / target_fps
        self.is_running = False

        # Dedicated pool to offload I/O injection without blocking inference
        self.io_executor = ThreadPoolExecutor(max_workers=1)

    def start(self):
        self.is_running = True
        self._run_loop()

    def stop(self):
        self.is_running = False
        self.io_executor.shutdown(wait=True)

    def _run_loop(self):
        import torch  # deferred to avoid forcing GPU init at module load
        print(
            f"[ENGINE] Initiating native {int(1.0 / self.frame_time)} Hz loop. "
            f"Target delta: {self.frame_time:.5f}s"
        )

        while self.is_running:
            loop_start = time.perf_counter()

            # 1. Zero-copy fetch from Shared Memory bridge
            state_vector, _tracking_id = self.shm_reader.read_latest_state()

            if state_vector is None:
                # Pacing fallback if SHM reader is not yet synchronised
                time.sleep(0.001)
                continue

            # 1b. Zero-state safety lock — skip inference on loading/corrupted frames
            if np.sum(np.abs(state_vector)) == 0.0:
                try:
                    self.controller.global_flush()
                except AttributeError:
                    pass
                time.sleep(0.1)
                continue

            # 2. Tensor conversion + inference
            # torch.inference_mode() selects the optimal C++ backend path
            with torch.inference_mode():
                state_tensor = torch.from_numpy(state_vector).unsqueeze(0).cuda()
                action_distribution, _value_pred = self.agent.policy(state_tensor)
                action = action_distribution.sample().cpu().numpy()[0]

            # 3. Asynchronous input injection — fire-and-forget
            self.io_executor.submit(self.controller.inject_inputs, action)

            # 4. Precision timing realignment
            elapsed = time.perf_counter() - loop_start
            sleep_time = self.frame_time - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(
                    f"[WARN] Frame deadline exceeded by "
                    f"{-sleep_time * 1000:.2f}ms — inference lag detected."
                )
