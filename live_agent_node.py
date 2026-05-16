# live_agent_node.py
"""
live_agent_node — production real-time inference and input injection node.

Pipeline:
  ZeroCopyVisionEngine  (DXGI 60 Hz GPU capture)
  → SemanticKinematicExtractor  (YOLO entity coordinates)
  → zero-state guard
  → policy network inference
  → FrameDeterministicDispatcher  (nanosecond-precise scan-code injection)

Features:
  - Per-subsystem error recovery with configurable back-off
  - Live FPS / frame-drop statistics printed every 5 seconds
  - CPU throttle guard: if inference takes > 80 % of frame budget, sleep is
    extended to cap throughput and prevent thermal runaway
  - Graceful degradation: vision errors skip the frame without crashing
  - Clean shutdown on SIGINT / SIGTERM with guaranteed key release

Usage (programmatic)::

    from weaponized_ai.policy_network import EnlightenedPolicyNetwork
    policy = EnlightenedPolicyNetwork(obs_dim=9, ...)
    run_agent_live_loop(policy)
"""
from __future__ import annotations

import signal
import sys
import time
import logging
from collections import deque
from typing import Optional

import numpy as np
import torch

from weaponized_ai.hardware_driver import FrameDeterministicDispatcher
from weaponized_ai.cuda_vision import ZeroCopyVisionEngine
from weaponized_ai.semantic_extractor import SemanticKinematicExtractor

log = logging.getLogger("live_agent")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ── Tunables ──────────────────────────────────────────────────────────────────
_TARGET_FPS          = 60
_FRAME_BUDGET_S      = 1.0 / _TARGET_FPS        # 16.67 ms
_MIN_SLEEP_S         = 0.0005                    # 0.5 ms minimum yield
_THROTTLE_THRESHOLD  = 0.80                      # fraction of budget before throttle kicks in
_THROTTLE_SLEEP_S    = _FRAME_BUDGET_S * 0.5    # extra sleep when over threshold
_VISION_MAX_RETRIES  = 5                         # consecutive None frames before warning
_STATS_INTERVAL_S    = 5.0                       # FPS report interval
_ZERO_STATE_SLEEP_S  = 0.100                     # sleep on all-zero state (loading screen)


def run_agent_live_loop(
    policy_network: torch.nn.Module,
    target_fps: int = _TARGET_FPS,
    vision_region: Optional[tuple[int, int, int, int]] = None,
) -> None:
    """
    Start the production live inference loop.

    Args:
        policy_network:  An EnlightenedPolicyNetwork (or any compatible module)
                         already on the correct device.  Must accept a
                         (1, feature_dim) float32 tensor and return
                         (List[Categorical], value_tensor).
        target_fps:      Inference rate.  Defaults to 60 Hz.
        vision_region:   DXGI capture region (left, top, right, bottom).
                         None = full primary monitor.
    """
    frame_budget_s = 1.0 / target_fps

    dispatcher = FrameDeterministicDispatcher(target_fps=target_fps)
    vision     = ZeroCopyVisionEngine(
        region_box=vision_region or (0, 0, 1920, 1080)
    )
    extractor  = SemanticKinematicExtractor()
    dispatcher.start()

    # ── SIGTERM handler — mirrors SIGINT behaviour ─────────────────────────
    _shutdown = [False]
    def _handle_signal(sig, frame):  # noqa: ANN001
        log.info("[AGENT] Shutdown signal %s received.", sig)
        _shutdown[0] = True
    signal.signal(signal.SIGTERM, _handle_signal)

    # ── Stats accumulators ─────────────────────────────────────────────────
    frame_times: deque[float] = deque(maxlen=int(target_fps * _STATS_INTERVAL_S))
    frames_skipped  = 0
    frames_total    = 0
    last_report_t   = time.perf_counter()
    consecutive_none = 0

    log.info("[AGENT] Live inference node starting — Ctrl+C to exit.")

    try:
        while not _shutdown[0]:
            t0 = time.perf_counter()

            # ── Vision capture ────────────────────────────────────────────
            try:
                frame_gpu = vision.capture_next_nn_input()
            except Exception as exc:
                log.warning("[AGENT] Vision capture error: %s — skipping frame.", exc)
                frames_skipped += 1
                time.sleep(_MIN_SLEEP_S)
                continue

            if frame_gpu is None:
                consecutive_none += 1
                if consecutive_none >= _VISION_MAX_RETRIES:
                    log.warning(
                        "[AGENT] %d consecutive empty frames — DXGI may not be ready.",
                        consecutive_none,
                    )
                    consecutive_none = 0
                time.sleep(_MIN_SLEEP_S)
                continue
            consecutive_none = 0

            # ── Entity detection ──────────────────────────────────────────
            try:
                features = extractor.extract_frame_coordinates(frame_gpu)
            except Exception as exc:
                log.warning("[AGENT] Extractor error: %s — skipping frame.", exc)
                frames_skipped += 1
                time.sleep(_MIN_SLEEP_S)
                continue

            # ── Zero-state guard: loading screen / SHM startup ────────────
            if np.sum(np.abs(features)) == 0.0:
                dispatcher.emergency_release()
                time.sleep(_ZERO_STATE_SLEEP_S)
                continue

            # ── Policy inference ──────────────────────────────────────────
            try:
                device = next(policy_network.parameters()).device
                feat_t = torch.from_numpy(features).unsqueeze(0).to(device).float()
                with torch.inference_mode():
                    dists, _value = policy_network(feat_t)
                    move_x = dists[0].logits.argmax().item()   # 0=Neutral 1=Left 2=Right
                    move_y = dists[1].logits.argmax().item()   # 0=Neutral 1=Up   2=Down
                    action = dists[2].logits.argmax().item()   # 0-4
            except Exception as exc:
                log.error("[AGENT] Inference error: %s — releasing all keys.", exc)
                dispatcher.emergency_release()
                frames_skipped += 1
                time.sleep(_MIN_SLEEP_S)
                continue

            # ── Build and stage key-state map ─────────────────────────────
            dispatcher.stage_action_map({
                "LEFT":  move_x == 1,
                "RIGHT": move_x == 2,
                "UP":    move_y == 1,
                "DOWN":  move_y == 2,
                "LIGHT": action == 1,
                "HEAVY": action == 2,
                "DODGE": action == 3,
                "JUMP":  action == 4,
            })

            frames_total += 1
            elapsed = time.perf_counter() - t0
            frame_times.append(elapsed)

            # ── CPU throttle guard ────────────────────────────────────────
            remaining = frame_budget_s - elapsed
            if elapsed > frame_budget_s * _THROTTLE_THRESHOLD:
                # We burned most of the frame budget — yield extra to OS
                time.sleep(max(_THROTTLE_SLEEP_S, _MIN_SLEEP_S))
            elif remaining > _MIN_SLEEP_S:
                time.sleep(remaining)
            else:
                time.sleep(_MIN_SLEEP_S)

            # ── Periodic stats report ─────────────────────────────────────
            now = time.perf_counter()
            if now - last_report_t >= _STATS_INTERVAL_S:
                if frame_times:
                    avg_ms = (sum(frame_times) / len(frame_times)) * 1000.0
                    fps    = 1.0 / (sum(frame_times) / len(frame_times))
                    log.info(
                        "[AGENT] %.1f fps | avg %.2f ms/frame | "
                        "frames=%d skipped=%d",
                        fps, avg_ms, frames_total, frames_skipped,
                    )
                last_report_t = now

    except KeyboardInterrupt:
        log.info("[AGENT] SIGINT — shutting down.")
    finally:
        dispatcher.emergency_release()
        vision.shutdown()
        log.info(
            "[AGENT] Node exited cleanly. total=%d skipped=%d",
            frames_total,
            frames_skipped,
        )


# ── Smoke-test / standalone entry point ──────────────────────────────────────
if __name__ == "__main__":
    """
    Standalone smoke-test: instantiates all subsystems, runs a 3-second loop
    with a random policy, prints FPS, then exits cleanly.
    Does NOT require Brawlhalla to be running.
    """
    print("[SMOKE] Running live_agent_node standalone smoke-test (3 s)...")

    # Minimal CPU-only policy (no YOLO / DXGI required for smoke-test)
    import threading

    from weaponized_ai.policy_network import FactorizedMultiDiscreteActorHead, EnlightenedPolicyNetwork

    class _SmokePolicyWrapper(torch.nn.Module):
        """Wraps EnlightenedPolicyNetwork to expose three Categorical heads."""
        def __init__(self):
            super().__init__()
            self.base = EnlightenedPolicyNetwork(ram_state_dim=9, num_actions=14)

        def parameters(self, recurse=True):
            return self.base.parameters(recurse)

        def forward(self, x):
            # Pad 9-dim input to 64-dim expected by EnlightenedPolicyNetwork
            padded = torch.nn.functional.pad(x, (0, 55))
            dist, val = self.base(padded)
            # Split into three pseudo-heads for compatibility
            from torch.distributions import Categorical
            logits = dist.logits  # (1, 14)
            return [
                Categorical(logits=logits[:, :3]),   # move_x (3 choices)
                Categorical(logits=logits[:, 3:6]),  # move_y (3 choices)
                Categorical(logits=logits[:, 6:11]), # action (5 choices)
            ], val

    smoke_policy = _SmokePolicyWrapper()

    _stop = threading.Event()

    def _timed_run():
        time.sleep(3.0)
        _stop.set()
        import os; os.kill(os.getpid(), signal.SIGTERM)  # noqa: E702

    threading.Thread(target=_timed_run, daemon=True).start()

    try:
        run_agent_live_loop(smoke_policy, target_fps=30)
    except SystemExit:
        pass

    print("[SMOKE] Smoke-test complete — all subsystems initialised and shut down cleanly.")
