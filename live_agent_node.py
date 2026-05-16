# live_agent_node.py
"""
live_agent_node — real-time inference and input injection entry point.

Connects ZeroCopyVisionEngine (screen capture) → SemanticKinematicExtractor
(YOLO entity coordinates) → policy network → FrameDeterministicDispatcher
(nanosecond-precise key injection) in a tight 1 ms poll loop.

Usage:
    from weaponized_ai.policy_network import EnlightenedPolicyNetwork
    policy = EnlightenedPolicyNetwork(obs_dim=9, ...)
    run_agent_live_loop(policy)
"""

import time

import numpy as np
import torch

from weaponized_ai.hardware_driver import FrameDeterministicDispatcher
from weaponized_ai.cuda_vision import ZeroCopyVisionEngine
from weaponized_ai.semantic_extractor import SemanticKinematicExtractor


def run_agent_live_loop(policy_network: torch.nn.Module):
    """
    Starts the live inference loop.

    Args:
        policy_network: An EnlightenedPolicyNetwork (or compatible module)
                        already moved to CUDA.  The model is called with a
                        (1, feature_dim) float32 CUDA tensor and must return
                        (List[Distribution], value_tensor).
    """
    dispatcher = FrameDeterministicDispatcher(target_fps=60)
    vision     = ZeroCopyVisionEngine()
    extractor  = SemanticKinematicExtractor()
    dispatcher.start()

    print("[AGENT] Live inference node running — press Ctrl+C to exit.")

    try:
        while True:
            # ── Vision capture ─────────────────────────────────────────────
            frame_gpu = vision.capture_next_nn_input()
            if frame_gpu is None:
                continue

            # ── Entity detection ───────────────────────────────────────────
            features = extractor.extract_frame_coordinates(frame_gpu)

            # ── Zero-state guard — skip inference on empty frames ──────────
            if np.sum(np.abs(features)) == 0.0:
                dispatcher.emergency_release()
                time.sleep(0.001)
                continue

            # ── Policy inference ───────────────────────────────────────────
            feat_tensor = torch.from_numpy(features).unsqueeze(0).cuda().float()
            with torch.inference_mode():
                action_distributions, _value = policy_network(feat_tensor)
                move_x  = action_distributions[0].argmax().item()  # 0=Neutral 1=Left 2=Right
                move_y  = action_distributions[1].argmax().item()  # 0=Neutral 1=Up   2=Down
                action  = action_distributions[2].argmax().item()  # 0-4

            # ── Build key-state map ────────────────────────────────────────
            action_payload: dict[str, bool] = {
                "LEFT":   move_x == 1,
                "RIGHT":  move_x == 2,
                "UP":     move_y == 1,
                "DOWN":   move_y == 2,
                "LIGHT":  action == 1,
                "HEAVY":  action == 2,
                "DODGE":  action == 3,
                "JUMP":   action == 4,
            }
            dispatcher.stage_action_map(action_payload)

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("[AGENT] Shutdown signal received — cleaning up.")
    finally:
        dispatcher.emergency_release()
        vision.shutdown()
        print("[AGENT] Node exited cleanly.")


if __name__ == "__main__":
    # Standalone smoke-test (no policy passed — exits immediately)
    pass
