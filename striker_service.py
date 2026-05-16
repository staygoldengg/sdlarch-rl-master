# striker_service.py
"""
Striker The Enlightened — background service entry point.

Bootstraps the persistent configuration store, elevates process priority,
and keeps the service alive until SIGINT (Ctrl+C).

Usage:
    python striker_service.py

To attach a live training loop, uncomment the HighFidelityTrainingLoop block
below and supply the required reader / policy / controller instances.
"""

import sys
import time

from weaponized_ai.config_manager import PersistentStorageEngine
from weaponized_ai.process_utils import HighPriorityExecutionShield


def main():
    print("[SERVICE] Initialising Striker Core subsystems...")

    # Persistent user configuration
    storage = PersistentStorageEngine()
    print(f"[SERVICE] Loaded settings: {storage.settings}")

    # Kernel priority elevation
    HighPriorityExecutionShield.claim_cpu_dominance()

    print("[SERVICE] Striker Core background service is fully operational.")

    # ── Optional: attach live RL loop ─────────────────────────────────────────
    # from weaponized_ai.training_loop import HighFidelityTrainingLoop
    # loop_engine = HighFidelityTrainingLoop(
    #     shm_reader=...,
    #     policy_network=...,
    #     controller=...,
    #     target_fps=storage.settings["target_fps"],
    # )
    # loop_engine.start()
    # ─────────────────────────────────────────────────────────────────────────

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[SERVICE] Silent termination signal processed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
