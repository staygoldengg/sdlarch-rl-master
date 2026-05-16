# striker_service.py
"""
Striker The Enlightened — background service entry point.

Responsibilities:
  - Bootstrap the persistent configuration store.
  - Elevate process scheduling priority.
  - Watchdog the FastAPI backend: if it dies, restart it automatically
    (up to MAX_BACKEND_RESTARTS, with exponential back-off).
  - Expose a clean Ctrl-C / SIGTERM shutdown path.

Usage:
    python striker_service.py

To attach a live training loop, uncomment the HighFidelityTrainingLoop block
near the bottom of _run_service() and supply the required instances.
"""

import logging
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from weaponized_ai.config_manager import PersistentStorageEngine
from weaponized_ai.process_utils import HighPriorityExecutionShield

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("striker_service")

# ── Watchdog tunables ─────────────────────────────────────────────────────────
_SERVER_HOST           = "127.0.0.1"
_SERVER_PORT           = 8000
_POLL_INTERVAL_S       = 2.0      # health-check cadence
_RESTART_BACK_OFF_BASE = 2.0      # seconds; doubles on each consecutive failure
_MAX_BACKEND_RESTARTS  = 10       # give up after this many crashes per session

# Path resolution: prefer .venv python, fall back to system python
_PROJECT_ROOT = Path(__file__).parent
_VENV_PYTHON  = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
_PYTHON_EXE   = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable


def _port_alive(host: str = _SERVER_HOST, port: int = _SERVER_PORT) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _launch_backend() -> subprocess.Popen:
    """Spawn the FastAPI server as a hidden child process."""
    entry = str(_PROJECT_ROOT / "server_entry.py")
    proc = subprocess.Popen(
        [_PYTHON_EXE, entry],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=0x08000000,  # CREATE_NO_WINDOW
    )
    log.info("Backend spawned (PID %d).", proc.pid)
    return proc


def _run_service() -> None:
    storage = PersistentStorageEngine()
    log.info("Settings loaded: %s", storage.settings)

    HighPriorityExecutionShield.claim_cpu_dominance()

    # ── Shutdown flag (shared between main loop and signal handler) ───────────
    _shutdown = [False]

    def _on_signal(sig, _frame):
        log.info("Signal %s received — initiating clean shutdown.", sig)
        _shutdown[0] = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT,  _on_signal)

    backend_proc:   subprocess.Popen | None = None
    restart_count   = 0
    back_off        = _RESTART_BACK_OFF_BASE

    log.info("Striker Core background service is fully operational.")

    while not _shutdown[0]:
        # ── Ensure backend is alive ───────────────────────────────────────────
        if not _port_alive():
            if backend_proc is not None:
                ret = backend_proc.poll()
                if ret is not None:
                    log.warning("Backend exited (code %d) — restarting...", ret)
                    backend_proc = None

            if backend_proc is None:
                if restart_count >= _MAX_BACKEND_RESTARTS:
                    log.critical(
                        "Backend crashed %d times — giving up. "
                        "Check server_crash.log for details.",
                        restart_count,
                    )
                    break

                if restart_count > 0:
                    log.info("Back-off: waiting %.1f s before restart.", back_off)
                    time.sleep(back_off)
                    back_off = min(back_off * 2.0, 60.0)
                else:
                    back_off = _RESTART_BACK_OFF_BASE

                try:
                    backend_proc = _launch_backend()
                    restart_count += 1
                    # Give it a moment to bind the port
                    time.sleep(1.5)
                except Exception as exc:
                    log.error("Failed to spawn backend: %s", exc)
        else:
            # Backend is healthy — reset back-off
            if restart_count > 0:
                log.info("Backend is healthy. Resetting restart counter.")
                restart_count = 0
                back_off = _RESTART_BACK_OFF_BASE

        # ── Optional: attach live RL loop ─────────────────────────────────────
        # Uncomment the block below and supply the required instances to run
        # the high-fidelity 60 Hz training loop from this service.
        #
        # from weaponized_ai.training_loop import HighFidelityTrainingLoop
        # loop_engine = HighFidelityTrainingLoop(
        #     shm_reader=...,
        #     policy_network=...,
        #     controller=...,
        #     target_fps=storage.settings["target_fps"],
        # )
        # loop_engine.start()
        # ─────────────────────────────────────────────────────────────────────

        time.sleep(_POLL_INTERVAL_S)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if backend_proc is not None:
        log.info("Terminating backend (PID %d)...", backend_proc.pid)
        backend_proc.terminate()
        try:
            backend_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            backend_proc.kill()

    log.info("Striker service shut down cleanly.")


if __name__ == "__main__":
    _run_service()
