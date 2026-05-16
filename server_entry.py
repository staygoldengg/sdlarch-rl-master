"""
server_entry.py
---------------
PyInstaller entry point for the Striker AI backend server.

When frozen into striker-server.exe, this script:
  1. Resolves the project root (next to the exe OR the real cwd in dev mode).
  2. Ensures weaponized_ai package is importable.
  3. Checks that port 8000 is not already bound (prevents double-launch).
  4. Starts uvicorn on 127.0.0.1:8000 with the FastAPI app.
  5. On any crash, logs a timestamped error to <data_dir>/server_crash.log
     and exits with code 1 so Tauri / the watchdog can restart it.

Data written by the AI (brain store, model checkpoints) goes into:
  <exe_dir>\\data\\    (release build)
  weaponized_ai\\brain\\  (dev / source run)
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import traceback
from datetime import datetime
from pathlib import Path

# ── Path bootstrap ─────────────────────────────────────────────────────────────
if getattr(sys, "frozen", False):
    _EXE_DIR = Path(sys.executable).parent
    sys.path.insert(0, str(sys._MEIPASS))  # type: ignore[attr-defined]
    os.environ.setdefault("STRIKER_DATA_DIR", str(_EXE_DIR / "data"))
else:
    _EXE_DIR = Path(__file__).parent
    sys.path.insert(0, str(_EXE_DIR))

# ── Brain store data-dir override ─────────────────────────────────────────────
_data_dir = Path(os.environ.get("STRIKER_DATA_DIR", str(_EXE_DIR / "weaponized_ai" / "brain")))
_data_dir.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_data_dir / "server.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("server_entry")


def _port_in_use(port: int = 8000) -> bool:
    """Return True if something is already listening on 127.0.0.1:<port>."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def _write_crash_log(exc: BaseException) -> None:
    crash_path = _data_dir / "server_crash.log"
    try:
        with open(crash_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"CRASH @ {datetime.now().isoformat()}\n")
            f.write(traceback.format_exc())
    except Exception:
        pass  # Never let the crash logger itself crash


# ── Start server ───────────────────────────────────────────────────────────────
import uvicorn  # noqa: E402 — must come after path bootstrap

if __name__ == "__main__":
    if _port_in_use(8000):
        log.info("Port 8000 already bound — server already running, exiting.")
        sys.exit(0)

    log.info("Striker AI backend starting on http://127.0.0.1:8000")
    log.info("Data directory: %s", _data_dir)

    try:
        uvicorn.run(
            "weaponized_ai.api_server:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info",
            workers=1,
            access_log=True,
        )
    except Exception as exc:
        log.critical("Striker backend crashed: %s", exc, exc_info=True)
        _write_crash_log(exc)
        sys.exit(1)
