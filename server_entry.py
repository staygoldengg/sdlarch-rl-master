"""
server_entry.py
---------------
PyInstaller entry point for the Striker AI backend server.

When frozen into striker-server.exe, this script:
  1. Resolves the project root (next to the exe OR the real cwd in dev mode).
  2. Ensures weaponized_ai package is importable.
  3. Starts uvicorn on 127.0.0.1:8000 with the FastAPI app.

Data written by the AI (brain store, model checkpoints) goes into:
  <exe_dir>\data\    (release build)
  weaponized_ai\brain\  (dev / source run)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Path bootstrap ─────────────────────────────────────────────────────────────
# When running as a PyInstaller frozen exe, sys._MEIPASS is the temp dir
# containing unpacked modules. We need our project root on sys.path so that
# `import weaponized_ai` works whether frozen or not.
if getattr(sys, "frozen", False):
    # Running from striker-server.exe
    _EXE_DIR = Path(sys.executable).parent
    # Add the frozen bundle root so weaponized_ai is found
    sys.path.insert(0, str(sys._MEIPASS))  # type: ignore[attr-defined]
    # Redirect brain_store to a writable data dir next to the exe
    os.environ.setdefault("STRIKER_DATA_DIR", str(_EXE_DIR / "data"))
else:
    # Running directly from source
    _EXE_DIR = Path(__file__).parent
    sys.path.insert(0, str(_EXE_DIR))

# ── Brain store data-dir override ──────────────────────────────────────────────
# brain_store.py reads STRIKER_DATA_DIR when set, falling back to the default.
# This keeps model files and knowledge base next to the exe (writable by user).
_data_dir = os.environ.get("STRIKER_DATA_DIR")
if _data_dir:
    Path(_data_dir).mkdir(parents=True, exist_ok=True)

# ── Start server ───────────────────────────────────────────────────────────────
import uvicorn  # noqa: E402 — must come after path bootstrap

if __name__ == "__main__":
    uvicorn.run(
        "weaponized_ai.api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,         # no reload in frozen builds
        log_level="info",
        workers=1,
    )
