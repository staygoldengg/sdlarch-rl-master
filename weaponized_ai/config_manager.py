# config_manager.py
"""
PersistentStorageEngine — JSON-backed settings store for Striker.

Settings are persisted in %APPDATA%\\StrikerEnlightened\\user_settings.json
and survive across sessions.  Any missing key falls back to the built-in
defaults defined below.
"""

import os
import json
from pathlib import Path


class PersistentStorageEngine:
    DEFAULTS: dict = {
        "target_fps": 60,
        "exploration_entropy_ratio": 0.25,
        "kl_divergence_tether": 0.01,
        "auto_start_on_game_launch": True,
        "emergency_panic_key": "ESCAPE",
    }

    def __init__(self, app_name: str = "StrikerEnlightened"):
        # Resolve %APPDATA% safely; fall back to home directory
        self.base_dir = Path(os.getenv("APPDATA") or Path.home()) / app_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.base_dir / "user_settings.json"

        self.settings: dict = dict(self.DEFAULTS)
        self.load_settings()

    def load_settings(self):
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self.settings.update(loaded)
            except Exception as exc:
                print(
                    f"[STORAGE] Failed reading configuration — reverting to defaults. "
                    f"Error: {exc}"
                )

    def save_settings(self, new_settings: dict):
        self.settings.update(new_settings)
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4)
            print(f"[STORAGE] Configuration written to {self.config_file}")
        except Exception as exc:
            print(f"[CRITICAL] Failed writing configuration: {exc}")

    def get(self, key: str, default=None):
        return self.settings.get(key, default)
