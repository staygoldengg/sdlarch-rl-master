# watchdog_reader.py
"""
TelemetryWatchdogMemoryReader — Parses and validates raw bytes from the
Shared Memory bridge before they are fed into the policy network.

Sanity checks applied every frame:
  1. Minimum buffer size check (struct layout verification)
  2. Out-of-bounds coordinate detection (blastzone teleportation / corruption)
  3. Impossible per-frame delta distance (pointer drift / map reload detection)
  4. Stock counter overflow guard (negative-stock wraparound from bad pointer)

If any check fails the reader returns the last validated state so the loop
continues without a training NaN instead of crashing.
"""

import numpy as np
import json
import os


class TelemetryWatchdogMemoryReader:
    # Raw buffer layout offsets (matches BrawlhallaStateBuffer in shm_bridge.cpp)
    # Byte 0-7:   alignment_checksum  (uint64)
    # Byte 8-11:  current_frame       (uint32)
    # Byte 12-15: player_x            (float32)
    # Byte 16-19: player_y            (float32)
    # Byte 20-23: opponent_x          (float32)
    # Byte 24-27: opponent_y          (float32)
    # Byte 28-31: player_damage       (float32)
    # Byte 32-35: opponent_damage     (float32)
    # Byte 36-39: player_stocks       (uint32)
    # Byte 40-43: opponent_stocks     (uint32)
    # Byte 44-235: feature_vector[48] (48 × float32)
    _MIN_BUFFER_SIZE = 236  # 44 + 48 * 4

    def __init__(
        self,
        cache_path: str = "weaponized_ai/_addr_cache.json",
        max_expected_velocity: float = 150.0,
    ):
        self.cache_path = cache_path
        self.max_expected_velocity = max_expected_velocity
        self.last_valid_state: np.ndarray | None = None

    def validate_and_parse(self, raw_buffer_bytes: bytes) -> tuple[np.ndarray, bool]:
        """
        Parses raw data from the Shared Memory buffer and validates integrity.

        Returns:
            (state_vector, is_valid) — state_vector is always non-None
              (falls back to last valid state or zeros on failure).
        """
        if len(raw_buffer_bytes) < self._MIN_BUFFER_SIZE:
            return self._fallback(), False

        # Unpack floats from player_x offset onwards
        extracted_floats = np.frombuffer(raw_buffer_bytes[12:236], dtype=np.float32)

        p_x  = float(extracted_floats[0])
        p_y  = float(extracted_floats[1])

        # Heuristic 1: Out-of-bounds coordinates (blastzone teleportation / corruption)
        if abs(p_x) > 5000.0 or abs(p_y) > 5000.0:
            print("[WATCHDOG] Out-of-bounds coordinate burst detected. Suppressing state.")
            return self._fallback(), False

        # Heuristic 2: Impossible delta shifts (game state loading / match swap)
        if self.last_valid_state is not None:
            delta = np.sqrt(
                (p_x - float(self.last_valid_state[0])) ** 2
                + (p_y - float(self.last_valid_state[1])) ** 2
            )
            if delta > self.max_expected_velocity:
                print("[WATCHDOG] Frame delta indicates pointer drift or map reload.")
                return self._fallback(), False

        # Heuristic 3: Stock counter overflow (bad pointer jump into negative territory)
        player_stocks = int(np.frombuffer(raw_buffer_bytes[36:40], dtype=np.uint32)[0])
        if player_stocks > 3:
            return self._fallback(), False

        # State passed all checks — commit to buffer
        self.last_valid_state = extracted_floats
        return extracted_floats, True

    def _fallback(self) -> np.ndarray:
        if self.last_valid_state is not None:
            return self.last_valid_state
        return np.zeros(53, dtype=np.float32)
