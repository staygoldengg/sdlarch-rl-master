# hardware_driver.py
"""
Thread-safe, nanosecond-precise Win32 keyboard injection drivers.

MutexHardwareDriver
  Simple synchronous scan-code driver with per-call mutex protection and
  LEFT+RIGHT / UP+DOWN conflict resolution.

FrameDeterministicDispatcher
  Asynchronous dispatcher running a dedicated nanosecond spin-lock thread
  pinned to the game's tick rate.  The AI enqueues action maps; the
  dispatcher thread consumes them at the exact right moment.
  Uses timeBeginPeriod(1) for OS timer precision.

Both classes support emergency_release() / global_flush() to guarantee the
keyboard is returned to a clean state on any shutdown path.
"""

import ctypes
import threading
import time
from collections import deque

# ── Win32 scan codes (physical keyboard layout) ───────────────────────────────
SCAN_CODES: dict[str, int] = {
    "LEFT":  0x1E,   # A
    "RIGHT": 0x20,   # D
    "UP":    0x11,   # W
    "DOWN":  0x1F,   # S
    "LIGHT": 0x24,   # J
    "HEAVY": 0x25,   # K
    "DODGE": 0x26,   # L
    "JUMP":  0x39,   # Spacebar
}

KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP    = 0x0002
_PUL = ctypes.POINTER(ctypes.c_ulong)


# ── ctypes Win32 input structures ─────────────────────────────────────────────

class _KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk",         ctypes.c_ushort),
        ("wScan",       ctypes.c_ushort),
        ("dwFlags",     ctypes.c_ulong),
        ("time",        ctypes.c_ulong),
        ("dwExtraInfo", _PUL),
    ]


class _HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg",    ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx",          ctypes.c_long),
        ("dy",          ctypes.c_long),
        ("mouseData",   ctypes.c_ulong),
        ("dwFlags",     ctypes.c_ulong),
        ("time",        ctypes.c_ulong),
        ("dwExtraInfo", _PUL),
    ]


class _Input_I(ctypes.Union):
    _fields_ = [
        ("ki", _KeyBdInput),
        ("mi", _MouseInput),
        ("hi", _HardwareInput),
    ]


class _Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", _Input_I)]


def _raw_send(scan_code: int, flags: int):
    """Low-level SendInput helper — one keystroke event."""
    extra = ctypes.c_ulong(0)
    ii = _Input_I()
    ii.ki = _KeyBdInput(0, scan_code, flags, 0, ctypes.pointer(extra))
    x = _Input(ctypes.c_ulong(1), ii)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def _resolve_conflicts(states: dict[str, bool]) -> dict[str, bool]:
    """Enforce mutual-exclusion rules to prevent engine input lockups."""
    if states.get("LEFT") and states.get("RIGHT"):
        states["LEFT"] = states["RIGHT"] = False
    if states.get("UP") and states.get("DOWN"):
        states["UP"] = False   # Prioritise fast-falls
    return states


# ─────────────────────────────────────────────────────────────────────────────
# MutexHardwareDriver — synchronous, thread-safe
# ─────────────────────────────────────────────────────────────────────────────

class MutexHardwareDriver:
    """
    Synchronous scan-code driver.  update_input_matrix() is safe to call from
    any thread; a mutex serialises concurrent callers.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.registry: dict[str, bool] = {k: False for k in SCAN_CODES}

    def update_input_matrix(self, target_states: dict[str, bool]):
        with self.lock:
            target_states = _resolve_conflicts(dict(target_states))
            for key, scancode in SCAN_CODES.items():
                target  = target_states.get(key, False)
                current = self.registry[key]
                if target and not current:
                    _raw_send(scancode, KEYEVENTF_SCANCODE)
                    self.registry[key] = True
                elif not target and current:
                    _raw_send(scancode, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP)
                    self.registry[key] = False

    def global_flush(self):
        """Release every held key unconditionally."""
        with self.lock:
            for key, scancode in SCAN_CODES.items():
                if self.registry[key]:
                    _raw_send(scancode, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP)
                    self.registry[key] = False
        print("[DRIVER] Global flush complete — all keys released.")


# ─────────────────────────────────────────────────────────────────────────────
# FrameDeterministicDispatcher — async nanosecond-precise
# ─────────────────────────────────────────────────────────────────────────────

class FrameDeterministicDispatcher:
    """
    Decoupled high-frequency input loop running on an isolated, time-pinned
    background thread.

    The AI inference cycle calls stage_action_map() asynchronously.
    The worker thread consumes the latest staged state at each frame boundary,
    keeping hardware injection strictly aligned to the game's tick rate.

    Uses timeBeginPeriod(1) to request 1 ms OS timer resolution and
    perf_counter_ns spin-locking for sub-millisecond frame precision.
    """

    def __init__(self, target_fps: int = 60):
        self.frame_time_ns = int((1.0 / target_fps) * 1_000_000_000)
        # maxlen=2 keeps latency minimal — inference always delivers the freshest state
        self.input_queue: deque = deque(maxlen=2)
        self.current_registry: dict[str, bool] = {k: False for k in SCAN_CODES}
        self.lock = threading.Lock()
        self.is_running = False
        self.worker = threading.Thread(
            target=self._hardware_spin_loop, daemon=True, name="InputDispatcher"
        )

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.worker.start()
            print("[DISPATCHER] High-priority hardware execution thread operational.")

    def stop(self):
        self.is_running = False

    def stage_action_map(self, action_map: dict[str, bool]):
        """Called by the AI inference cycle to schedule the next key state."""
        self.input_queue.append(action_map)

    def _hardware_spin_loop(self):
        # Request 1 ms OS timer resolution for reliable spin-lock behaviour
        ctypes.windll.winmm.timeBeginPeriod(1)
        next_tick = time.perf_counter_ns() + self.frame_time_ns

        try:
            while self.is_running:
                # Consume newest staged state (or hold current if queue is empty)
                if self.input_queue:
                    self._diff_and_execute(self.input_queue.pop())

                # Nanosecond spin-lock to hit exact frame boundary
                while time.perf_counter_ns() < next_tick:
                    pass

                next_tick += self.frame_time_ns
        finally:
            ctypes.windll.winmm.timeEndPeriod(1)

    def _diff_and_execute(self, target_states: dict[str, bool]):
        with self.lock:
            target_states = _resolve_conflicts(dict(target_states))
            for key, scancode in SCAN_CODES.items():
                target  = target_states.get(key, False)
                current = self.current_registry[key]
                if target and not current:
                    _raw_send(scancode, KEYEVENTF_SCANCODE)
                    self.current_registry[key] = True
                elif not target and current:
                    _raw_send(scancode, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP)
                    self.current_registry[key] = False

    def emergency_release(self):
        """Hard release of every tracked scan code — call on fault or shutdown."""
        with self.lock:
            for key, scancode in SCAN_CODES.items():
                if self.current_registry[key]:
                    _raw_send(scancode, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP)
                    self.current_registry[key] = False
        print("[DISPATCHER] Emergency cleanup complete — all keys unmapped.")

    # Alias so callers that expect global_flush() also work
    global_flush = emergency_release
