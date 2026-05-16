# input_controller.py
"""
Handles real and simulated input for AI agents, including keypresses, combos,
macros, and input confirmation. Weaponized from fight_engine.py.
"""

import threading
import ctypes
import ctypes.wintypes
import time
from typing import Optional, List, Tuple

_input_lock = threading.Lock()

# Virtual key codes for common Brawlhalla actions (customizable)
VK_MAP: dict = {
    'N': 0x4E, 'M': 0x4D, 'SPACE': 0x20,
    'A': 0x41, 'D': 0x44, 'W': 0x57, 'S': 0x53,
    'E': 0x45, 'F': 0x46,
    'LEFT': 0x25, 'RIGHT': 0x27, 'UP': 0x26, 'DOWN': 0x28,
}


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk",         ctypes.wintypes.WORD),
        ("wScan",       ctypes.wintypes.WORD),
        ("dwFlags",     ctypes.wintypes.DWORD),
        ("time",        ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("_pad", ctypes.c_byte * 28)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.wintypes.DWORD), ("iu", _INPUT_UNION)]


def _sendinput_key(vk: int, down: bool):
    flags = 0 if down else 0x0002
    ki  = _KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0,
                      dwExtraInfo=ctypes.pointer(ctypes.c_ulong(0)))
    inp = _INPUT(type=1, iu=_INPUT_UNION(ki=ki))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))


def send_key(vk: Optional[int], down: bool):
    """Press or release a virtual key."""
    if vk is not None:
        with _input_lock:
            _sendinput_key(vk, down)


def tap(vk: Optional[int], hold_s: float = 0.016):
    """Tap a key: press, hold, release."""
    send_key(vk, True)
    time.sleep(hold_s)
    send_key(vk, False)


def tap_name(key_name: str, hold_s: float = 0.016):
    """Tap a key by name (e.g. 'A', 'SPACE')."""
    vk = VK_MAP.get(key_name.upper())
    if vk:
        tap(vk, hold_s)


def combo(steps: List[Tuple[int, float, float]]):
    """Execute a combo sequence. steps: list of (vk, hold_s, delay_after_s)."""
    for vk, hold_s, delay in steps:
        tap(vk, hold_s)
        if delay > 0:
            time.sleep(delay)


def macro_thread(steps: List[Tuple[int, float, float]]):
    """Run a combo in a background thread (non-blocking)."""
    t = threading.Thread(target=combo, args=(steps,), daemon=True)
    t.start()
    return t


# ── Predefined BH move macros ─────────────────────────────────────────────────
MACROS = {
    "nlight":     [("N", 0.016, 0.0)],
    "slight":     [("D", 0.016, 0.02), ("N", 0.016, 0.0)],
    "dlight":     [("S", 0.016, 0.02), ("N", 0.016, 0.0)],
    "nheavy":     [("M", 0.016, 0.0)],
    "sheavy":     [("D", 0.016, 0.02), ("M", 0.016, 0.0)],
    "dheavy":     [("S", 0.016, 0.02), ("M", 0.016, 0.0)],
    "jump":       [("SPACE", 0.016, 0.0)],
    "dodge":      [("E", 0.016, 0.0)],
    "dash_right": [("D", 0.016, 0.02), ("D", 0.016, 0.0)],
    "dash_left":  [("A", 0.016, 0.02), ("A", 0.016, 0.0)],
}


def execute_macro(name: str) -> bool:
    """Execute a named macro in a background thread."""
    steps_named = MACROS.get(name.lower())
    if not steps_named:
        return False
    steps = [(VK_MAP[k.upper()], h, d) for k, h, d in steps_named
             if k.upper() in VK_MAP]
    macro_thread(steps)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Hardware Scan-Code Controller (used by HighFidelityTrainingLoop)
# ─────────────────────────────────────────────────────────────────────────────

# Physical scan codes for standard Brawlhalla key bindings
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


class _ScanInput(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", _Input_I)]


class WindowsHardwareController:
    """
    Low-level kernel-mode input injection via hardware scan codes.

    Uses SendInput with KEYEVENTF_SCANCODE so inputs are
    indistinguishable from physical keyboard hardware signals.
    """

    def press_key(self, scan_code: int):
        extra = ctypes.c_ulong(0)
        ii = _Input_I()
        ii.ki = _KeyBdInput(0, scan_code, KEYEVENTF_SCANCODE, 0, ctypes.pointer(extra))
        x = _ScanInput(ctypes.c_ulong(1), ii)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def release_key(self, scan_code: int):
        extra = ctypes.c_ulong(0)
        ii = _Input_I()
        ii.ki = _KeyBdInput(
            0, scan_code, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, ctypes.pointer(extra)
        )
        x = _ScanInput(ctypes.c_ulong(1), ii)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


class ActionTranslationEngine:
    """
    Translates FactorizedMultiDiscreteActorHead output dicts into
    hardware scan-code key presses.

    Maintains a diff of current key states so only changed keys trigger
    SendInput calls, minimising Win32 API overhead.
    """

    def __init__(self, hardware_controller: WindowsHardwareController):
        self.hw = hardware_controller
        self.current_key_states: dict[str, bool] = {k: False for k in SCAN_CODES}

    def execute_macro_dict(self, action_map: dict[str, int]):
        """
        Accepts a factorised action dict and updates hardware key states.

        action_map format (from FactorizedMultiDiscreteActorHead):
            {"move_x": 0-2, "move_y": 0-2, "action": 0-4}
        """
        target_states: dict[str, bool] = {k: False for k in SCAN_CODES}

        if action_map["move_x"] == 1:
            target_states["LEFT"] = True
        elif action_map["move_x"] == 2:
            target_states["RIGHT"] = True

        if action_map["move_y"] == 1:
            target_states["UP"] = True
        elif action_map["move_y"] == 2:
            target_states["DOWN"] = True

        if action_map["action"] == 1:
            target_states["LIGHT"] = True
        elif action_map["action"] == 2:
            target_states["HEAVY"] = True
        elif action_map["action"] == 3:
            target_states["DODGE"] = True
        elif action_map["action"] == 4:
            target_states["JUMP"] = True

        # Diff-based update — only trigger SendInput when state changes
        for key, scan_code in SCAN_CODES.items():
            should = target_states[key]
            current = self.current_key_states[key]
            if should and not current:
                self.hw.press_key(scan_code)
                self.current_key_states[key] = True
            elif not should and current:
                self.hw.release_key(scan_code)
                self.current_key_states[key] = False

    def inject_inputs(self, action_map: dict[str, int]):
        """Alias used by HighFidelityTrainingLoop.io_executor.submit()."""
        self.execute_macro_dict(action_map)

    def emergency_flush(self):
        """Releases every tracked key — call on crash or session end."""
        for key, scan_code in SCAN_CODES.items():
            if self.current_key_states[key]:
                self.hw.release_key(scan_code)
                self.current_key_states[key] = False
        print("[TRANSLATOR] Emergency Input Registry Flush Completed.")
