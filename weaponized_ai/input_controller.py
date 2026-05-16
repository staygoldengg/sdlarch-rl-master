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
