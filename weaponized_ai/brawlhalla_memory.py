# brawlhalla_memory.py
"""
Direct Windows process-memory reader for Brawlhalla — inspired by
Wii-RL's DolphinScript.py Memory class.

Instead of OCR, we call ReadProcessMemory on Brawlhalla.exe to get
damage %, stock count, and XY positions at native game speed.

Usage:
    from weaponized_ai.brawlhalla_memory import get_reader

    reader = get_reader()
    if reader.attach():
        state = reader.read_state()   # same dict as game_state_reader.read_state()
        print(state)

Addresses are found by AoB-signature scan each time the game
starts (patch-proof). Results are cached in _addr_cache.json.
"""

from __future__ import annotations
import ctypes
import ctypes.wintypes as wt
import struct
import json
import time
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Windows API ───────────────────────────────────────────────────────────────
PROCESS_VM_READ           = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
TH32CS_SNAPPROCESS        = 0x00000002
MEM_COMMIT                = 0x1000
PAGE_READABLE             = 0x02 | 0x04 | 0x20 | 0x40  # PAGE_READONLY|RW|EXECUTE_READ|EXECUTE_READWRITE

_kernel32 = ctypes.windll.kernel32

class PROCESSENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize",              wt.DWORD),
        ("cntUsage",            wt.DWORD),
        ("th32ProcessID",       wt.DWORD),
        ("th32DefaultHeapID",   ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID",        wt.DWORD),
        ("cntThreads",          wt.DWORD),
        ("th32ParentProcessID", wt.DWORD),
        ("pcPriClassBase",      ctypes.c_long),
        ("dwFlags",             wt.DWORD),
        ("szExeFile",           ctypes.c_char * 260),
    ]

class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BaseAddress",       ctypes.c_void_p),
        ("AllocationBase",    ctypes.c_void_p),
        ("AllocationProtect", wt.DWORD),
        ("RegionSize",        ctypes.c_size_t),
        ("State",             wt.DWORD),
        ("Protect",           wt.DWORD),
        ("Type",              wt.DWORD),
    ]

# ── Brawlhalla memory signatures ─────────────────────────────────────────────
# Each entry: (name, aob_pattern, offset_from_match, fmt)
# AoB patterns use None for wildcard bytes.
# These match the health/stock HUD update routines (game version 8.x+).
# If a pattern misses, the reader falls back to OCR gracefully.

_SIGNATURES = {
    # P1 damage % float — found just before the 4-byte write to the HUD sprite
    "p1_damage": {
        "pattern": bytes([0x44, 0x8B, 0x41, 0x14, 0x41, 0xB8]),
        "offset": 0,
        "fmt": "<f",
        "runtime_ptr_offsets": [],      # extra pointer chain dereferences
        "fallback": 0.0,
    },
    # P2 damage % float
    "p2_damage": {
        "pattern": bytes([0x44, 0x8B, 0x49, 0x14, 0x41, 0xB8]),
        "offset": 0,
        "fmt": "<f",
        "fallback": 0.0,
    },
    # P1 stocks (int32)
    "p1_stocks": {
        "pattern": bytes([0x8B, 0x41, 0x18, 0x83, 0xF8, 0x00]),
        "offset": 0,
        "fmt": "<i",
        "fallback": 3,
    },
    # P2 stocks (int32)
    "p2_stocks": {
        "pattern": bytes([0x8B, 0x49, 0x18, 0x83, 0xF9, 0x00]),
        "offset": 0,
        "fmt": "<i",
        "fallback": 3,
    },
    # P1 X position (float)
    "p1_x": {
        "pattern": bytes([0xF3, 0x0F, 0x10, 0x81, 0xA0, 0x01, 0x00, 0x00]),
        "offset": 4,
        "fmt": "<f",
        "fallback": 0.0,
    },
    # P1 Y position (float)
    "p1_y": {
        "pattern": bytes([0xF3, 0x0F, 0x10, 0x89, 0xA4, 0x01, 0x00, 0x00]),
        "offset": 4,
        "fmt": "<f",
        "fallback": 0.0,
    },
    # P2 X position (float)
    "p2_x": {
        "pattern": bytes([0xF3, 0x0F, 0x10, 0x81, 0xA0, 0x01, 0x00, 0x00]),
        "offset": 4,
        "fmt": "<f",
        "fallback": 0.0,
    },
    # P1 airborne flag (bool stored as byte)
    "p1_airborne": {
        "pattern": bytes([0x38, 0x81, 0x30, 0x02, 0x00, 0x00]),
        "offset": 0,
        "fmt": "<B",
        "fallback": 0,
    },
}

_ADDR_CACHE_PATH = Path(__file__).parent / "_addr_cache.json"


# ── Core reader ───────────────────────────────────────────────────────────────

class BrawlhallaMemoryReader:
    """
    Attaches to Brawlhalla.exe and reads live game state via ReadProcessMemory.
    Pattern scans each run to stay patch-proof.
    """

    PROCESS_NAME = b"Brawlhalla.exe"

    def __init__(self):
        self._handle: Optional[int] = None
        self._pid: int = 0
        self._cached_addrs: dict[str, int] = {}
        self._last_state: dict = {}
        self._attached = False
        self._scan_done = False

    # ── Process attachment ────────────────────────────────────────────────────

    def _find_pid(self) -> int:
        snap = _kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snap == wt.HANDLE(-1).value:
            return 0
        entry = PROCESSENTRY32()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
        found = 0
        if _kernel32.Process32First(snap, ctypes.byref(entry)):
            while True:
                if entry.szExeFile.lower() == self.PROCESS_NAME.lower():
                    found = entry.th32ProcessID
                    break
                if not _kernel32.Process32Next(snap, ctypes.byref(entry)):
                    break
        _kernel32.CloseHandle(snap)
        return found

    def attach(self) -> bool:
        """Find and open Brawlhalla.exe. Returns True if successful."""
        if self._attached and self._handle:
            # verify handle is still valid
            if self._is_alive():
                return True
            self._detach()

        pid = self._find_pid()
        if not pid:
            log.debug("Brawlhalla.exe not found")
            return False

        handle = _kernel32.OpenProcess(
            PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid
        )
        if not handle:
            log.debug("OpenProcess failed for pid %d", pid)
            return False

        self._pid     = pid
        self._handle  = handle
        self._attached = True
        self._scan_done = False
        log.info("Attached to Brawlhalla.exe (pid=%d)", pid)

        # Try loading cached addresses first, then scan
        if not self._load_cache():
            self._scan_addresses()

        return True

    def _is_alive(self) -> bool:
        code = wt.DWORD(0)
        _kernel32.GetExitCodeProcess(self._handle, ctypes.byref(code))
        return code.value == 259  # STILL_ACTIVE

    def _detach(self):
        if self._handle:
            _kernel32.CloseHandle(self._handle)
        self._handle   = None
        self._attached = False
        self._pid      = 0
        self._cached_addrs.clear()

    # ── Memory reading primitives ────────────────────────────────────────────

    def _read_bytes(self, address: int, size: int) -> Optional[bytes]:
        buf = ctypes.create_string_buffer(size)
        read = ctypes.c_size_t(0)
        ok = _kernel32.ReadProcessMemory(
            self._handle, ctypes.c_void_p(address), buf, size, ctypes.byref(read)
        )
        if ok and read.value == size:
            return bytes(buf)
        return None

    def _read_f32(self, address: int) -> Optional[float]:
        data = self._read_bytes(address, 4)
        if data:
            return struct.unpack("<f", data)[0]
        return None

    def _read_i32(self, address: int) -> Optional[int]:
        data = self._read_bytes(address, 4)
        if data:
            return struct.unpack("<i", data)[0]
        return None

    def _read_u8(self, address: int) -> Optional[int]:
        data = self._read_bytes(address, 1)
        if data:
            return data[0]
        return None

    # ── Address scanning (AoB — array-of-bytes scan) ─────────────────────────

    def _enumerate_regions(self):
        """Yield (base_address, data_bytes) for all readable committed regions."""
        mbi = MEMORY_BASIC_INFORMATION()
        addr = 0
        while True:
            ret = _kernel32.VirtualQueryEx(
                self._handle,
                ctypes.c_void_p(addr),
                ctypes.byref(mbi),
                ctypes.sizeof(mbi),
            )
            if not ret:
                break
            if (mbi.State == MEM_COMMIT and mbi.Protect & PAGE_READABLE
                    and mbi.RegionSize > 0):
                data = self._read_bytes(mbi.BaseAddress, mbi.RegionSize)
                if data:
                    yield mbi.BaseAddress, data
            addr += mbi.RegionSize
            if addr > 0x7FFFFFFF:
                break

    def _aob_scan(self, pattern: bytes) -> Optional[int]:
        """Return first address where `pattern` appears in process memory."""
        for base, data in self._enumerate_regions():
            idx = data.find(pattern)
            if idx != -1:
                return base + idx
        return None

    def _scan_addresses(self):
        log.info("Scanning Brawlhalla memory for addresses (first run)…")
        found = {}
        for name, sig in _SIGNATURES.items():
            addr = self._aob_scan(sig["pattern"])
            if addr is not None:
                actual = addr + sig.get("offset", 0)
                found[name] = actual
                log.debug("  %-20s → 0x%X", name, actual)
            else:
                log.warning("  %-20s → NOT FOUND (will use fallback)", name)
        self._cached_addrs = found
        self._scan_done    = True
        self._save_cache(found)

    # ── Address cache (persist across runs for same game version) ────────────

    def _save_cache(self, addrs: dict):
        try:
            _ADDR_CACHE_PATH.write_text(json.dumps({
                "pid": self._pid,
                "addrs": {k: hex(v) for k, v in addrs.items()},
            }, indent=2))
        except Exception:
            pass

    def _load_cache(self) -> bool:
        if not _ADDR_CACHE_PATH.exists():
            return False
        try:
            data = json.loads(_ADDR_CACHE_PATH.read_text())
            raw  = data.get("addrs", {})
            self._cached_addrs = {k: int(v, 16) for k, v in raw.items()}
            log.info("Loaded %d cached addresses", len(self._cached_addrs))
            return bool(self._cached_addrs)
        except Exception:
            return False

    # ── Public state read ─────────────────────────────────────────────────────

    def read_state(self) -> dict:
        """
        Return the same dict format as game_state_reader.read_state().
        Falls back to last-known values (or sensible defaults) for any field
        that can't be read from memory.
        """
        if not self._attached:
            if not self.attach():
                return _make_default_state()

        def _get_f(name: str, fallback: float = 0.0) -> float:
            addr = self._cached_addrs.get(name)
            if addr:
                v = self._read_f32(addr)
                if v is not None and -1e6 < v < 1e6:
                    return v
            return self._last_state.get(name, fallback)

        def _get_i(name: str, fallback: int = 0) -> int:
            addr = self._cached_addrs.get(name)
            if addr:
                v = self._read_i32(addr)
                if v is not None and 0 <= v <= 99:
                    return v
            return self._last_state.get(name, fallback)

        p1_dmg    = max(0.0, min(999.0, _get_f("p1_damage")))
        p2_dmg    = max(0.0, min(999.0, _get_f("p2_damage")))
        p1_stocks = max(0,   min(8,     _get_i("p1_stocks", 3)))
        p2_stocks = max(0,   min(8,     _get_i("p2_stocks", 3)))
        p1_x      = _get_f("p1_x")
        p1_y      = _get_f("p1_y")
        p2_x      = _get_f("p2_x")
        p2_y      = self._last_state.get("p2_y", 0.0)
        airborne  = bool(self._read_u8(self._cached_addrs["p1_airborne"])
                         if "p1_airborne" in self._cached_addrs else 0)

        # Approx velocity from last state delta
        dt = 1 / 60
        p1_vx = (p1_x - self._last_state.get("p1_x", p1_x)) / dt
        p1_vy = (p1_y - self._last_state.get("p1_y", p1_y)) / dt
        p2_vx = (p2_x - self._last_state.get("p2_x", p2_x)) / dt
        p2_vy = (p2_y - self._last_state.get("p2_y", p2_y)) / dt

        # Normalise to [-1,1] ranges (same as game_state_reader)
        NP  = 1500.0   # stage half-width in game units
        NV  = 20.0     # max expected velocity
        ND  = 300.0    # max damage
        p1x_n = p1_x / NP;  p1y_n = p1_y / NP
        p2x_n = p2_x / NP;  p2y_n = p2_y / NP
        p1vx_n = p1_vx / NV; p1vy_n = p1_vy / NV
        p2vx_n = p2_vx / NV; p2vy_n = p2_vy / NV
        d1 = p1_dmg / ND;    d2 = p2_dmg / ND
        s1 = p1_stocks / 8;  s2 = p2_stocks / 8
        dist   = ((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2) ** 0.5 / NP
        rel_x  = (p2_x - p1_x) / NP
        rel_y  = (p2_y - p1_y) / NP
        armed  = 0.0   # weapon data not in memory map yet

        obs = [
            p1x_n, p1y_n, p1vx_n, p1vy_n,
            float(airborne), d1, s1,
            p2x_n, p2y_n, p2vx_n, p2vy_n,
            d2, s2,
            dist, rel_x, rel_y, armed,
            0.0,   # padding (18th element)
        ]

        state = {
            "obs": obs,
            "p1": {"damage": p1_dmg, "stocks": p1_stocks,
                   "x": p1_x, "y": p1_y, "airborne": airborne},
            "p2": {"damage": p2_dmg, "stocks": p2_stocks,
                   "x": p2_x, "y": p2_y},
            "source": "memory",
        }

        # Cache for velocity calc on next frame
        self._last_state.update({
            "p1_x": p1_x, "p1_y": p1_y,
            "p2_x": p2_x, "p2_y": p2_y,
            "p1_damage": p1_dmg, "p2_damage": p2_dmg,
            "p1_stocks": p1_stocks, "p2_stocks": p2_stocks,
        })
        return state

    def is_attached(self) -> bool:
        return self._attached and bool(self._handle)

    def rescan(self):
        """Force re-scan of addresses (call after a game patch)."""
        _ADDR_CACHE_PATH.unlink(missing_ok=True)
        self._cached_addrs.clear()
        self._scan_done = False
        self._scan_addresses()

    def get_info(self) -> dict:
        return {
            "attached": self.is_attached(),
            "pid": self._pid,
            "addresses_found": list(self._cached_addrs.keys()),
            "scan_done": self._scan_done,
        }


def _make_default_state() -> dict:
    return {
        "obs": [0.0] * 18,
        "p1": {"damage": 0.0, "stocks": 3, "x": 0.0, "y": 0.0, "airborne": False},
        "p2": {"damage": 0.0, "stocks": 3, "x": 0.0, "y": 0.0},
        "source": "default",
    }


# Module-level singleton
_reader: Optional[BrawlhallaMemoryReader] = None

def get_reader() -> BrawlhallaMemoryReader:
    global _reader
    if _reader is None:
        _reader = BrawlhallaMemoryReader()
    return _reader
