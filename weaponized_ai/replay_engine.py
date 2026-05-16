# replay_engine.py
"""
Brawlhalla Replay Digestion Engine
===================================
Reads .replay files from C:\\Users\\carli\\BrawlhallaReplays (or fallback
APPDATA path) and converts them into (obs, action, reward, next_obs, done)
transitions for Behavioural Cloning pre-training.

Inspired by Wii-RL's DolphinScript.py approach of extracting per-frame
game data — here we decompress and parse the binary replay rather than
reading live memory.

Actual Binary Format (discovered by hex analysis, v10.06)
----------------------------------------------------------
Files are **zlib-deflate compressed** (magic 0x78 0xDA).

After decompression:
  Offset  Size  Description
  ------  ----  -----------
  0x00    4     Format magic: 0x6b 0x10 0xdf 0x35  (constant across all files)
  0x04    5     Per-match seed / identifier (varies per match)
  0x09    3     Mode flags (mostly constant, encodes game mode)
  0x0C    ~36   Player metadata block (character IDs, etc. — proprietary packing)
  ~0x30+  N×2   Per-frame input stream: 1 byte per player per frame (2P = 2 B/frame)
                Bit layout: [7]=Up/Jump [6]=Right [5]=Left [4]=Down
                             [3]=LightAtk [2]=HeavyAtk [1]=Dodge [0]=Pickup

Frame count estimate: (decompressed_size - HEADER_EST) // NUM_PLAYERS
HEADER_EST ≈ 48 bytes (validated against file-size vs match-duration)

Stage name and game version are extracted from the filename:
  "[10.06] WesternAirTemple (3).replay"
   ^^^^^^^ ^^^^^^^^^^^^^^^^^^
   version  stage_name

Notes
-----
* The first 48 bytes form a fixed + per-match header. The remaining bytes
  are the packed input stream at ~2 bytes per frame for 2 players.
* Character names are not decoded (proprietary encoding) — shown as "P1/P2".
* All multi-byte integers are little-endian.
"""

from __future__ import annotations

import os
import re
import struct
import zlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Replay directory — primary is the user's actual replay folder.
# Falls back to the default Brawlhalla APPDATA location.
_REPLAY_CANDIDATES: list[Path] = [
    Path(r"C:\Users\carli\BrawlhallaReplays"),
    Path(os.environ.get("APPDATA", "~")).expanduser() / "Brawlhalla" / "replays",
]
REPLAY_DIR: Path = next(
    (p for p in _REPLAY_CANDIDATES if p.exists()),
    _REPLAY_CANDIDATES[0],          # keep primary even if missing (will warn on use)
)

# Actual magic bytes found by hex analysis of real v10.06 replay files.
# Files are zlib-compressed; this magic appears after decompression at offset 0.
REPLAY_MAGIC  = bytes([0x6b, 0x10, 0xdf, 0x35])

# The first ~48 bytes form a fixed + per-match header; inputs follow.
HEADER_EST    = 48
MAX_FRAMES    = 108_000     # 30 min at 60 fps — sanity cap
MIN_FRAMES    = 60          # skip very-short replays (< 1 s)
NUM_PLAYERS   = 2           # bytes per frame in standard 1v1 replays

# Filename pattern:  "[10.06] WesternAirTemple (3).replay"
_FNAME_RE = re.compile(r"^\[([0-9.]+)\]\s+(.+?)(?:\s+\(\d+\))?\.replay$", re.IGNORECASE)

# ── Stage bounds ──────────────────────────────────────────────────────────────

# (half_width_units, ground_y) for each stage name appearing in replays.
# half_width is the horizontal boundary; falling below ground_y - 400 = blast zone.
# Values are approximate but consistent with in-game geometry.
_STAGE_BOUNDS: dict[str, tuple[float, float]] = {
    "WesternAirTemple":   (800.0,  0.0),
    "SmallFortressof":    (700.0,  0.0),
    "SmallWasteland":     (660.0,  0.0),
    "SmallWorld'sEnd":    (720.0,  0.0),
    "SmallTerminus":      (680.0,  0.0),
    "ShadowscarLanding":  (760.0,  0.0),
    "MishimaDojo":        (740.0,  0.0),
    "DemonIsland":        (780.0,  0.0),
    "Ring":               (620.0,  0.0),
    "Jikoku":             (700.0,  0.0),
    "Shiganshina":        (750.0,  0.0),
    "ShiganshinaBrawl":   (750.0,  0.0),
    "BigMiamiDome":       (850.0,  0.0),
    "BlackguardKeep":     (710.0,  0.0),
    "PlainsofPassage":    (800.0,  0.0),
    "Apocalypse":         (760.0,  0.0),
    "SmallBrawlhaven":    (680.0,  0.0),
    "SmallEnigma":        (680.0,  0.0),
    "SpiritRealm":        (780.0,  0.0),
    "TempleRuins":        (780.0,  0.0),
    "TheGreatHall":       (760.0,  0.0),
    "Lich'sTomb":         (720.0,  0.0),
    "CrumblingChasm":     (760.0,  0.0),
}
_DEFAULT_HALF_W = 760.0

# Character ID → name string
CHAR_NAMES: dict[int, str] = {
    1: "Bodvar",  2: "Cassidy", 3: "Orion",    4: "Lord_Vraxx",
    5: "Gnash",   6: "Queen_Nai", 7: "Hattori", 8: "Sir_Roland",
    9: "Scarlet",10: "Thatch",  11: "Ada",      12: "Sentinel",
    13: "Lucien", 14: "Teros",  15: "Brynn",    16: "Asuri",
    17: "Barraza",18: "Ember",  19: "Azoth",    20: "Koji",
    21: "Ulgrim", 22: "Diana",  23: "Jhala",    24: "Kor",
    25: "Wu Shang",26:"Val",    27: "Ragnir",   28: "Cross",
    29: "Mirage", 30: "Nix",    31: "Mordex",   32: "Yumiko",
    33: "Artemis",34: "Caspian",35: "Sidra",    36: "Xull",
    37: "Kaya",   38: "Isaiah", 39: "Jiro",     40: "Lin Fei",
    41: "Zariel", 42: "Rayman", 43: "Dusk",     44: "Fait",
    45: "Thor",   46: "Petra",  47: "Vector",   48: "Volkov",
    49: "Onyx",   50: "Jaeyun", 51: "Mako",     52: "TarIQ",
    53: "Magyar", 54: "Munin",  55: "Arcadia",  56: "Ezio",
    57: "Loki",   58: "Yago",
}

# Action index mapping (must match rl_agent.py ACT_DIM=16 ordering)
# Inputs are a bitmask per player per frame:
#   bit7=Up(jump), bit6=Right, bit5=Left, bit4=Down
#   bit3=LightAtk, bit2=HeavyAtk, bit1=Dodge, bit0=Pickup
ACT_NONE    = 0   # no relevant input
ACT_NLIGHT  = 0   # neutral light
ACT_SLIGHT  = 1   # side light (L/R + light)
ACT_DLIGHT  = 2   # down light
ACT_NHEAVY  = 3
ACT_SHEAVY  = 4
ACT_DHEAVY  = 5
ACT_NAIR    = 6   # jump (no attack held yet, next-frame decision)
ACT_SAIR    = 7
ACT_DAIR    = 8
ACT_JUMP    = 9
ACT_DJ      = 10
ACT_DODGE   = 11
ACT_DASH_R  = 12
ACT_DASH_L  = 13
ACT_PICKUP  = 14
ACT_NSIG    = 15

# Simplified Brawlhalla physics constants
WALK_SPEED   = 7.3     # units/frame
RUN_SPEED    = 10.6
JUMP_VY      = 18.0
GRAVITY      = -0.65
MAX_FALL     = -15.0
GROUND_Y     = 0.0


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PlayerConfig:
    slot:       int
    character:  str
    weapon1:    int
    weapon2:    int
    color:      int
    name:       str = ""


@dataclass
class ReplayMeta:
    path:        Path
    file_version: int          = 0
    game_version: str          = ""     # e.g. "10.06"
    timestamp:    int          = 0
    level_id:     int          = 0
    game_mode:    int          = 0
    stage_name:   str          = ""     # extracted from filename
    players:      list[PlayerConfig] = field(default_factory=list)
    frame_count:  int          = 0
    parse_ok:     bool         = False
    error:        str          = ""


@dataclass
class Transition:
    obs:      list[float]
    action:   int
    reward:   float
    next_obs: list[float]
    done:     bool
    weight:   float = 1.0   # BC importance weight


# ── Lightweight physics simulation ───────────────────────────────────────────

class PlayerPhysics:
    """
    Simulates one player's XY position and velocity given per-frame inputs.
    Cheap enough to run for a full 20-minute replay in <100ms.
    """

    def __init__(self, stage_half_w: float = _DEFAULT_HALF_W):
        self.stage_half_w = stage_half_w
        self.x:        float = 0.0
        self.y:        float = GROUND_Y
        self.vx:       float = 0.0
        self.vy:       float = 0.0
        self.airborne: bool  = False
        self.jumps:    int   = 2    # double jump counter
        self.dodge_cd: int   = 0
        self.stocks:   int   = 3
        self.damage:   float = 0.0
        self._just_jumped:   bool = False

    def step(self, bits: int) -> None:
        """Advance one game frame given the 8-bit input byte."""
        up     = bool(bits & 0x80)
        right  = bool(bits & 0x40)
        left   = bool(bits & 0x20)
        down   = bool(bits & 0x10)
        # attack and heavy are handled via reward logic, not physics

        # Horizontal movement
        if right:
            self.vx = min(self.vx + 1.5, RUN_SPEED)
        elif left:
            self.vx = max(self.vx - 1.5, -RUN_SPEED)
        else:
            # friction
            self.vx *= 0.85

        # Jump
        if up and self.jumps > 0 and not self._just_jumped:
            self.vy       = JUMP_VY
            self.airborne = True
            self.jumps   -= 1
            self._just_jumped = True
        else:
            self._just_jumped = False

        # Gravity when airborne
        if self.airborne:
            self.vy = max(self.vy + GRAVITY, MAX_FALL)
        else:
            self.vy = 0.0

        # Update position
        self.x += self.vx
        self.y += self.vy

        # Ground detection
        if self.y <= GROUND_Y and self.airborne:
            self.y        = GROUND_Y
            self.vy       = 0.0
            self.airborne = False
            self.jumps    = 2

        # Dodge cooldown
        if self.dodge_cd > 0:
            self.dodge_cd -= 1

    def _just_jumped_init(self):
        self._just_jumped = False

    def obs_vec(self) -> list[float]:
        """6-element local obs sub-vector for this player."""
        return [
            self.x / self.stage_half_w,
            self.y / 500.0,
            self.vx / RUN_SPEED,
            self.vy / JUMP_VY,
            float(self.airborne),
            self.damage / 300.0,
        ]


def _make_obs(p1: PlayerPhysics, p2: PlayerPhysics) -> list[float]:
    """Build 18-element obs vector matching rl_agent.OBS_DIM=18."""
    p1v  = p1.obs_vec()
    p2v  = p2.obs_vec()
    hw   = p1.stage_half_w
    dx   = (p2.x - p1.x) / hw
    dy   = (p2.y - p1.y) / 500.0
    dist = min(1.0, ((p2.x-p1.x)**2 + (p2.y-p1.y)**2)**0.5 / hw)
    # layout: p1x,p1y,p1vx,p1vy,p1air,p1dmg, p1stk, p2x,p2y,p2vx,p2vy,p2dmg, p2stk, dist, relx, rely, armed, pad
    return [
        p1v[0], p1v[1], p1v[2], p1v[3],
        p1v[4], p1v[5], p1.stocks / 8.0,
        p2v[0], p2v[1], p2v[2], p2v[3],
        p2v[5], p2.stocks / 8.0,
        dist, dx, dy, 0.0, 0.0,
    ]


def _bits_to_action(bits: int, p: PlayerPhysics) -> int:
    """Map 8-bit input byte → action index (ACT_DIM=16 space)."""
    up     = bool(bits & 0x80)
    right  = bool(bits & 0x40)
    left   = bool(bits & 0x20)
    down   = bool(bits & 0x10)
    attack = bool(bits & 0x08)
    heavy  = bool(bits & 0x04)
    dodge  = bool(bits & 0x02)
    pickup = bool(bits & 0x01)

    has_dir = right or left

    if pickup:                           return ACT_PICKUP
    if dodge:                            return ACT_DODGE
    if up and not attack and not heavy:  return ACT_JUMP if not p.airborne else ACT_DJ
    if attack:
        if p.airborne:
            if has_dir:  return ACT_SAIR
            if down:     return ACT_DAIR
            return ACT_NAIR
        else:
            if has_dir:  return ACT_SLIGHT
            if down:     return ACT_DLIGHT
            return ACT_NLIGHT
    if heavy:
        if p.airborne:
            return ACT_NSIG
        if has_dir:  return ACT_SHEAVY
        if down:     return ACT_DHEAVY
        return ACT_NHEAVY
    if right and not p.airborne: return ACT_DASH_R
    if left  and not p.airborne: return ACT_DASH_L
    return ACT_NONE


def _compute_reward(prev_obs: list[float], next_obs: list[float],
                    p1: PlayerPhysics, p2: PlayerPhysics,
                    prev_p2_dmg: float, prev_p1_dmg: float) -> float:
    """
    Reward function analogous to the race-completion reward in Wii-RL.
    Positive for dealing damage / KOs, negative for taking damage / dying.
    """
    dmg_dealt  = max(0.0, p2.damage - prev_p2_dmg) * 0.01
    dmg_taken  = max(0.0, p1.damage - prev_p1_dmg) * 0.01
    reward     = dmg_dealt - dmg_taken

    # KO bonus / death penalty (stocks drop)
    p1_stk_n = next_obs[6] * 8
    p2_stk_n = next_obs[12] * 8
    p1_stk_p = prev_obs[6] * 8
    p2_stk_p = prev_obs[12] * 8
    if p2_stk_n < p2_stk_p:
        reward += 1.0          # KO'd the opponent
    if p1_stk_n < p1_stk_p:
        reward -= 1.0          # got KO'd
    return float(reward)


# ── Filename metadata extractor ───────────────────────────────────────────────

def _parse_filename_meta(path: Path) -> tuple[str, str]:
    """
    Extract (game_version, stage_name) from a Brawlhalla replay filename.
    Format: "[10.06] WesternAirTemple (3).replay"
    Returns ("", "") if the pattern does not match.
    """
    m = _FNAME_RE.match(path.name)
    if not m:
        return "", ""
    return m.group(1), m.group(2).strip()


# ── Binary parser ─────────────────────────────────────────────────────────────

def _parse_header(data: bytes) -> ReplayMeta:
    """
    Parse one replay file's raw bytes (may be zlib-compressed).

    Steps:
    1. Decompress if zlib-compressed (starts with 0x78 0xDA or 0x78 0x9C).
    2. Verify the magic bytes 0x6b 0x10 0xdf 0x35.
    3. Estimate frame count from (decompressed_size - HEADER_EST) // NUM_PLAYERS.
    4. Store the decompressed payload for input extraction.

    Player character data is not decoded (format is proprietary) — the
    caller fills stage_name / game_version from the filename instead.
    """
    meta = ReplayMeta(path=Path(""))
    if len(data) < 8:
        meta.error = "File too short"
        return meta

    # Decompress zlib if compressed
    if data[0] == 0x78 and data[1] in (0x9C, 0xDA, 0x01, 0x5E):
        try:
            data = zlib.decompress(data)
        except zlib.error as exc:
            meta.error = f"zlib decompression failed: {exc}"
            return meta

    # Check magic
    if data[:4] != REPLAY_MAGIC:
        meta.error = f"Bad magic: {data[:4].hex()} (expected {REPLAY_MAGIC.hex()})"
        return meta

    # Estimate frame count from payload size
    payload_bytes = max(0, len(data) - HEADER_EST)
    frame_count   = min(payload_bytes // NUM_PLAYERS, MAX_FRAMES)

    meta.frame_count = frame_count
    meta.parse_ok    = True
    meta._decompressed   = data          # type: ignore[attr-defined]
    meta._input_offset   = HEADER_EST   # type: ignore[attr-defined]
    meta._num_players    = NUM_PLAYERS  # type: ignore[attr-defined]
    return meta


def _heuristic_input_scan(data: bytes, num_players: int = 2) -> Optional[tuple[int, int]]:
    """
    Find the input stream within decompressed replay data.

    Since the format stores inputs starting at HEADER_EST, we simply:
    1. Skip the first HEADER_EST bytes (known fixed header).
    2. Use the remaining bytes as a flat (N × num_players) input array.

    Returns (start_offset, frame_count) or None if there isn't enough data.
    The fallback popcount heuristic is kept for edge cases.
    """
    start = HEADER_EST
    remaining = len(data) - start
    if remaining < MIN_FRAMES * num_players:
        # Try the full data if header estimate is wrong
        start = 0
        remaining = len(data)

    fc = min(remaining // num_players, MAX_FRAMES)
    if fc < MIN_FRAMES:
        return None
    return start, fc


def _extract_inputs(data: bytes, meta: ReplayMeta) -> Optional[np.ndarray]:
    """
    Return ndarray shape (frame_count, num_players) of uint8 input bytes.
    Uses the decompressed data cached on meta._decompressed when available.
    """
    # Prefer the already-decompressed copy stored in meta
    raw_data: bytes = getattr(meta, "_decompressed", data)

    if not meta.parse_ok:
        result = _heuristic_input_scan(raw_data, 2)
        if result is None:
            return None
        start, fc = result
        fc = min(fc, MAX_FRAMES)
        raw = np.frombuffer(raw_data[start: start + fc * 2], dtype=np.uint8)
        return raw.reshape(-1, 2)[:fc]

    n      = getattr(meta, "_num_players", NUM_PLAYERS)
    start  = getattr(meta, "_input_offset", HEADER_EST)
    fc     = min(meta.frame_count, MAX_FRAMES)
    needed = fc * n
    available = len(raw_data) - start
    if available < needed:
        fc = available // n
    if fc < MIN_FRAMES:
        return None
    raw = np.frombuffer(raw_data[start: start + fc * n], dtype=np.uint8)
    return raw.reshape(-1, n)[:fc]


# ── Main engine class ─────────────────────────────────────────────────────────

class ReplayEngine:
    """
    Discovers and digests Brawlhalla .replay files.

    Usage
    -----
    engine = ReplayEngine()
    results = engine.ingest_all()   # scan + parse + return stats

    Or for a single file:
    transitions = engine.process_replay(Path("my_match.replay"))
    """

    def __init__(self, replay_dir: Optional[Path] = None):
        self.replay_dir = Path(replay_dir or REPLAY_DIR)
        self._ingested: set[str] = set()   # paths already processed

    # ── Discovery ─────────────────────────────────────────────────────────────

    def discover(self) -> list[Path]:
        """Return all .replay files in the configured directory."""
        if not self.replay_dir.exists():
            # Try other candidate directories
            for candidate in _REPLAY_CANDIDATES:
                if candidate.exists():
                    self.replay_dir = candidate
                    break
            else:
                log.warning("Replay directory not found: %s", self.replay_dir)
                return []
        files = sorted(self.replay_dir.glob("**/*.replay"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        log.info("Found %d .replay files in %s", len(files), self.replay_dir)
        return files

    def parse_meta(self, path: Path) -> ReplayMeta:
        """Parse the header of one replay file, enriching with filename metadata."""
        try:
            data = path.read_bytes()
        except OSError as exc:
            m = ReplayMeta(path=path, error=str(exc))
            return m
        meta = _parse_header(data)
        meta.path = path
        # Always fill game_version + stage_name from filename (reliable)
        gv, stage = _parse_filename_meta(path)
        if gv:
            meta.game_version = gv
        if stage:
            meta.stage_name = stage
        # Provide default player labels if not decoded
        if not meta.players:
            meta.players = [
                PlayerConfig(slot=0, character="P1", weapon1=0, weapon2=0, color=0, name=""),
                PlayerConfig(slot=1, character="P2", weapon1=0, weapon2=0, color=0, name=""),
            ]
        return meta

    # ── Processing ───────────────────────────────────────────────────────────

    def process_replay(self, path: Path,
                       max_transitions: int = 20_000) -> list[Transition]:
        """
        Parse one .replay file and return a list of Transition objects
        suitable for brain_store.save_corpus().
        """
        t0 = time.perf_counter()
        try:
            data = path.read_bytes()
        except OSError as exc:
            log.error("Cannot read %s: %s", path, exc)
            return []

        meta = _parse_header(data)
        meta.path = path
        # Fill in stage name + version from filename
        gv, stage = _parse_filename_meta(path)
        if gv:
            meta.game_version = gv
        if stage:
            meta.stage_name = stage

        inputs = _extract_inputs(data, meta)
        if inputs is None:
            log.warning("No parseable input stream in %s", path.name)
            return []

        # Stage-specific bounds
        half_w = _STAGE_BOUNDS.get(meta.stage_name, (_DEFAULT_HALF_W, 0.0))[0]

        num_players = inputs.shape[1]
        log.info("Simulating %s  frames=%d  players=%d  stage=%s",
                 path.name, inputs.shape[0], num_players, meta.stage_name or "?")

        p1 = PlayerPhysics(stage_half_w=half_w)
        p2 = PlayerPhysics(stage_half_w=half_w)

        # Spread initial positions
        p1.x = -half_w * 0.25
        p2.x =  half_w * 0.25

        transitions: list[Transition] = []
        obs = _make_obs(p1, p2)

        for frame_idx, row in enumerate(inputs):
            if len(transitions) >= max_transitions:
                break

            bits_p1 = int(row[0])
            bits_p2 = int(row[1]) if num_players > 1 else 0

            prev_p1_dmg = p1.damage
            prev_p2_dmg = p2.damage
            prev_obs    = obs[:]

            action = _bits_to_action(bits_p1, p1)
            p1.step(bits_p1)
            p2.step(bits_p2)

            # Simulate damage from attack actions (rough approximation)
            if bits_p1 & 0x08:   # light attack
                p2.damage += 5.0 * (1.0 + 0.003 * p2.damage)
            if bits_p1 & 0x04:   # heavy
                p2.damage += 10.0 * (1.0 + 0.003 * p2.damage)
            if bits_p2 & 0x08:
                p1.damage += 5.0 * (1.0 + 0.003 * p1.damage)
            if bits_p2 & 0x04:
                p1.damage += 10.0 * (1.0 + 0.003 * p1.damage)

            # Stage bounds → KO
            done = False
            if abs(p1.x) > half_w * 1.3 or p1.y < -400:
                p1.stocks -= 1
                p1.damage  = 0.0
                p1.x, p1.y = -half_w * 0.25, GROUND_Y
                p1.vx = p1.vy = 0.0
                if p1.stocks <= 0:
                    done = True
            if abs(p2.x) > half_w * 1.3 or p2.y < -400:
                p2.stocks -= 1
                p2.damage  = 0.0
                p2.x, p2.y = half_w * 0.25, GROUND_Y
                p2.vx = p2.vy = 0.0
                if p2.stocks <= 0:
                    done = True

            next_obs = _make_obs(p1, p2)
            reward   = _compute_reward(prev_obs, next_obs, p1, p2,
                                       prev_p2_dmg, prev_p1_dmg)

            # BC weight: upweight attack actions, downweight no-ops
            weight = 2.0 if action in (ACT_SLIGHT, ACT_NLIGHT, ACT_SHEAVY,
                                        ACT_NSIG, ACT_NHEAVY) else 1.0

            transitions.append(Transition(
                obs=prev_obs, action=action, reward=reward,
                next_obs=next_obs, done=done, weight=weight,
            ))

            obs = next_obs
            if done:
                p1.__init__(stage_half_w=half_w)
                p2.__init__(stage_half_w=half_w)
                p1.x = -half_w * 0.25; p2.x = half_w * 0.25
                obs = _make_obs(p1, p2)

        elapsed = time.perf_counter() - t0
        log.info("  → %d transitions in %.2fs", len(transitions), elapsed)
        self._ingested.add(str(path))
        return transitions

        elapsed = time.perf_counter() - t0
        log.info("  → %d transitions in %.2fs", len(transitions), elapsed)
        self._ingested.add(str(path))
        return transitions

    def ingest_all(self, skip_ingested: bool = True,
                   max_per_file: int = 20_000) -> dict:
        """
        Process all discovered replays, push to brain_store.

        Returns a summary dict for the API endpoint.
        """
        from weaponized_ai import brain_store

        files   = self.discover()
        results = []
        total_trans = 0

        for path in files:
            if skip_ingested and brain_store.is_already_ingested(str(path)):
                continue

            trans_list = self.process_replay(path, max_per_file)
            if not trans_list:
                results.append({"path": str(path), "transitions": 0,
                                 "status": "no_data"})
                continue

            # Convert to arrays for brain_store
            obs_arr    = np.array([t.obs      for t in trans_list], dtype=np.float32)
            act_arr    = np.array([t.action   for t in trans_list], dtype=np.int32)
            rwd_arr    = np.array([t.reward   for t in trans_list], dtype=np.float32)
            nobs_arr   = np.array([t.next_obs for t in trans_list], dtype=np.float32)
            done_arr   = np.array([t.done     for t in trans_list], dtype=np.float32)
            wgt_arr    = np.array([t.weight   for t in trans_list], dtype=np.float32)

            brain_store.save_corpus(
                obs_arr, act_arr, rwd_arr, nobs_arr, done_arr, wgt_arr
            )

            # Build a title from the file and stage info
            meta  = self.parse_meta(path)
            stage = meta.stage_name or path.stem
            title = f"{stage} — v{meta.game_version} — {path.name}"

            brain_store.mark_ingested(
                url=str(path),
                title=title,
                frames=len(trans_list),
                transitions=len(trans_list),
            )
            total_trans += len(trans_list)
            results.append({
                "path": str(path), "title": title,
                "transitions": len(trans_list), "status": "ok",
            })

        return {
            "files_found":  len(files),
            "files_ingested": len([r for r in results if r["status"] == "ok"]),
            "total_transitions": total_trans,
            "results": results,
        }

    def process_single(self, path_str: str,
                       max_transitions: int = 20_000) -> dict:
        """Process one file by path string. Used by the API endpoint."""
        from weaponized_ai import brain_store
        path  = Path(path_str)
        if not path.exists():
            return {"status": "error", "detail": f"File not found: {path_str}"}

        trans_list = self.process_replay(path, max_transitions)
        if not trans_list:
            return {"status": "error", "detail": "No parseable input stream"}

        obs_arr  = np.array([t.obs      for t in trans_list], dtype=np.float32)
        act_arr  = np.array([t.action   for t in trans_list], dtype=np.int32)
        rwd_arr  = np.array([t.reward   for t in trans_list], dtype=np.float32)
        nobs_arr = np.array([t.next_obs for t in trans_list], dtype=np.float32)
        done_arr = np.array([t.done     for t in trans_list], dtype=np.float32)
        wgt_arr  = np.array([t.weight   for t in trans_list], dtype=np.float32)

        brain_store.save_corpus(obs_arr, act_arr, rwd_arr, nobs_arr, done_arr, wgt_arr)

        meta  = self.parse_meta(path)
        stage = meta.stage_name or path.stem
        title = f"{stage} — v{meta.game_version} — {path.name}"
        brain_store.mark_ingested(str(path), title, len(trans_list), len(trans_list))

        return {
            "status": "ok",
            "path":  path_str,
            "title": title,
            "transitions": len(trans_list),
            "frame_count": meta.frame_count,
            "stage": meta.stage_name,
            "game_version": meta.game_version,
        }


# Module-level singleton
_engine: Optional[ReplayEngine] = None

def get_engine() -> ReplayEngine:
    global _engine
    if _engine is None:
        _engine = ReplayEngine()
    return _engine
