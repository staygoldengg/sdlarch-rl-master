"""Smoke test for the updated replay_engine.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from weaponized_ai.replay_engine import (
    get_engine, REPLAY_DIR, _STAGE_BOUNDS, _parse_filename_meta,
    _parse_header, _extract_inputs, PlayerPhysics, _make_obs,
)
from pathlib import Path

print("[TEST] replay_engine smoke test")

# 1: REPLAY_DIR
assert REPLAY_DIR.exists(), f"REPLAY_DIR missing: {REPLAY_DIR}"
print(f"  [OK] REPLAY_DIR = {REPLAY_DIR}")

# 2: filename parser
cases = [
    ("[10.06] WesternAirTemple (96).replay",  "10.06", "WesternAirTemple"),
    ("[10.06] SmallWorld'sEnd (49).replay",   "10.06", "SmallWorld'sEnd"),
    ("[10.06] Apocalypse.replay",             "10.06", "Apocalypse"),
    ("bad_file.replay",                       "",      ""),
]
for fname, exp_v, exp_s in cases:
    v, s = _parse_filename_meta(Path(fname))
    assert v == exp_v and s == exp_s, f"FAIL {fname}: got ({v!r},{s!r})"
print("  [OK] _parse_filename_meta")

# 3: stage bounds
for name, bounds in _STAGE_BOUNDS.items():
    assert len(bounds) == 2 and bounds[0] > 0, f"Bad bounds for {name}"
print(f"  [OK] _STAGE_BOUNDS ({len(_STAGE_BOUNDS)} stages)")

# 4: physics uses stage_half_w
p = PlayerPhysics(stage_half_w=700.0)
assert p.stage_half_w == 700.0
obs = p.obs_vec()
assert len(obs) == 6
print("  [OK] PlayerPhysics(stage_half_w)")

# 5: full engine pipeline on 5 real files
engine = get_engine()
files = engine.discover()
assert len(files) > 0, "No files discovered"
print(f"  [OK] discover() found {len(files)} files")

for f in files[:5]:
    meta = engine.parse_meta(f)
    assert meta.parse_ok,              f"parse failed: {f.name}: {meta.error}"
    assert meta.stage_name,            f"no stage: {f.name}"
    assert meta.game_version == "10.06", f"bad version: {f.name}"
    assert meta.frame_count > 0,       f"zero frames: {f.name}"
    trans = engine.process_replay(f, max_transitions=200)
    assert len(trans) > 0,             f"no transitions: {f.name}"
    t = trans[0]
    assert len(t.obs) == 18,           f"wrong obs len: {f.name}"
    assert 0 <= t.action <= 15,        f"bad action: {f.name}"
    print(f"    {f.name[-42:]:<42}  frames={meta.frame_count:5d}  trans={len(trans)}")

print()
print("All checks passed.")
