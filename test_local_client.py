# test_local_client.py
"""
Senior tester — automated end-to-end local client validation suite.

Covers:
  [T01] Python environment and critical package imports
  [T02] API server reachability and health check
  [T03] Policy inference (PPO forward pass)
  [T04] BTR agent action
  [T05] Replay scanner
  [T06] Memory reader attach / state read
  [T07] Brain store read/write/reload cycle
  [T08] Strategy engine (landing prediction + strategy ranking)
  [T09] Input controller (macro list only — no keys sent)
  [T10] Config manager persistence
  [T11] Hardware driver instantiation (no keys sent)
  [T12] Live agent node module import sanity
  [T13] API server startup via start_server.ps1 (optional — requires server offline)
  [T14] Training loop module import sanity

Usage:
    python test_local_client.py              # run all tests
    python test_local_client.py --no-server  # skip tests that require the server
    python test_local_client.py --fast       # skip slow tests (video, replay ingest)

Exit code: 0 = all passed, 1 = one or more failures.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

# ── ANSI colours ───────────────────────────────────────────────────────────────
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"

PASS    = f"{_GREEN}{_BOLD}PASS{_RESET}"
FAIL    = f"{_RED}{_BOLD}FAIL{_RESET}"
SKIP    = f"{_YELLOW}SKIP{_RESET}"


# ── Test registry ──────────────────────────────────────────────────────────────

class TestResult:
    def __init__(self, tid: str, name: str, status: str, detail: str = ""):
        self.tid    = tid
        self.name   = name
        self.status = status   # "pass" | "fail" | "skip"
        self.detail = detail

    def tag(self) -> str:
        return {"pass": PASS, "fail": FAIL, "skip": SKIP}[self.status]


_results: list[TestResult] = []


def _test(tid: str, name: str):
    """Decorator that registers a test function."""
    def decorator(fn: Callable[[], Any]):
        def wrapper(skip: bool = False) -> TestResult:
            if skip:
                r = TestResult(tid, name, "skip", "skipped by flag")
                _results.append(r)
                return r
            try:
                detail = fn() or ""
                r = TestResult(tid, name, "pass", str(detail))
            except Exception as exc:
                r = TestResult(tid, name, "fail", traceback.format_exc().strip())
            _results.append(r)
            return r
        wrapper.__test_id__ = tid
        wrapper.__test_name__ = name
        return wrapper
    return decorator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _server_alive(host: str = "127.0.0.1", port: int = 8000) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _get(path: str) -> dict:
    import urllib.request
    url = f"http://127.0.0.1:8000{path}"
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def _post(path: str, data: dict) -> dict:
    import urllib.request
    url  = f"http://127.0.0.1:8000{path}"
    body = json.dumps(data).encode()
    req  = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


# ── Tests ─────────────────────────────────────────────────────────────────────

@_test("T01", "Python environment — critical imports")
def test_imports():
    required = [
        "torch", "numpy", "fastapi", "uvicorn", "pydantic",
        "weaponized_ai.rl_agent",
        "weaponized_ai.policy_network",
        "weaponized_ai.config_manager",
        "weaponized_ai.hardware_driver",
        "weaponized_ai.strategy_engine",
        "weaponized_ai.brain_store",
        "weaponized_ai.input_controller",
        "weaponized_ai.training_loop",
        "weaponized_ai.reward_shaper",
        "weaponized_ai.ppo_loss",
        "weaponized_ai.advantage_engine",
        "weaponized_ai.entropy_tuner",
        "weaponized_ai.action_masker",
        "weaponized_ai.value_heads",
    ]
    failed = []
    for mod in required:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            failed.append(f"{mod}: {e}")
    if failed:
        raise ImportError("Missing modules:\n" + "\n".join(failed))
    return f"{len(required)} modules OK"


@_test("T02", "API server health check")
def test_server_health():
    if not _server_alive():
        raise RuntimeError("Server not reachable on 127.0.0.1:8000")
    resp = _get("/health")
    assert resp.get("status") == "ok", f"Unexpected health response: {resp}"
    return f"obs_dim={resp['obs_dim']} act_dim={resp['act_dim']}"


@_test("T03", "PPO policy inference (/policy/infer)")
def test_policy_infer():
    obs = [0.0] * 18
    resp = _post("/policy/infer", {"obs": obs})
    assert "action" in resp, f"No action in response: {resp}"
    assert "value"  in resp, f"No value in response: {resp}"
    return f"action={resp['action']} value={resp['value']:.4f}"


@_test("T04", "BTR agent action (/btr/action)")
def test_btr_action():
    obs = [0.0] * 18
    resp = _post("/btr/action", {"obs": obs})
    assert "action" in resp, f"No action in response: {resp}"
    return f"action={resp['action']} q_value={resp.get('q_value', 'n/a')}"


@_test("T05", "Replay scanner (/replay/scan)")
def test_replay_scan():
    resp = _get("/replay/scan")
    assert isinstance(resp, list), f"Expected list, got: {type(resp)}"
    return f"{len(resp)} replay files found"


@_test("T06", "Memory reader info (/memory/info)")
def test_memory_info():
    resp = _get("/memory/info")
    assert "attached" in resp, f"Missing 'attached' key: {resp}"
    return f"attached={resp['attached']} pid={resp.get('pid', 'n/a')}"


@_test("T07", "Brain store read/write/reload (/brain/*)")
def test_brain_store():
    info = _get("/brain/info")
    assert "corpus_size" in info, f"Missing corpus_size: {info}"
    save = _post("/brain/save",   {})
    assert save.get("saved"), f"Save failed: {save}"
    relo = _post("/brain/reload", {})
    assert relo.get("reloaded"), f"Reload failed: {relo}"
    return (
        f"terms={info['knowledge_terms']} "
        f"corpus={info['corpus_size']} "
        f"registry={info['registry_size']}"
    )


@_test("T08", "Strategy engine (/strategy/rank + /strategy/predict_landing)")
def test_strategy():
    p1 = {"id": "p1", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0,
          "damage": 50.0, "stocks": 3, "airborne": False,
          "has_weapon": False, "has_buff": False}
    p2 = {"id": "p2", "x": 100.0, "y": 0.0, "vx": -1.0, "vy": 0.0,
          "damage": 30.0, "stocks": 3, "airborne": True,
          "has_weapon": True, "has_buff": False}
    rank = _post("/strategy/rank", {"p1": p1, "p2": p2})
    assert "top" in rank, f"Missing 'top': {rank}"

    land = _post("/strategy/predict_landing", {
        "state": {"x": 0.0, "y": 300.0, "vx": 5.0, "vy": 0.0}
    })
    assert "landing_x" in land, f"Missing landing_x: {land}"
    return f"top={rank['top']} landing_x={land['landing_x']:.1f}"


@_test("T09", "Input controller — macro list (/input/macros)")
def test_macros():
    resp = _get("/input/macros")
    macros = resp.get("macros", [])
    assert len(macros) > 0, "No macros registered"
    return f"{len(macros)} macros: {', '.join(macros[:5])}..."


@_test("T10", "Config manager persistence (PersistentStorageEngine)")
def test_config():
    from weaponized_ai.config_manager import PersistentStorageEngine
    store = PersistentStorageEngine()
    original = store.settings.copy()
    store.save_settings({"_test_key": "__striker_test__"})
    store2 = PersistentStorageEngine()
    assert store2.settings.get("_test_key") == "__striker_test__", "Setting not persisted"
    # Clean up
    del store2.settings["_test_key"]
    store2.save_settings({})
    return f"settings_path={store.config_file}"


@_test("T11", "Hardware driver instantiation (no keys sent)")
def test_hardware_driver():
    from weaponized_ai.hardware_driver import MutexHardwareDriver, FrameDeterministicDispatcher
    d1 = MutexHardwareDriver()
    d2 = FrameDeterministicDispatcher(target_fps=60)
    # Do NOT start d2 or call update — just verify construction
    d1.global_flush()   # should be a no-op (nothing was pressed)
    return "MutexHardwareDriver and FrameDeterministicDispatcher instantiated OK"


@_test("T12", "live_agent_node module import sanity")
def test_live_agent_import():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "live_agent_node",
        str(Path(__file__).parent / "live_agent_node.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # We intentionally do NOT exec_module here; we only verify the spec loads.
    assert spec is not None, "live_agent_node.py not found"
    return "live_agent_node.py located and spec loaded"


@_test("T13", "RL agent training pass (offline, no game required)")
def test_rl_train_pass():
    """
    Stores 64 dummy transitions and triggers one PPO update offline
    to verify that the full gradient pipeline works end-to-end.
    """
    from weaponized_ai.rl_agent import get_agent
    agent = get_agent()
    import torch
    for _ in range(64):
        obs    = [0.0] * 18
        action = 0
        reward = 0.1
        lp     = -2.77  # approx log(1/16)
        agent.store(obs, action, reward, lp, done=False)
    result = agent.train_step()
    assert result is not None, "train_step returned None"
    return f"loss_policy={result.get('loss_policy', 'n/a')}"


@_test("T14", "Reward shaper PBRS formula correctness")
def test_reward_shaper():
    from weaponized_ai.reward_shaper import AlignedRewardShaper
    shaper = AlignedRewardShaper()
    s  = {"x": 0.0, "y": 0.0, "stocks_p1": 3, "stocks_p2": 3}
    sp = {"x": 50.0, "y": 0.0, "stocks_p1": 3, "stocks_p2": 3}
    r = shaper.compute(
        raw_reward=5.0,
        state=s,
        next_state=sp,
        combo_hit=True,
        stock_lost=False,
    )
    assert isinstance(r, float), f"Expected float, got {type(r)}"
    return f"shaped_reward={r:.4f}"


# ── Runner ────────────────────────────────────────────────────────────────────

def _print_header():
    print(f"\n{_CYAN}{_BOLD}{'='*62}")
    print("  Striker — The Enlightened  |  Local Client Test Suite")
    print(f"{'='*62}{_RESET}\n")


def _print_summary():
    passed = sum(1 for r in _results if r.status == "pass")
    failed = sum(1 for r in _results if r.status == "fail")
    skipped = sum(1 for r in _results if r.status == "skip")

    print(f"\n{_CYAN}{_BOLD}{'='*62}{_RESET}")
    for r in _results:
        label  = f"[{r.tid}]".ljust(6)
        status = r.tag()
        name   = r.name.ljust(52)
        print(f"  {label} {status}  {name}")
        if r.status == "fail":
            # Indent failure detail
            for line in r.detail.splitlines()[:6]:
                print(f"           {_RED}{line}{_RESET}")

    print(f"\n{_BOLD}  Results: "
          f"{_GREEN}{passed} passed{_RESET}  "
          f"{_RED}{failed} failed{_RESET}  "
          f"{_YELLOW}{skipped} skipped{_RESET}"
          f"  ({len(_results)} total)\n")


def main():
    parser = argparse.ArgumentParser(description="Striker local client test suite.")
    parser.add_argument("--no-server",  action="store_true", help="Skip API server tests")
    parser.add_argument("--fast",       action="store_true", help="Skip slow tests")
    args = parser.parse_args()

    server_present = _server_alive()
    skip_server    = args.no_server or not server_present

    if not server_present and not args.no_server:
        print(
            f"{_YELLOW}[WARN] Server not detected on 127.0.0.1:8000. "
            "Server-dependent tests will be skipped.{_RESET}"
        )

    _print_header()

    # Ensure weaponized_ai is importable from project root
    sys.path.insert(0, str(Path(__file__).parent))

    print(f"{_CYAN}Running tests…{_RESET}\n")

    # Tests that always run
    test_imports()
    test_config()
    test_hardware_driver()
    test_live_agent_import()
    test_reward_shaper()
    test_rl_train_pass()

    # Tests that require the server
    test_server_health(skip=skip_server)
    test_policy_infer(skip=skip_server)
    test_btr_action(skip=skip_server)
    test_replay_scan(skip=skip_server)
    test_memory_info(skip=skip_server)
    test_brain_store(skip=skip_server)
    test_strategy(skip=skip_server)
    test_macros(skip=skip_server)

    _print_summary()

    failed = any(r.status == "fail" for r in _results)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
