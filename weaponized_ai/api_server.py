# api_server.py
"""
FastAPI server exposing weaponized AI endpoints for RL, input, and strategy.
Run: python -m uvicorn weaponized_ai.api_server:app --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import torch

from weaponized_ai.rl_agent import get_agent, OBS_DIM, ACT_DIM
from weaponized_ai.input_controller import tap, tap_name, execute_macro, MACROS
from weaponized_ai.strategy_engine import (
    Vec2, PlayerState, predict_landing, projectile_lead,
    rank_strategies, ALL_STRATEGIES
)

app = FastAPI(title="Weaponized AI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "obs_dim": OBS_DIM, "act_dim": ACT_DIM}


# ── RL Policy Inference ───────────────────────────────────────────────────────
class InferRequest(BaseModel):
    obs: List[float]

@app.post("/policy/infer")
def policy_infer(req: InferRequest):
    agent = get_agent()
    if len(req.obs) != agent.obs_dim:
        raise HTTPException(400, f"obs must have {agent.obs_dim} elements, got {len(req.obs)}")
    action, log_prob = agent.select_action(req.obs)
    t = torch.tensor(req.obs, dtype=torch.float32)
    with torch.no_grad():
        logits = agent.policy(t).tolist()
        value  = agent.value(t).item()
    return {"action": action, "log_prob": log_prob, "logits": logits, "value": value}


# ── RL Training ───────────────────────────────────────────────────────────────
class StoreRequest(BaseModel):
    obs:      List[float]
    action:   int
    reward:   float
    log_prob: float
    done:     bool = False

@app.post("/rl/store")
def rl_store(req: StoreRequest):
    get_agent().store(req.obs, req.action, req.reward, req.log_prob, req.done)
    return {"buffered": len(get_agent().buffer)}

@app.post("/rl/train")
def rl_train():
    result = get_agent().train_step()
    return result

@app.post("/rl/save")
def rl_save():
    get_agent().save()
    return {"saved": True}

@app.post("/rl/load")
def rl_load():
    ok = get_agent().load()
    return {"loaded": ok}

@app.get("/rl/buffer_size")
def rl_buffer_size():
    return {"size": len(get_agent().buffer)}

@app.get("/rl/stats")
def rl_stats():
    return get_agent().stats()


# ── Input ─────────────────────────────────────────────────────────────────────
class TapRequest(BaseModel):
    vk:     Optional[int]  = None
    key:    Optional[str]  = None
    hold_s: float          = 0.016

@app.post("/input/tap")
def input_tap(req: TapRequest):
    if req.key:
        tap_name(req.key, req.hold_s)
        return {"status": "tapped", "key": req.key}
    elif req.vk is not None:
        tap(req.vk, req.hold_s)
        return {"status": "tapped", "vk": req.vk}
    raise HTTPException(400, "Provide 'key' (name) or 'vk' (int)")

class MacroRequest(BaseModel):
    name: str

@app.post("/input/macro")
def input_macro(req: MacroRequest):
    ok = execute_macro(req.name)
    if not ok:
        raise HTTPException(404, f"Unknown macro '{req.name}'. Available: {list(MACROS.keys())}")
    return {"status": "executing", "macro": req.name}

@app.get("/input/macros")
def list_macros():
    return {"macros": list(MACROS.keys())}


# ── Strategy ──────────────────────────────────────────────────────────────────
def _parse_state(d: dict, default_id: str = "p") -> PlayerState:
    pos = Vec2(d.get("pos", {}).get("x", 0), d.get("pos", {}).get("y", 0))
    vel = Vec2(d.get("vel", {}).get("x", 0), d.get("vel", {}).get("y", 0))
    return PlayerState(
        id=d.get("player_id", default_id),
        pos=pos, vel=vel,
        is_airborne=d.get("is_airborne", False),
        is_invulnerable=d.get("is_invulnerable", False),
        is_attacking=d.get("is_attacking", False),
        current_move=d.get("current_move"),
        last_move=d.get("last_move"),
        buff_active=d.get("buff_active", False),
        buff_expires_at=d.get("buff_expires_at", 0),
        stocks_remaining=d.get("stocks_remaining", 3),
        damage=d.get("damage", 0),
    )

class StrategyRequest(BaseModel):
    me:  dict
    opp: dict

@app.post("/strategy/rank")
def strategy_rank(req: StrategyRequest):
    me  = _parse_state(req.me,  "me")
    opp = _parse_state(req.opp, "opp")
    ranked = rank_strategies(me, opp)
    top = ranked[0] if ranked else None
    return {
        "strategy":    {"id": top.id, "name": top.name, "description": top.description,
                        "priority": top.priority, "suggested_moves": top.suggested_moves} if top else None,
        "all_ranked":  [{"id": s.id, "name": s.name, "priority": s.priority,
                         "suggested_moves": s.suggested_moves} for s in ranked],
        "lead_direction": None,
        "target_landing": predict_landing(opp),
    }

@app.get("/strategy/all")
def strategy_all():
    return [{"id": s.id, "name": s.name, "description": s.description,
             "priority": s.priority, "suggested_moves": s.suggested_moves}
            for s in ALL_STRATEGIES]

class VecRequest(BaseModel):
    x1: float; y1: float
    x2: float; y2: float

@app.post("/vec/dist")
def vec_dist(req: VecRequest):
    return {"distance": Vec2(req.x1, req.y1).dist_to(Vec2(req.x2, req.y2))}

class LeadRequest(BaseModel):
    shooter: dict
    target:  dict
    projectile_speed: float = 800.0

@app.post("/strategy/lead")
def strategy_lead(req: LeadRequest):
    shooter = _parse_state(req.shooter, "shooter")
    target  = _parse_state(req.target,  "target")
    pt = projectile_lead(shooter, target, req.projectile_speed)
    return {"lead": {"x": pt.x, "y": pt.y} if pt else None}


# ── Training Loop ─────────────────────────────────────────────────────────────
class LoopStartRequest(BaseModel):
    capture_mode:      str   = "mss"
    obs_camera_index:  int   = 1
    tick_s:            float = 0.12
    region_left:       int   = 0
    region_top:        int   = 0
    region_width:      int   = 1920
    region_height:     int   = 1080
    auto_find_window:  bool  = True   # auto-detect Brawlhalla window

@app.post("/loop/start")
def loop_start(req: LoopStartRequest):
    from weaponized_ai.training_loop import get_loop
    import weaponized_ai.training_loop as tl
    loop = get_loop()
    if loop.running:
        return {"status": "already_running", "step_count": loop.step_count}
    # Use None region when auto_find_window=True so the loop discovers it at runtime
    region = None if req.auto_find_window else {
        "left": req.region_left, "top": req.region_top,
        "width": req.region_width, "height": req.region_height
    }
    tl._loop = tl.TrainingLoop(
        capture_mode=req.capture_mode,
        capture_region=region,
        obs_camera_index=req.obs_camera_index,
        tick_s=req.tick_s,
        auto_find_window=req.auto_find_window,
    )
    tl._loop.start()
    return {"status": "started"}

@app.post("/loop/stop")
def loop_stop():
    from weaponized_ai.training_loop import get_loop
    get_loop().stop()
    return {"status": "stopping"}

@app.get("/loop/status")
def loop_status():
    from weaponized_ai.training_loop import get_loop
    return get_loop().status()


# ── OBS Management ────────────────────────────────────────────────────────────
@app.get("/obs/status")
def obs_status():
    from weaponized_ai.obs_manager import get_status
    return get_status()

@app.post("/obs/launch")
def obs_launch():
    from weaponized_ai.obs_manager import launch_obs
    return launch_obs(start_virtual_cam=True)

@app.post("/obs/install")
def obs_install():
    """Start OBS download+install in background. Poll /obs/status for progress."""
    from weaponized_ai.obs_manager import download_and_install_obs, get_status
    download_and_install_obs()
    return {"status": "download_started", "message": "Downloading OBS installer in background. Poll /obs/status for progress."}

@app.post("/obs/ensure")
def obs_ensure():
    """One-shot: check OBS → install if missing → launch if not running."""
    from weaponized_ai.obs_manager import ensure_obs
    return ensure_obs(auto_install=True, auto_launch=True)

@app.get("/obs/camera-index")
def obs_camera_index():
    from weaponized_ai.obs_manager import find_obs_camera_index
    idx = find_obs_camera_index()
    return {"camera_index": idx}

class CaptureRegionRequest(BaseModel):
    left:   int = 0
    top:    int = 0
    width:  int = 1920
    height: int = 1080

@app.post("/obs/set-region")
def obs_set_region(req: CaptureRegionRequest):
    from weaponized_ai.obs_capture import set_obs_region
    set_obs_region(req.left, req.top, req.width, req.height)
    return {"status": "ok", "region": {"left": req.left, "top": req.top, "width": req.width, "height": req.height}}

@app.post("/obs/calibrate")
def obs_calibrate():
    """
    Auto-detect the Brawlhalla window and set it as the active capture region.
    Updates obs_capture DEFAULT_REGION and returns the detected bounds.
    """
    from weaponized_ai.obs_manager import find_brawlhalla_window
    from weaponized_ai.obs_capture import set_obs_region
    from weaponized_ai.game_state_reader import set_resolution
    window = find_brawlhalla_window()
    if not window:
        raise HTTPException(404, "Brawlhalla window not found. Is the game running?")
    set_obs_region(window["left"], window["top"], window["width"], window["height"])
    set_resolution(window["width"], window["height"])
    return {"status": "calibrated", "window": window}


# ── YouTube Video Learning ────────────────────────────────────────────────────

class VideoIngestRequest(BaseModel):
    url:        str
    max_frames: int  = 8000
    transcribe: bool = True    # use Whisper to extract Brawlhalla terms from audio


class VideoPretrainRequest(BaseModel):
    n_epochs:   int   = 20
    batch_size: int   = 128
    lr:         float = 3e-4


@app.post("/video/ingest")
def video_ingest(req: VideoIngestRequest):
    """Download a YouTube URL and extract (obs, action, reward) transitions."""
    from weaponized_ai.video_learner import get_learner
    learner = get_learner()
    try:
        learner.ingest(req.url, max_frames=req.max_frames, transcribe=req.transcribe)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"status": "started", "url": req.url}


@app.get("/video/status")
def video_status():
    """Poll ingestion / pretraining progress."""
    from weaponized_ai.video_learner import get_learner
    return get_learner().get_status()


@app.post("/video/pretrain")
def video_pretrain(req: VideoPretrainRequest):
    """Run behavioural cloning on accumulated video corpus."""
    from weaponized_ai.video_learner import get_learner
    learner = get_learner()
    try:
        learner.pretrain(n_epochs=req.n_epochs, batch_size=req.batch_size, lr=req.lr)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"status": "pretraining_started", "corpus_size": len(learner.corpus)}


@app.post("/video/reset")
def video_reset():
    """Clear corpus and reset state."""
    from weaponized_ai.video_learner import get_learner
    get_learner().reset()
    return {"status": "reset"}


@app.get("/video/knowledge")
def video_knowledge():
    """
    Return all extracted Brawlhalla terms sorted by frequency.
    Each entry: {"term": str, "count": int}.
    """
    from weaponized_ai.video_learner import get_learner
    kb = get_learner().get_knowledge()
    return {"terms": [{"term": t, "count": c} for t, c in kb]}


# ── Brain Storage ─────────────────────────────────────────────────────────────

@app.get("/brain/info")
def brain_info():
    """
    Return a summary of what the AI has learned and stored on disk.
    Includes knowledge terms, corpus size, ingested video registry, model size.
    """
    from weaponized_ai import brain_store
    return brain_store.get_info()


@app.post("/brain/save")
def brain_save():
    """
    Flush the current in-memory knowledge base and BC corpus to disk immediately.
    This happens automatically after ingest and after pretrain; use this for a
    manual checkpoint.
    """
    from weaponized_ai import brain_store
    from weaponized_ai.video_learner import get_learner
    from weaponized_ai.rl_agent import get_agent
    learner = get_learner()
    saved_corpus = brain_store.save_corpus(learner.corpus, learner.corpus_weights)
    brain_store.save_knowledge(learner.knowledge_base)
    get_agent().save()
    return {
        "status":           "saved",
        "corpus_written":   saved_corpus,
        "knowledge_terms":  len(learner.knowledge_base),
    }


@app.post("/brain/reload")
def brain_reload():
    """
    Reload knowledge base and corpus from disk into the live VideoLearner.
    Useful after a manual brain/save from another process or after clearing
    in-memory state without wiping disk.
    """
    from weaponized_ai import brain_store
    from weaponized_ai.video_learner import get_learner
    learner = get_learner()
    # Clear in-memory accumulations then re-bootstrap from disk
    learner.knowledge_base.clear()
    learner.corpus.clear()
    learner.corpus_weights.clear()
    brain_store.bootstrap_learner(learner)
    return {
        "status":          "reloaded",
        "knowledge_terms": len(learner.knowledge_base),
        "corpus_size":     len(learner.corpus),
    }


class BrainClearRequest(BaseModel):
    confirm: bool = False

@app.post("/brain/clear")
def brain_clear(req: BrainClearRequest):
    """
    Permanently wipe the brain directory (knowledge, corpus, registry).
    Requires {"confirm": true} in the request body.
    Does NOT delete model.pt — use /rl/save to overwrite that separately.
    """
    if not req.confirm:
        raise HTTPException(400, "Send {\"confirm\": true} to wipe the brain store.")
    from weaponized_ai import brain_store
    from weaponized_ai.video_learner import get_learner
    brain_store.clear(confirm=True)
    # Also clear in-memory state
    learner = get_learner()
    learner.knowledge_base.clear()
    learner.corpus.clear()
    learner.corpus_weights.clear()
    return {"status": "cleared"}


# ── Brawlhalla Replay Engine ──────────────────────────────────────────────────

@app.get("/replay/scan")
def replay_scan():
    """List all .replay files found in the Brawlhalla replay folder."""
    from weaponized_ai.replay_engine import get_engine, REPLAY_DIR
    engine = get_engine()
    files  = engine.discover()
    results = []
    for p in files:
        try:
            meta = engine.parse_meta(p)
            results.append({
                "path":         str(p),
                "name":         p.name,
                "size_kb":      round(p.stat().st_size / 1024, 1),
                "parse_ok":     meta.parse_ok,
                "frame_count":  meta.frame_count,
                "stage":        meta.stage_name,
                "game_version": meta.game_version,
                "level_id":     meta.level_id,
                "game_mode":    meta.game_mode,
                "characters":   [pl.character for pl in meta.players],
            })
        except Exception as e:
            results.append({"path": str(p), "name": p.name, "error": str(e)})
    return {"replay_dir": str(REPLAY_DIR), "count": len(files), "replays": results}


class ReplayIngestRequest(BaseModel):
    path: str
    max_transitions: int = 20_000

@app.post("/replay/ingest")
def replay_ingest(req: ReplayIngestRequest):
    """Parse one replay file and push transitions to brain_store."""
    from weaponized_ai.replay_engine import get_engine
    engine = get_engine()
    result = engine.process_single(req.path, req.max_transitions)
    if result["status"] == "error":
        raise HTTPException(400, result["detail"])
    return result


@app.post("/replay/ingest_all")
def replay_ingest_all():
    """Process every not-yet-ingested replay in the AppData folder."""
    from weaponized_ai.replay_engine import get_engine
    engine = get_engine()
    result = engine.ingest_all()
    return result


# ── Live Memory Reader ────────────────────────────────────────────────────────

@app.get("/memory/info")
def memory_info():
    """Return attachment status and discovered memory addresses."""
    from weaponized_ai.brawlhalla_memory import get_reader
    reader = get_reader()
    reader.attach()   # no-op if already attached
    return reader.get_info()

@app.get("/memory/state")
def memory_state():
    """Read live game state directly from Brawlhalla process memory."""
    from weaponized_ai.brawlhalla_memory import get_reader
    reader = get_reader()
    if not reader.attach():
        raise HTTPException(503, "Brawlhalla.exe not running or not attachable")
    return reader.read_state()

@app.post("/memory/rescan")
def memory_rescan():
    """Force re-scan of memory addresses (call after a game patch)."""
    from weaponized_ai.brawlhalla_memory import get_reader
    reader = get_reader()
    if not reader.attach():
        raise HTTPException(503, "Brawlhalla.exe not running")
    reader.rescan()
    return reader.get_info()


# ── BTR Agent ─────────────────────────────────────────────────────────────────

class BTRStoreRequest(BaseModel):
    obs:      List[float]
    action:   int
    reward:   float
    next_obs: List[float]
    done:     bool = False

@app.post("/btr/store")
def btr_store(req: BTRStoreRequest):
    from weaponized_ai.rl_agent import get_btr_agent
    get_btr_agent().store(req.obs, req.action, req.reward, req.next_obs, req.done)
    return {"replay_size": len(get_btr_agent().replay)}

@app.post("/btr/action")
def btr_action(req: InferRequest):
    from weaponized_ai.rl_agent import get_btr_agent
    agent = get_btr_agent()
    if len(req.obs) != agent.obs_dim:
        raise HTTPException(400, f"obs must have {agent.obs_dim} elements")
    action = agent.select_action(req.obs)
    return {"action": action}

@app.get("/btr/stats")
def btr_stats():
    from weaponized_ai.rl_agent import get_btr_agent
    return get_btr_agent().stats()

@app.post("/btr/save")
def btr_save():
    from weaponized_ai.rl_agent import get_btr_agent
    get_btr_agent().save()
    return {"saved": True}

@app.post("/btr/load")
def btr_load():
    from weaponized_ai.rl_agent import get_btr_agent
    ok = get_btr_agent().load()
    return {"loaded": ok}

@app.post("/btr/pretrain")
def btr_pretrain():
    """BC pre-train BTR agent on current brain_store corpus."""
    from weaponized_ai.rl_agent import get_btr_agent
    from weaponized_ai import brain_store
    import numpy as np
    data = brain_store.load_corpus()
    if data is None or len(data[0]) < 8:
        raise HTTPException(400, "Brain corpus is empty — ingest replays or videos first")
    obs_arr, act_arr, _, _, _, wgt_arr = data
    loss = get_btr_agent().pretrain_bc(obs_arr, act_arr, wgt_arr, epochs=3)
    get_btr_agent().save()
    return {"status": "ok", "bc_loss": round(loss, 5), "samples": len(obs_arr)}

