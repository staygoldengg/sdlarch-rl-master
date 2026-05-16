# brain_store.py
"""
Persistent AI learning storage.

Directory layout (weaponized_ai/brain/):
  knowledge.json   — Brawlhalla term→count dict, accumulated across all sessions.
                     Grows every time a video is ingested with Whisper active.
  corpus.npz       — Behavioural-Cloning transitions (obs, acts, rwds, next_obs,
                     dones, weights) as compressed numpy arrays. Capped at
                     MAX_CORPUS entries; newer transitions replace oldest.
  registry.json    — Metadata for every ingested video so we know what has
                     already been processed and how many transitions it produced.

Auto-save triggers:
  • After every successful video ingest  → knowledge + registry
  • After every BC pretrain              → corpus + knowledge
  • POST /brain/save                     → everything (manual flush)

Loading on startup:
  get_learner() in video_learner.py calls brain_store.bootstrap_learner(learner)
  which pre-populates knowledge_base and corpus from disk so the AI continues
  exactly where it left off after a server restart.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

BRAIN_DIR  = Path(
    os.environ.get("STRIKER_DATA_DIR") or (Path(__file__).parent / "brain")
)
MAX_CORPUS = 50_000   # maximum transitions kept on disk

_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure() -> None:
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Knowledge base ────────────────────────────────────────────────────────────

def load_knowledge() -> Dict[str, int]:
    """Load the accumulated term→count dict from disk."""
    path = BRAIN_DIR / "knowledge.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_knowledge(kb: Dict[str, int]) -> None:
    """Overwrite knowledge.json with the full current in-memory dict."""
    _ensure()
    with _lock:
        with open(BRAIN_DIR / "knowledge.json", "w", encoding="utf-8") as f:
            json.dump(kb, f, indent=2, ensure_ascii=False, sort_keys=True)


# ── BC Corpus ─────────────────────────────────────────────────────────────────

def load_corpus() -> Tuple[list, List[float]]:
    """
    Load transitions and weights from corpus.npz.
    Returns (list_of_Transition, list_of_float) or ([], []) if nothing stored.
    Import is deferred to avoid circular dependency at module load time.
    """
    path = BRAIN_DIR / "corpus.npz"
    if not path.exists():
        return [], []
    try:
        data = np.load(str(path), allow_pickle=False)
        obs_arr      = data["obs"]       # (N, 18) float32
        acts_arr     = data["acts"]      # (N,)    int32
        rwds_arr     = data["rwds"]      # (N,)    float32
        next_obs_arr = data["next_obs"]  # (N, 18) float32
        dones_arr    = data["dones"]     # (N,)    bool
        weights      = data["weights"].tolist()

        from weaponized_ai.video_learner import Transition
        corpus = [
            Transition(
                obs_arr[i],
                int(acts_arr[i]),
                float(rwds_arr[i]),
                next_obs_arr[i],
                bool(dones_arr[i]),
            )
            for i in range(len(obs_arr))
        ]
        return corpus, weights
    except Exception:
        return [], []


def save_corpus(corpus: list, weights: List[float]) -> int:
    """
    Persist corpus + weights as a compressed npz file.
    Caps at MAX_CORPUS by keeping the newest entries.
    Returns the number of transitions written.
    """
    if not corpus:
        return 0
    _ensure()

    # Keep newest MAX_CORPUS entries
    if len(corpus) > MAX_CORPUS:
        corpus  = corpus[-MAX_CORPUS:]
        weights = weights[-MAX_CORPUS:]

    obs_arr      = np.array([t.obs      for t in corpus], dtype=np.float32)
    acts_arr     = np.array([t.action   for t in corpus], dtype=np.int32)
    rwds_arr     = np.array([t.reward   for t in corpus], dtype=np.float32)
    next_obs_arr = np.array([t.next_obs for t in corpus], dtype=np.float32)
    dones_arr    = np.array([t.done     for t in corpus], dtype=bool)
    wts_arr      = np.array(weights,                      dtype=np.float32)

    with _lock:
        np.savez_compressed(
            str(BRAIN_DIR / "corpus.npz"),
            obs=obs_arr, acts=acts_arr, rwds=rwds_arr,
            next_obs=next_obs_arr, dones=dones_arr, weights=wts_arr,
        )
    return len(corpus)


# ── Ingestion Registry ────────────────────────────────────────────────────────

def load_registry() -> List[Dict[str, Any]]:
    """Load the list of ingested-video metadata records."""
    path = BRAIN_DIR / "registry.json"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def is_already_ingested(url: str) -> bool:
    """Return True if this URL has been ingested at least once before."""
    return any(e.get("url") == url for e in load_registry())


def mark_ingested(url: str, title: str, frames: int, transitions: int) -> None:
    """Upsert an ingestion record for this URL."""
    _ensure()
    with _lock:
        registry = load_registry()
        for entry in registry:
            if entry.get("url") == url:
                entry["times_ingested"]  = entry.get("times_ingested", 1) + 1
                entry["last_ingested"]   = _now()
                entry["frames"]          = frames
                entry["transitions"]     = transitions
                break
        else:
            registry.append({
                "url":            url,
                "title":          title,
                "frames":         frames,
                "transitions":    transitions,
                "times_ingested": 1,
                "first_ingested": _now(),
                "last_ingested":  _now(),
            })
        with open(BRAIN_DIR / "registry.json", "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_learner(learner) -> None:
    """
    Pre-populate a freshly created VideoLearner with everything stored on disk.
    Called once by get_learner() so the AI picks up where it left off after
    a server restart.
    """
    # Knowledge base
    kb = load_knowledge()
    if kb:
        learner.knowledge_base.update(kb)

    # BC corpus
    corpus, weights = load_corpus()
    if corpus:
        learner.corpus.extend(corpus)
        learner.corpus_weights.extend(weights)


# ── Summary / Info ────────────────────────────────────────────────────────────

def get_info() -> Dict[str, Any]:
    """Return a dict suitable for the /brain/info endpoint."""
    kb           = load_knowledge()
    registry     = load_registry()
    corpus_path  = BRAIN_DIR / "corpus.npz"
    model_path   = Path(__file__).parent / "model.pt"

    corpus_size  = 0
    corpus_bytes = 0
    if corpus_path.exists():
        corpus_bytes = corpus_path.stat().st_size
        try:
            data        = np.load(str(corpus_path), allow_pickle=False)
            corpus_size = int(len(data["acts"]))
        except Exception:
            pass

    return {
        "knowledge_terms": len(kb),
        "top_terms":       sorted(kb.items(), key=lambda x: -x[1])[:20],
        "corpus_size":     corpus_size,
        "corpus_bytes":    corpus_bytes,
        "corpus_mb":       round(corpus_bytes / (1024 * 1024), 2),
        "videos_ingested": len(registry),
        "registry":        registry,
        "model_exists":    model_path.exists(),
        "model_bytes":     model_path.stat().st_size if model_path.exists() else 0,
        "model_mb":        round(model_path.stat().st_size / (1024 * 1024), 2) if model_path.exists() else 0.0,
        "brain_dir":       str(BRAIN_DIR),
    }


# ── Clear ─────────────────────────────────────────────────────────────────────

def clear(confirm: bool = False) -> None:
    """Wipe the entire brain directory. Requires confirm=True."""
    if not confirm:
        raise ValueError("Pass confirm=True to wipe the brain store.")
    import shutil
    if BRAIN_DIR.exists():
        shutil.rmtree(BRAIN_DIR)
