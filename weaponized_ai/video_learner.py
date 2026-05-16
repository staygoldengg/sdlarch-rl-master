# video_learner.py
"""
YouTube → Brawlhalla knowledge pipeline.

Stages
------
1. DOWNLOAD  — yt-dlp pulls the best ≤1080p stream to a temp mp4.
2. EXTRACT   — cv2 reads frames at ~4 fps; game_state_reader converts each to
               an 18-dim observation vector.
3. INFER     — Optical-flow + heuristics label each frame with the most likely
               player action (action index matching ACTION_MACROS in training_loop).
4. FILTER    — Only frames where the game is "active" (both players alive, damage
               numbers visible) are kept.
5. STORE     — Transitions (obs, action, next_obs, reward) written to an in-memory
               replay corpus separate from the live RL buffer.
6. PRETRAIN  — Behavioural Cloning (BC): supervised cross-entropy on (obs→action)
               pairs to seed the policy before self-play RL.

All heavy work runs in a background thread; `get_status()` streams live progress.
"""

import os
import tempfile
import threading
import time
import logging
import math
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

log = logging.getLogger("video_learner")

# ── Action indices ────────────────────────────────────────────────────────────
_ACT_NLIGHT = 0
_ACT_SLIGHT = 1
_ACT_DLIGHT = 2
_ACT_NHEAVY = 3
_ACT_SHEAVY = 4
_ACT_DHEAVY = 5
_ACT_JUMP   = 6
_ACT_DASH_R = 12
_ACT_DASH_L = 13
_ACT_IDLE   = 0

# ── Brawlhalla vocabulary: spoken/shown term → action index (None = no action) ─
_BRAWLHALLA_VOCAB: Dict[str, Optional[int]] = {
    # Lights
    "nlight": _ACT_NLIGHT, "n light": _ACT_NLIGHT, "n-light": _ACT_NLIGHT,
    "neutral light": _ACT_NLIGHT, "nair": _ACT_NLIGHT,
    "slight": _ACT_SLIGHT, "s light": _ACT_SLIGHT, "s-light": _ACT_SLIGHT,
    "side light": _ACT_SLIGHT, "sair": _ACT_SLIGHT,
    "dlight": _ACT_DLIGHT, "d light": _ACT_DLIGHT, "d-light": _ACT_DLIGHT,
    "down light": _ACT_DLIGHT, "dair": _ACT_DLIGHT,
    # Heavies / Sigs
    "nheavy": _ACT_NHEAVY, "nsig": _ACT_NHEAVY, "n sig": _ACT_NHEAVY,
    "neutral sig": _ACT_NHEAVY, "neutral heavy": _ACT_NHEAVY,
    "sheavy": _ACT_SHEAVY, "ssig": _ACT_SHEAVY, "s sig": _ACT_SHEAVY,
    "side sig": _ACT_SHEAVY, "side heavy": _ACT_SHEAVY,
    "dheavy": _ACT_DHEAVY, "dsig": _ACT_DHEAVY, "d sig": _ACT_DHEAVY,
    "down sig": _ACT_DHEAVY, "down heavy": _ACT_DHEAVY,
    # Movement
    "jump": _ACT_JUMP, "double jump": _ACT_JUMP, "short hop": _ACT_JUMP,
    "gravity cancel": _ACT_JUMP, "gc": _ACT_JUMP,
    "wall jump": _ACT_JUMP, "recovery": _ACT_JUMP,
    "dash": _ACT_DASH_R,
    # Tactical / positional — no action index, pure knowledge-base entries
    "combo": None, "true combo": None, "punish": None, "confirm": None,
    "zero to death": None, "0 to death": None, "zero to ko": None,
    "edgeguard": None, "edge guard": None, "offstage": None,
    "ko": None, "kill": None, "stocks": None,
    "dodge": None, "dodge cancel": None,
    "wall tech": None, "wt": None, "tech": None,
    "platform": None, "platform cancel": None, "plat": None,
    "optimal": None, "weapon throw": None, "throw": None,
}

# Transitions near these spoken terms get 3× weight in BC training
_QUALITY_TERMS = frozenset({
    "combo", "true combo", "punish", "confirm", "zero to death",
    "0 to death", "zero to ko", "edgeguard", "edge guard",
    "gravity cancel", "gc", "platform cancel", "optimal",
})

# Detection tuning
_FLOW_THRESH       = 2.5    # pixels/frame for meaningful player movement
_FLASH_SIGMA       = 1.8    # brightness z-score → attack flash
_SCENE_DIFF_THRESH = 4.0    # mean-pixel diff below which a frame is a duplicate
_TERM_WINDOW_S     = 1.5    # ± seconds around a frame to search transcript


# ─────────────────────────────────────────────────────────────────────────────
class Transition:
    __slots__ = ("obs", "action", "reward", "next_obs", "done")
    def __init__(self, obs, action, reward, next_obs, done):
        self.obs      = obs
        self.action   = action
        self.reward   = reward
        self.next_obs = next_obs
        self.done     = done


class VideoLearner:
    """
    Stateful singleton — one background thread at a time.
    """

    def __init__(self):
        self._lock    = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        # Progress / status
        self.state        = "idle"   # idle | downloading | extracting | pretraining | done | error
        self.progress     = 0.0      # 0..1
        self.current_url  = ""
        self.video_title  = ""
        self.frames_total = 0
        self.frames_done  = 0
        self.transitions_extracted = 0
        self.bc_epochs_done  = 0
        self.bc_loss_last    = 0.0
        self.bc_loss_initial = 0.0
        self.error_msg    = ""
        self.log_tail: List[str] = []   # last 20 log lines shown in UI

        # Accumulated corpus from all ingested videos
        self.corpus: List[Transition] = []
        # Parallel sample weights for BC (3× for quality frames near combo/punish commentary)
        self.corpus_weights: List[float] = []
        # Knowledge base: term → count across all ingested videos
        self.knowledge_base: Dict[str, int] = {}
        # Whisper transcript segments from the current/last ingest
        self._timed_segs: List[Dict[str, Any]] = []
        self.transcribed = False

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest(self, url: str, max_frames: int = 8000, transcribe: bool = True):
        """Download and extract transitions from a YouTube URL (non-blocking).
        Every frame is evaluated; a scene-change gate skips duplicate frames.
        If transcribe=True the audio is passed through Whisper so spoken terms
        (e.g. "nlight", "gravity cancel", "combo") override optical-flow labels.
        """
        with self._lock:
            if self.state in ("downloading", "extracting"):
                raise RuntimeError("Already processing a video. Wait or call reset().")
            self.current_url = url
            self.state       = "downloading"
            self.progress    = 0.0
            self.error_msg   = ""
            self.log_tail    = []
        self._thread = threading.Thread(
            target=self._run_ingest,
            args=(url, max_frames, transcribe),
            daemon=True,
            name="VideoIngester",
        )
        self._thread.start()

    def pretrain(self, n_epochs: int = 20, batch_size: int = 128, lr: float = 3e-4):
        """Run behavioural cloning on the accumulated corpus (non-blocking)."""
        with self._lock:
            if len(self.corpus) < batch_size:
                raise RuntimeError(f"Need at least {batch_size} transitions (have {len(self.corpus)}).")
            if self.state == "pretraining":
                raise RuntimeError("Already pretraining.")
            self.state       = "pretraining"
            self.progress    = 0.0
            self.bc_epochs_done = 0
        self._thread = threading.Thread(
            target=self._run_pretrain,
            args=(n_epochs, batch_size, lr),
            daemon=True,
            name="BCTrainer",
        )
        self._thread.start()

    def reset(self):
        """Clear in-memory session state, then reload persisted knowledge+corpus
        from the brain store so accumulated learning is never lost."""
        with self._lock:
            self.corpus.clear()
            self.corpus_weights.clear()
            self.knowledge_base.clear()
            self._timed_segs.clear()
            self.state                 = "idle"
            self.progress              = 0.0
            self.current_url           = ""
            self.video_title           = ""
            self.frames_total          = 0
            self.frames_done           = 0
            self.transitions_extracted = 0
            self.bc_epochs_done        = 0
            self.bc_loss_last          = 0.0
            self.bc_loss_initial       = 0.0
            self.log_tail              = []
            self.transcribed           = False
        # Reload accumulated knowledge + corpus from brain store
        from weaponized_ai import brain_store
        brain_store.bootstrap_learner(self)

    def get_status(self) -> Dict[str, Any]:
        # Top-10 knowledge-base terms by frequency
        top_terms = sorted(self.knowledge_base.items(), key=lambda x: -x[1])[:10]
        return {
            "state":                 self.state,
            "progress":              round(self.progress, 3),
            "current_url":           self.current_url,
            "video_title":           self.video_title,
            "frames_total":          self.frames_total,
            "frames_done":           self.frames_done,
            "transitions_extracted": self.transitions_extracted,
            "corpus_size":           len(self.corpus),
            "bc_epochs_done":        self.bc_epochs_done,
            "bc_loss_last":          round(self.bc_loss_last, 5),
            "bc_loss_initial":       round(self.bc_loss_initial, 5),
            "transcribed":           self.transcribed,
            "knowledge_terms":       len(self.knowledge_base),
            "top_terms":             top_terms,
            "error":                 self.error_msg,
            "log":                   list(self.log_tail[-20:]),
        }

    def get_knowledge(self) -> List[Tuple[str, int]]:
        """Return all knowledge-base terms sorted by frequency (descending)."""
        return sorted(self.knowledge_base.items(), key=lambda x: -x[1])

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log(self, msg: str):
        log.info(msg)
        ts = time.strftime("%H:%M:%S")
        with self._lock:
            self.log_tail.append(f"[{ts}] {msg}")

    def _run_ingest(self, url: str, max_frames: int, transcribe: bool):
        try:
            import yt_dlp                               # lazy
            import cv2                                  # lazy
            import numpy as np                          # lazy
            from weaponized_ai.game_state_reader import read_state, compute_reward  # lazy

            tmp_dir  = tempfile.mkdtemp(prefix="bhalla_vid_")
            out_path = os.path.join(tmp_dir, "video.%(ext)s")

            # ── Stage 1: Download ─────────────────────────────────────────────
            self._log(f"Downloading: {url}")
            ydl_opts = {
                "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
                "outtmpl": out_path,
                "quiet": True,
                "no_warnings": True,
                "merge_output_format": "mp4",
                "progress_hooks": [self._ydl_progress_hook],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                self.video_title = info.get("title", "Unknown")

            vid_files = (list(Path(tmp_dir).glob("*.mp4")) +
                         list(Path(tmp_dir).glob("*.webm")) +
                         list(Path(tmp_dir).glob("*.mkv")))
            if not vid_files:
                raise FileNotFoundError("yt-dlp did not produce an output file.")
            video_path = str(vid_files[0])
            self._log(f"Downloaded: {self.video_title!r} → {Path(video_path).name}")

            # ── Stage 2: Whisper transcription (optional) ─────────────────────
            timed_segs: List[Dict[str, Any]] = []
            if transcribe:
                with self._lock:
                    self.state    = "transcribing"
                    self.progress = 0.10
                timed_segs = self._try_transcribe(video_path)
                with self._lock:
                    self._timed_segs  = timed_segs
                    self.transcribed  = len(timed_segs) > 0

            # Build frame-number → (term_action_override, is_quality) lookup
            term_lookup: Dict[int, Tuple[Optional[int], bool]] = {}
            if timed_segs:
                cap_probe = cv2.VideoCapture(video_path)
                _fps_probe = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
                cap_probe.release()
                term_lookup = self._build_term_lookup(timed_segs, _fps_probe)
                self._log(f"Term lookup: {len(term_lookup)} frames have transcript guidance")

            # ── Stage 3: Frame-by-frame extraction ───────────────────────────
            with self._lock:
                self.state    = "extracting"
                self.progress = 0.15

            cap = cv2.VideoCapture(video_path)
            total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps              = cap.get(cv2.CAP_PROP_FPS) or 30.0

            with self._lock:
                self.frames_total = min(max_frames, total_vid_frames)

            self._log(f"Video: {total_vid_frames} frames @ {fps:.1f}fps — "
                      f"processing every frame (scene-change gated)")

            transitions: List[Transition]   = []
            weights:     List[float]        = []
            new_kb:      Dict[str, int]     = {}

            prev_state: Optional[dict] = None
            prev_gray:  Optional[Any]  = None
            mean_bright = 80.0
            var_bright  = 400.0
            frame_idx   = 0
            kept        = 0

            while kept < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # Resize to 1280×720 for consistent HUD layout
                frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # ── Scene-change gate: skip duplicate / near-identical frames ─
                if prev_gray is not None:
                    diff = float(cv2.absdiff(gray, prev_gray).mean())
                    if diff < _SCENE_DIFF_THRESH:
                        continue   # frame too similar to previous — skip

                kept += 1

                try:
                    curr_state = read_state(frame)
                except Exception:
                    prev_state = None
                    prev_gray  = gray
                    continue

                # ── Optical-flow action inference ─────────────────────────────
                action    = _ACT_IDLE
                flow_conf = 0.5   # confidence: 0=low, 1=high (boosted by transcript)

                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    mean_mag  = float(mag.mean())
                    w = frame.shape[1] // 3
                    p1_mag = float(mag[:, :w].mean())
                    p1_ang = float(ang[:, :w].mean()) * 180.0 / math.pi

                    bright = float(gray.mean())
                    delta        = bright - mean_bright
                    mean_bright += delta * 0.05
                    var_bright  += (bright - mean_bright) * delta * 0.05
                    std_bright   = math.sqrt(max(var_bright, 1.0))
                    z_score      = (bright - mean_bright) / std_bright

                    if p1_mag > _FLOW_THRESH:
                        if 315 <= p1_ang or p1_ang < 45:
                            action = _ACT_DASH_R
                        elif 135 <= p1_ang < 225:
                            action = _ACT_DASH_L
                        elif 225 <= p1_ang < 315:
                            action = _ACT_JUMP
                        else:
                            action = _ACT_SLIGHT
                        flow_conf = 0.7
                    elif z_score > _FLASH_SIGMA:
                        action    = _ACT_SHEAVY if mean_mag > _FLOW_THRESH * 2 else _ACT_NLIGHT
                        flow_conf = 0.6
                    elif curr_state.get("p1_airborne"):
                        action = _ACT_JUMP

                # ── Transcript term override ──────────────────────────────────
                is_quality = False
                term_override, is_quality = term_lookup.get(frame_idx, (None, False))

                # Also check multi-word terms in nearby segments
                frame_time = frame_idx / fps
                for seg in timed_segs:
                    if abs(seg["start"] - frame_time) <= _TERM_WINDOW_S:
                        txt = seg["text"].lower()
                        for term, act_idx in _BRAWLHALLA_VOCAB.items():
                            if term in txt:
                                new_kb[term] = new_kb.get(term, 0) + 1
                                if act_idx is not None and flow_conf < 0.9:
                                    term_override = act_idx
                                    flow_conf     = 0.95  # transcript = high conf
                                if term in _QUALITY_TERMS:
                                    is_quality = True

                if term_override is not None:
                    action = term_override

                # ── Store transition ──────────────────────────────────────────
                if prev_state is not None:
                    reward = compute_reward(prev_state, curr_state)
                    done   = (
                        curr_state["p1_stocks"] != prev_state["p1_stocks"] or
                        curr_state["p2_stocks"] != prev_state["p2_stocks"]
                    )
                    transitions.append(
                        Transition(prev_state["obs"], action, reward, curr_state["obs"], done)
                    )
                    weights.append(3.0 if is_quality else 1.0)

                prev_state = curr_state
                prev_gray  = gray

                with self._lock:
                    self.frames_done = kept
                    self.progress    = 0.15 + 0.80 * (kept / max(self.frames_total, 1))

                if kept % 500 == 0:
                    self._log(f"  {kept} frames kept, {len(transitions)} transitions, "
                              f"{len(new_kb)} terms seen")

            cap.release()
            try:
                os.remove(video_path)
                os.rmdir(tmp_dir)
            except Exception:
                pass

            # ── Stage 4: Merge into corpus ────────────────────────────────────
            with self._lock:
                self.corpus.extend(transitions)
                self.corpus_weights.extend(weights)
                for k, v in new_kb.items():
                    self.knowledge_base[k] = self.knowledge_base.get(k, 0) + v
                self.transitions_extracted = len(transitions)
                self.state    = "done"
                self.progress = 1.0

            from weaponized_ai import brain_store
            brain_store.save_knowledge(self.knowledge_base)
            brain_store.mark_ingested(
                url, self.video_title, kept, len(transitions)
            )

            quality_count = sum(1 for w in weights if w > 1.0)
            self._log(
                f"Done. {len(transitions)} transitions ({quality_count} quality-weighted) "
                f"from {self.video_title!r}. "
                f"Terms: {len(new_kb)} unique. Corpus total: {len(self.corpus)}"
            )

        except Exception as e:
            self._log(f"ERROR: {e}")
            with self._lock:
                self.state     = "error"
                self.error_msg = str(e)

    def _ydl_progress_hook(self, d: dict):
        if d.get("status") == "downloading":
            total   = d.get("total_bytes") or d.get("total_bytes_estimate") or 1
            done    = d.get("downloaded_bytes", 0)
            frac    = min(done / total, 1.0)
            with self._lock:
                self.progress = frac * 0.10
            if done % (5 * 1024 * 1024) < 256 * 1024:
                self._log(f"  Downloading {done/1e6:.1f}/{total/1e6:.1f} MB…")

    def _try_transcribe(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Run Whisper on the video file.  Returns a list of
        {"start": float, "end": float, "text": str} segments.
        Gracefully returns [] if Whisper is unavailable or fails.
        Uses the 'tiny' model by default (fast, ~39 MB, CPU-capable).
        """
        try:
            import whisper   # openai-whisper (lazy)
        except ImportError:
            self._log("Whisper not available — skipping transcription. "
                      "Install openai-whisper for term-guided labelling.")
            return []

        self._log("Loading Whisper tiny model…")
        try:
            model = whisper.load_model("tiny")
            self._log("Transcribing audio… (this may take a moment)")
            result = model.transcribe(video_path, language="en", fp16=False, verbose=False)
            segs = [
                {"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in result.get("segments", [])
            ]
            # Flatten into single-word entries for faster lookup
            self._log(f"Transcript: {len(segs)} segments, "
                      f"{sum(len(s['text'].split()) for s in segs)} words")
            return segs
        except Exception as e:
            self._log(f"Transcription error (continuing without): {e}")
            return []

    def _build_term_lookup(
        self,
        segs: List[Dict[str, Any]],
        fps: float,
    ) -> Dict[int, Tuple[Optional[int], bool]]:
        """
        Map frame index → (action_override | None, is_quality_frame).
        For each transcript segment that contains a known Brawlhalla term,
        all frame indices within ±_TERM_WINDOW_S of the segment's midpoint
        receive that action override.
        Multi-word terms are matched against the full segment text.
        """
        lookup: Dict[int, Tuple[Optional[int], bool]] = {}
        # Sort terms longest-first so multi-word matches win over sub-words
        sorted_vocab = sorted(_BRAWLHALLA_VOCAB.items(), key=lambda x: -len(x[0]))

        for seg in segs:
            txt  = seg["text"].lower()
            mid  = (seg["start"] + seg["end"]) / 2.0
            t0   = seg["start"] - _TERM_WINDOW_S
            t1   = seg["end"]   + _TERM_WINDOW_S

            matched_action:    Optional[int] = None
            matched_quality:   bool          = False

            for term, act_idx in sorted_vocab:
                if term in txt:
                    if act_idx is not None and matched_action is None:
                        matched_action = act_idx
                    if term in _QUALITY_TERMS:
                        matched_quality = True
                    break   # stop at first (longest) match per segment

            if matched_action is not None or matched_quality:
                f0 = max(1, int(t0 * fps))
                f1 = int(t1 * fps) + 1
                for fi in range(f0, f1):
                    if fi not in lookup or matched_quality:
                        lookup[fi] = (matched_action, matched_quality)

        return lookup

    def _run_pretrain(self, n_epochs: int, batch_size: int, lr: float):
        """
        Behavioural Cloning on the accumulated corpus.
        Quality frames (near combo/punish commentary) have 3× sample weight,
        implemented via torch.multinomial weighted random sampling each epoch.
        """
        try:
            import torch                              # lazy
            import numpy as np                        # lazy
            from weaponized_ai.rl_agent import get_agent  # lazy

            agent     = get_agent()
            optimizer = torch.optim.Adam(agent.policy.parameters(), lr=lr)

            corpus  = self.corpus
            n       = len(corpus)
            obs_t   = torch.tensor([t.obs    for t in corpus], dtype=torch.float32)
            act_t   = torch.tensor([t.action for t in corpus], dtype=torch.int64)
            # Sample weights: quality frames get 3×, others 1×
            raw_w   = self.corpus_weights if len(self.corpus_weights) == n else [1.0] * n
            w_t     = torch.tensor(raw_w, dtype=torch.float32)
            w_t     = w_t / w_t.sum()

            quality_frac = sum(1 for w in raw_w if w > 1.0) / max(n, 1)
            self._log(
                f"Behavioural Cloning: {n} samples ({quality_frac*100:.1f}% quality-weighted), "
                f"{n_epochs} epochs, batch={batch_size}"
            )

            initial_loss = None
            for epoch in range(n_epochs):
                # Weighted random draw of n indices (with replacement)
                indices = torch.multinomial(w_t, n, replacement=True)
                obs_e   = obs_t[indices]
                act_e   = act_t[indices]

                epoch_loss = 0.0
                steps      = 0
                for i in range(0, n - batch_size + 1, batch_size):
                    ob = obs_e[i: i + batch_size]
                    ac = act_e[i: i + batch_size]
                    log_probs, _ = agent.policy.evaluate(ob, ac)
                    loss = -log_probs.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 0.5)
                    optimizer.step()

                    epoch_loss += loss.item()
                    steps      += 1

                epoch_loss /= max(steps, 1)
                if initial_loss is None:
                    initial_loss = epoch_loss
                    with self._lock:
                        self.bc_loss_initial = initial_loss

                with self._lock:
                    self.bc_loss_last   = epoch_loss
                    self.bc_epochs_done = epoch + 1
                    self.progress       = (epoch + 1) / n_epochs

                self._log(f"  BC epoch {epoch+1}/{n_epochs}  loss={epoch_loss:.4f}")

            agent.save()
            from weaponized_ai import brain_store
            saved = brain_store.save_corpus(self.corpus, self.corpus_weights)
            brain_store.save_knowledge(self.knowledge_base)
            self._log(
                f"BC complete. Loss: {initial_loss:.4f} → {self.bc_loss_last:.4f}. "
                f"Model + brain saved ({saved} transitions on disk)."
            )

            with self._lock:
                self.state    = "done"
                self.progress = 1.0

        except Exception as e:
            self._log(f"ERROR during pretraining: {e}")
            with self._lock:
                self.state     = "error"
                self.error_msg = str(e)


# ── Module-level singleton ────────────────────────────────────────────────────
_learner: Optional[VideoLearner] = None


def get_learner() -> VideoLearner:
    global _learner
    if _learner is None:
        _learner = VideoLearner()
        from weaponized_ai import brain_store
        brain_store.bootstrap_learner(_learner)
    return _learner
