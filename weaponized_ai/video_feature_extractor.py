# video_feature_extractor.py
"""
VideoFeatureAnchorExtractor — produces 64-dimensional feature vectors from
raw screen captures using dense optical flow and UI anchor sampling.

Feature layout:
  [0-7]  Optical flow magnitude by screen quadrant (Q1-Q4 × X+Y)
  [8]    UI damage zone pixel density estimate
  [9-63] Padding (zeros) — reserved for future structural features
"""

import cv2
import numpy as np


class VideoFeatureAnchorExtractor:
    def __init__(self, target_w: int = 1280, target_h: int = 720):
        self.w = target_w
        self.h = target_h
        self.prev_gray: np.ndarray | None = None

    def process_frame(self, raw_bgr_frame: np.ndarray) -> np.ndarray:
        """
        Extracts a 64-dim feature vector from a raw BGR video frame.

        Args:
            raw_bgr_frame: uint8 BGR frame from capture source.

        Returns:
            float32 array of shape (64,).
        """
        resized = cv2.resize(raw_bgr_frame, (self.w, self.h))
        gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        feature_vector = np.zeros(64, dtype=np.float32)

        # ── 1. Quadrant optical flow ──────────────────────────────────────────
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mid_y, mid_x = self.h // 2, self.w // 2
            quadrants = [
                flow[0:mid_y, 0:mid_x],       # Q1 top-left
                flow[0:mid_y, mid_x:self.w],   # Q2 top-right
                flow[mid_y:self.h, 0:mid_x],   # Q3 bottom-left
                flow[mid_y:self.h, mid_x:self.w],  # Q4 bottom-right
            ]
            for i, q in enumerate(quadrants):
                feature_vector[i * 2]     = float(np.mean(np.abs(q[..., 0])))  # X
                feature_vector[i * 2 + 1] = float(np.mean(np.abs(q[..., 1])))  # Y

        self.prev_gray = gray

        # ── 2. UI damage zone anchor ──────────────────────────────────────────
        # Sample the top-right corner where player damage cards are rendered
        ui_zone = gray[20:100, (self.w - 250):(self.w - 50)]
        _, thresh = cv2.threshold(ui_zone, 200, 255, cv2.THRESH_BINARY)
        feature_vector[8] = float(np.sum(thresh == 255) / max(ui_zone.size, 1))

        return feature_vector

    def reset(self):
        """Clear frame history (call at match start to avoid inter-match flow artifacts)."""
        self.prev_gray = None
