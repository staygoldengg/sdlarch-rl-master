# obs_capture.py
"""
Screen capture for RL observation extraction.
Supports two backends:
  1. mss  — direct fast screen grab (no OBS needed)
  2. OBS Virtual Camera — reads from the OBS virtual camera device via OpenCV

Usage:
    cap = ScreenCapture(mode="mss", region=(x, y, w, h))
    frame = cap.grab()   # returns numpy BGR array
"""

from typing import Optional, Tuple

# ── region defaults — tune these to your Brawlhalla window ───────────────────
DEFAULT_REGION = {"left": 0, "top": 0, "width": 1920, "height": 1080}


class ScreenCapture:
    def __init__(
        self,
        mode: str = "mss",
        region: Optional[dict] = None,
        obs_camera_index: int = 1,
    ):
        """
        mode: "mss" (direct grab) or "obs" (OBS virtual camera via OpenCV)
        region: {"left": x, "top": y, "width": w, "height": h}
        obs_camera_index: device index of the OBS virtual camera (usually 1 or 2)
        """
        self.mode = mode
        self.region = region or DEFAULT_REGION
        self._sct = None
        self._cap = None

        if mode == "mss":
            import mss
            self._sct = mss.mss()
        elif mode == "obs":
            import cv2
            # Auto-detect OBS camera index if not specified
            if obs_camera_index == 1:
                try:
                    from weaponized_ai.obs_manager import find_obs_camera_index, is_obs_running, launch_obs
                    if not is_obs_running():
                        launch_obs(start_virtual_cam=True)
                        import time; time.sleep(3)   # let OBS start
                    obs_camera_index = find_obs_camera_index()
                except Exception:
                    pass
            self._cap = cv2.VideoCapture(obs_camera_index)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"OBS virtual camera not found at index {obs_camera_index}. "
                    "Start OBS and enable Virtual Camera, or use mode='mss'."
                )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'mss' or 'obs'.")

    def grab(self) -> np.ndarray:
        """Capture a frame. Returns a BGR numpy array."""
        import cv2
        if self.mode == "mss":
            import numpy as np
            img = self._sct.grab(self.region)
            frame = np.array(img)[:, :, :3]  # BGRA → BGR
            return frame
        else:
            ret, frame = self._cap.read()
            if not ret:
                raise RuntimeError("Failed to read from OBS virtual camera.")
            return frame

    def grab_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Grab a sub-region of the screen."""
        frame = self.grab()
        return frame[y:y+h, x:x+w]

    def close(self):
        if self._sct:
            self._sct.close()
        if self._cap:
            self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Convenience: grab one frame from the whole screen ────────────────────────
def grab_screen(region: Optional[dict] = None) -> np.ndarray:
    with ScreenCapture(mode="mss", region=region or DEFAULT_REGION) as cap:
        return cap.grab()


# ── OBS WebSocket integration (optional — requires obs-websocket-py) ──────────
def set_obs_region(x: int, y: int, w: int, h: int):
    """Update the DEFAULT_REGION for future captures."""
    global DEFAULT_REGION
    DEFAULT_REGION = {"left": x, "top": y, "width": w, "height": h}
