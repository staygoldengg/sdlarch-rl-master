# visual_engine.py
"""
High-speed DXGI Desktop Duplication screen capture for the policy visual input pipeline.

HighSpeedVisualEngine polls the Brawlhalla window at 120 Hz using DXCAM and
exposes the latest frame as a normalised float32 CUDA tensor in CHW layout.

Requires: pip install dxcam opencv-python-headless torch
"""

import threading
import time
import numpy as np
import cv2
import torch

try:
    import dxcam
except ImportError:
    raise ImportError(
        "dxcam is required for HighSpeedVisualEngine. "
        "Install it with: pip install dxcam"
    )


class HighSpeedVisualEngine:
    def __init__(self, target_width: int = 256, target_height: int = 256):
        # Initialise DXCAM with performance optimisations hooked to GPU 0
        self.camera = dxcam.create(device_idx=0, output_idx=0, max_buffer_len=3)
        self.target_dim = (target_width, target_height)

        self.latest_frame: np.ndarray | None = None
        self.lock = threading.Lock()
        self.is_running = False
        self.stream_thread: threading.Thread | None = None

    def start(self, region: tuple = None):
        """
        Begin the 120 Hz capture pipeline.

        Args:
            region: Optional bounding box (left, top, right, bottom) to
                    capture only the Brawlhalla window.
        """
        self.is_running = True
        self.camera.start(region=region, target_fps=120)
        self.stream_thread = threading.Thread(
            target=self._worker_cycle, daemon=True, name="VisualEngine"
        )
        self.stream_thread.start()
        print("[VISUAL] DXGI Desktop Duplication pipeline active at 120 Hz.")

    def _worker_cycle(self):
        while self.is_running:
            frame = self.camera.get_latest_frame()
            if frame is not None:
                # Downsample to policy input dimensions
                resized = cv2.resize(frame, self.target_dim, interpolation=cv2.INTER_LINEAR)
                with self.lock:
                    self.latest_frame = resized
            else:
                # Yield thread to prevent CPU core spinning
                time.sleep(0.001)

    def get_gpu_tensor(self) -> torch.Tensor:
        """
        Returns the latest captured frame as a normalised float32 CUDA tensor.

        Layout: (1, C, H, W) in [0.0, 1.0] range.
        Falls back to a zero tensor if no frame has been captured yet.
        """
        with self.lock:
            if self.latest_frame is None:
                return torch.zeros(
                    (1, 3, self.target_dim[1], self.target_dim[0]),
                    dtype=torch.float32,
                ).cuda()
            frame_copy = np.copy(self.latest_frame)

        # Permute HWC → CHW, push to VRAM, normalise [0, 255] → [0.0, 1.0]
        tensor = torch.from_numpy(frame_copy).permute(2, 0, 1).float().cuda()
        return tensor.unsqueeze(0).div(255.0)

    def stop(self):
        self.is_running = False
        self.camera.stop()
        if self.stream_thread:
            self.stream_thread.join()
        print("[VISUAL] Capture pipeline stopped.")
