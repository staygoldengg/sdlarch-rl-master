# cuda_vision.py
"""
ZeroCopyVisionEngine — DXGI Desktop Duplication capture pipeline.

Grabs the game window directly from the GPU via dxcam and returns a
normalised float32 CUDA tensor in NCHW layout without any CPU round-trip.

Requires: pip install dxcam torch
"""

import torch

try:
    import dxcam
except ImportError:
    raise ImportError(
        "dxcam is required for ZeroCopyVisionEngine. "
        "Install it with: pip install dxcam"
    )


class ZeroCopyVisionEngine:
    """
    Args:
        region_box: Screen region as (left, top, right, bottom).
                    Defaults to the full primary monitor (0, 0, 1920, 1080).
    """

    def __init__(self, region_box: tuple[int, int, int, int] = (0, 0, 1920, 1080)):
        self.region = region_box
        self.camera = dxcam.create(device_idx=0, output_color="BGRA")
        self.camera.start(region=self.region, target_fps=60, video_mode=True)
        print(f"[VISION] Zero-Copy D3D11 pipeline bound to region {self.region}")

    def capture_next_nn_input(self) -> torch.Tensor | None:
        """
        Returns the latest captured frame as a normalised float32 CUDA tensor.

        Shape: (1, 3, H, W) in [0.0, 1.0].
        Returns None if no new frame is available yet.
        """
        raw = self.camera.get_latest_frame()
        if raw is None:
            return None

        with torch.device("cuda"):
            tensor = torch.as_tensor(raw, dtype=torch.uint8)
            # Drop the alpha channel: BGRA → RGB (first 3 channels)
            tensor = tensor[:, :, :3]
            # HWC → NCHW
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            return tensor.float() / 255.0

    def shutdown(self):
        self.camera.stop()
        print("[VISION] DXGI capture pipeline safely detached.")
