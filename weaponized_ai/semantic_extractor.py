# semantic_extractor.py
"""
SemanticKinematicExtractor — YOLO-based entity tracker.

Detects Brawlhalla entities in a GPU frame tensor and returns a 9-dimensional
kinematic feature vector:
  [0-3]  Player 1 bounding box in xywh format (normalised)
  [4-7]  Player 2 bounding box in xywh format (normalised)
  [8]    Weapon count on stage

Requires: pip install ultralytics
Model:    models/brawlhalla_yolo_nano.pt  (place at project root)
"""

import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "ultralytics is required for SemanticKinematicExtractor. "
        "Install it with: pip install ultralytics"
    )


class SemanticKinematicExtractor:
    """
    Args:
        model_weights_path: Path to the YOLO `.pt` file.
        conf_threshold:     Minimum detection confidence to accept.
    """

    # Class indices matching the custom training dataset
    CLS_PLAYER_1 = 0
    CLS_PLAYER_2 = 1
    CLS_WEAPON   = 2

    def __init__(
        self,
        model_weights_path: str = "models/brawlhalla_yolo_nano.pt",
        conf_threshold: float = 0.5,
    ):
        self.model = YOLO(model_weights_path)
        self.conf = conf_threshold
        print(f"[YOLO] Entity tracker loaded: {model_weights_path}")

    def extract_frame_coordinates(
        self, frame_tensor: torch.Tensor
    ) -> np.ndarray:
        """
        Args:
            frame_tensor: Float32 CUDA tensor, shape (1, 3, H, W) in [0.0, 1.0]
                          as returned by ZeroCopyVisionEngine.

        Returns:
            float32 ndarray of shape (9,) — zeros for absent entities.
        """
        coords = np.zeros(9, dtype=np.float32)

        results = self.model(frame_tensor, verbose=False, conf=self.conf)
        if not results:
            return coords

        boxes = results[0].boxes
        weapon_count = 0

        for box in boxes:
            cls_id = int(box.cls[0].item())
            xywh   = box.xywh[0].cpu().numpy()

            if cls_id == self.CLS_PLAYER_1:
                coords[0:4] = xywh
            elif cls_id == self.CLS_PLAYER_2:
                coords[4:8] = xywh
            elif cls_id == self.CLS_WEAPON:
                weapon_count += 1

        coords[8] = float(weapon_count)
        return coords
