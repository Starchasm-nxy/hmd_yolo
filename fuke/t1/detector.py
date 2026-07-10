"""
YOLO-based object detector with full-frame and crop inference modes.

Detection: single detection record in full-frame coordinates.
YOLODetector: loads a single YOLO model, provides detect_full() and
detect_crop() methods with automatic coordinate conversion and dynamic imgsz.
"""

import time
import logging
from dataclasses import dataclass
from typing import List, Any, Dict

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection record, all coordinates in full-frame space."""

    ux: int
    uy: int
    cls: int
    conf: float
    r: List[int]  # [x1, y1, x2, y2] full-frame
    w: int
    h: int
    area: int
    dis: float  # Euclidean distance from full-frame center
    box: Any  # Original ultralytics Box (for draw_square)
    names: Dict[int, str]  # Class name dict (for draw_square)


class YOLODetector:
    """Loads a YOLO model and provides two inference modes.

    detect_full:  full-frame inference with mode-specific conf/imgsz.
    detect_crop:  crop inference with dynamic imgsz, coordinates converted
                  to full-frame by adding (offset_x, offset_y).
    Both filter detections by area <= max_area.
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        iou: float,
        locked_conf: float,
        unlocked_1m_conf: float,
        unlocked_1m_imgsz: int,
        unlocked_2m_conf: float,
        unlocked_2m_imgsz: int,
        imgsz_step: int,
        max_area: int,
        frame_w: int,
        frame_h: int,
    ) -> None:
        self.model = YOLO(model_path)
        self.device = device
        self.iou = iou
        self.locked_conf = locked_conf
        self.unlocked_1m_conf = unlocked_1m_conf
        self.unlocked_1m_imgsz = unlocked_1m_imgsz
        self.unlocked_2m_conf = unlocked_2m_conf
        self.unlocked_2m_imgsz = unlocked_2m_imgsz
        self.imgsz_step = imgsz_step
        self.max_area = max_area
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._center_x = frame_w // 2
        self._center_y = frame_h // 2

    def detect_full(self, frame: np.ndarray, mode: str) -> List[Detection]:
        """Run full-frame YOLO inference.

        Selects conf and imgsz based on mode ('1m' or '2m').
        Detection coordinates are already in full-frame space.
        """
        conf = self.unlocked_1m_conf if mode == '1m' else self.unlocked_2m_conf
        imgsz = self.unlocked_1m_imgsz if mode == '1m' else self.unlocked_2m_imgsz

        t0 = time.time()
        results = self.model.predict(
            source=frame, device=self.device, show=False,
            stream=False, verbose=False, iou=self.iou,
            conf=conf, imgsz=imgsz,
        )
        infer_ms = int((time.time() - t0) * 1000)
        logger.debug(f"Full-frame inference: {infer_ms}ms  imgsz={imgsz}  conf={conf}")

        return self._results_to_detections(results, offset_x=0, offset_y=0)

    def detect_crop(
        self, crop: np.ndarray, offset_x: int, offset_y: int
    ) -> List[Detection]:
        """Run YOLO inference on a cropped region.

        imgsz is dynamically computed from the crop dimensions (rounded up
        to the nearest imgsz_step). Box coordinates are converted from
        crop-local to full-frame by adding (offset_x, offset_y).
        """
        crop_h, crop_w = crop.shape[:2]
        crop_max_dim = max(crop_w, crop_h)
        imgsz = ((crop_max_dim + self.imgsz_step - 1) // self.imgsz_step) * self.imgsz_step

        t0 = time.time()
        results = self.model.predict(
            source=crop, device=self.device, show=False,
            stream=False, verbose=False, iou=self.iou,
            conf=self.locked_conf, imgsz=imgsz,
        )
        infer_ms = int((time.time() - t0) * 1000)
        logger.debug(f"Crop inference: {infer_ms}ms  imgsz={imgsz}  crop={crop_w}x{crop_h}")

        return self._results_to_detections(results, offset_x=offset_x, offset_y=offset_y)

    def _results_to_detections(
        self, results: Any, offset_x: int = 0, offset_y: int = 0
    ) -> List[Detection]:
        """Convert ultralytics Results into a list of Detection objects.

        Applies coordinate offset, area filter, and distance-from-center.
        Distance is always computed against the full-frame center.
        """
        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            names = result.names
            if boxes is None:
                continue
            for box in boxes:
                r = box.xyxy[0].cpu().numpy().astype(int)
                # Apply offset for crop inference
                r[0] += offset_x
                r[1] += offset_y
                r[2] += offset_x
                r[3] += offset_y

                ux = int((r[0] + r[2]) / 2)
                uy = int((r[1] + r[3]) / 2)
                w = abs(int(r[2] - r[0]))
                h = abs(int(r[3] - r[1]))
                area = w * h

                if area > self.max_area:
                    continue

                dis = ((ux - self._center_x) ** 2 + (uy - self._center_y) ** 2) ** 0.5

                detections.append(Detection(
                    ux=ux, uy=uy,
                    cls=int(box.cls[0]),
                    conf=float(box.conf[0]),
                    r=r.tolist(),
                    w=w, h=h, area=area, dis=dis,
                    box=box, names=names,
                ))

        return detections
