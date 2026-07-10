"""Stateless helper functions for drawing, geometry, and file I/O."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Any, Dict


def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp value v to the range [lo, hi]."""
    return max(lo, min(hi, v))


def draw_square(
    image: np.ndarray,
    box: Any,
    names: Dict[int, str],
    r: Tuple[int, int, int, int],
) -> Tuple[int, int]:
    """Draw detection bounding box, label, and center dot. Returns (ux, uy)."""
    ux = int((r[0] + r[2]) / 2)
    uy = int((r[1] + r[3]) / 2)
    cls = int(box.cls[0])
    conf = box.conf[0]
    label = f"{names[cls]} {conf:.2f}"
    cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (221, 185, 193), 2)
    cv2.putText(image, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (176, 196, 222), 2)
    cv2.circle(image, (ux, uy), 5, (240, 240, 240), -1)
    return ux, uy


def draw_lock_rect(
    canvas: np.ndarray,
    x: int,
    y: int,
    shw: int,
    shh: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw lock search rectangle border and center crosshair."""
    cv2.rectangle(canvas, (x - shw, y - shh), (x + shw, y + shh), color, 2)
    cv2.circle(canvas, (x, y), 4, color, -1)


def search_rect(
    x: float,
    y: float,
    w: float,
    h: float,
    search_ratio: float,
    min_search_radius: int,
    max_search_radius: int,
) -> Tuple[int, int, int, int, int, int, int]:
    """Compute an adaptive search rectangle around a target.

    Returns (sx1, sy1, sx2, sy2, shw, shh, area).
    """
    shw = clamp(w * search_ratio / 2, min_search_radius, max_search_radius)
    shh = clamp(h * search_ratio / 2, min_search_radius, max_search_radius)
    sx1 = int(x - shw)
    sy1 = int(y - shh)
    sx2 = int(x + shw)
    sy2 = int(y + shh)
    return sx1, sy1, sx2, sy2, int(shw), int(shh), (sx2 - sx1) * (sy2 - sy1)


def pick_nearest(items: List) -> Optional[Any]:
    """From a list of (dis, ...) tuples, return the one with smallest dis.

    Returns None if the list is empty.
    """
    if not items:
        return None
    return min(items, key=lambda x: x[0])


def clear_files(files: List[str]) -> None:
    """Truncate each listed file (create empty if it doesn't exist)."""
    for file_name in files:
        with open(file_name, 'w'):
            pass
