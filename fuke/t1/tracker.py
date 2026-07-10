"""Lock-based visual tracking state machine.

LockTracker maintains a lock on the nearest detected target. In the LOCKED
state it expects detections within an adaptive search window around the
locked target; in the UNLOCKED state it selects the nearest detection from
the full frame to establish a new lock.

8 state transition paths (preserved from original):
  1. LOCKED + detections non-empty + candidate found in search rect  → DETECT (update)
  2. LOCKED + detections non-empty + no candidate, miss ≤ max_miss   → PREDICT (old pos)
  3. LOCKED + detections non-empty + miss > max_miss                 → DETECT (re-lock nearest)
  4. LOCKED + detections empty + miss ≤ max_miss                     → PREDICT
  5. LOCKED + detections empty + miss > max_miss                     → LOST
  6. UNLOCKED + detections non-empty                                 → DETECT (new lock)
  7. UNLOCKED + detections empty                                     → LOST
  8. (pre-inference) lock_frame_count ≥ max_hit                      → force_unlock → Path 6

Path 8 is triggered by the pipeline calling force_unlock() before inference;
update() handles paths 1–7.
"""

import threading
import logging
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Tuple

from detector import Detection
from utils import search_rect, pick_nearest

logger = logging.getLogger(__name__)


class TrackAction(Enum):
    DETECT = auto()   # Real detection: write [1, ux, uy]
    PREDICT = auto()  # Miss but still locked: write [2, ox, oy]
    LOST = auto()     # No target: caller should invoke OutputWriter.write_fallback()


@dataclass
class TrackResult:
    """Output of LockTracker.update()."""
    action: TrackAction
    x: int = 0
    y: int = 0
    lock_miss_count: int = 0
    lock_frame_count: int = 0
    is_locked: bool = False


class LockTracker:
    """Lock-based single-target visual tracker.

    The pipeline should call force_unlock() before inference when
    lock_frame_count reaches max_hit (Path 8). Then run inference
    with full-frame mode, and call update() with the results.
    """

    def __init__(
        self,
        max_hit: int,
        max_miss: int,
        search_ratio: float,
        min_search_radius: int,
        max_search_radius: int,
    ) -> None:
        self.max_hit = max_hit
        self.max_miss = max_miss
        self.search_ratio = search_ratio
        self.min_search_radius = min_search_radius
        self.max_search_radius = max_search_radius
        self._lock = threading.Lock()

        # Mutable state
        self.lock_target: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
        self.lock_miss_count: int = 0
        self.lock_frame_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_locked(self) -> bool:
        return self.lock_target is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def force_unlock(self) -> None:
        """Force unlock before inference (Path 8: max_hit reached)."""
        with self._lock:
            logger.info(f"锁定满{self.max_hit}帧，强制重判")
            self.lock_target = None
            self.lock_miss_count = 0
            self.lock_frame_count = 0

    def update(self, detections: List[Detection]) -> TrackResult:
        """Main state machine entry point (paths 1–7).

        Args:
            detections: List of Detection objects from the current frame
                        (full-frame or crop, depending on lock state).
        Returns:
            TrackResult with the appropriate action and coordinates.
        """
        with self._lock:
            if self.is_locked:
                if detections:
                    return self._handle_locked_with_detections(detections)
                else:
                    return self._handle_locked_no_detections()
            else:
                if detections:
                    return self._handle_unlocked_with_detections(detections)
                else:
                    return self._handle_unlocked_no_detections()

    def reset(self) -> None:
        """Fully reset state to UNLOCKED."""
        with self._lock:
            self.lock_target = None
            self.lock_miss_count = 0
            self.lock_frame_count = 0

    def increment_frame_count(self) -> None:
        """Increment lock_frame_count. Called by pipeline before inference."""
        with self._lock:
            if self.is_locked:
                self.lock_frame_count += 1

    # ------------------------------------------------------------------
    # Internal state transitions (caller must hold self._lock)
    # ------------------------------------------------------------------

    def _handle_locked_with_detections(
        self, detections: List[Detection]
    ) -> TrackResult:
        """Paths 1, 2, 3: LOCKED + detections non-empty."""
        lock_ox, lock_oy, lock_w, lock_h = self.lock_target  # type: ignore[misc]
        sx1, sy1, sx2, sy2, _shw, _shh, _search_area = search_rect(
            lock_ox, lock_oy, lock_w, lock_h,
            self.search_ratio, self.min_search_radius, self.max_search_radius,
        )

        # Filter detections within the search rectangle
        candidates = [
            d for d in detections
            if sx1 <= d.ux <= sx2 and sy1 <= d.uy <= sy2
        ]
        # Convert to (dis, cls, ux, uy, r) tuple format for pick_nearest
        candidate_tuples = [
            (d.dis, d.cls, d.ux, d.uy, d.r) for d in candidates
        ]
        best_match = pick_nearest(candidate_tuples)

        if best_match is not None:
            # Path 1: candidate found in search rect → update lock
            _dis, _cls, ux, uy, r = best_match
            w, h = abs(r[2] - r[0]), abs(r[3] - r[1])
            self.lock_target = (ux, uy, w, h)
            self.lock_miss_count = 0
            logger.info(
                f"锁定桶：({ux},{uy}) 目标面积={w*h} "
                f"搜索框=({sx1},{sy1},{sx2},{sy2}) 搜索面积={_search_area}"
            )
            return TrackResult(
                action=TrackAction.DETECT, x=ux, y=uy,
                lock_miss_count=self.lock_miss_count,
                lock_frame_count=self.lock_frame_count,
                is_locked=True,
            )

        # No candidate in search rect
        self.lock_miss_count += 1
        if self.lock_miss_count > self.max_miss:
            # Path 3: exceeded miss limit with detections available → re-lock
            logger.info(f"丢帧满{self.max_miss}帧，强制重判")
            self.lock_target = None
            self.lock_miss_count = 0
            self.lock_frame_count = 0
            # Re-lock on nearest from the FULL detection list (guaranteed non-empty)
            best = pick_nearest([
                (d.dis, d.cls, d.ux, d.uy, d.r) for d in detections
            ])
            _dis, _cls, ux, uy, r = best
            w, h = abs(r[2] - r[0]), abs(r[3] - r[1])
            self.lock_target = (ux, uy, w, h)
            _nsx1, _nsy1, _nsx2, _nsy2, nshw, nshh, narea = search_rect(
                ux, uy, w, h,
                self.search_ratio, self.min_search_radius, self.max_search_radius,
            )
            logger.info(
                f"解锁-最近桶：({ux},{uy}) 目标面积={w*h} "
                f"搜索框=({ux-nshw},{uy-nshh},{ux+nshw},{uy+nshh}) 搜索面积={narea}"
            )
            return TrackResult(
                action=TrackAction.DETECT, x=ux, y=uy,
                lock_miss_count=self.lock_miss_count,
                lock_frame_count=self.lock_frame_count,
                is_locked=True,
            )
        else:
            # Path 2: miss within limit → predict old position
            logger.info(
                f"丢帧：({lock_ox},{lock_oy}) miss={self.lock_miss_count} "
                f"搜索框=({sx1},{sy1},{sx2},{sy2}) 搜索面积={_search_area}"
            )
            return TrackResult(
                action=TrackAction.PREDICT, x=lock_ox, y=lock_oy,
                lock_miss_count=self.lock_miss_count,
                lock_frame_count=self.lock_frame_count,
                is_locked=True,
            )

    def _handle_locked_no_detections(self) -> TrackResult:
        """Paths 4, 5: LOCKED + detections empty."""
        lock_ox, lock_oy, lock_w, lock_h = self.lock_target  # type: ignore[misc]
        sx1, sy1, sx2, sy2, _shw, _shh, _search_area = search_rect(
            lock_ox, lock_oy, lock_w, lock_h,
            self.search_ratio, self.min_search_radius, self.max_search_radius,
        )

        self.lock_miss_count += 1
        if self.lock_miss_count > self.max_miss:
            # Path 5: exceeded miss limit → unlock, report LOST
            logger.info(f"丢帧满{self.max_miss}帧，强制重判")
            self.lock_target = None
            self.lock_miss_count = 0
            self.lock_frame_count = 0
            return TrackResult(
                action=TrackAction.LOST, x=0, y=0,
                lock_miss_count=self.lock_miss_count,
                lock_frame_count=self.lock_frame_count,
                is_locked=False,
            )
        else:
            # Path 4: miss within limit → predict old position
            logger.info(
                f"丢帧(无检测)：({lock_ox},{lock_oy}) miss={self.lock_miss_count} "
                f"搜索框=({sx1},{sy1},{sx2},{sy2}) 搜索面积={_search_area}"
            )
            return TrackResult(
                action=TrackAction.PREDICT, x=lock_ox, y=lock_oy,
                lock_miss_count=self.lock_miss_count,
                lock_frame_count=self.lock_frame_count,
                is_locked=True,
            )

    def _handle_unlocked_with_detections(
        self, detections: List[Detection]
    ) -> TrackResult:
        """Path 6: UNLOCKED + detections non-empty → establish new lock."""
        best = pick_nearest([
            (d.dis, d.cls, d.ux, d.uy, d.r) for d in detections
        ])
        _dis, _cls, ux, uy, r = best
        w, h = abs(r[2] - r[0]), abs(r[3] - r[1])
        self.lock_target = (ux, uy, w, h)
        self.lock_miss_count = 0
        self.lock_frame_count = 0
        _sx1, _sy1, _sx2, _sy2, shw, shh, sarea = search_rect(
            ux, uy, w, h,
            self.search_ratio, self.min_search_radius, self.max_search_radius,
        )
        logger.info(
            f"最近桶：({ux},{uy}) 目标面积={w*h} "
            f"搜索框=({ux-shw},{uy-shh},{ux+shw},{uy+shh}) 搜索面积={sarea}"
        )
        return TrackResult(
            action=TrackAction.DETECT, x=ux, y=uy,
            lock_miss_count=self.lock_miss_count,
            lock_frame_count=self.lock_frame_count,
            is_locked=True,
        )

    def _handle_unlocked_no_detections(self) -> TrackResult:
        """Path 7: UNLOCKED + detections empty → LOST."""
        return TrackResult(
            action=TrackAction.LOST, x=0, y=0,
            lock_miss_count=self.lock_miss_count,
            lock_frame_count=self.lock_frame_count,
            is_locked=False,
        )
