"""Detection pipeline: integrates camera, detector, tracker, and output writer.

DetectionPipeline.run() is the main detection loop, called from a dedicated
thread managed by main.py / pipeline_manager_loop.
"""

import time
import queue
import logging
from collections import deque
from typing import Optional

import cv2
import numpy as np

from config_schema import Config
from camera import CameraSource
from detector import YOLODetector, Detection
from tracker import LockTracker, TrackAction
from communication import FileMonitor, OutputWriter
from utils import draw_square, draw_lock_rect, search_rect

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Integrates all detection modules into a single processing loop.

    The pipeline thread runs only when the mode is '1m' or '2m'.
    It reads frames from CameraSource, runs YOLO inference, updates
    the LockTracker, writes results via OutputWriter, and pushes
    rendered canvases to the display queue.
    """

    def __init__(
        self,
        config: Config,
        camera: CameraSource,
        detector: YOLODetector,
        tracker: LockTracker,
        writer: OutputWriter,
        file_monitor: FileMonitor,
    ) -> None:
        self.config = config
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.writer = writer
        self.file_monitor = file_monitor

    def run(self, stop_event, display_queue) -> None:
        """Main detection loop. Blocks until stop_event is set."""
        self.camera.start()
        last_mode: Optional[str] = None
        last_canvas: Optional[np.ndarray] = None
        last_seq: int = 0
        frame_count: int = 0
        fps_times: deque = deque(maxlen=30)
        fps_print_count: int = 0
        skip_enabled = self.config.frame_skip.enabled
        skip_n = self.config.frame_skip.n

        try:
            while not stop_event.is_set():
                mode = self.file_monitor.read_command()

                # Idle when mode is not active
                if mode not in ('1m', '2m'):
                    if last_mode in ('1m', '2m'):
                        self._write_stop()
                    last_mode = mode
                    time.sleep(0.01)
                    continue

                # Mode switch handling
                if mode != last_mode:
                    self._handle_mode_switch(mode, last_mode)
                    last_mode = mode
                    frame_count = 0
                    fps_print_count = 5

                # Get next frame (non-blocking)
                got, frame, last_seq = self.camera.get_frame(last_seq)
                if not got:
                    time.sleep(0.001)
                    continue

                # FPS tracking
                now = time.time()
                fps_times.append(now)
                fps = (
                    len(fps_times) / (fps_times[-1] - fps_times[0])
                    if len(fps_times) > 1 else 0
                )
                if fps_print_count > 0:
                    logger.info(f"实时帧率: {fps:.1f} FPS")
                    fps_print_count -= 1

                frame_count += 1
                do_inference = not skip_enabled or frame_count % skip_n == 0

                if do_inference:
                    canvas = frame.copy()
                    self._process_frame(frame, canvas, mode)
                    last_canvas = canvas.copy()
                else:
                    canvas = (
                        last_canvas.copy()
                        if last_canvas is not None
                        else frame.copy()
                    )

                # Draw mode and FPS overlay
                cv2.putText(
                    canvas, f"{mode}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    canvas, f"FPS:{fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
                )

                # Push to display queue
                try:
                    display_queue.put((canvas, self.config.display.window_name), block=False)
                except queue.Full:
                    pass

        finally:
            self.camera.stop()

    # ------------------------------------------------------------------
    # Internal: mode switching
    # ------------------------------------------------------------------

    def _handle_mode_switch(self, new_mode: str, old_mode: Optional[str]) -> None:
        """Handle mode transitions, preserving state appropriately."""
        logger.info(f"模式切换: {old_mode} -> {new_mode}")

        # Cold start: from non-active mode into 1m or 2m
        if old_mode not in ('1m', '2m'):
            self.writer.reset(cold_start=True)
            self.tracker.reset()
            logger.info("冷启动：历史坐标已清空")
        else:
            # Warm switch (1m <-> 2m): reset lock state only, preserve last_value
            self.tracker.reset()

    def _write_stop(self) -> None:
        """Write '0' to output file when mode deactivates."""
        with open(self.config.files.output, 'w') as f:
            f.write('0')

    # ------------------------------------------------------------------
    # Internal: per-frame processing
    # ------------------------------------------------------------------

    def _process_frame(
        self, frame: np.ndarray, canvas: np.ndarray, mode: str
    ) -> None:
        """Core processing for a single inference frame."""
        frame_w = self.config.camera.width
        frame_h = self.config.camera.height

        # --- Pre-inference: max_hit check (Path 8) ---
        self.tracker.increment_frame_count()
        if self.tracker.is_locked and self.tracker.lock_frame_count >= self.tracker.max_hit:
            self.tracker.force_unlock()

        is_locked = self.tracker.is_locked

        # --- YOLO inference ---
        if is_locked:
            lock_ox, lock_oy, lock_w, lock_h = self.tracker.lock_target  # type: ignore[misc]
            sx1, sy1, sx2, sy2, shw, shh, _search_area = search_rect(
                lock_ox, lock_oy, lock_w, lock_h,
                self.config.lock_tracker.search_ratio,
                self.config.lock_tracker.min_search_radius,
                self.config.lock_tracker.max_search_radius,
            )
            # Clamp crop region to frame boundaries
            csx1 = max(0, int(sx1))
            csy1 = max(0, int(sy1))
            csx2 = min(frame_w, int(sx2))
            csy2 = min(frame_h, int(sy2))
            crop = frame[csy1:csy2, csx1:csx2]
            detections = self.detector.detect_crop(crop, csx1, csy1)
            # Draw lock search rect
            draw_lock_rect(canvas, lock_ox, lock_oy, shw, shh, (0, 255, 255))
        else:
            detections = self.detector.detect_full(frame, mode)

        # --- Draw all detections ---
        for d in detections:
            ux, uy = draw_square(canvas, d.box, d.names, d.r)
            cv2.circle(canvas, (d.ux, d.uy), 4, (255, 255, 255), 5)
            cv2.putText(
                canvas, str([d.ux, d.uy]), (d.ux + 20, d.uy + 10),
                0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA,
            )

        # --- Draw orange lock rect when LOCKED + no detections (before tracker may unlock) ---
        if is_locked and not detections:
            lock_ox, lock_oy, lock_w, lock_h = self.tracker.lock_target  # type: ignore[misc]
            _sx1, _sy1, _sx2, _sy2, shw, shh, _sarea = search_rect(
                lock_ox, lock_oy, lock_w, lock_h,
                self.config.lock_tracker.search_ratio,
                self.config.lock_tracker.min_search_radius,
                self.config.lock_tracker.max_search_radius,
            )
            draw_lock_rect(canvas, lock_ox, lock_oy, shw, shh, (0, 165, 255))

        # --- Run tracker state machine ---
        track_result = self.tracker.update(detections)

        # Log debug info
        lock_status = "锁定" if track_result.is_locked else ""
        logger.debug(
            f"模式：{mode}{lock_status} 检测数={len(detections)} "
            f"锁帧数={track_result.lock_frame_count} 丢帧数={track_result.lock_miss_count}"
        )

        # --- Write result ---
        if track_result.action == TrackAction.DETECT:
            self.writer.write_detection(mode, track_result.x, track_result.y)
            # Draw green lock rect for newly established locks (was UNLOCKED → now LOCKED)
            if not is_locked and track_result.is_locked:
                lock_ox, lock_oy, lock_w, lock_h = self.tracker.lock_target  # type: ignore[misc]
                _sx1, _sy1, _sx2, _sy2, shw, shh, _sarea = search_rect(
                    lock_ox, lock_oy, lock_w, lock_h,
                    self.config.lock_tracker.search_ratio,
                    self.config.lock_tracker.min_search_radius,
                    self.config.lock_tracker.max_search_radius,
                )
                draw_lock_rect(canvas, lock_ox, lock_oy, shw, shh, (0, 255, 0))
        elif track_result.action == TrackAction.PREDICT:
            self.writer.write_prediction(mode, track_result.x, track_result.y)
        else:  # LOST
            self.writer.write_fallback(mode)
