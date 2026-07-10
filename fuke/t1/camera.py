"""RealSense D435 camera source with thread-safe frame buffering.

FrameBuffer: single-producer, single-consumer buffer keeping only the latest frame.
CameraSource: manages D435 pipeline and internal capture thread.
"""

import threading
import logging
from typing import Optional, Tuple

import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)


class FrameBuffer:
    """Thread-safe buffer that retains only the latest frame.

    Uses a sequence number so the consumer can detect new frames
    without blocking.
    """

    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._seq: int = 0
        self._lock = threading.Lock()

    def put(self, frame: np.ndarray) -> None:
        """Store a new frame (producer side)."""
        with self._lock:
            self._frame = frame
            self._seq += 1

    def get_latest(self, seen_seq: int = 0) -> Tuple[bool, Optional[np.ndarray], int]:
        """Non-blocking check for a new frame.

        Returns (has_new, frame_copy, current_seq). If no new frame,
        returns (False, None, current_seq).
        """
        with self._lock:
            if self._seq <= seen_seq or self._frame is None:
                return False, None, self._seq
            return True, self._frame.copy(), self._seq


class CameraSource:
    """Manages the D435 camera pipeline and internal capture thread.

    Frames are continuously captured into a FrameBuffer, from which
    the consumer pulls them non-blockingly.
    """

    def __init__(self, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._pipeline: Optional[rs.pipeline] = None
        self._config: Optional[rs.config] = None
        self._frame_buf = FrameBuffer()
        self._cap_stop = threading.Event()
        self._cap_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Open the D435 pipeline and start the capture thread.

        Raises RuntimeError if no device is found.
        """
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        self._pipeline.start(self._config)
        self._cap_stop.clear()
        self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._cap_thread.start()
        logger.debug("D435 capture thread started")

    def stop(self) -> None:
        """Signal capture thread to stop, join, and release the pipeline."""
        self._cap_stop.set()
        if self._cap_thread is not None:
            self._cap_thread.join(timeout=1)
        if self._pipeline is not None:
            self._pipeline.stop()
            logger.info("D435摄像头已释放")

    def get_frame(self, seen_seq: int) -> Tuple[bool, Optional[np.ndarray], int]:
        """Non-blocking: retrieve the latest frame from the buffer."""
        return self._frame_buf.get_latest(seen_seq)

    @staticmethod
    def check_device() -> bool:
        """Return True if at least one RealSense device is connected."""
        ctx = rs.context()
        return len(ctx.devices) > 0

    def _capture_loop(self) -> None:
        """Internal loop: wait_for_frames -> FrameBuffer.put."""
        while not self._cap_stop.is_set():
            frames = self._pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                frame = np.asanyarray(color_frame.get_data())
                self._frame_buf.put(frame)
