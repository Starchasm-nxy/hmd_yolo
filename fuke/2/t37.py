"""
YOLO + RealSense D435 real-time object detection with lock-based tracking.

Single-file refactor of t36.py. The nested if/else state machine in
camera_detection_thread is extracted into LockTracker with 8 explicit
transition paths. All major components are classes with clear boundaries.
"""

import cv2
import time
import queue
import threading
import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple, Any, Dict

import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# ======================================================================
# 1. Configuration
# ======================================================================

FILES_TO_CLEAR = ['data.txt', 'gaozhi.txt']

D435_WIDTH = 848
D435_HEIGHT = 480

MODEL_LOCKED_PATH = "/home/fu/weights/tong_blue_v0.pt"
MODEL_UNLOCKED_PATH = "/home/fu/weights/tongv3.pt"

HISTORY_CLEAR_ENABLED = True
HISTORY_CLEAR_TIMEOUT = 10.0

FRAME_SKIP_ENABLED = True
FRAME_SKIP_N = 2

LOCK_MAX_HIT = 15
LOCK_MAX_MISS = 7
LOCK_SEARCH_RATIO = 2.5
LOCK_MIN_SEARCH_RADIUS = 130
LOCK_MAX_SEARCH_RADIUS = 270

MODEL_LOCKED_CONF = 0.5
MODEL_UNLOCKED_2M_IMGSZ = 640
MODEL_UNLOCKED_1M_IMGSZ = 640
MODEL_UNLOCKED_2M_CONF = 0.5
MODEL_UNLOCKED_1M_CONF = 0.5
MODEL_IOU = 0.45
MODEL_IMGSZ_STEP = 32
MODEL_DEVICE = 'cpu'
MAX_AREA = 100000

DISPLAY_WIN_NAME = "d435"
DISPLAY_QUEUE_SIZE = 3

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ======================================================================
# 2. Utility functions
# ======================================================================

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def search_rect(x: float, y: float, w: float, h: float) -> Tuple[int, int, int, int, int, int, int]:
    """Adaptive search window around a target.
    Returns (sx1, sy1, sx2, sy2, shw, shh, area)."""
    shw = clamp(w * LOCK_SEARCH_RATIO / 2, LOCK_MIN_SEARCH_RADIUS, LOCK_MAX_SEARCH_RADIUS)
    shh = clamp(h * LOCK_SEARCH_RATIO / 2, LOCK_MIN_SEARCH_RADIUS, LOCK_MAX_SEARCH_RADIUS)
    sx1, sy1 = int(x - shw), int(y - shh)
    sx2, sy2 = int(x + shw), int(y + shh)
    return sx1, sy1, sx2, sy2, int(shw), int(shh), (sx2 - sx1) * (sy2 - sy1)


def pick_nearest(items: List) -> Optional[Any]:
    """Pick the item with smallest first element (dis)."""
    if not items:
        return None
    return min(items, key=lambda x: x[0])


def draw_square(image: np.ndarray, box: Any, names: Dict[int, str],
                r: List[int]) -> Tuple[int, int]:
    """Draw bounding box, label, and center dot. Returns (ux, uy)."""
    ux = int((r[0] + r[2]) / 2)
    uy = int((r[1] + r[3]) / 2)
    cls = int(box.cls[0])
    conf = box.conf[0]
    label = f"{names[cls]} {conf:.2f}"
    cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (221, 185, 193), 2)
    cv2.putText(image, label, (r[0], r[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (176, 196, 222), 2)
    cv2.circle(image, (ux, uy), 5, (240, 240, 240), -1)
    return ux, uy


def draw_lock_rect(canvas: np.ndarray, x: int, y: int, shw: int, shh: int,
                   color: Tuple[int, int, int]) -> None:
    """Draw search window border and center crosshair."""
    cv2.rectangle(canvas, (x - shw, y - shh), (x + shw, y + shh), color, 2)
    cv2.circle(canvas, (x, y), 4, color, -1)


def clear_files(files: List[str]) -> None:
    for fn in files:
        with open(fn, 'w'):
            pass
    logger.info("通讯txt文件已建立并清空")


# ======================================================================
# 3. Data types
# ======================================================================

class TrackAction(Enum):
    DETECT = auto()   # real detection → [1, x, y]
    PREDICT = auto()  # missed but tracking → [2, x, y]
    LOST = auto()     # no target → [0] or fallback


@dataclass
class TrackResult:
    action: TrackAction
    x: int = 0
    y: int = 0
    is_locked: bool = False


@dataclass
class Detection:
    """Single detection in full-frame coordinates."""
    ux: int
    uy: int
    cls: int
    conf: float
    r: List[int]          # [x1, y1, x2, y2] in full-frame coords
    w: int
    h: int
    area: int
    dis: float             # distance from frame center
    box: Any               # ultralytics Box (for draw_square)
    names: Dict[int, str]  # class names (for draw_square)


# ======================================================================
# 4. FrameBuffer — thread-safe single-slot frame buffer
# ======================================================================

class FrameBuffer:
    """Single-producer single-consumer buffer, keeps only the latest frame."""

    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._seq: int = 0
        self._lock = threading.Lock()

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._seq += 1

    def get_latest(self, seen_seq: int = 0) -> Tuple[bool, Optional[np.ndarray], int]:
        with self._lock:
            if self._seq <= seen_seq or self._frame is None:
                return False, None, self._seq
            return True, self._frame.copy(), self._seq


# ======================================================================
# 5. FileMonitor — polls data.txt, signals mode changes
# ======================================================================

class FileMonitor:
    """Polls data.txt for command changes. Signals via threading.Event."""

    def __init__(self, file_path: str, poll_interval: float = 0.05) -> None:
        self.file_path = file_path
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.command_active = threading.Event()
        self._lock = threading.Lock()
        self.current_command: Optional[str] = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def read_command(self) -> Optional[str]:
        with self._lock:
            return self.current_command

    def _run(self) -> None:
        last_content: Optional[str] = None
        while not self._stop.is_set():
            try:
                with open(self.file_path, 'r') as f:
                    content = f.read().strip()
            except Exception:
                time.sleep(self.poll_interval)
                continue

            if content != last_content:
                with self._lock:
                    self.current_command = content
                if content in ('1m', '2m'):
                    logger.info(f"检测到{content}模式")
                    self.command_active.set()
                elif content == '0':
                    logger.info("检测到0模式，关闭摄像头")
                    self.command_active.clear()
                else:
                    logger.info("未检测到有效内容，等待...")
                last_content = content

            time.sleep(self.poll_interval)


# ======================================================================
# 6. OutputWriter — writes gaozhi.txt, manages history
# ======================================================================

class OutputWriter:
    """Writes detection results to gaozhi.txt. Manages last_value, history
    timeout clearing, and b_zero one-shot log."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._lock = threading.Lock()
        self.last_value: Optional[str] = None
        self.last_detection_time: float = 0.0
        self.b_zero: bool = True

    def write_detection(self, mode: str, ux: int, uy: int) -> None:
        """Write [1, ux, uy]. Updates last_value and timestamp."""
        with self._lock:
            self._log_2m_entry(mode)
            self._write(f"1 {ux} {uy}")
            self.last_value = f"{ux} {uy}"
            self.last_detection_time = time.time()

    def write_prediction(self, mode: str, ux: int, uy: int) -> None:
        """Write [2, ux, uy]. Updates last_value but not timestamp (prediction
        is not a real detection)."""
        with self._lock:
            self._log_2m_entry(mode)
            self._write(f"2 {ux} {uy}")
            self.last_value = f"{ux} {uy}"

    def write_fallback(self, mode: str) -> None:
        """Write fallback when tracker reports LOST.
        In 2m mode: replays last_value if available, otherwise writes [0].
        In 1m mode: writes [0]."""
        with self._lock:
            self._check_timeout()
            self._log_2m_entry(mode)

            if mode == '2m' and self.last_value is not None:
                content = "0" if self.last_value == '0' else "2 " + self.last_value
            else:
                content = "0"

            self._write(content)

            # In 2m mode, writing '0' preserves last_value
            if mode == '2m' and content.strip() == '0':
                return

            parts = content.strip().split()
            self.last_value = '0' if parts[0] == '0' else ' '.join(parts[1:])

    def reset(self, cold_start: bool = False) -> None:
        """Reset state on mode switch. cold_start=True clears everything."""
        with self._lock:
            if cold_start:
                self.last_value = None
                self.last_detection_time = time.time()
                self.b_zero = True
                self._write("0")

    def _check_timeout(self) -> None:
        if (HISTORY_CLEAR_ENABLED and self.last_value is not None
                and time.time() - self.last_detection_time > HISTORY_CLEAR_TIMEOUT):
            self.last_value = None
            logger.info(f"超过{HISTORY_CLEAR_TIMEOUT}秒未检测到目标，清空历史坐标")

    def _log_2m_entry(self, mode: str) -> None:
        if mode == '2m' and self.b_zero:
            logger.info("我方即将进入对桶程序※※")
            self.b_zero = False

    def _write(self, content: str) -> None:
        with open(self.file_path, 'w') as f:
            f.write(content)


# ======================================================================
# 7. LockTracker — lock-based tracking state machine
# ======================================================================
#
# States: UNLOCKED / LOCKED (is_locked property).
#
# 8 transition paths, called from update():
#
#   State     | Detections | Condition              | Path | Output
#   ----------+------------+------------------------+------+--------
#   LOCKED    | yes        | candidate in window    |  1   | DETECT
#   LOCKED    | yes        | no candidate, miss≤max |  2   | PREDICT
#   LOCKED    | yes        | no candidate, miss>max |  3   | DETECT (relock)
#   LOCKED    | no         | miss ≤ max_miss        |  4   | PREDICT
#   LOCKED    | no         | miss > max_miss        |  5   | LOST
#   UNLOCKED  | yes        | —                      |  6   | DETECT (acquire)
#   UNLOCKED  | no         | —                      |  7   | LOST
#   (pre-inf) | —          | frame≥max_hit          |  8   | force_unlock
#

class LockTracker:
    """Lock-based single-target tracker. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.lock_target: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
        self.lock_miss_count: int = 0
        self.lock_frame_count: int = 0

    @property
    def is_locked(self) -> bool:
        return self.lock_target is not None

    def reset(self) -> None:
        with self._lock:
            self.lock_target = None
            self.lock_miss_count = 0
            self.lock_frame_count = 0

    def increment_frame_count(self) -> None:
        with self._lock:
            if self.is_locked:
                self.lock_frame_count += 1

    def force_unlock(self) -> None:
        """Path 8: max_hit reached → full-frame re-evaluation this frame."""
        with self._lock:
            logger.info(f"锁定满{LOCK_MAX_HIT}帧，强制重判")
            self.lock_target = None
            self.lock_miss_count = 0
            self.lock_frame_count = 0

    def update(self, detections: List[Detection]) -> TrackResult:
        """Main FSM entry point. Thread-safe."""
        with self._lock:
            if self.is_locked:
                return (self._locked_with_dets(detections) if detections
                        else self._locked_no_dets())
            else:
                return (self._unlocked_with_dets(detections) if detections
                        else self._unlocked_no_dets())

    # ---- LOCKED + detections (Paths 1, 2, 3) ----

    def _locked_with_dets(self, detections: List[Detection]) -> TrackResult:
        ox, oy, ow, oh = self.lock_target  # type: ignore[misc]
        sx1, sy1, sx2, sy2, shw, shh, sarea = search_rect(ox, oy, ow, oh)

        candidates = [(d.dis, d.cls, d.ux, d.uy, d.r)
                      for d in detections
                      if sx1 <= d.ux <= sx2 and sy1 <= d.uy <= sy2]
        best = pick_nearest(candidates)

        if best is not None:
            # Path 1: hit → update lock
            _, _, ux, uy, r = best
            w, h = abs(r[2] - r[0]), abs(r[3] - r[1])
            self.lock_target = (ux, uy, w, h)
            self.lock_miss_count = 0
            logger.info(f"锁定桶：({ux},{uy}) 目标面积={w*h} "
                        f"搜索框=({sx1},{sy1},{sx2},{sy2}) 搜索面积={sarea}")
            return TrackResult(TrackAction.DETECT, ux, uy, True)

        self.lock_miss_count += 1
        if self.lock_miss_count > LOCK_MAX_MISS:
            # Path 3: exceeded miss limit → unlock & relock on nearest
            logger.info(f"丢帧满{LOCK_MAX_MISS}帧，强制重判")
            self.lock_target = None
            self.lock_miss_count = 0
            self.lock_frame_count = 0
            best = pick_nearest([(d.dis, d.cls, d.ux, d.uy, d.r) for d in detections])
            _, _, ux, uy, r = best
            w, h = abs(r[2] - r[0]), abs(r[3] - r[1])
            self.lock_target = (ux, uy, w, h)
            _, _, _, _, nshw, nshh, narea = search_rect(ux, uy, w, h)
            logger.info(f"解锁-最近桶：({ux},{uy}) 目标面积={w*h} "
                        f"搜索框=({ux-nshw},{uy-nshh},{ux+nshw},{uy+nshh}) 搜索面积={narea}")
            return TrackResult(TrackAction.DETECT, ux, uy, True)
        else:
            # Path 2: miss within limit → predict old position
            logger.info(f"丢帧：({ox},{oy}) miss={self.lock_miss_count} "
                        f"搜索框=({sx1},{sy1},{sx2},{sy2}) 搜索面积={sarea}")
            return TrackResult(TrackAction.PREDICT, ox, oy, True)

    # ---- LOCKED + no detections (Paths 4, 5) ----

    def _locked_no_dets(self) -> TrackResult:
        ox, oy, ow, oh = self.lock_target  # type: ignore[misc]
        sx1, sy1, sx2, sy2, shw, shh, sarea = search_rect(ox, oy, ow, oh)

        self.lock_miss_count += 1
        if self.lock_miss_count > LOCK_MAX_MISS:
            # Path 5: lost → unlock
            logger.info(f"丢帧满{LOCK_MAX_MISS}帧，强制重判")
            self.lock_target = None
            self.lock_miss_count = 0
            self.lock_frame_count = 0
            return TrackResult(TrackAction.LOST, 0, 0, False)
        else:
            # Path 4: miss within limit → predict
            logger.info(f"丢帧(无检测)：({ox},{oy}) miss={self.lock_miss_count} "
                        f"搜索框=({sx1},{sy1},{sx2},{sy2}) 搜索面积={sarea}")
            return TrackResult(TrackAction.PREDICT, ox, oy, True)

    # ---- UNLOCKED + detections (Path 6) ----

    def _unlocked_with_dets(self, detections: List[Detection]) -> TrackResult:
        best = pick_nearest([(d.dis, d.cls, d.ux, d.uy, d.r) for d in detections])
        _, _, ux, uy, r = best
        w, h = abs(r[2] - r[0]), abs(r[3] - r[1])
        self.lock_target = (ux, uy, w, h)
        self.lock_miss_count = 0
        self.lock_frame_count = 0
        _, _, _, _, shw, shh, sarea = search_rect(ux, uy, w, h)
        logger.info(f"最近桶：({ux},{uy}) 目标面积={w*h} "
                    f"搜索框=({ux-shw},{uy-shh},{ux+shw},{uy+shh}) 搜索面积={sarea}")
        return TrackResult(TrackAction.DETECT, ux, uy, True)

    # ---- UNLOCKED + no detections (Path 7) ----

    def _unlocked_no_dets(self) -> TrackResult:
        return TrackResult(TrackAction.LOST, 0, 0, False)


# ======================================================================
# 8. YOLODetector — dual-model YOLO inference
# ======================================================================

class YOLODetector:
    """Dual-model YOLO detector.

    detect_full  → model_unlocked (full-frame, mode-specific conf/imgsz)
    detect_crop  → model_locked  (cropped region, dynamic imgsz)
    """

    def __init__(self) -> None:
        self.model_locked = YOLO(MODEL_LOCKED_PATH)
        self.model_unlocked = YOLO(MODEL_UNLOCKED_PATH)
        self._cx = D435_WIDTH // 2
        self._cy = D435_HEIGHT // 2

    def detect_full(self, frame: np.ndarray, mode: str) -> List[Detection]:
        conf = MODEL_UNLOCKED_1M_CONF if mode == '1m' else MODEL_UNLOCKED_2M_CONF
        imgsz = MODEL_UNLOCKED_1M_IMGSZ if mode == '1m' else MODEL_UNLOCKED_2M_IMGSZ
        results = self.model_unlocked.predict(
            source=frame, device=MODEL_DEVICE, show=False,
            stream=False, verbose=False, iou=MODEL_IOU, conf=conf, imgsz=imgsz)
        return self._parse(results, 0, 0)

    def detect_crop(self, crop: np.ndarray, ox: int, oy: int) -> List[Detection]:
        h, w = crop.shape[:2]
        imgsz = ((max(w, h) + MODEL_IMGSZ_STEP - 1) // MODEL_IMGSZ_STEP) * MODEL_IMGSZ_STEP
        results = self.model_locked.predict(
            source=crop, device=MODEL_DEVICE, show=False,
            stream=False, verbose=False, iou=MODEL_IOU,
            conf=MODEL_LOCKED_CONF, imgsz=imgsz)
        return self._parse(results, ox, oy)

    def _parse(self, results: Any, ox: int, oy: int) -> List[Detection]:
        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            names = result.names
            if boxes is None:
                continue
            for box in boxes:
                r = box.xyxy[0].cpu().numpy().astype(int)
                r[0] += ox; r[1] += oy; r[2] += ox; r[3] += oy
                ux = int((r[0] + r[2]) / 2)
                uy = int((r[1] + r[3]) / 2)
                w = abs(int(r[2] - r[0]))
                h = abs(int(r[3] - r[1]))
                area = w * h
                if area > MAX_AREA:
                    continue
                dis = ((ux - self._cx) ** 2 + (uy - self._cy) ** 2) ** 0.5
                detections.append(Detection(
                    ux=ux, uy=uy, cls=int(box.cls[0]), conf=float(box.conf[0]),
                    r=r.tolist(), w=w, h=h, area=area, dis=dis, box=box, names=names))
        return detections


# ======================================================================
# 9. CameraSource — D435 camera with internal capture thread
# ======================================================================

class CameraSource:
    """D435 camera wrapper. Internal capture thread feeds a FrameBuffer."""

    def __init__(self) -> None:
        self._pipeline: Optional[rs.pipeline] = None
        self._buf = FrameBuffer()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, D435_WIDTH, D435_HEIGHT, rs.format.bgr8, 30)
        self._pipeline.start(cfg)
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture, daemon=True)
        self._thread.start()
        logger.debug("D435 capture thread started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._pipeline is not None:
            self._pipeline.stop()
            logger.info("D435摄像头已释放")

    def get_frame(self, seen_seq: int) -> Tuple[bool, Optional[np.ndarray], int]:
        return self._buf.get_latest(seen_seq)

    @staticmethod
    def check_device() -> bool:
        return len(rs.context().devices) > 0

    def _capture(self) -> None:
        while not self._stop.is_set():
            frames = self._pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if color:
                self._buf.put(np.asanyarray(color.get_data()))


# ======================================================================
# 10. DetectionPipeline — main processing loop
# ======================================================================

class DetectionPipeline:
    """Orchestrates: capture → inference → tracking → output → display."""

    def __init__(self, camera: CameraSource, detector: YOLODetector,
                 tracker: LockTracker, writer: OutputWriter,
                 monitor: FileMonitor) -> None:
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.writer = writer
        self.monitor = monitor

    def run(self, stop_event: threading.Event,
            display_queue: queue.Queue) -> None:
        self.camera.start()
        last_mode: Optional[str] = None
        last_canvas: Optional[np.ndarray] = None
        last_seq: int = 0
        frame_count: int = 0
        fps_times: deque = deque(maxlen=30)
        fps_print_count: int = 0

        try:
            while not stop_event.is_set():
                mode = self.monitor.read_command()

                if mode not in ('1m', '2m'):
                    if last_mode in ('1m', '2m'):
                        with open('gaozhi.txt', 'w') as f:
                            f.write('0')
                    last_mode = mode
                    time.sleep(0.01)
                    continue

                # Mode switch
                if mode != last_mode:
                    logger.info(f"模式切换: {last_mode} -> {mode}")
                    cold = last_mode not in ('1m', '2m')
                    self.writer.reset(cold_start=cold)
                    self.tracker.reset()
                    if cold:
                        logger.info("冷启动：历史坐标已清空")
                    last_mode = mode
                    frame_count = 0
                    fps_print_count = 5

                # Get latest frame
                got, frame, last_seq = self.camera.get_frame(last_seq)
                if not got:
                    time.sleep(0.001)
                    continue

                # FPS
                now = time.time()
                fps_times.append(now)
                fps = (len(fps_times) / (fps_times[-1] - fps_times[0])
                       if len(fps_times) > 1 else 0)
                if fps_print_count > 0:
                    logger.info(f"实时帧率: {fps:.1f} FPS")
                    fps_print_count -= 1

                frame_count += 1
                do_inference = (not FRAME_SKIP_ENABLED
                                or frame_count % FRAME_SKIP_N == 0)

                if do_inference:
                    canvas = frame.copy()
                    self._process_frame(frame, canvas, mode)
                    last_canvas = canvas.copy()
                else:
                    canvas = (last_canvas.copy() if last_canvas is not None
                              else frame.copy())

                # Overlay mode and FPS
                cv2.putText(canvas, f"{mode}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"FPS:{fps:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                try:
                    display_queue.put((canvas, DISPLAY_WIN_NAME), block=False)
                except queue.Full:
                    pass

        finally:
            self.camera.stop()

    def _process_frame(self, frame: np.ndarray, canvas: np.ndarray,
                       mode: str) -> None:
        # Pre-inference: force_unlock if frame_count ≥ max_hit (Path 8)
        self.tracker.increment_frame_count()
        if self.tracker.is_locked and self.tracker.lock_frame_count >= LOCK_MAX_HIT:
            self.tracker.force_unlock()

        was_locked = self.tracker.is_locked

        # YOLO inference
        if was_locked:
            ox, oy, ow, oh = self.tracker.lock_target  # type: ignore[misc]
            sx1, sy1, sx2, sy2, shw, shh, _ = search_rect(ox, oy, ow, oh)
            csx1 = max(0, int(sx1)); csy1 = max(0, int(sy1))
            csx2 = min(D435_WIDTH, int(sx2)); csy2 = min(D435_HEIGHT, int(sy2))
            crop = frame[csy1:csy2, csx1:csx2]
            detections = self.detector.detect_crop(crop, csx1, csy1)
            draw_lock_rect(canvas, ox, oy, shw, shh, (0, 255, 255))
        else:
            detections = self.detector.detect_full(frame, mode)

        # Draw all detections
        for d in detections:
            ux, uy = draw_square(canvas, d.box, d.names, d.r)
            cv2.circle(canvas, (d.ux, d.uy), 4, (255, 255, 255), 5)
            cv2.putText(canvas, str([d.ux, d.uy]), (d.ux + 20, d.uy + 10),
                        0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

        # Orange lock rect when LOCKED + no detections (drawn before tracker may unlock)
        if was_locked and not detections:
            ox, oy, ow, oh = self.tracker.lock_target  # type: ignore[misc]
            _, _, _, _, shw, shh, _ = search_rect(ox, oy, ow, oh)
            draw_lock_rect(canvas, ox, oy, shw, shh, (0, 165, 255))

        # Run tracker state machine
        result = self.tracker.update(detections)

        logger.debug(f"模式：{mode}{'锁定' if result.is_locked else ''} "
                     f"检测数={len(detections)} "
                     f"锁帧数={self.tracker.lock_frame_count} "
                     f"丢帧数={self.tracker.lock_miss_count}")

        # Write output
        if result.action == TrackAction.DETECT:
            self.writer.write_detection(mode, result.x, result.y)
            # Green lock rect for newly acquired lock
            if not was_locked and result.is_locked:
                ox, oy, ow, oh = self.tracker.lock_target  # type: ignore[misc]
                _, _, _, _, shw, shh, _ = search_rect(ox, oy, ow, oh)
                draw_lock_rect(canvas, ox, oy, shw, shh, (0, 255, 0))
        elif result.action == TrackAction.PREDICT:
            self.writer.write_prediction(mode, result.x, result.y)
        else:  # LOST
            self.writer.write_fallback(mode)


# ======================================================================
# 11. main()
# ======================================================================

def main() -> None:
    logger.info("Yolo26n目标检测-程序启动")
    logger.info("开始Yolo26n模型加载")

    clear_files(FILES_TO_CLEAR)

    # Init modules
    detector = YOLODetector()
    camera = CameraSource()
    tracker = LockTracker()
    writer = OutputWriter('gaozhi.txt')
    monitor = FileMonitor('data.txt')
    pipeline = DetectionPipeline(camera, detector, tracker, writer, monitor)

    # Check D435
    if not CameraSource.check_device():
        for _ in range(5):
            logger.warning("未检测到D435相机，请检查连接！")
    else:
        logger.info("D435相机已连接。")

    monitor.start()

    main_stop = threading.Event()
    display_queue: queue.Queue = queue.Queue(maxsize=DISPLAY_QUEUE_SIZE)

    # Pipeline manager: start/stop DetectionPipeline based on FileMonitor signal
    def pipeline_manager() -> None:
        p_stop: Optional[threading.Event] = None
        p_thread: Optional[threading.Thread] = None
        try:
            while not main_stop.is_set():
                monitor.command_active.wait()
                if main_stop.is_set():
                    break
                if p_thread is None or not p_thread.is_alive():
                    p_stop = threading.Event()
                    p_thread = threading.Thread(
                        target=pipeline.run, args=(p_stop, display_queue), daemon=True)
                    p_thread.start()
                    logger.info("检测管道已启动")
                while monitor.command_active.is_set() and not main_stop.is_set():
                    time.sleep(0.1)
                if p_stop is not None:
                    p_stop.set()
                if p_thread is not None and p_thread.is_alive():
                    p_thread.join(timeout=2)
                cv2.destroyAllWindows()
                p_thread = None
                logger.info("检测管道已停止")
        finally:
            if p_stop is not None:
                p_stop.set()
            if p_thread is not None and p_thread.is_alive():
                p_thread.join(timeout=2)
            cv2.destroyAllWindows()

    manager_thread = threading.Thread(target=pipeline_manager, daemon=True)
    manager_thread.start()

    logger.info("完成Yolo26n模型加载")
    logger.info("系统初始化完成，等待指令...")

    # Display loop
    window_created = False
    try:
        while True:
            try:
                frame, win_name = display_queue.get(timeout=0.05)
            except queue.Empty:
                if main_stop.is_set():
                    break
                continue

            if not window_created:
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL |
                                cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                cv2.resizeWindow(win_name, D435_WIDTH, D435_HEIGHT)
                window_created = True

            cv2.imshow(win_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                main_stop.set()
                break

    except KeyboardInterrupt:
        logger.info("键盘中断，正在退出...")
    finally:
        main_stop.set()
        monitor.stop()
        manager_thread.join(timeout=3)
        cv2.destroyAllWindows()
        logger.info("程序已退出")


if __name__ == '__main__':
    main()
