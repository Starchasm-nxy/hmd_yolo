"""
2m/4m 遍历方案
yolo D435 + USB 双相机目标检测

D435 部分：t39.py 架构（LockTracker / FrameBuffer / DetectionPipeline）
USB 部分：t44.py 原有逻辑（DBSCAN 聚类 + 多目标选通）
"""

import os
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
from sklearn.cluster import DBSCAN

# ======================================================================
# 1. Logging setup
# ======================================================================

RESET  = "\033[0m"
RED    = "\033[31m"
ORANGE = "\033[38;5;214m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"

class ColoredFormatter(logging.Formatter):
    """按日志级别着色：WARNING→橙色, ERROR/CRITICAL→红色"""
    COLORS = {
        logging.WARNING:  ORANGE,
        logging.ERROR:    RED,
        logging.CRITICAL: RED,
    }
    def format(self, record):
        msg = super().format(record)
        color = self.COLORS.get(record.levelno)
        return f"{color}{msg}{RESET}" if color else msg

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s [%(levelname)s] %(message)s'))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

def yellow_log(msg: str) -> None:
    """输出黄色高亮的 INFO 日志"""
    logger.info(f"{YELLOW}{msg}{RESET}")


# ======================================================================
# 2. Configuration
# ======================================================================

# ---- 启动时清空的文件 ----
FILES_TO_CLEAR = ['data.txt', 'gaozhi.txt']

# ---- D435 RealSense 参数 ----
D435_WIDTH = 848
D435_HEIGHT = 480
D435_FPS = 30

# ---- D435 模型文件路径 ----
MODEL_2M_PATH = "/home/fu/weights/tong_blue_v0.pt"
MODEL_1M_PATH = "/home/fu/weights/tong_blue_v0.pt"

# ---- USB 模型文件路径 ----
MODEL_USB_PATH = "/home/fu/weights/4m12kd.pt"

# ---- USB 摄像头路径 ----
USB_CAM_PATH = '/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0'

# ---- USB 标定文件 ----
CALIB_FILE = 'calib_resultA.npz'

# ---- 历史坐标超时清空（D435）----
HISTORY_CLEAR_ENABLED = True
HISTORY_CLEAR_TIMEOUT = 10.0

# ---- 跳帧推理（D435）----
FRAME_SKIP_ENABLED = False
FRAME_SKIP_N = 2

# ---- D435 锁定跟踪参数 ----
LOCK_MAX_HIT = 15
LOCK_MAX_MISS = 7
LOCK_SEARCH_RATIO = 2.5
LOCK_MIN_SEARCH_RADIUS = 130
LOCK_MAX_SEARCH_RADIUS = 270

# ---- D435 模型推理参数 ----
MODEL_2M_CONF = 0.5
MODEL_2M_IMGSZ = 640
MODEL_1M_IMGSZ = 640
MODEL_1M_CONF = 0.5
MODEL_IOU = 0.45
MODEL_IMGSZ_STEP = 32
MODEL_DEVICE = 'cpu'
MAX_AREA = 223300

# ---- USB 模型推理参数 ----
USB_CONF = 0.6

# ---- USB DBSCAN 聚类参数 ----
DBSCAN_EPS = 50
DBSCAN_MIN_SAMPLES = 15
USB_MAX_WAIT_TIME = 5.0

# ---- USB 像素→世界坐标转换参数 ----
Z_CAMERA_HEIGHT = 4.31
USB_FRAME_W = 640
USB_FRAME_H = 480

# ---- 显示参数 ----
DISPLAY_WIN_NAME_D435 = "d435"
DISPLAY_WIN_NAME_USB = "usb"
DISPLAY_QUEUE_SIZE = 3

# ---- FileMonitor 轮询间隔 ----
FILE_MONITOR_POLL_INTERVAL = 0.05


# ======================================================================
# 3. Data types
# ======================================================================

class TrackAction(Enum):
    """跟踪动作枚举：DETECT真实检测，PREDICT预测，LOST丢失"""
    DETECT = auto()
    PREDICT = auto()
    LOST = auto()


@dataclass
class TrackResult:
    """跟踪结果数据类"""
    action: TrackAction
    x: int = 0
    y: int = 0
    is_locked: bool = False


@dataclass
class Detection:
    """单个检测框信息，坐标均为全图坐标"""
    ux: int
    uy: int
    cls: int
    conf: float
    r: List[int]
    w: int
    h: int
    area: int
    dis: float
    box: Any
    names: Dict[int, str]


@dataclass
class Command:
    """解析后的 data.txt 指令"""
    kind: str             # 'usb' | 'd435' | 'none'
    usb_target: int = 0   # 1-6（USB 方向选通，对应原 d 变量）
    d435_mode: str = ''   # '1m' | '2m'（D435 模式，对应原 b 变量）

    # data.txt 内容 → Command 映射表
    _MAP = {
        'ml': ('usb', 6), 'rm': ('usb', 5), 'rl': ('usb', 4),
        'lm': ('usb', 3), 'lr': ('usb', 2), 'mr': ('usb', 1),
        '2m': ('d435', 0, '2m'), '1m': ('d435', 0, '1m'),
    }

    @classmethod
    def parse(cls, content: str) -> 'Command':
        """从 data.txt 内容解析 Command"""
        content = content.strip()
        if content in cls._MAP:
            entry = cls._MAP[content]
            if entry[0] == 'usb':
                return cls(kind='usb', usb_target=entry[1])
            else:
                return cls(kind='d435', d435_mode=entry[2])
        return cls(kind='none')


# ======================================================================
# 4. Utility functions
# ======================================================================

def clamp(v: float, lo: float, hi: float) -> float:
    """将浮点数v限制在[lo, hi]区间内"""
    return max(lo, min(hi, v))


def search_rect(x: float, y: float, w: float, h: float) -> Tuple[int, int, int, int, int, int, int]:
    """
    根据目标中心(x,y)和宽高(w,h)计算自适应搜索窗口
    返回: (sx1, sy1, sx2, sy2, shw, shh, area)
    """
    shw = clamp(w * LOCK_SEARCH_RATIO / 2, LOCK_MIN_SEARCH_RADIUS, LOCK_MAX_SEARCH_RADIUS)
    shh = clamp(h * LOCK_SEARCH_RATIO / 2, LOCK_MIN_SEARCH_RADIUS, LOCK_MAX_SEARCH_RADIUS)
    sx1, sy1 = int(x - shw), int(y - shh)
    sx2, sy2 = int(x + shw), int(y + shh)
    return sx1, sy1, sx2, sy2, int(shw), int(shh), (sx2 - sx1) * (sy2 - sy1)


def pick_nearest(items: List) -> Optional[Any]:
    """从列表中选取第一个元素（距离）最小的项，若列表为空返回None"""
    if not items:
        return None
    return min(items, key=lambda x: x[0])


def draw_square(image: np.ndarray, box: Any, names: Dict[int, str],
                r: List[int]) -> Tuple[int, int]:
    """
    在图像上绘制检测框、标签和中心点。
    返回: (ux, uy) 框的中心点坐标
    """
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
    """在画布上绘制搜索窗口边框和中心十字准星"""
    cv2.rectangle(canvas, (x - shw, y - shh), (x + shw, y + shh), color, 2)
    cv2.circle(canvas, (x, y), 4, color, -1)


def pixel_to_camera(ux: float, uy: float, mtx: np.ndarray, w: int, h: int) -> Tuple[float, float]:
    """
    USB 像素坐标 → 世界坐标转换（t44.py 原有逻辑）
    """
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = w / 2
    cy = h / 2
    X = (ux - cx) * Z_CAMERA_HEIGHT / fx
    Y = (uy - cy) * Z_CAMERA_HEIGHT / fy
    return X, Y


def clear_files(files: List[str]) -> None:
    """清空指定的文件列表（创建空文件）"""
    for fn in files:
        with open(fn, 'w'):
            pass
    yellow_log("通讯txt文件已建立并清空")


# ======================================================================
# 5. FrameBuffer — thread-safe single-slot frame buffer
# ======================================================================

class FrameBuffer:
    """单生产者单消费者帧缓冲，只保留最新帧，线程安全"""

    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._seq: int = 0
        self._lock = threading.Lock()

    def put(self, frame: np.ndarray) -> None:
        """写入新帧，覆盖旧帧，递增序号"""
        with self._lock:
            self._frame = frame
            self._seq += 1

    def get_latest(self, seen_seq: int = 0) -> Tuple[bool, Optional[np.ndarray], int]:
        """
        获取最新帧，如果序号未更新则返回(False, None, 当前序号)
        返回: (是否新帧, 帧数据拷贝, 最新序号)
        """
        with self._lock:
            if self._seq <= seen_seq or self._frame is None:
                return False, None, self._seq
            return True, self._frame.copy(), self._seq


# ======================================================================
# 6. FileMonitor — polls data.txt, signals mode changes
# ======================================================================

class FileMonitor:
    """
    轮询 data.txt 文件内容变化，解析为 Command 对象。
    支持 D435 指令（2m/1m）和 USB 指令（ml/rm/rl/lm/lr/mr）。
    """

    def __init__(self, file_path: str, poll_interval: float = FILE_MONITOR_POLL_INTERVAL) -> None:
        self.file_path = file_path
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.command_active = threading.Event()
        self._lock = threading.Lock()
        self.current_command: Optional[Command] = None
        self.clearfile = False
        self.connectfile = False

    def start(self) -> None:
        """启动后台轮询线程"""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止轮询线程"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def read_command(self) -> Optional[Command]:
        """线程安全地读取当前命令"""
        with self._lock:
            return self.current_command

    def _run(self) -> None:
        """后台轮询主循环"""
        last_content: Optional[str] = None
        while not self._stop.is_set():
            try:
                if not self.clearfile:
                    logger.info("读取成功and去除空白")
                    self.clearfile = True
                    self.connectfile = True
                with open(self.file_path, 'r') as f:
                    content = f.read().strip()
            except Exception:
                if not self.connectfile:
                    logger.warning("读取失败and等待重试")
                time.sleep(self.poll_interval)
                continue

            if content != last_content:
                cmd = Command.parse(content)
                with self._lock:
                    self.current_command = cmd
                # 根据指令类型设置激活事件
                if cmd.kind in ('d435', 'usb'):
                    if cmd.kind == 'd435':
                        for _ in range(5):
                            yellow_log(f"检测到{cmd.d435_mode}模式")
                    else:
                        yellow_log(f"检测到USB模式(d={cmd.usb_target})")
                    self.command_active.set()
                elif cmd.kind == 'none' and content == '0':
                    yellow_log("检测到0模式>_<已关闭")
                    self.command_active.clear()
                elif cmd.kind == 'none' and content:
                    yellow_log("目前无效内容>_<等待ing...")
                last_content = content

            time.sleep(self.poll_interval)


# ======================================================================
# 7. OutputWriter — writes gaozhi.txt (D435), manages history
# ======================================================================

class OutputWriter:
    """将 D435 检测结果写入 gaozhi.txt，管理历史坐标、超时清空"""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._lock = threading.Lock()
        self.last_value: Optional[str] = None
        self.last_detection_time: float = 0.0
        self.b_zero: bool = True
        self.a_zero: bool = True

    def write_detection(self, mode: str, ux: int, uy: int) -> None:
        """写入真实检测结果 [1, ux, uy]"""
        with self._lock:
            self._log_entry(mode)
            self._write(f"1 {ux} {uy}")
            self.last_value = f"{ux} {uy}"
            self.last_detection_time = time.time()

    def write_prediction(self, mode: str, ux: int, uy: int) -> None:
        """写入预测结果 [2, ux, uy]"""
        with self._lock:
            self._log_entry(mode)
            self._write(f"2 {ux} {uy}")
            self.last_value = f"{ux} {uy}"

    def write_fallback(self, mode: str) -> None:
        """跟踪器报告 LOST 时的回退写入策略"""
        with self._lock:
            self._check_timeout()
            self._log_entry(mode)
            if mode != '0' and self.last_value is not None:
                content = "0" if self.last_value == '0' else "2 " + self.last_value
            else:
                content = "0"
            self._write(content)
            if mode == '2m' and content.strip() == '0':
                return
            parts = content.strip().split()
            self.last_value = '0' if parts[0] == '0' else ' '.join(parts[1:])

    def reset(self, cold_start: bool = False) -> None:
        """模式切换时重置状态"""
        with self._lock:
            if cold_start:
                self.last_value = None
                self.last_detection_time = time.time()
                self.b_zero = True
                self.a_zero = True
                self._write("0")

    def _check_timeout(self) -> None:
        """若启用历史清空且超时，则清空 last_value"""
        if (HISTORY_CLEAR_ENABLED and self.last_value is not None
                and time.time() - self.last_detection_time > HISTORY_CLEAR_TIMEOUT):
            self.last_value = None
            logger.info(f"超过{HISTORY_CLEAR_TIMEOUT}秒未检测到目标，清空历史坐标")

    def _log_entry(self, mode: str) -> None:
        """首次写入非0值时打印提示日志"""
        if mode == '2m' and self.b_zero:
            for _ in range(5):
                yellow_log("2m对桶开始※※※")
            self.b_zero = False
        elif mode == '1m' and self.a_zero:
            for _ in range(5):
                yellow_log("1m对桶开始※※※")
            self.a_zero = False

    def _write(self, content: str) -> None:
        """实际写入文件"""
        with open(self.file_path, 'w') as f:
            f.write(content)


# ======================================================================
# 8. LockTracker — lock-based tracking state machine
# ======================================================================

class LockTracker:
    """基于锁定的单目标跟踪器，线程安全。8 路径状态机。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.lock_target: Optional[Tuple[int, int, int, int]] = None
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
        """Path 8: 锁定帧数达到上限，强制解锁"""
        with self._lock:
            yellow_log(f"锁定满{LOCK_MAX_HIT}帧，强制重判")
            self.lock_target = None
            self.lock_miss_count = 0
            self.lock_frame_count = 0

    def update(self, detections: List[Detection]) -> TrackResult:
        """状态机主入口，线程安全"""
        with self._lock:
            if self.is_locked:
                return (self._locked_with_dets(detections) if detections
                        else self._locked_no_dets())
            else:
                return (self._unlocked_with_dets(detections) if detections
                        else self._unlocked_no_dets())

    # ---- LOCKED + detections (Paths 1, 2, 3) ----

    def _locked_with_dets(self, detections: List[Detection]) -> TrackResult:
        ox, oy, ow, oh = self.lock_target
        sx1, sy1, sx2, sy2, shw, shh, sarea = search_rect(ox, oy, ow, oh)

        candidates = [(d.dis, d.cls, d.ux, d.uy, d.r)
                      for d in detections
                      if sx1 <= d.ux <= sx2 and sy1 <= d.uy <= sy2]
        best = pick_nearest(candidates)

        if best is not None:
            # Path 1: 窗口内有匹配 → 更新锁定目标
            _, _, ux, uy, r = best
            w, h = abs(r[2] - r[0]), abs(r[3] - r[1])
            self.lock_target = (ux, uy, w, h)
            self.lock_miss_count = 0
            logger.info(f"锁定桶：({ux},{uy}) 目标面积={w*h}")
            return TrackResult(TrackAction.DETECT, ux, uy, True)

        # 窗口内无匹配，丢失计数+1
        self.lock_miss_count += 1
        if self.lock_miss_count > LOCK_MAX_MISS:
            # Path 3: 丢失超过上限 → 解锁并立即重锁定
            yellow_log(f"丢帧满{LOCK_MAX_MISS}帧，强制重判")
            self.lock_target = None
            self.lock_miss_count = 0
            self.lock_frame_count = 0
            best = pick_nearest([(d.dis, d.cls, d.ux, d.uy, d.r) for d in detections])
            _, _, ux, uy, r = best
            w, h = abs(r[2] - r[0]), abs(r[3] - r[1])
            self.lock_target = (ux, uy, w, h)
            _, _, _, _, nshw, nshh, narea = search_rect(ux, uy, w, h)
            yellow_log(f"解锁-最近桶：({ux},{uy}) 目标面积={w*h}")
            return TrackResult(TrackAction.DETECT, ux, uy, True)
        else:
            # Path 2: 丢失但未超限 → 沿用旧位置进行预测
            logger.warning(f"锁定-历史：({ox},{oy}) miss={self.lock_miss_count} ")
            return TrackResult(TrackAction.PREDICT, ox, oy, True)

    # ---- LOCKED + no detections (Paths 4, 5) ----

    def _locked_no_dets(self) -> TrackResult:
        ox, oy, ow, oh = self.lock_target
        sx1, sy1, sx2, sy2, shw, shh, sarea = search_rect(ox, oy, ow, oh)

        self.lock_miss_count += 1
        if self.lock_miss_count > LOCK_MAX_MISS:
            # Path 5: 连续丢失超限 → 解锁
            yellow_log(f"丢帧满{LOCK_MAX_MISS}帧，强制重判")
            self.lock_target = None
            self.lock_miss_count = 0
            self.lock_frame_count = 0
            return TrackResult(TrackAction.LOST, 0, 0, False)
        else:
            # Path 4: 丢失但未超限 → 预测旧位置
            logger.warning(f"锁定-历史：({ox},{oy}) miss={self.lock_miss_count}")
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
        yellow_log(f"锁定-最近桶：({ux},{uy}) 目标面积={w*h}")
        return TrackResult(TrackAction.DETECT, ux, uy, True)

    # ---- UNLOCKED + no detections (Path 7) ----

    def _unlocked_no_dets(self) -> TrackResult:
        return TrackResult(TrackAction.LOST, 0, 0, False)


# ======================================================================
# 9. YOLODetector — dual-model YOLO inference (D435)
# ======================================================================

class YOLODetector:
    """双模型YOLO检测器（2m/1m），全图+裁剪检测"""

    def __init__(self) -> None:
        self.model_2m = YOLO(MODEL_2M_PATH)
        self.model_1m = YOLO(MODEL_1M_PATH)
        self._cx = D435_WIDTH // 2
        self._cy = D435_HEIGHT // 2

    def _select_model(self, mode: str):
        """根据模式返回对应的模型和参数"""
        if mode == '1m':
            return self.model_1m, MODEL_1M_CONF, MODEL_1M_IMGSZ
        return self.model_2m, MODEL_2M_CONF, MODEL_2M_IMGSZ

    def detect_full(self, frame: np.ndarray, mode: str) -> List[Detection]:
        model, conf, imgsz = self._select_model(mode)
        t0 = time.time()
        results = model.predict(
            source=frame, device=MODEL_DEVICE, show=False,
            stream=False, verbose=False, iou=MODEL_IOU, conf=conf, imgsz=imgsz)
        t1 = time.time()
        infer_time = int((t1-t0) * 1000)
        yellow_log(f"全图推理：{infer_time}ms")
        return self._parse(results, 0, 0)

    def detect_crop(self, crop: np.ndarray, ox: int, oy: int,
                    mode: str) -> List[Detection]:
        h, w = crop.shape[:2]
        imgsz = ((max(w, h) + MODEL_IMGSZ_STEP - 1) // MODEL_IMGSZ_STEP) * MODEL_IMGSZ_STEP
        model, conf, _ = self._select_model(mode)
        t0 = time.time()
        results = model.predict(
            source=crop, device=MODEL_DEVICE, show=False,
            stream=False, verbose=False, iou=MODEL_IOU,
            conf=conf, imgsz=imgsz)
        t1 = time.time()
        infer_time = int((t1-t0) * 1000)
        logger.info(f"裁减推理：{infer_time}ms")
        return self._parse(results, ox, oy)

    def _parse(self, results: Any, ox: int, oy: int) -> List[Detection]:
        """将YOLO推理结果解析为 Detection 列表"""
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
# 10. CameraSource — D435 camera with internal capture thread
# ======================================================================

class CameraSource:
    """D435相机封装，内部使用独立线程持续采集帧放入 FrameBuffer"""

    def __init__(self) -> None:
        self._pipeline: Optional[rs.pipeline] = None
        self._buf = FrameBuffer()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """启动RealSense管道和采集线程"""
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, D435_WIDTH, D435_HEIGHT, rs.format.bgr8, D435_FPS)
        self._pipeline.start(cfg)
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture, daemon=True)
        self._thread.start()
        logger.info("D435 capture thread started")

    def stop(self) -> None:
        """停止采集线程并释放相机资源"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._pipeline is not None:
            self._pipeline.stop()
            logger.info("D435摄像头已释放")

    def get_frame(self, seen_seq: int) -> Tuple[bool, Optional[np.ndarray], int]:
        """从缓冲区获取最新帧"""
        return self._buf.get_latest(seen_seq)

    @staticmethod
    def wait_for_camera(timeout: float = 30.0, poll_interval: float = 2.0) -> bool:
        """阻塞等待 D435 相机连接"""
        start = time.time()
        while True:
            if CameraSource.check_device():
                return True
            elapsed = time.time() - start
            if timeout is not None and elapsed >= timeout:
                logger.error(f"等待 {timeout}s 后仍未检测到 D435 相机，退出。")
                return False
            logger.warning(f"未检测到 D435 相机，{poll_interval}s 后重试...（已等待 {elapsed:.0f}s）")
            time.sleep(poll_interval)

    @staticmethod
    def check_device() -> bool:
        """检查是否有D435设备连接"""
        return len(rs.context().devices) > 0

    def _capture(self) -> None:
        """后台采集线程主循环"""
        while not self._stop.is_set():
            frames = self._pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if color:
                self._buf.put(np.asanyarray(color.get_data()))


# ======================================================================
# 11. DetectionPipeline — D435 main processing loop
# ======================================================================

class DetectionPipeline:
    """D435 主处理管道：采集 → 推理 → 跟踪 → 输出 → 显示"""

    MODEL_DELAY = 3.0  # 2m→1m 切换时延用2m模型的秒数

    def __init__(self, camera: CameraSource, detector: YOLODetector,
                 tracker: LockTracker, writer: OutputWriter,
                 monitor: FileMonitor) -> None:
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.writer = writer
        self.monitor = monitor
        self._model_mode: Optional[str] = None
        self._delay_until: float = 0.0

    def run(self, stop_event: threading.Event,
            display_queue: queue.Queue) -> None:
        """主循环，运行在独立线程中"""
        self.camera.start()
        last_mode: Optional[str] = None
        last_canvas: Optional[np.ndarray] = None
        last_seq: int = 0
        frame_count: int = 0
        fps_times: deque = deque(maxlen=60)

        try:
            while not stop_event.is_set():
                cmd = self.monitor.read_command()

                # 非 D435 模式：idle
                if cmd is None or cmd.kind != 'd435':
                    if last_mode in ('1m', '2m'):
                        with open('gaozhi.txt', 'w') as f:
                            f.write('0')
                    last_mode = None if (cmd is None or cmd.kind == 'none') else last_mode
                    time.sleep(0.01)
                    continue

                mode = cmd.d435_mode
                delay_time = 0.0

                # 模式切换处理
                if mode != last_mode:
                    logger.info(f"模式切换: {last_mode} -> {mode}")
                    cold = last_mode not in ('1m', '2m')
                    self.writer.reset(cold_start=cold)
                    self.tracker.reset()
                    if cold:
                        logger.info("冷启动：历史坐标已清空")
                    if last_mode == '2m' and mode == '1m':
                        self._delay_until = time.time() + self.MODEL_DELAY
                        logger.info(f"2m→1m: 保持2m模型 {self.MODEL_DELAY}s")
                    else:
                        self._model_mode = mode
                        self._delay_until = 0
                    last_mode = mode
                    frame_count = 0

                # 计算当前帧实际使用的模型模式（含2m→1m延时）
                if (self._model_mode == '2m' and mode == '1m'
                        and self._delay_until > time.time()):
                    delay_time = self._delay_until - time.time()
                    effective_mode = '2m'
                else:
                    if self._model_mode != mode:
                        self._model_mode = mode
                    effective_mode = mode

                # 从相机获取最新帧
                got, frame, last_seq = self.camera.get_frame(last_seq)
                if not got:
                    time.sleep(0.001)
                    continue

                # FPS
                now = time.time()
                fps_times.append(now)
                fps = (len(fps_times) / (fps_times[-1] - fps_times[0])
                       if len(fps_times) > 1 else 0)

                frame_count += 1
                do_inference = (not FRAME_SKIP_ENABLED
                                or frame_count % FRAME_SKIP_N == 0)

                if do_inference:
                    canvas = frame.copy()
                    self._process_frame(frame, canvas, effective_mode)
                    last_canvas = canvas.copy()
                else:
                    canvas = (last_canvas.copy() if last_canvas is not None
                              else frame.copy())

                # 叠加模式/FPS
                if effective_mode == mode:
                    cv2.putText(canvas, f"{mode}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(canvas, f"{effective_mode}->{mode}...{round(delay_time,1)}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"FPS:{fps:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                try:
                    display_queue.put((canvas, DISPLAY_WIN_NAME_D435), block=False)
                except queue.Full:
                    pass

        finally:
            self.camera.stop()

    def _process_frame(self, frame: np.ndarray, canvas: np.ndarray,
                       mode: str) -> None:
        """单帧核心处理"""
        self.tracker.increment_frame_count()
        if self.tracker.is_locked and self.tracker.lock_frame_count >= LOCK_MAX_HIT:
            self.tracker.force_unlock()

        was_locked = self.tracker.is_locked

        if was_locked:
            ox, oy, ow, oh = self.tracker.lock_target
            sx1, sy1, sx2, sy2, shw, shh, _ = search_rect(ox, oy, ow, oh)
            csx1 = max(0, int(sx1)); csy1 = max(0, int(sy1))
            csx2 = min(D435_WIDTH, int(sx2)); csy2 = min(D435_HEIGHT, int(sy2))
            crop = frame[csy1:csy2, csx1:csx2]
            detections = self.detector.detect_crop(crop, csx1, csy1, mode)
            if detections:
                draw_lock_rect(canvas, ox, oy, shw, shh, (0, 255, 255))
            else:
                draw_lock_rect(canvas, ox, oy, shw, shh, (0, 165, 255))
        else:
            detections = self.detector.detect_full(frame, mode)

        for d in detections:
            ux, uy = draw_square(canvas, d.box, d.names, d.r)
            cv2.circle(canvas, (d.ux, d.uy), 4, (255, 255, 255), 5)
            cv2.putText(canvas, str([d.ux, d.uy]), (d.ux + 20, d.uy + 10),
                        0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

        result = self.tracker.update(detections)

        if result.action == TrackAction.DETECT:
            self.writer.write_detection(mode, result.x, result.y)
            if not was_locked and result.is_locked:
                ox, oy, ow, oh = self.tracker.lock_target
                _, _, _, _, shw, shh, _ = search_rect(ox, oy, ow, oh)
                draw_lock_rect(canvas, ox, oy, shw, shh, (0, 255, 0))
        elif result.action == TrackAction.PREDICT:
            self.writer.write_prediction(mode, result.x, result.y)
        else:
            self.writer.write_fallback(mode)


# ======================================================================
# 12. USBCameraSource — USB camera with distortion correction
# ======================================================================

class USBCameraSource:
    """USB 摄像头封装：畸变校正 + 采集线程（封装 t44.py test() 逻辑）"""

    def __init__(self, cam_path: str = USB_CAM_PATH, calib_file: str = CALIB_FILE) -> None:
        self.cam_path = cam_path
        self.calib_file = calib_file
        self._cap: Optional[cv2.VideoCapture] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=3)
        self._mtx: Optional[np.ndarray] = None
        self._mapx: Optional[np.ndarray] = None
        self._mapy: Optional[np.ndarray] = None

    @staticmethod
    def check_available(cam_path: str = USB_CAM_PATH) -> bool:
        """检查 USB 摄像头是否可用"""
        cap = cv2.VideoCapture(cam_path)
        if cap.isOpened():
            cap.release()
            return True
        cap.release()
        return False

    def start(self) -> None:
        """打开摄像头、加载标定、启动采集线程"""
        calib = np.load(self.calib_file)
        self._mtx = calib['mtx']
        dist = calib['dist']
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
            self._mtx, dist, (USB_FRAME_W, USB_FRAME_H), 0, (USB_FRAME_W, USB_FRAME_H))
        self._mapx, self._mapy = cv2.initUndistortRectifyMap(
            self._mtx, dist, None, newcameramtx, (USB_FRAME_W, USB_FRAME_H), 5)

        self._cap = cv2.VideoCapture(self.cam_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"{GREEN}[INFO] usb摄像头已成功连接{RESET}")

        logger.info("USB摄像头已连接，畸变校正已加载")
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture, daemon=True)
        self._thread.start()
        logger.info("USB capture thread started")

    def stop(self) -> None:
        """停止采集线程并释放摄像头"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._cap is not None:
            self._cap.release()
            logger.info("USB摄像头已释放")

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """获取畸变校正后的帧（阻塞，带超时）"""
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def camera_matrix(self) -> Optional[np.ndarray]:
        """返回相机内参矩阵，供 pixel_to_camera 使用"""
        return self._mtx

    def _capture(self) -> None:
        """后台采集线程主循环（与 t44.py test() 逻辑完全一致）"""
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if ret:
                undistorted = cv2.remap(frame, self._mapx, self._mapy, cv2.INTER_LINEAR)
                if not self._frame_queue.full():
                    self._frame_queue.put(undistorted)
            else:
                logger.warning("无法读取USB摄像头画面")
                self._stop.set()
            time.sleep(0.01)


# ======================================================================
# 13. USBDetectionPipeline — USB detect + DBSCAN cluster + output
# ======================================================================

class USBDetectionPipeline:
    """
    USB 摄像头检测流水线（封装 t44.py 的 usb_detect + dbscan_cluster_and_draw 逻辑）。
    一次性聚类：累积点到触发条件后运行 DBSCAN，然后持续输出最终结果。
    """

    def __init__(self, camera_source: USBCameraSource, model_path: str,
                 output_path: str, display_queue: queue.Queue) -> None:
        self.camera = camera_source
        self.model_path = model_path
        self.output_path = output_path
        self.display_queue = display_queue
        self.model: Optional[YOLO] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._done = threading.Event()
        self._command: Optional[Command] = None

    def start(self, command: Command) -> None:
        """启动 USB 流水线（一次性聚类）"""
        if self.model is None:
            logger.info(f"加载USB模型: {self.model_path}")
            self.model = YOLO(self.model_path)
        self._command = command
        self._stop.clear()
        self._done.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止 USB 流水线"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def is_done(self) -> bool:
        """检查一次性聚类是否已完成"""
        return self._done.is_set()

    def _run(self) -> None:
        """USB 检测主循环（与 t44.py usb_detect() 逻辑完全一致）"""
        # 启动时立即写一次
        with open(self.output_path, 'w') as f:
            f.write("0 0 0 0")

        result_str = "0 0 0 0"
        last_result = None
        points = []
        last_cluster_time = time.time()
        clustered_once = False
        final_result_str = None
        mtx = self.camera.camera_matrix

        while not self._stop.is_set():
            if clustered_once:
                if final_result_str is not None:
                    with open(self.output_path, 'w') as f:
                        f.write(final_result_str)
                self._done.set()
                time.sleep(0.1)
                continue

            frame = self.camera.get_frame(timeout=0.1)
            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            results = self.model.predict(
                source=frame, device='cpu', show=False, stream=False,
                verbose=False, iou=MODEL_IOU, conf=USB_CONF)

            for result in results:
                image = result.orig_img
                names = result.names
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    r = box.xyxy[0].cpu().numpy().astype(int)
                    ux, uy = draw_square(image, box, names, r)
                    x, y = pixel_to_camera(ux, uy, mtx, w, h)
                    cv2.putText(image, f"({x:.2f}m, {y:.2f}m)",
                               (ux + 10, uy), cv2.FONT_HERSHEY_SIMPLEX,
                               1, (240, 240, 240), 3)
                    points.append((ux, uy))

            current_time = time.time()
            last_time = current_time - last_cluster_time
            keydoor = False
            if len(points) >= DBSCAN_MIN_SAMPLES:
                keydoor = True
            elif last_time >= USB_MAX_WAIT_TIME and len(points) > 0:
                keydoor = True

            if keydoor:
                centers = self._dbscan_cluster(image, points)
                camera_centers = []
                for center in centers:
                    x_cam, y_cam = pixel_to_camera(center[0], center[1], mtx, w, h)
                    camera_centers.append((x_cam, y_cam))
                camera_centers.sort(key=lambda c: c[0])

                d = self._command.usb_target
                if len(camera_centers) >= 3:
                    # 方向选通逻辑（与 t44.py 完全一致）
                    if d == 6:
                        center1, center2 = camera_centers[1], camera_centers[0]
                    elif d == 5:
                        center1, center2 = camera_centers[2], camera_centers[1]
                    elif d == 4:
                        center1, center2 = camera_centers[2], camera_centers[0]
                    elif d == 3:
                        center1, center2 = camera_centers[0], camera_centers[1]
                    elif d == 2:
                        center1, center2 = camera_centers[0], camera_centers[2]
                    elif d == 1:
                        center1, center2 = camera_centers[1], camera_centers[2]

                    c1_valid = -4 <= center1[0] <= 4 and -2.5 <= center1[1] <= 2.5
                    c2_valid = -4 <= center2[0] <= 4 and -2.5 <= center2[1] <= 2.5

                    if c1_valid and c2_valid:
                        result_str = f"{center1[0]:.1f} {center1[1]:.1f} {center2[0]:.1f} {center2[1]:.1f}"
                    elif c1_valid:
                        result_str = f"{center1[0]:.1f} {center1[1]:.1f} 0 0"
                    elif c2_valid:
                        result_str = f"0 0 {center2[0]:.1f} {center2[1]:.1f}"
                    else:
                        result_str = "0 0 0 0"

                    if result_str != last_result:
                        with open(self.output_path, 'w') as f:
                            f.write(str(result_str))
                        last_result = result_str
                    clustered_once = True
                    final_result_str = result_str
                else:
                    if last_result != "0 0 0 0":
                        with open(self.output_path, 'w') as f:
                            f.write("0 0 0 0")
                        last_result = "0 0 0 0"

            last_cluster_time = current_time

            if 'image' in locals():
                if not self.display_queue.full():
                    self.display_queue.put((image, DISPLAY_WIN_NAME_USB))

        self._done.set()

    def _dbscan_cluster(self, image: np.ndarray, points: List) -> List:
        """
        DBSCAN 聚类并绘制（与 t44.py dbscan_cluster_and_draw() 逻辑完全一致）
        """
        if len(points) == 0:
            logger.info("没有桶")
            return []
        X = np.array(points)
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(X)
        labels = db.fit_predict(X)  # 保留冗余调用，与原 t44.py 行为一致
        cluster_centers = []
        colors = [tuple(np.random.randint(0, 255, 3).tolist())
                  for _ in range(max(labels) + 2)]
        for i in range(max(labels) + 1):
            cluster_points = X[labels == i]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append(cluster_center)
            if image is not None:
                for pt in cluster_points:
                    cv2.circle(image, (int(pt[0]), int(pt[1])), 8, colors[i], -1)
                cv2.circle(image, (int(cluster_center[0]), int(cluster_center[1])),
                          15, colors[i], 3)
                cv2.putText(image, f"cluster{i+1}",
                          (int(cluster_center[0]), int(cluster_center[1]) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)
        # 创建输出目录
        os.makedirs("聚类结果图", exist_ok=True)
        cv2.imwrite(f"聚类结果图/cluster_{int(time.time())}.jpg", image)
        return cluster_centers


# ======================================================================
# 14. PipelineManager — orchestrates D435 and USB pipelines
# ======================================================================

class PipelineManager:
    """
    统一管理 D435 和 USB 两条流水线的生命周期。
    替代 t44.py 的 main_control() + read_file_content()。
    """

    def __init__(self, display_queue: queue.Queue,
                 d435_output_path: str = 'gaozhi.txt',
                 usb_output_path: str = 'gaozhi.txt') -> None:
        self.display_queue = display_queue
        self.d435_output_path = d435_output_path
        self.usb_output_path = usb_output_path

        # D435 组件
        self.d435_detector = YOLODetector()
        self.d435_camera = CameraSource()
        self.d435_tracker = LockTracker()
        self.d435_writer = OutputWriter(d435_output_path)
        self.d435_pipeline: Optional[DetectionPipeline] = None
        self.d435_thread: Optional[threading.Thread] = None
        self.d435_stop_event: Optional[threading.Event] = None

        # USB 组件（按需创建）
        self.usb_camera: Optional[USBCameraSource] = None
        self.usb_pipeline: Optional[USBDetectionPipeline] = None

        # 状态
        self._monitor = FileMonitor('data.txt')
        self._stop = threading.Event()
        self._last_kind: Optional[str] = None
        self._last_usb_target: int = 0

    def start(self) -> None:
        """启动 FileMonitor"""
        self._monitor.start()
        self._stop.clear()
        logger.info("PipelineManager started, waiting for commands...")

    def run(self) -> None:
        """
        主管理循环（独立线程，替代 main_control + read_file_content）。
        """
        while not self._stop.is_set():
            cmd = self._monitor.read_command()

            if cmd is None:
                time.sleep(0.1)
                continue

            camera_kind = cmd.kind

            # 模式切换
            if camera_kind != self._last_kind:
                logger.info(f"相机切换: {self._last_kind} -> {camera_kind}")
                self._stop_current_pipeline()

                if camera_kind == 'd435':
                    self._start_d435_pipeline(cmd)
                elif camera_kind == 'usb':
                    self._start_usb_pipeline(cmd)

                self._last_kind = camera_kind
                self._last_usb_target = cmd.usb_target

            elif camera_kind == 'usb':
                # 同为 USB 但目标方向变化 → 重启
                if cmd.usb_target != self._last_usb_target:
                    logger.info(f"USB 方向变化: d={self._last_usb_target} -> d={cmd.usb_target}")
                    self._stop_current_pipeline()
                    self._start_usb_pipeline(cmd)
                    self._last_usb_target = cmd.usb_target

            # D435 模式内切换由 DetectionPipeline 自行处理

            time.sleep(0.1)

    def stop(self) -> None:
        """停止一切"""
        self._stop.set()
        self._stop_current_pipeline()
        self._monitor.stop()

    def _stop_current_pipeline(self) -> None:
        """停止当前活跃的流水线"""
        # 停止 D435
        if self.d435_stop_event is not None:
            self.d435_stop_event.set()
        if self.d435_thread is not None and self.d435_thread.is_alive():
            logger.info("停止 D435 pipeline")
            self.d435_thread.join(timeout=3)
            self.d435_pipeline = None
            self.d435_thread = None
            self.d435_stop_event = None
        # 重置 D435 组件状态
        self.d435_tracker.reset()
        self.d435_writer.reset(cold_start=True)

        # 停止 USB
        if self.usb_pipeline is not None:
            logger.info("停止 USB pipeline")
            self.usb_pipeline.stop()
            self.usb_camera.stop()
            self.usb_pipeline = None
            self.usb_camera = None

    def _start_d435_pipeline(self, cmd: Command) -> None:
        """启动 D435 检测流水线"""
        if not CameraSource.wait_for_camera(timeout=60):
            logger.error("D435 相机不可用，无法启动 D435 流水线")
            return

        logger.info("D435 相机已连接。")
        self.d435_stop_event = threading.Event()
        self.d435_pipeline = DetectionPipeline(
            self.d435_camera, self.d435_detector,
            self.d435_tracker, self.d435_writer,
            self._monitor
        )
        self.d435_thread = threading.Thread(
            target=self.d435_pipeline.run,
            args=(self.d435_stop_event, self.display_queue),
            daemon=True
        )
        self.d435_thread.start()
        logger.info("D435 pipeline started")

    def _start_usb_pipeline(self, cmd: Command) -> None:
        """启动 USB 检测流水线（一次性聚类）"""
        if not USBCameraSource.check_available():
            for _ in range(5):
                logger.error(f"{RED}[WARN] 无法打开usb摄像头，请检查连接或权限。{RESET}")
            return

        logger.info(f"{GREEN}[INFO] usb摄像头已成功连接{RESET}")
        self.usb_camera = USBCameraSource()
        self.usb_pipeline = USBDetectionPipeline(
            self.usb_camera, MODEL_USB_PATH,
            self.usb_output_path, self.display_queue
        )
        self.usb_camera.start()
        self.usb_pipeline.start(cmd)
        logger.info(f"USB pipeline started for d={cmd.usb_target}")


# ======================================================================
# 15. main() — entry point
# ======================================================================

def main() -> None:
    """程序主入口"""
    logger.info("Yolo目标检测-程序启动 (D435 + USB)")

    clear_files(FILES_TO_CLEAR)

    display_queue: queue.Queue = queue.Queue(maxsize=DISPLAY_QUEUE_SIZE)

    manager = PipelineManager(display_queue)
    manager.start()

    manager_thread = threading.Thread(target=manager.run, daemon=True)
    manager_thread.start()

    logger.info("系统初始化完成，等待指令...")

    # 显示循环（主线程）
    window_names: set = set()

    try:
        while True:
            try:
                frame, win_name = display_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            # 按需创建窗口
            if win_name not in window_names:
                if win_name == DISPLAY_WIN_NAME_D435:
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL |
                                    cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                    cv2.resizeWindow(win_name, D435_WIDTH, D435_HEIGHT)
                else:
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(win_name, 900, 800)
                window_names.add(win_name)

            # USB 显示：四角世界坐标标注（t44.py 原有逻辑）
            if win_name == DISPLAY_WIN_NAME_USB:
                h, w = frame.shape[:2]
                corners = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
                corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
                calib = np.load(CALIB_FILE)
                mtx = calib['mtx']
                for i, (cx, cy) in enumerate(corners):
                    X, Y = pixel_to_camera(cx, cy, mtx, w, h)
                    cv2.circle(frame, (cx, cy), 8, corner_colors[i], -1)
                    if cx < w // 2 and cy < h // 2:       # 左上
                        tx, ty = cx + 10, cy + 30
                    elif cx >= w // 2 and cy < h // 2:    # 右上
                        tx, ty = cx - 220, cy + 30
                    elif cx < w // 2 and cy >= h // 2:    # 左下
                        tx, ty = cx + 10, cy - 10
                    else:                                 # 右下
                        tx, ty = cx - 220, cy - 10
                    cv2.putText(frame, f"({cx},{cy}) ({X:.2f},{Y:.2f})m",
                               (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                               corner_colors[i], 2)

            cv2.imshow(win_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("键盘中断，正在退出...")
    finally:
        manager.stop()
        manager_thread.join(timeout=3)
        cv2.destroyAllWindows()
        logger.info("程序已退出")


if __name__ == '__main__':
    main()
