"""
2m遍历方案 + 4m解算国赛
yolo + d435 LockTracker + USB DBSCAN 
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
# 1. Configuration
# ======================================================================

# ---- 全局参数 ----
Z = 4.31                                         # 相机高度（米）
eps = 50                                         # DBSCAN 聚类半径（像素）
min_samples = 15                                 # DBSCAN 最小聚类点数
max_time = 5.0                                   # 最大等待时间（秒）
files_to_clear = ['data.txt', 'gaozhi.txt']      # 启动时需要清空的文件列表

# ---- USB 相机配置 ----
files_for_pixel = 'calib_resultA.npz'            # 畸变校正文件（根据相机改变）
cam_num = '/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0'
model_usb = YOLO("/home/fu/weights/tong_blue_v0.pt")

# ---- D435 配置 ----
D435_WIDTH = 848
D435_HEIGHT = 480
MODEL_2M_PATH = "/home/fu/weights/tong_blue_v0.pt"
MODEL_1M_PATH = "/home/fu/weights/tong_blue_v0.pt"

# ---- 锁定跟踪参数 ----
LOCK_MAX_HIT = 15                                # 锁定最大命中帧数（达到后强制解锁重判）
LOCK_MAX_MISS = 7                                # 锁定最大丢失帧数（达到后解锁）
LOCK_SEARCH_RATIO = 2.5                          # 搜索框相对于目标尺寸的放大比例
LOCK_MIN_SEARCH_RADIUS = 130                     # 搜索框最小半边长
LOCK_MAX_SEARCH_RADIUS = 270                     # 搜索框最大半边长

# ---- 模型推理参数 ----
MODEL_2M_CONF = 0.5                              # 2m 模式置信度阈值
MODEL_2M_IMGSZ = 640                             # 2m 模式全图推理尺寸
MODEL_1M_CONF = 0.5                              # 1m 模式置信度阈值
MODEL_1M_IMGSZ = 640                             # 1m 模式全图推理尺寸
MODEL_IOU = 0.45                                 # NMS IoU 阈值
MODEL_IMGSZ_STEP = 32                            # 动态 imgsz 步长（对齐到 32 的倍数）
MODEL_DEVICE = 'cpu'                             # 推理设备
MAX_AREA = 223300                                # 最大目标面积，超过则过滤

# ---- 跳帧与历史清空 ----
FRAME_SKIP_ENABLED = False
FRAME_SKIP_N = 2
HISTORY_CLEAR_ENABLED = True
HISTORY_CLEAR_TIMEOUT = 10.0

# ---- 显示 ----
display_queue: queue.Queue = queue.Queue(maxsize=3)
start_time: Optional[float] = None

# ---- 全局模式变量（由 read_file_content 写入，main_control/d435_detect 读取）----
c: int = 0                                       # 0=关闭, 2=D435, 3=USB
b: int = 0                                       # D435 子模式: 0=无, 1=1m, 2=2m
d: int = 0                                       # USB 目标选择: 1~6

# ---- 全局线程/资源引用（由 main_control 管理）----
t_test: Optional[threading.Thread] = None
t_detect: Optional[threading.Thread] = None
t_d435: Optional[threading.Thread] = None
camera: Optional[cv2.VideoCapture] = None


# ======================================================================
# 2. Logging Setup
# ======================================================================

# ANSI 颜色码
RESET  = "\033[0m"
RED    = "\033[31m"
ORANGE = "\033[38;5;214m"
YELLOW = "\033[33m"


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
# 3. Utility functions
# ======================================================================

def clamp(v: float, lo: float, hi: float) -> float:
    """将浮点数 v 限制在 [lo, hi] 区间内"""
    return max(lo, min(hi, v))


def search_rect(x: float, y: float, w: float, h: float
                ) -> Tuple[int, int, int, int, int, int, int]:
    """
    根据目标中心 (x,y) 和宽高 (w,h) 计算自适应搜索窗口
    返回: (sx1, sy1, sx2, sy2, shw, shh, area)
    """
    shw = clamp(w * LOCK_SEARCH_RATIO / 2, LOCK_MIN_SEARCH_RADIUS, LOCK_MAX_SEARCH_RADIUS)
    shh = clamp(h * LOCK_SEARCH_RATIO / 2, LOCK_MIN_SEARCH_RADIUS, LOCK_MAX_SEARCH_RADIUS)
    sx1, sy1 = int(x - shw), int(y - shh)
    sx2, sy2 = int(x + shw), int(y + shh)
    return sx1, sy1, sx2, sy2, int(shw), int(shh), (sx2 - sx1) * (sy2 - sy1)


def pick_nearest(items: List) -> Optional[Any]:
    """从列表中选取第一个元素（距离）最小的项，若列表为空返回 None"""
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


def pixel_to_camera(ux: int, uy: int, mtx: np.ndarray,
                    w: int, h: int) -> Tuple[float, float]:
    """像素坐标转相机坐标系（米）"""
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = w / 2
    cy = h / 2
    X = (ux - cx) * Z / fx
    Y = (uy - cy) * Z / fy
    return X, Y


def clear_files(files: List[str]) -> None:
    """清空指定的文件列表（创建空文件）"""
    for file_name in files:
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, 'w') as f:
            pass
    yellow_log("通讯txt文件已建立并清空")


def check_camera(cam_num: str) -> None:
    """检查 USB 和 D435 摄像头连接状态"""
    # 检查 USB 摄像头
    cap = cv2.VideoCapture(cam_num)
    if not cap.isOpened():
        for _ in range(5):
            logger.warning("无法打开usb摄像头，请检查连接或权限。")
        cap.release()
    else:
        yellow_log("usb摄像头已成功连接")
        cap.release()

    # 检查 D435 相机
    ctx = rs.context()
    if len(ctx.devices) == 0:
        for _ in range(5):
            logger.warning("未检测到D435相机，请检查连接！")
    else:
        yellow_log("D435相机已连接。")


# ======================================================================
# 4. Data types
# ======================================================================

class TrackAction(Enum):
    """跟踪动作枚举：DETECT 真实检测，PREDICT 预测，LOST 丢失"""
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
    r: list
    w: int
    h: int
    area: int
    dis: float
    box: object
    names: dict


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

    def get_latest(self, seen_seq: int = 0
                   ) -> Tuple[bool, Optional[np.ndarray], int]:
        """获取最新帧，如果序号未更新则返回 (False, None, 当前序号)"""
        with self._lock:
            if self._seq <= seen_seq or self._frame is None:
                return False, None, self._seq
            return True, self._frame.copy(), self._seq


# ======================================================================
# 6. OutputWriter — writes gaozhi.txt, manages history
# ======================================================================

class OutputWriter:
    """将检测结果写入 gaozhi.txt，管理历史坐标、超时清空和首次日志"""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._lock = threading.Lock()
        self.last_value: Optional[str] = None
        self.last_detection_time: float = 0.0
        self.b_zero: bool = True
        self.a_zero: bool = True

    def write_detection(self, mode: str, ux: int, uy: int) -> None:
        """写入真实检测结果 [1, ux, uy]，更新 last_value 和时间戳"""
        with self._lock:
            self._log_2m_entry(mode)
            self._log_1m_entry(mode)
            self._write(f"1 {ux} {uy}")
            self.last_value = f"{ux} {uy}"
            self.last_detection_time = time.time()

    def write_prediction(self, mode: str, ux: int, uy: int) -> None:
        """写入预测结果 [2, ux, uy]，更新 last_value 但不更新时间戳"""
        with self._lock:
            self._log_2m_entry(mode)
            self._log_1m_entry(mode)
            self._write(f"2 {ux} {uy}")
            self.last_value = f"{ux} {uy}"

    def write_fallback(self, mode: str) -> None:
        """跟踪器报告 LOST 时的回退写入策略"""
        with self._lock:
            self._check_timeout()
            self._log_2m_entry(mode)
            self._log_1m_entry(mode)

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
            logger.info(f"超过{HISTORY_CLEAR_TIMEOUT}s未检测到目标，清空历史坐标")

    def _log_2m_entry(self, mode: str) -> None:
        """2m 模式下首次写入非0值时打印提示日志"""
        if mode == '2m' and self.b_zero:
            for _ in range(5):
                yellow_log("2m对桶开始※※※")
            self.b_zero = False

    def _log_1m_entry(self, mode: str) -> None:
        """1m 模式下首次写入非0值时打印提示日志"""
        if mode == '1m' and self.a_zero:
            for _ in range(5):
                yellow_log("1m对桶开始※※※")
            self.a_zero = False

    def _write(self, content: str) -> None:
        """实际写入文件"""
        with open(self.file_path, 'w') as f:
            f.write(content)


# ======================================================================
# 7. LockTracker — lock-based tracking state machine
# ======================================================================
#
# States: UNLOCKED / LOCKED (is_locked property).
#
# 8 transition paths:
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
    """基于锁定的单目标跟踪器，线程安全"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.lock_target: Optional[Tuple[int, int, int, int]] = None
        self.lock_miss_count: int = 0
        self.lock_frame_count: int = 0

    @property
    def is_locked(self) -> bool:
        """是否处于锁定状态"""
        return self.lock_target is not None

    def reset(self) -> None:
        """重置跟踪器到未锁定状态"""
        with self._lock:
            self.lock_target = None
            self.lock_miss_count = 0
            self.lock_frame_count = 0

    def increment_frame_count(self) -> None:
        """锁定状态下帧计数+1（由外部每帧调用）"""
        with self._lock:
            if self.is_locked:
                self.lock_frame_count += 1

    def force_unlock(self) -> None:
        """Path 8: 锁定帧数达到上限，强制解锁以重新进行全图评估"""
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
        """锁定状态下收到检测结果的处理"""
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
            # Path 3: 丢失超过上限 → 解锁并立即在全部检测中选最近目标重新锁定
            yellow_log(f"丢帧满{LOCK_MAX_MISS}帧，强制重判")
            self.lock_target = None
            self.lock_miss_count = 0
            self.lock_frame_count = 0
            best = pick_nearest([(d.dis, d.cls, d.ux, d.uy, d.r) for d in detections])
            _, _, ux, uy, r = best
            w, h = abs(r[2] - r[0]), abs(r[3] - r[1])
            self.lock_target = (ux, uy, w, h)
            yellow_log(f"解锁-最近桶：({ux},{uy}) 目标面积={w*h}")
            return TrackResult(TrackAction.DETECT, ux, uy, True)
        else:
            # Path 2: 丢失但未超限 → 沿用旧位置进行预测
            logger.warning(f"锁定-历史：({ox},{oy}) miss={self.lock_miss_count}")
            return TrackResult(TrackAction.PREDICT, ox, oy, True)

    # ---- LOCKED + no detections (Paths 4, 5) ----

    def _locked_no_dets(self) -> TrackResult:
        """锁定状态下无任何检测时的处理"""
        ox, oy, ow, oh = self.lock_target

        self.lock_miss_count += 1
        if self.lock_miss_count > LOCK_MAX_MISS:
            # Path 5: 连续丢失超限 → 解锁，返回 LOST
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
        """未锁定状态下收到检测：选择最近目标并锁定"""
        best = pick_nearest([(d.dis, d.cls, d.ux, d.uy, d.r) for d in detections])
        _, _, ux, uy, r = best
        w, h = abs(r[2] - r[0]), abs(r[3] - r[1])
        self.lock_target = (ux, uy, w, h)
        self.lock_miss_count = 0
        self.lock_frame_count = 0
        yellow_log(f"锁定-最近桶：({ux},{uy}) 目标面积={w*h}")
        return TrackResult(TrackAction.DETECT, ux, uy, True)

    # ---- UNLOCKED + no detections (Path 7) ----

    def _unlocked_no_dets(self) -> TrackResult:
        """未锁定且无检测：返回 LOST"""
        return TrackResult(TrackAction.LOST, 0, 0, False)


# ======================================================================
# 8. YOLODetector — dual-model YOLO inference
# ======================================================================

class YOLODetector:
    """
    双模型 YOLO 检测器（按模式区分：2m/1m，不分锁定/未锁定）
    detect_full / detect_crop 都根据 mode 选择对应模型。
    """

    def __init__(self) -> None:
        self.model_2m = YOLO(MODEL_2M_PATH)
        self.model_1m = YOLO(MODEL_1M_PATH)
        self._cx = D435_WIDTH // 2
        self._cy = D435_HEIGHT // 2

    def _select_model(self, mode: str):
        """根据模式返回对应的模型和参数 (model, conf, full_imgsz)"""
        if mode == '1m':
            return self.model_1m, MODEL_1M_CONF, MODEL_1M_IMGSZ
        return self.model_2m, MODEL_2M_CONF, MODEL_2M_IMGSZ

    def detect_full(self, frame: np.ndarray, mode: str) -> List[Detection]:
        """全图推理"""
        model, conf, imgsz = self._select_model(mode)
        t0 = time.time()
        results = model.predict(
            source=frame, device=MODEL_DEVICE, show=False,
            stream=False, verbose=False, iou=MODEL_IOU, conf=conf, imgsz=imgsz)
        t1 = time.time()
        infer_time = int((t1 - t0) * 1000)
        yellow_log(f"全图推理：{infer_time}ms")
        return self._parse(results, 0, 0)

    def detect_crop(self, crop: np.ndarray, ox: int, oy: int,
                    mode: str) -> List[Detection]:
        """裁剪区域推理，ox/oy 为裁剪区域左上角在全图中的偏移"""
        h, w = crop.shape[:2]
        imgsz = ((max(w, h) + MODEL_IMGSZ_STEP - 1) // MODEL_IMGSZ_STEP) * MODEL_IMGSZ_STEP
        model, conf, _ = self._select_model(mode)
        t0 = time.time()
        results = model.predict(
            source=crop, device=MODEL_DEVICE, show=False,
            stream=False, verbose=False, iou=MODEL_IOU,
            conf=conf, imgsz=imgsz)
        t1 = time.time()
        infer_time = int((t1 - t0) * 1000)
        logger.info(f"裁减推理：{infer_time}ms")
        return self._parse(results, ox, oy)

    def _parse(self, results: Any, ox: int, oy: int) -> List[Detection]:
        """将 YOLO 推理结果解析为 Detection 列表"""
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
                    ux=ux, uy=uy, cls=int(box.cls[0]),
                    conf=float(box.conf[0]), r=r.tolist(), w=w, h=h,
                    area=area, dis=dis, box=box, names=names))
        return detections


# ======================================================================
# 9. CameraSource — D435 camera with internal capture thread
# ======================================================================

class CameraSource:
    """D435 相机封装，内部使用独立线程持续采集帧放入 FrameBuffer"""

    def __init__(self) -> None:
        self._pipeline: Optional[rs.pipeline] = None
        self._buf = FrameBuffer()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """启动 RealSense 管道和采集线程"""
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, D435_WIDTH, D435_HEIGHT,
                          rs.format.bgr8, 30)
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

    def get_frame(self, seen_seq: int
                  ) -> Tuple[bool, Optional[np.ndarray], int]:
        """从缓冲区获取最新帧"""
        return self._buf.get_latest(seen_seq)

    def _capture(self) -> None:
        """后台采集线程主循环"""
        while not self._stop.is_set():
            frames = self._pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if color:
                self._buf.put(np.asanyarray(color.get_data()))


# ======================================================================
# 10. USB functions
# ======================================================================

def usb_test(camera: cv2.VideoCapture, frame_queue: queue.Queue,
             start_event: threading.Event,
             stop_event: threading.Event) -> None:
    """USB 畸变校正 + 帧捕获线程"""
    start_event.wait()
    logger.info("test线程已启动")

    calib = np.load(files_for_pixel)
    mtx = calib['mtx']
    dist = calib['dist']
    h, w = 480, 640
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx,
                                              (w, h), 5)

    while not stop_event.is_set():
        ret, frame = camera.read()
        if ret:
            if c == 3:
                undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
                if not frame_queue.full():
                    frame_queue.put(undistorted)
        else:
            logger.error("无法读取摄像头画面，请检查摄像头连接或权限设置。")
            stop_event.set()
        time.sleep(0.01)


def usb_detect(model: YOLO, frame_queue: queue.Queue,
               shuchu_file_path: str, stop_event: threading.Event,
               start_event: threading.Event) -> None:
    """USB 相机 YOLO 检测 + DBSCAN 聚类 + 多目标输出"""
    global start_time

    start_event.wait()
    logger.info("usb_detect线程已启动")

    # 启动时立即写一次
    with open(shuchu_file_path, 'w') as file:
        file.write("0 0 0 0")
    result_str = "0 0 0 0"
    last_result: Optional[str] = None

    calib = np.load(files_for_pixel)
    mtx = calib['mtx']

    points: List[Tuple[int, int]] = []
    last_cluster_time = time.time()
    clustered_once = False
    final_result_str: Optional[str] = None

    while not stop_event.is_set():
        if clustered_once:
            if final_result_str is not None:
                with open(shuchu_file_path, 'w') as file:
                    file.write(final_result_str)
            time.sleep(0.1)
            continue

        if not frame_queue.empty():
            frame = frame_queue.get()
            h, w = frame.shape[:2]
            results = model.predict(
                source=frame, device='cpu', show=False,
                stream=False, verbose=False, iou=0.45, conf=0.6)

            for result in results:
                image = result.orig_img
                names = result.names
                boxes = result.boxes
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
            if len(points) >= min_samples:
                keydoor = True
            elif last_time >= max_time and len(points) > 0:
                keydoor = True

            if keydoor:
                centers = dbscan_cluster_and_draw(image, points, eps, min_samples)
                camera_centers = []
                for center in centers:
                    x_cam, y_cam = pixel_to_camera(center[0], center[1], mtx, w, h)
                    camera_centers.append((x_cam, y_cam))

                # 按 x 轴从小到大排序
                camera_centers.sort(key=lambda ctr: ctr[0])

                if len(camera_centers) >= 3:
                    if d == 6:
                        center1 = camera_centers[1]
                        center2 = camera_centers[0]
                    elif d == 5:
                        center1 = camera_centers[2]
                        center2 = camera_centers[1]
                    elif d == 4:
                        center1 = camera_centers[2]
                        center2 = camera_centers[0]
                    elif d == 3:
                        center1 = camera_centers[0]
                        center2 = camera_centers[1]
                    elif d == 2:
                        center1 = camera_centers[0]
                        center2 = camera_centers[2]
                    elif d == 1:
                        center1 = camera_centers[1]
                        center2 = camera_centers[2]

                    if (-4 <= center1[0] <= 4 and 
                        -4 <= center2[0] <= 4 and
                        -2.5 <= center1[1] <= 2.5 and 
                        -2.5 <= center2[1] <= 2.5):
                        result_str = (f"{center1[0]:.1f} {center1[1]:.1f} "
                                      f"{center2[0]:.1f} {center2[1]:.1f}")
                    elif -4 <= center1[0] <= 4 and -2.5 <= center1[1] <= 2.5:
                        result_str = f"{center1[0]:.1f} {center1[1]:.1f} 0 0"
                    elif -4 <= center2[0] <= 4 and -2.5 <= center2[1] <= 2.5:
                        result_str = f"0 0 {center2[0]:.1f} {center2[1]:.1f}"
                    else:
                        result_str = "0 0 0 0"

                    if result_str != last_result:
                        with open(shuchu_file_path, 'w') as file:
                            file.write(str(result_str))
                        last_result = result_str

                    clustered_once = True
                    final_result_str = result_str
                else:
                    if last_result != "0 0 0 0":
                        with open(shuchu_file_path, 'w') as file:
                            file.write("0 0 0 0")
                        last_result = "0 0 0 0"

            last_cluster_time = current_time

            if 'image' in locals():
                if not display_queue.full():
                    display_queue.put((image, "usb"))

            with open('gaozhi.txt', 'r') as file3:
                content = file3.read().strip()
                logger.info(f"gaozhi.txt 内容: {content}")
        else:
            time.sleep(0.01)


def dbscan_cluster_and_draw(image: np.ndarray, points: List,
                            eps: float, min_samples: int) -> List:
    """DBSCAN 聚类并绘制可视化，返回聚类中心列表"""
    if len(points) == 0:
        logger.info("没有桶")
        return []

    X = np.array(points)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.fit_predict(X)

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

    cv2.imwrite(f"聚类结果图/cluster_{int(time.time())}.jpg", image)
    return cluster_centers


# ======================================================================
# 11. File monitor & mode control
# ======================================================================

def read_file_content(shuru_file_path: str, start_event: threading.Event,
                      stop_event: threading.Event) -> None:
    """轮询 data.txt 文件内容变化，通过全局变量 c/b/d 通知模式切换"""
    global c, b, start_time, d

    last_result: Optional[str] = None

    with open(shuru_file_path, 'r') as file2:
        logger.info(f"data.txt 内容: {file2.read().strip()}")
    time.sleep(0.25)

    while not stop_event.is_set():
        with open(shuru_file_path, 'r') as file2:
            content = file2.read().strip()
            if content != last_result:
                if content == 'ml':
                    for _ in range(10):
                        yellow_log("已经到达4m，即将打开usb相机ml模式")
                    start_event.set()
                    c = 3
                    d = 6
                    start_time = time.time()
                elif content == 'rm':
                    for _ in range(10):
                        yellow_log("已经到达4m，即将打开usb相机rm模式")
                    start_event.set()
                    c = 3
                    d = 5
                    start_time = time.time()
                elif content == 'rl':
                    for _ in range(10):
                        yellow_log("已经到达4m，即将打开usb相机rl模式")
                    start_event.set()
                    c = 3
                    d = 4
                    start_time = time.time()
                elif content == 'lm':
                    for _ in range(10):
                        yellow_log("已经到达4m，即将打开usb相机lm模式")
                    start_event.set()
                    c = 3
                    d = 3
                    start_time = time.time()
                elif content == 'lr':
                    for _ in range(10):
                        yellow_log("已经到达4m，即将打开usb相机lr模式")
                    start_event.set()
                    c = 3
                    d = 2
                    start_time = time.time()
                elif content == 'mr':
                    for _ in range(10):
                        yellow_log("已经到达4m，即将打开usb相机mr模式")
                    start_event.set()
                    c = 3
                    d = 1
                    start_time = time.time()
                elif content == '2m':
                    for _ in range(10):
                        yellow_log("即将切换D435i相机，开始锁定目标桶")
                    start_event.set()
                    c = 2
                    b = 2
                elif content == '1m':
                    for _ in range(10):
                        yellow_log("目标桶已锁定，即将投放")
                    start_event.set()
                    c = 2
                    b = 1
                elif content == '0':
                    for _ in range(5):
                        yellow_log("检测到0模式，关闭摄像头")
                    c = 0
                    b = 0
                else:
                    logger.info("未检测到内容，等待...")
                last_result = content


def main_control() -> None:
    """相机切换及权重切换 — 根据全局 c 启停 USB/D435 线程"""
    global c, cam_num, t_test, t_detect, t_d435, camera

    last_c: Optional[int] = None
    running = False
    camera = None  # type: ignore[assignment]
    stop_event: Optional[threading.Event] = None
    frame_queue: Optional[queue.Queue] = None

    check_camera(cam_num)

    while True:
        if c != last_c:
            logger.info(f"main_control 检测到 c 变化: {last_c} -> {c}")

            # 关闭旧线程和相机
            if running:
                stop_event.set()
                if t_test: t_test.join()
                if t_detect: t_detect.join()
                if camera: camera.release()
                if t_d435: t_d435.join()
                cv2.destroyAllWindows()
                time.sleep(0.2)
                running = False

            stop_event = threading.Event()

            # 启动新线程和相机
            if c == 3:
                camera = cv2.VideoCapture(cam_num)
                frame_queue = queue.Queue(maxsize=3)
                t_test = threading.Thread(
                    target=usb_test,
                    args=(camera, frame_queue, start_event, stop_event))
                t_detect = threading.Thread(
                    target=usb_detect,
                    args=(model_usb, frame_queue, shuchu_file_path,
                          stop_event, start_event))
                t_test.start()
                t_detect.start()
                t_d435 = None
                running = True
            elif c == 2:
                t_d435 = threading.Thread(
                    target=d435_detect,
                    args=(shuchu_file_path, stop_event, start_event))
                t_d435.start()
                t_test = None
                t_detect = None
                camera = None
                running = True

            if c == 0:
                with open('gaozhi.txt', 'w') as f:
                    f.write('0')

            last_c = c
        time.sleep(0.1)


# ======================================================================
# 12. D435 detection pipeline
# ======================================================================

def d435_detect(shuchu_file_path: str, stop_event: threading.Event,
                start_event: threading.Event) -> None:
    """D435 相机 LockTracker 检测管道线程"""
    start_event.wait()
    logger.info("d435_detect线程已启动 (LockTracker pipeline)")

    with open(shuchu_file_path, 'w') as f:
        f.write("0")

    detector = YOLODetector()
    camera_src = CameraSource()
    tracker = LockTracker()
    writer = OutputWriter(shuchu_file_path)

    camera_src.start()

    last_b: Optional[int] = None
    last_mode: Optional[str] = None
    delay_until: float = 0.0
    last_seq: int = 0
    frame_count: int = 0
    last_canvas: Optional[np.ndarray] = None

    try:
        while not stop_event.is_set():
            current_b = b
            if current_b not in (1, 2):
                if last_b in (1, 2):
                    with open(shuchu_file_path, 'w') as f:
                        f.write('0')
                last_b = current_b
                time.sleep(0.01)
                continue

            mode = '2m' if current_b == 2 else '1m'

            if mode != last_mode:
                logger.info(f"模式切换: {last_mode} -> {mode}")
                cold = last_mode is None
                writer.reset(cold_start=cold)
                tracker.reset()
                if last_mode == '2m' and mode == '1m':
                    delay_until = time.time() + 3.0
                    logger.info("2m->1m: 保持2m模型3s")
                else:
                    delay_until = 0.0
                last_mode = mode
                frame_count = 0

            last_b = current_b

            # 计算当前帧实际使用的模型模式（含 2m→1m 延时）
            if delay_until > time.time() and mode == '1m':
                effective_mode = '2m'
            else:
                effective_mode = mode

            got, frame, last_seq = camera_src.get_frame(last_seq)
            if not got:
                time.sleep(0.001)
                continue

            frame_count += 1
            do_inference = (not FRAME_SKIP_ENABLED
                            or frame_count % FRAME_SKIP_N == 0)

            if do_inference:
                canvas = frame.copy()

                # 预推理：检查是否需要强制解锁 (Path 8)
                tracker.increment_frame_count()
                if (tracker.is_locked
                        and tracker.lock_frame_count >= LOCK_MAX_HIT):
                    tracker.force_unlock()

                was_locked = tracker.is_locked

                # YOLO 推理
                if was_locked:
                    ox, oy, ow, oh = tracker.lock_target
                    sx1, sy1, sx2, sy2, shw, shh, _ = search_rect(
                        ox, oy, ow, oh)
                    csx1 = max(0, int(sx1))
                    csy1 = max(0, int(sy1))
                    csx2 = min(D435_WIDTH, int(sx2))
                    csy2 = min(D435_HEIGHT, int(sy2))
                    crop = frame[csy1:csy2, csx1:csx2]
                    detections = detector.detect_crop(
                        crop, csx1, csy1, effective_mode)
                    if detections:
                        draw_lock_rect(canvas, ox, oy, shw, shh,
                                       (0, 255, 255))
                    else:
                        draw_lock_rect(canvas, ox, oy, shw, shh,
                                       (0, 165, 255))
                else:
                    detections = detector.detect_full(frame, effective_mode)

                # 绘制所有检测框
                for d in detections:
                    draw_square(canvas, d.box, d.names, d.r)
                    cv2.circle(canvas, (d.ux, d.uy), 4,
                               (255, 255, 255), 5)
                    cv2.putText(canvas, f"[{d.ux},{d.uy}]",
                                (d.ux + 20, d.uy + 10),
                                0, 1, (225, 255, 255),
                                thickness=2, lineType=cv2.LINE_AA)

                # 运行跟踪状态机
                result = tracker.update(detections)

                # 根据跟踪结果写入输出文件
                if result.action == TrackAction.DETECT:
                    writer.write_detection(mode, result.x, result.y)
                    if not was_locked and result.is_locked:
                        ox, oy, ow, oh = tracker.lock_target
                        _, _, _, _, shw, shh, _ = search_rect(ox, oy, ow, oh)
                        draw_lock_rect(canvas, ox, oy, shw, shh, (0, 255, 0))
                elif result.action == TrackAction.PREDICT:
                    writer.write_prediction(mode, result.x, result.y)
                else:  # LOST
                    writer.write_fallback(mode)

                last_canvas = canvas.copy()
            else:
                canvas = (last_canvas.copy() if last_canvas is not None
                          else frame.copy())

            cv2.putText(canvas, f"D435:{effective_mode}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                        cv2.LINE_AA)

            if not display_queue.full():
                display_queue.put((canvas, "d435"))

            if os.path.exists('gaozhi.txt'):
                with open('gaozhi.txt', 'r') as f3:
                    logger.info(f"gaozhi.txt 内容: {f3.read().strip()}")
    finally:
        camera_src.stop()


# ======================================================================
# 13. main()
# ======================================================================

if __name__ == '__main__':
    try:
        t_test = None
        t_detect = None
        t_d435 = None
        camera = None
        c = 0
        b = 0

        clear_files(files_to_clear)

        shuru_file_path = 'data.txt'
        shuchu_file_path = 'gaozhi.txt'

        start_event = threading.Event()
        stop_event = threading.Event()

        t1 = threading.Thread(
            target=read_file_content,
            args=(shuru_file_path, start_event, stop_event))
        t1.start()

        main_thread = threading.Thread(target=main_control)
        main_thread.start()

        # 主线程负责显示窗口
        window_created = False
        frame_count = 0
        start_time = time.time()

        while True:
            if not display_queue.empty():
                frame, window_name = display_queue.get()

                if not window_created:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 900, 800)
                    window_created = True

                if c == 3:
                    # USB 画面四角坐标叠加
                    h, w = frame.shape[:2]
                    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
                    colors = [(0, 0, 255), (0, 255, 0),
                              (255, 0, 0), (0, 255, 255)]
                    calib = np.load(files_for_pixel)
                    mtx = calib['mtx']

                    for i, (x, y) in enumerate(corners):
                        X, Y = pixel_to_camera(x, y, mtx, w, h)
                        cv2.circle(frame, (x, y), 8, colors[i], -1)

                        if x < w // 2 and y < h // 2:          # 左上
                            tx, ty = x + 10, y + 30
                        elif x >= w // 2 and y < h // 2:       # 右上
                            tx, ty = x - 220, y + 30
                        elif x < w // 2 and y >= h // 2:       # 左下
                            tx, ty = x + 10, y - 10
                        else:                                  # 右下
                            tx, ty = x - 220, y - 10

                        cv2.putText(
                            frame,
                            f"({x},{y}) ({X:.2f},{Y:.2f})m",
                            (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2
                        )

                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)

        cv2.destroyAllWindows()
        main_thread.join()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        
