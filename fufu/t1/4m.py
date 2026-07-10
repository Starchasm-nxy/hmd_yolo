"""
2m遍历 + 4m解算
yolo + d435 LockTracker + USB DBSCAN 
"""

import os
import cv2
import time
import queue
import threading
import logging
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
model_usb = YOLO("/home/fu/yolo/weights/tong_blue_v0.pt")

# ---- D435 配置 ----
D435_WIDTH = 848                                  # 彩色流宽度
D435_HEIGHT = 480                                 # 彩色流高度
# MODEL_FULL_PATH = "/home/fu/weights/tong_blue_v0.pt"   # 全图推理权重
# MODEL_CROP_PATH = "/home/fu/weights/tong_blue_v0.pt"   # 裁剪推理权重
MODEL_PATH = "/home/fu/yolo/weights/tong_blue_v0.pt"


# ---- 锁定跟踪参数 ----
LOCK_MAX_HIT = 15                                # 锁定最大命中帧数（达到后强制解锁重判）
LOCK_MAX_MISS = 7                                # 锁定最大丢失帧数（达到后解锁）
LOCK_SEARCH_RATIO = 2.5                          # 搜索框相对于目标尺寸的放大比例
LOCK_MIN_SEARCH_RADIUS = 130                     # 搜索框最小半边长
LOCK_MAX_SEARCH_RADIUS = 270                     # 搜索框最大半边长

# ---- 模型推理参数 ----
MODEL_2M_CONF = 0.5                              # 2m 模式置信度阈值
MODEL_2M_IMGSZ = 640                             # 2m 模式全图推理尺寸
MODEL_2M_IOU = 0.45                              # NMS IoU 阈值

MODEL_1M_CONF = 0.5                              # 1m 模式置信度阈值
MODEL_1M_IMGSZ = 640                             # 1m 模式全图推理尺寸
MODEL_1M_IOU = 0.45                              # NMS IoU 阈值

MODEL_IMGSZ_STEP = 32                            # 动态 imgsz 步长（对齐到 32 的倍数）
MODEL_DEVICE = 'cpu'                             # 推理设备
MAX_AREA = 112233                                # 最大目标面积，超过则过滤

# ---- D435历史清空 ----
HISTORY_CLEAR_ENABLED = True
HISTORY_CLEAR_TIMEOUT = 10.0

# ---- 显示 ----
display_queue: queue.Queue = queue.Queue(maxsize=3)  # 帧显示队列

# ---- 超椭圆绘制参数 ----
SUPERELLIPSE_P_LOCK = 2.5                      # draw_lock_rect / LockTracker 超椭圆指数
SUPERELLIPSE_NUM_POINTS = 100                  # 超椭圆采样点数

# ---- USB 推理参数 ----
USB_CONF = 0.6                                  # USB YOLO 置信度阈值
USB_IMGSZ = 640                                 # USB YOLO 置信推理尺寸
USB_IOU = 0.45                                  # USB YOLO IoU 阈值

# ---- USB 坐标有效性阈值（米） ----
COORD_X_MIN = -4.0
COORD_X_MAX = 4.0
COORD_Y_MIN = -2.5
COORD_Y_MAX = 2.5

# ---- USB 目标选择查表 ----
USB_TARGET_MAP = {6: (1, 0), 5: (2, 1), 4: (2, 0),
                  3: (0, 1), 2: (0, 2), 1: (1, 2)}

# ---- 窗口尺寸 ----
DISPLAY_WIN_W = 1000                               # 显示窗口宽度
DISPLAY_WIN_H = 800                                # 显示窗口高度

# ---- 时序参数 ----
D435_FRAME_TIMEOUT_MS = 5000                     # D435 wait_for_frames 超时（毫秒）
CAPTURE_RETRY_SLEEP = 0.1                        # 采集异常重试间隔（秒）
CLUSTERED_IDLE_SLEEP = 0.1                       # USB 聚类完成后空闲间隔（秒）
ORCHESTRATOR_POLL_SLEEP = 0.1                    # PipelineOrchestrator 轮询间隔（秒）
FRAME_NOT_READY_SLEEP = 0.001                    # 帧未就绪时的等待间隔（秒）
POLL_SLEEP_FAST = 0.01                           # 快速轮询间隔（秒）
FILE_POLL_SLEEP = 0.25                           # data.txt 文件轮询间隔（秒）
MODE_SWITCH_DELAY = 0.2                          # 模式切换后延时（秒）
THREAD_JOIN_TIMEOUT = 1.0                        # 线程 join 超时（秒）
DISPLAY_WAITKEY_MS = 1                           # cv2.waitKey 延迟（毫秒）

CLUSTER_TIMEOUT = 6.0                             # USB 聚类超时（秒）

# ---- USB 相机标定参数 ----
UNDISTORT_ALPHA = 0.0                            # cv2.getOptimalNewCameraMatrix alpha
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
    参数:
        image: 画布图像
        box: YOLO输出的box对象，包含类别和置信度
        names: 类别名称字典
        r: 框坐标 [x1, y1, x2, y2]
    返回: (ux, uy) 框的中心点坐标
    """
    ux = int((r[0] + r[2]) / 2)          # 框中心x
    uy = int((r[1] + r[3]) / 2)          # 框中心y
    cls = int(box.cls[0])                # 类别索引
    conf = box.conf[0]                   # 置信度
    label = f"{names[cls]} {conf:.2f}"   # 标签文字
    cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (221, 185, 193), 2)  # 画框
    cv2.putText(image, label, (r[0], r[1] - 10),                          # 画文字
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (176, 196, 222), 2)
    cv2.circle(image, (ux, uy), 5, (240, 240, 240), -1)                  # 画中心点
    return ux, uy


def draw_lock_rect(canvas: np.ndarray, x: int, y: int, shw: int, shh: int,
                   color: Tuple[int, int, int],
                   p: float = SUPERELLIPSE_P_LOCK,
                   num_points: int = SUPERELLIPSE_NUM_POINTS) -> None:
    """
    绘制超椭圆搜索框（p=4 时近似圆角矩形），中心 (x, y)，半轴 shw, shh。
    """
    theta = np.linspace(0, 2 * np.pi, num_points)      # 角度采样点
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # 计算超椭圆半径
    denom = np.abs(cos_t / shw) ** p + np.abs(sin_t / shh) ** p
    r = denom ** (-1.0 / p)                             # 每个角度上的半径

    # 得到超椭圆上的点坐标（浮点）
    xs = r * cos_t + x
    ys = r * sin_t + y

    # 转换为 int32 的点数组 (N, 1, 2)
    pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)

    # 绘制超椭圆边框
    cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

    # 绘制中心点（可选）
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
# 4.5 ModeState — thread-safe shared mode state
# ======================================================================

class ModeState:
    """线程安全共享模式状态，同时管理摄像头选择、子模式、启动时间"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._c: int = 0
        self._b: int = 0
        self._d: int = 0
        self.start_time: Optional[float] = None

    def get(self) -> Tuple[int, int, int]:
        """读取完整状态 (c, b, d)"""
        with self._lock:
            return self._c, self._b, self._d

    def get_c(self) -> int:
        """读取摄像头选择: 0=关闭, 2=D435, 3=USB"""
        with self._lock:
            return self._c

    def get_b(self) -> int:
        """读取 D435 子模式: 0=无, 1=1m, 2=2m"""
        with self._lock:
            return self._b

    def get_d(self) -> int:
        """读取 USB 目标选择编号 1~6"""
        with self._lock:
            return self._d

    def set(self, c: int, b: int = 0, d: int = 0) -> None:
        """写入模式状态"""
        with self._lock:
            self._c = c
            self._b = b
            self._d = d

    def set_from_command(self, content: str) -> bool:
        """解析 data.txt 命令，返回 True 表示识别成功"""
        with self._lock:
            if content == 'ml':
                for _ in range(10):
                    yellow_log("已经到达4m，即将打开usb相机ml模式")
                self._c = 3
                self._d = 6
                self.start_time = time.time()
            elif content == 'rm':
                for _ in range(10):
                    yellow_log("已经到达4m，即将打开usb相机rm模式")
                self._c = 3
                self._d = 5
                self.start_time = time.time()
            elif content == 'rl':
                for _ in range(10):
                    yellow_log("已经到达4m，即将打开usb相机rl模式")
                self._c = 3
                self._d = 4
                self.start_time = time.time()
            elif content == 'lm':
                for _ in range(10):
                    yellow_log("已经到达4m，即将打开usb相机lm模式")
                self._c = 3
                self._d = 3
                self.start_time = time.time()
            elif content == 'lr':
                for _ in range(10):
                    yellow_log("已经到达4m，即将打开usb相机lr模式")
                self._c = 3
                self._d = 2
                self.start_time = time.time()
            elif content == 'mr':
                for _ in range(10):
                    yellow_log("已经到达4m，即将打开usb相机mr模式")
                self._c = 3
                self._d = 1
                self.start_time = time.time()
            elif content == '2m':
                for _ in range(10):
                    yellow_log("即将切换D435i相机，开始锁定目标桶")
                self._c = 2
                self._b = 2
            elif content == '1m':
                for _ in range(10):
                    yellow_log("目标桶已锁定，即将投放")
                self._c = 2
                self._b = 1
            elif content == '0':
                for _ in range(5):
                    yellow_log("检测到0模式，关闭摄像头")
                self._c = 0
                self._b = 0
            else:
                logger.info("未检测到内容，等待...")
                return False
            return True


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
    """统一的文件输出写入器，同时支持 D435 和 USB 格式"""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._lock = threading.Lock()
        self._last_written: str = "0"     # 最后一次实际写入内容，供 read_last() 使用
        # ---- D435 侧状态 ----
        self.last_value: Optional[str] = None
        self.last_detection_time: float = 0.0
        self.b_zero: bool = True
        self.a_zero: bool = True
        # ---- USB 侧状态 ----
        self._last_usb_result: Optional[str] = None

    # ========== D435 接口 ==========

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
        """模式切换时重置 D435 状态"""
        with self._lock:
            if cold_start:
                self.last_value = None
                self.last_detection_time = time.time()
                self.b_zero = True
                self.a_zero = True
                self._write("0")

    # ========== USB 接口 ==========

    def write_usb_initial(self) -> None:
        """USB 启动时写入初始值"""
        with self._lock:
            self._write("0 0 0 0")

    def write_usb(self, result_str: str) -> None:
        """USB 去重写入（仅当内容变化时写入）"""
        with self._lock:
            if result_str != self._last_usb_result:
                self._write(result_str)
                self._last_usb_result = result_str

    def write_usb_final(self, result_str: str) -> None:
        """USB 无条件写入（聚类完成后的最终结果）"""
        with self._lock:
            self._write(result_str)

    def reset_usb(self) -> None:
        """重置 USB 侧去重状态"""
        with self._lock:
            self._last_usb_result = None

    def read_last(self) -> str:
        """线程安全读取最后写入的内容（同时覆盖 D435 和 USB）"""
        with self._lock:
            return self._last_written

    # ========== 内部方法 ==========

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
        """实际写入文件，同时更新 _last_written 缓存"""
        self._last_written = content
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

        p = SUPERELLIPSE_P_LOCK

        candidates = [(d.dis, d.cls, d.ux, d.uy, d.r)
                      for d in detections 
                      if abs((d.ux - ox) / shw) ** p + abs((d.uy - oy) / shh) ** p <= 1]
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
        # self.model_full = YOLO(MODEL_FULL_PATH)
        # self.model_crop = YOLO(MODEL_CROP_PATH)
        self.model = YOLO(MODEL_PATH)
        self._cx = D435_WIDTH // 2
        self._cy = D435_HEIGHT // 2

    def _select_model(self, mode: str):
        """根据模式返回对应的模型和参数 (conf, imgsz, iou)"""
        if mode == '1m':
            return MODEL_1M_CONF, MODEL_1M_IMGSZ, MODEL_1M_IOU
        return MODEL_2M_CONF, MODEL_2M_IMGSZ, MODEL_2M_IOU

    def detect_full(self, frame: np.ndarray, mode: str) -> List[Detection]:
        """全图推理"""
        conf, imgsz, iou = self._select_model(mode)
        t0 = time.time()
        results = self.model.predict(
            source=frame, device=MODEL_DEVICE, show=False,
            stream=False, verbose=False, iou=iou, conf=conf, imgsz=imgsz)
        t1 = time.time()
        infer_time = int((t1 - t0) * 1000)
        yellow_log(f"全图推理：{infer_time}ms")
        return self._parse(results, 0, 0)

    def detect_crop(self, crop: np.ndarray, ox: int, oy: int,
                    mode: str) -> List[Detection]:
        """裁剪区域推理，ox/oy 为裁剪区域左上角在全图中的偏移"""
        h, w = crop.shape[:2]
        imgsz = ((max(w, h) + MODEL_IMGSZ_STEP - 1) // MODEL_IMGSZ_STEP) * MODEL_IMGSZ_STEP
        conf, _, iou = self._select_model(mode)
        t0 = time.time()
        results = self.model.predict(
            source=crop, device=MODEL_DEVICE, show=False,
            stream=False, verbose=False, iou=iou,
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
            self._thread.join(timeout=THREAD_JOIN_TIMEOUT)
            if self._thread.is_alive():
                logger.warning("D435 采集线程未在 1s 内退出")
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
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=D435_FRAME_TIMEOUT_MS)
                color = frames.get_color_frame()
                if color:
                    self._buf.put(np.asanyarray(color.get_data()))
            except Exception:
                if not self._stop.is_set():
                    logger.warning("D435 帧采集异常，重试中...")
                time.sleep(CAPTURE_RETRY_SLEEP)


# ======================================================================
# 9.5 USBCameraSource — USB camera with internal capture thread
# ======================================================================

class USBCameraSource:
    """USB 相机封装，内部使用独立线程持续采集校正后帧放入 FrameBuffer"""

    def __init__(self, cam_num: str, calib_path: str,
                 mode_state: 'ModeState') -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._buf = FrameBuffer()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cam_num = cam_num
        self._calib_path = calib_path
        self._mode_state = mode_state
        self._mtx: Optional[np.ndarray] = None
        self._w: int = 640
        self._h: int = 480

    def start(self) -> None:
        """打开相机，加载标定参数，启动采集线程"""
        self._cap = cv2.VideoCapture(self._cam_num)
        calib = np.load(self._calib_path)
        self._mtx = calib['mtx']
        dist = calib['dist']
        self._h, self._w = 480, 640
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self._mtx, dist, (self._w, self._h), UNDISTORT_ALPHA, (self._w, self._h))
        self._mapx, self._mapy = cv2.initUndistortRectifyMap(
            self._mtx, dist, None, newcameramtx, (self._w, self._h), 5)
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止采集线程并释放相机资源"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=THREAD_JOIN_TIMEOUT)
            if self._thread.is_alive():
                logger.warning("USB 采集线程未在 1s 内退出")
        if self._cap is not None:
            self._cap.release()
            logger.info("USB摄像头已释放")

    def get_frame(self, seen_seq: int = 0
                  ) -> Tuple[bool, Optional[np.ndarray], int]:
        """从缓冲区获取最新帧，与 D435 CameraSource 统一接口"""
        return self._buf.get_latest(seen_seq)

    def get_calibration(self) -> Tuple[np.ndarray, int, int]:
        """返回 (mtx, w, h) 供像素坐标转换"""
        return self._mtx, self._w, self._h

    def _capture(self) -> None:
        """后台采集线程主循环：读取、畸变校正、写入 FrameBuffer"""
        while not self._stop.is_set():
            try:
                ret, frame = self._cap.read()
                if ret:
                    if self._mode_state.get_c() == 3:
                        undistorted = cv2.remap(frame, self._mapx, self._mapy,
                                                cv2.INTER_LINEAR)
                        self._buf.put(undistorted)
                else:
                    logger.error("无法读取摄像头画面，请检查摄像头连接或权限设置。")
                    self._stop.set()
                time.sleep(POLL_SLEEP_FAST)
            except Exception:
                if not self._stop.is_set():
                    logger.warning("USB 帧采集异常，重试中...")
                time.sleep(CAPTURE_RETRY_SLEEP)


# ======================================================================
# 10. DBSCANClusterer — DBSCAN 聚类 + 可视化
# ======================================================================

class DBSCANClusterer:
    """DBSCAN 聚类器，封装聚类计算和可视化绘制"""

    def __init__(self, eps: float, min_samples: int) -> None:
        self._eps = eps
        self._min_samples = min_samples

    def cluster(self, image: np.ndarray,
                points: List[Tuple[int, int]]) -> List[np.ndarray]:
        """运行 DBSCAN，绘制并保存聚类结果，返回聚类中心列表"""
        if len(points) == 0:
            logger.info("没有桶")
            return []

        X = np.array(points)
        db = DBSCAN(eps=self._eps, min_samples=self._min_samples).fit(X)
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
# 11. FileModeMonitor — data.txt 文件模式监控器
# ======================================================================

class FileModeMonitor:
    """轮询 data.txt，解析命令更新 ModeState"""

    def __init__(self, file_path: str, mode_state: 'ModeState') -> None:
        self._file_path = file_path
        self._mode_state = mode_state

    def run(self, stop_event: threading.Event,
            start_event: threading.Event) -> None:
        """轮询 data.txt 变化，解析命令更新 ModeState"""
        last_result: Optional[str] = None

        with open(self._file_path, 'r') as f:
            logger.info(f"data.txt 内容: {f.read().strip()}")
        time.sleep(FILE_POLL_SLEEP)

        while not stop_event.is_set():
            with open(self._file_path, 'r') as f:
                content = f.read().strip()
                if content != last_result:
                    if self._mode_state.set_from_command(content):
                        start_event.set()
                    last_result = content
            time.sleep(FILE_POLL_SLEEP)


# ======================================================================
# 12. USBDetectionPipeline — USB 检测管道
# ======================================================================

class USBDetectionPipeline:
    """USB YOLO 检测 + DBSCAN 聚类 + 多目标输出管道"""

    def __init__(self, model: YOLO, camera: 'USBCameraSource',
                 writer: OutputWriter, display_queue: queue.Queue,
                 mode_state: 'ModeState', eps: float,
                 min_samples: int) -> None:
        self._model = model
        self._camera = camera
        self._writer = writer
        self._clusterer = DBSCANClusterer(eps, min_samples)
        self._display_queue = display_queue
        self._mode_state = mode_state

    def run(self, stop_event: threading.Event,
            start_event: threading.Event) -> None:
        """USB YOLO 检测 + DBSCAN 聚类 + 多目标输出主循环"""
        start_event.wait()
        logger.info("usb_detect线程已启动")

        self._writer.write_usb_initial()
        result_str = "0 0 0 0"
        last_result: Optional[str] = None
        last_seq: int = 0

        mtx: Optional[np.ndarray] = None
        w: int = 640
        h: int = 480

        points: List[Tuple[int, int]] = []
        last_cluster_time = time.time()
        start_cluster_time = last_cluster_time
        clustered_once = False
        final_result_str: Optional[str] = None

        while not stop_event.is_set():
            if (not clustered_once
                    and time.time() - start_cluster_time > CLUSTER_TIMEOUT):
                yellow_log("聚类超时(9s)，输出 0 0 0 0 并进入 0 模式")
                self._writer.write_usb("0 0 0 0")
                self._mode_state.set(0)
                break

            if clustered_once:
                if final_result_str is not None:
                    self._writer.write_usb_final(final_result_str)
                time.sleep(CLUSTERED_IDLE_SLEEP)
                continue

            got, frame, last_seq = self._camera.get_frame(last_seq)
            if not got:
                time.sleep(POLL_SLEEP_FAST)
                continue

            if mtx is None:
                mtx, w, h = self._camera.get_calibration()

            results = self._model.predict(
                source=frame, device=MODEL_DEVICE, show=False,
                stream=False, verbose=False, iou=USB_IOU, conf=USB_CONF, imgsz = USB_IMGSZ)

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
                centers = self._clusterer.cluster(image, points)
                camera_centers = []
                for center in centers:
                    x_cam, y_cam = pixel_to_camera(
                        int(center[0]), int(center[1]), mtx, w, h)
                    camera_centers.append((x_cam, y_cam))

                camera_centers.sort(key=lambda ctr: ctr[0])

                if len(camera_centers) == 3:
                    d = self._mode_state.get_d()
                    i1, i2 = USB_TARGET_MAP.get(d, (1, 0))
                    center1 = camera_centers[i1]
                    center2 = camera_centers[i2]

                    c1x_ok = COORD_X_MIN <= center1[0] <= COORD_X_MAX
                    c1y_ok = COORD_Y_MIN <= center1[1] <= COORD_Y_MAX
                    c2x_ok = COORD_X_MIN <= center2[0] <= COORD_X_MAX
                    c2y_ok = COORD_Y_MIN <= center2[1] <= COORD_Y_MAX

                    if c1x_ok and c2x_ok and c1y_ok and c2y_ok:
                        result_str = (f"{center1[0]:.1f} {center1[1]:.1f} "
                                      f"{center2[0]:.1f} {center2[1]:.1f}")
                    else:
                        result_str = "0 0 0 0"

                    if result_str != last_result:
                        self._writer.write_usb(result_str)
                        last_result = result_str

                    clustered_once = True
                    final_result_str = result_str
                else:
                    if last_result != "0 0 0 0":
                        self._writer.write_usb("0 0 0 0")
                        last_result = "0 0 0 0"

            last_cluster_time = current_time

            if 'image' in locals():
                if not self._display_queue.full():
                    self._display_queue.put((image, "usb"))

            content = self._writer.read_last()
            logger.info(f"gaozhi.txt 内容: {content}")


# ======================================================================
# 13. D435DetectionPipeline — D435 检测管道
# ======================================================================

class D435DetectionPipeline:
    """D435 LockTracker 检测管道"""

    MODEL_DELAY = 3.0

    def __init__(self, writer: OutputWriter, display_queue: queue.Queue,
                 mode_state: 'ModeState') -> None:
        self._writer = writer
        self._display_queue = display_queue
        self._mode_state = mode_state

    def run(self, stop_event: threading.Event,
            start_event: threading.Event) -> None:
        """D435 LockTracker 检测管道主循环"""
        start_event.wait()
        logger.info("d435_detect线程已启动 (LockTracker pipeline)")

        self._writer.reset(cold_start=True)

        detector = YOLODetector()
        camera_src = CameraSource()
        tracker = LockTracker()

        camera_src.start()

        last_b: Optional[int] = None
        last_mode: Optional[str] = None
        delay_until: float = 0.0
        last_seq: int = 0

        try:
            while not stop_event.is_set():
                current_b = self._mode_state.get_b()
                if current_b not in (1, 2):
                    if last_b in (1, 2):
                        self._writer.reset(cold_start=True)
                    last_b = current_b
                    time.sleep(POLL_SLEEP_FAST)
                    continue

                mode = '2m' if current_b == 2 else '1m'

                if mode != last_mode:
                    logger.info(f"模式切换: {last_mode} -> {mode}")
                    cold = last_mode is None
                    self._writer.reset(cold_start=cold)
                    tracker.reset()
                    if last_mode == '2m' and mode == '1m':
                        delay_until = time.time() + self.MODEL_DELAY
                        logger.info("2m->1m: 保持2m模型3s")
                    else:
                        delay_until = 0.0
                    last_mode = mode

                last_b = current_b

                if delay_until > time.time() and mode == '1m':
                    effective_mode = '2m'
                else:
                    effective_mode = mode

                got, frame, last_seq = camera_src.get_frame(last_seq)
                if not got:
                    time.sleep(FRAME_NOT_READY_SLEEP)
                    continue
                
                canvas = frame.copy()

                tracker.increment_frame_count()
                if (tracker.is_locked
                        and tracker.lock_frame_count >= LOCK_MAX_HIT):
                    tracker.force_unlock()

                was_locked = tracker.is_locked

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

                for d in detections:
                    draw_square(canvas, d.box, d.names, d.r)
                    cv2.circle(canvas, (d.ux, d.uy), 4,
                                (255, 255, 255), 5)
                    cv2.putText(canvas, f"[{d.ux},{d.uy}]",
                                (d.ux + 20, d.uy + 10),
                                0, 1, (225, 255, 255),
                                thickness=2, lineType=cv2.LINE_AA)

                result = tracker.update(detections)

                if result.action == TrackAction.DETECT:
                    self._writer.write_detection(mode, result.x, result.y)
                    if not was_locked and result.is_locked:
                        ox, oy, ow, oh = tracker.lock_target
                        _, _, _, _, shw, shh, _ = search_rect(ox, oy, ow, oh)
                        draw_lock_rect(canvas, ox, oy, shw, shh, (0, 255, 0))
                elif result.action == TrackAction.PREDICT:
                    self._writer.write_prediction(mode, result.x, result.y)
                else:
                    self._writer.write_fallback(mode)


                cv2.putText(canvas, f"D435:{effective_mode}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                            cv2.LINE_AA)

                if not self._display_queue.full():
                    self._display_queue.put((canvas, "d435"))

                logger.info(f"gaozhi.txt 内容: {self._writer.read_last()}")
        finally:
            camera_src.stop()


# ======================================================================
# 14. PipelineOrchestrator — 管道生命周期管理器
# ======================================================================

class PipelineOrchestrator:
    """根据 ModeState 变化启停 USB/D435 管道，管理线程生命周期"""

    def __init__(self, mode_state: 'ModeState', writer: OutputWriter,
                 display_queue: queue.Queue) -> None:
        self._mode_state = mode_state
        self._writer = writer
        self._display_queue = display_queue
        self._usb_camera: Optional['USBCameraSource'] = None
        self._usb_pipeline: Optional['USBDetectionPipeline'] = None
        self._d435_pipeline: Optional['D435DetectionPipeline'] = None
        self._usb_capture_thread: Optional[threading.Thread] = None
        self._usb_detect_thread: Optional[threading.Thread] = None
        self._d435_thread: Optional[threading.Thread] = None
        self._running: bool = False

    def run(self, start_event: threading.Event) -> None:
        """监测 ModeState.c 变化，启停对应相机管道"""
        last_c: Optional[int] = None
        stop_event: Optional[threading.Event] = None

        check_camera(cam_num)

        while True:
            c = self._mode_state.get_c()
            if c != last_c:
                logger.info(f"main_control 检测到 c 变化: {last_c} -> {c}")

                if self._running:
                    stop_event.set()
                    if self._usb_capture_thread:
                        self._usb_capture_thread.join()
                    if self._usb_detect_thread:
                        self._usb_detect_thread.join()
                    if self._usb_camera:
                        self._usb_camera.stop()
                    if self._d435_thread:
                        self._d435_thread.join()
                    cv2.destroyAllWindows()
                    time.sleep(MODE_SWITCH_DELAY)
                    self._running = False

                stop_event = threading.Event()

                if c == 3:
                    self._usb_camera = USBCameraSource(
                        cam_num, files_for_pixel, self._mode_state)
                    self._usb_camera.start()
                    self._usb_pipeline = USBDetectionPipeline(
                        model_usb, self._usb_camera, self._writer,
                        self._display_queue, self._mode_state, eps, min_samples)
                    self._usb_capture_thread = None  # camera has its own thread
                    self._usb_detect_thread = threading.Thread(
                        target=self._usb_pipeline.run,
                        args=(stop_event, start_event))
                    self._usb_detect_thread.start()
                    self._d435_thread = None
                    self._running = True
                elif c == 2:
                    self._d435_pipeline = D435DetectionPipeline(
                        self._writer, self._display_queue, self._mode_state)
                    self._d435_thread = threading.Thread(
                        target=self._d435_pipeline.run,
                        args=(stop_event, start_event))
                    self._d435_thread.start()
                    self._usb_camera = None
                    self._usb_pipeline = None
                    self._running = True

                if c == 0:
                    self._writer.reset(cold_start=True)

                last_c = c
            time.sleep(ORCHESTRATOR_POLL_SLEEP)


# ======================================================================
# 15. main()
# ======================================================================

if __name__ == '__main__':
    orch_thread: Optional[threading.Thread] = None
    t_monitor: Optional[threading.Thread] = None
    stop_event: Optional[threading.Event] = None
    try:
        clear_files(files_to_clear)

        shuru_file_path = 'data.txt'
        shuchu_file_path = 'gaozhi.txt'

        mode_state = ModeState()
        writer = OutputWriter(shuchu_file_path)

        start_event = threading.Event()
        stop_event = threading.Event()

        file_monitor = FileModeMonitor(shuru_file_path, mode_state)
        t_monitor = threading.Thread(
            target=file_monitor.run,
            args=(stop_event, start_event))
        t_monitor.start()

        orchestrator = PipelineOrchestrator(
            mode_state, writer, display_queue)
        orch_thread = threading.Thread(
            target=orchestrator.run, args=(start_event,))
        orch_thread.start()

        # 预加载 USB 标定数据，避免显示循环中每帧重复读取
        try:
            calib_data = np.load(files_for_pixel)
            calib_mtx = calib_data['mtx']
        except Exception:
            calib_mtx = None
            logger.warning("USB 标定文件加载失败，四角坐标叠加将跳过")

        # 主线程负责显示窗口
        last_window_name: str = ""

        while True:
            if not display_queue.empty():
                frame, window_name = display_queue.get()

                if window_name != last_window_name:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, DISPLAY_WIN_W, DISPLAY_WIN_H)
                    last_window_name = window_name

                if mode_state.get_c() == 3 and calib_mtx is not None:
                    # USB 画面四角坐标叠加
                    h, w = frame.shape[:2]
                    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
                    colors = [(0, 0, 255), (0, 255, 0),
                              (255, 0, 0), (0, 255, 255)]

                    for i, (x, y) in enumerate(corners):
                        X, Y = pixel_to_camera(x, y, calib_mtx, w, h)
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
                if cv2.waitKey(DISPLAY_WAITKEY_MS) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(POLL_SLEEP_FAST)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if stop_event is not None:
            stop_event.set()
        cv2.destroyAllWindows()
        if orch_thread is not None:
            orch_thread.join(timeout=THREAD_JOIN_TIMEOUT)
        if t_monitor is not None:
            t_monitor.join(timeout=THREAD_JOIN_TIMEOUT)
        logger.info("程序已退出")
