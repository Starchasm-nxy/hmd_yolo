"""
2m遍历方案
yolo26+d435
"""

import cv2                                # OpenCV库，用于图像显示、绘制
import time                               # 时间操作，延时、时间戳
import queue                              # 线程安全队列，用于帧传递
import threading                          # 多线程支持
import logging                            # 日志记录
from dataclasses import dataclass         # 数据类装饰器
from enum import Enum, auto               # 枚举类型
from typing import List, Optional, Tuple, Any, Dict  # 类型注解

import numpy as np                        # 数值计算，处理图像数组
import pyrealsense2 as rs                 # Intel RealSense SDK
from ultralytics import YOLO              # YOLO模型加载与推理

# ======================================================================
# 1. Configuration
# ======================================================================

# 启动时需要清空的文件列表
FILES_TO_CLEAR = ['data.txt', 'gaozhi.txt']

# RealSense D435 彩色流分辨率（宽度，高度）
D435_WIDTH = 848
D435_HEIGHT = 480

# 模型文件路径（按模式区分，不分锁定/未锁定）
# MODEL_2M_PATH = "/home/fu/weights/tong_blue_v0.pt"
# MODEL_1M_PATH = "/home/fu/weights/tong_blue_v0.pt"
# MODEL_FULL_PATH = "/home/fu/weights/tong_blue_v0.pt"
# MODEL_CROP_PATH = "/home/fu/weights/tong_blue_v0.pt"
MODEL_PATH = "/home/fu/yolo/weights/tong_blue_v0.pt"

# 历史坐标超时清空开关及超时时间（秒）
HISTORY_CLEAR_ENABLED = True
HISTORY_CLEAR_TIMEOUT = 10.0

# 跳帧推理开关及每N帧推理一次
FRAME_SKIP_ENABLED = False
FRAME_SKIP_N = 2

# 锁定跟踪参数
LOCK_MAX_HIT = 15                     # 锁定最大命中帧数（达到后强制解锁重判）
LOCK_MAX_MISS = 7                     # 锁定最大丢失帧数（达到后解锁）
LOCK_SEARCH_RATIO = 2.5               # 搜索框相对于目标尺寸的放大比例
LOCK_MIN_SEARCH_RADIUS = 110          # 搜索框最小半边长
LOCK_MAX_SEARCH_RADIUS = 270          # 搜索框最大半边长
LOCK_CONF_GAP = 0.2                   # 锁定候选置信度差距阈值
D435_PAUSE_TIMEOUT = 20.0             # 0 模式暂停超时（秒），超时释放相机

# 模型推理参数（按模式区分）
MODEL_2M_CONF = 0.5                   # 2m模式置信度阈值（全图+裁剪共用）
MODEL_2M_IMGSZ = 640                  # 2m模式全图推理尺寸
MODEL_2M_IOU = 0.45                   # NMS IoU阈值

MODEL_1M_IMGSZ = 640                  # 1m模式全图推理尺寸
MODEL_1M_CONF = 0.5                   # 1m模式置信度阈值（全图+裁剪共用）
MODEL_1M_IOU = 0.45                   # NMS IoU阈值

MODEL_IMGSZ_STEP = 32                 # 动态imgsz步长（对齐到32的倍数）
MODEL_DEVICE = 'cpu'                  # 推理设备，'cpu'或'cuda:0'
MAX_AREA = 66666                      # 最大目标面积，超过则过滤

# 显示窗口名称
DISPLAY_WIN_NAME = "d435"
# 显示帧队列最大长度
DISPLAY_QUEUE_SIZE = 3

# ---- 超椭圆绘制参数 ----
SUPERELLIPSE_P_LOCK = 2.3                      # draw_lock_rect / LockTracker 超椭圆指数
SUPERELLIPSE_NUM_POINTS = 100                  # 超椭圆采样点数

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
logger = logging.getLogger(__name__)   # 获取本模块的logger

def yellow_log(msg: str) -> None:
    """输出黄色高亮的 INFO 日志"""
    logger.info(f"{YELLOW}{msg}{RESET}")


# ======================================================================
# 2. Utility functions
# ======================================================================

def clamp(v: float, lo: float, hi: float) -> float:
    """将浮点数v限制在[lo, hi]区间内"""
    return max(lo, min(hi, v))


def search_rect(x: float, y: float, w: float, h: float) -> Tuple[int, int, int, int, int, int, int]:
    """
    根据目标中心(x,y)和宽高(w,h)计算自适应搜索窗口
    返回: (sx1, sy1, sx2, sy2, shw, shh, area)
    """
    # 根据目标尺寸和放大比例计算搜索半宽、半高，并限制在最小/最大半径内
    shw = clamp(w * LOCK_SEARCH_RATIO / 2, LOCK_MIN_SEARCH_RADIUS, LOCK_MAX_SEARCH_RADIUS)
    shh = clamp(h * LOCK_SEARCH_RATIO / 2, LOCK_MIN_SEARCH_RADIUS, LOCK_MAX_SEARCH_RADIUS)
    # 搜索框左上角和右下角坐标（整型）
    sx1, sy1 = int(x - shw), int(y - shh)
    sx2, sy2 = int(x + shw), int(y + shh)
    # 返回边界坐标、半宽半高、面积
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


def clear_files(files: List[str]) -> None:
    """清空指定的文件列表（创建空文件）"""
    for fn in files:
        with open(fn, 'w'):      # 以写模式打开，立即清空
            pass
    yellow_log("通讯txt文件已建立并清空")


# ======================================================================
# 3. Data types
# ======================================================================

class TrackAction(Enum):
    """跟踪动作枚举：DETECT真实检测，PREDICT预测，LOST丢失"""
    DETECT = auto()    # 输出 [1, x, y]
    PREDICT = auto()   # 输出 [2, x, y]
    LOST = auto()      # 输出 [0] 或回退


@dataclass
class TrackResult:
    """跟踪结果数据类"""
    action: TrackAction                # 动作类型
    x: int = 0                         # 目标x坐标（像素）
    y: int = 0                         # 目标y坐标（像素）
    is_locked: bool = False            # 当前是否处于锁定状态


@dataclass
class Detection:
    """单个检测框信息，坐标均为全图坐标"""
    ux: int                            # 中心点x
    uy: int                            # 中心点y
    cls: int                           # 类别索引
    conf: float                        # 置信度
    r: List[int]                       # 边界框 [x1, y1, x2, y2]
    w: int                             # 框宽度
    h: int                             # 框高度
    area: int                          # 框面积
    dis: float                         # 框中心到图像中心的距离
    box: Any                           # YOLO原始box对象（用于绘制）
    names: Dict[int, str]              # 类别名映射（用于绘制）


# ======================================================================
# 4. FrameBuffer — thread-safe single-slot frame buffer
# ======================================================================

class FrameBuffer:
    """单生产者单消费者帧缓冲，只保留最新帧，线程安全"""

    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None   # 当前最新帧图像
        self._seq: int = 0                         # 帧序号，每put一次递增
        self._lock = threading.Lock()              # 互斥锁，保护帧数据

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
                return False, None, self._seq       # 无新帧
            # 有新帧，返回深拷贝避免外部修改内部数据
            return True, self._frame.copy(), self._seq


# ======================================================================
# 5. FileMonitor — polls data.txt, signals mode changes
# ======================================================================

class FileMonitor:
    """轮询 data.txt 文件内容变化，通过 Event 通知模式切换"""

    def __init__(self, file_path: str, poll_interval: float = 0.05) -> None:
        self.file_path = file_path                         # 监控的文件路径
        self.poll_interval = poll_interval                 # 轮询间隔（秒）
        self._stop = threading.Event()                     # 停止信号
        self._thread: Optional[threading.Thread] = None    # 后台线程
        self.command_active = threading.Event()            # 模式激活事件（1m/2m时置位，0时清除）
        self._lock = threading.Lock()                      # 保护 current_command
        self.current_command: Optional[str] = None         # 当前命令字符串
        self.clearfile = None
        self.connectfile = None

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

    def read_command(self) -> Optional[str]:
        """线程安全地读取当前命令"""
        with self._lock:
            return self.current_command

    def _run(self) -> None:
        """后台轮询主循环"""
        last_content: Optional[str] = None   # 记录上一次文件内容
        while not self._stop.is_set():
            try:
                if not self.clearfile:
                    logger.info("读取成功and去除空白")
                    self.clearfile = True
                    self.connectfile = True
                with open(self.file_path, 'r') as f:
                    content = f.read().strip()          # 读取并去除空白
            except Exception:
                if not self.connectfile:
                    logger.warning("读取失败and等待重试")
                time.sleep(self.poll_interval)          # 读取失败则等待后重试
                continue

            if content != last_content:                 # 内容有变化
                with self._lock:
                    self.current_command = content      # 更新当前命令
                # 根据内容设置模式激活事件
                if content in ('1m', '2m'):
                    for _ in range(5):
                        yellow_log(f"检测到{content}模式")
                    self.command_active.set()           # 激活
                elif content == '0':
                    yellow_log("检测到0模式>_<已关闭摄像头and清空历史")
                    logger.info("（下面可能还有一条不用管它，是最后位置")
                    self.command_active.clear()         # 取消激活
                else:
                    yellow_log("目前无效内容>_<等待ing...")
                last_content = content                  # 更新上次内容

            time.sleep(self.poll_interval)              # 等待下一轮查询


# ======================================================================
# 6. OutputWriter — writes gaozhi.txt, manages history
# ======================================================================

class OutputWriter:
    """将检测结果写入 gaozhi.txt，管理历史坐标、超时清空和 2m 首次日志"""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path                        # 输出文件路径
        self._lock = threading.Lock()                     # 线程锁
        self.last_value: Optional[str] = None             # 上次非丢失的有效坐标 "x y"
        self.last_detection_time: float = 0.0             # 上次真实检测的时间戳
        self.b_zero: bool = True                          # 2m模式下是否未输出过非0值
        self.a_zero: bool = True                          # 1m模式下是否未输出过非0值

    def write_detection(self, mode: str, ux: int, uy: int) -> None:
        """
        写入真实检测结果 [1, ux, uy]，更新 last_value 和时间戳
        """
        with self._lock:
            self._log_2m_entry(mode)                      # 2m模式首次输出日志
            self._log_1m_entry(mode)
            self._write(f"1 {ux} {uy}")                  # 写入格式 "1 ux uy"
            self.last_value = f"{ux} {uy}"                # 记录当前坐标
            self.last_detection_time = time.time()        # 更新最后检测时间

    def write_prediction(self, mode: str, ux: int, uy: int) -> None:
        """
        写入预测结果 [2, ux, uy]，更新 last_value 但不更新时间戳
        （预测并非真实检测，不重置超时计时器）
        """
        with self._lock:
            self._log_2m_entry(mode)
            self._log_1m_entry(mode)
            self._write(f"2 {ux} {uy}")
            self.last_value = f"{ux} {uy}"

    def write_fallback(self, mode: str) -> None:
        """
        跟踪器报告 LOST 时的回退写入策略：
        - 1/2m模式：若有 last_value 且不为 '0' 则输出 "2 last_value"，否则输出 "0"
        """
        with self._lock:
            self._check_timeout()                          # 先检查历史坐标是否超时
            self._log_2m_entry(mode)
            self._log_1m_entry(mode)

            if mode != '0' and self.last_value is not None:
                # 非0模式且有历史坐标，回退为预测模式 "2 x y"
                content = "0" if self.last_value == '0' else "2 " + self.last_value
            else:
                content = "0"

            self._write(content)

            # 2m模式下写 "0" 时保留 last_value 不变
            if mode == '2m' and content.strip() == '0':
                return

            # 更新 last_value：若为 "0" 则记录 "0"，否则提取坐标
            parts = content.strip().split()
            self.last_value = '0' if parts[0] == '0' else ' '.join(parts[1:])

    def reset(self, cold_start: bool = False) -> None:
        """
        模式切换时重置状态
        cold_start=True 表示冷启动（从非1m/2m进入），清空历史坐标并写 '0'
        """
        with self._lock:
            if cold_start:
                self.last_value = None                # 清空历史坐标
                self.last_detection_time = time.time() # 重置检测计时
                self.b_zero = True                    # 重置2m首次标志
                self.a_zero = True                    # 重置1m首次标志
                self._write("0")

    def _check_timeout(self) -> None:
        """若启用历史清空且超时，则清空 last_value"""
        if (HISTORY_CLEAR_ENABLED and self.last_value is not None
                and time.time() - self.last_detection_time > HISTORY_CLEAR_TIMEOUT):
            self.last_value = None
            logger.info(f"超过{HISTORY_CLEAR_TIMEOUT}秒未检测到目标，清空历史坐标")

    def _log_2m_entry(self, mode: str) -> None:
        """2m模式下首次写入非0值时打印提示日志"""
        if mode == '2m' and self.b_zero:
            for _ in range(5):
                yellow_log("2m对桶开始※※※")
            self.b_zero = False                       # 标记已输出过

    def _log_1m_entry(self, mode: str) -> None:
        """1m模式下首次写入非0值时打印提示日志"""
        if mode == '1m' and self.a_zero:
            for _ in range(5):
                yellow_log("1m对桶开始※※※")
            self.a_zero = False                       # 标记已输出过

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
    """基于锁定的单目标跟踪器，线程安全"""

    def __init__(self) -> None:
        self._lock = threading.Lock()                          # 线程锁
        self.lock_target: Optional[Tuple[int, int, int, int]] = None  # 锁定目标 (x, y, w, h)
        self.lock_miss_count: int = 0                          # 锁定状态下连续丢失帧计数
        self.lock_frame_count: int = 0                         # 锁定状态下累计帧计数（用于强制解锁）

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

    @staticmethod
    def _pick_best(candidates: List) -> Optional[Any]:
        """置信度梯队优先：最高置信度 ±0.2 内的候选中，选离画面中心最近的"""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        max_conf = max(c[1] for c in candidates)
        high = [c for c in candidates if max_conf - c[1] <= LOCK_CONF_GAP]
        return min(high, key=lambda c: c[0])

    def update(self, detections: List[Detection]) -> TrackResult:
        """
        状态机主入口，线程安全。
        根据当前锁定状态和检测列表返回 TrackResult。
        """
        with self._lock:
            if self.is_locked:
                # 锁定状态 + 有检测 → 走 Path 1/2/3
                return (self._locked_with_dets(detections) if detections
                        else self._locked_no_dets())
            else:
                # 未锁定状态 → 走 Path 6/7
                return (self._unlocked_with_dets(detections) if detections
                        else self._unlocked_no_dets())

    # ---- LOCKED + detections (Paths 1, 2, 3) ----

    def _locked_with_dets(self, detections: List[Detection]) -> TrackResult:
        """锁定状态下收到检测结果的处理"""
        ox, oy, ow, oh = self.lock_target  # 当前锁定目标位置和尺寸
        # 计算搜索窗口
        sx1, sy1, sx2, sy2, shw, shh, sarea = search_rect(ox, oy, ow, oh)

        p= SUPERELLIPSE_P_LOCK # 超椭圆参数

        # 筛选搜索窗口内的候选检测
        candidates = [(d.dis, d.conf, d.cls, d.ux, d.uy, d.r)
                      for d in detections
                      if abs((d.ux - ox) / shw) ** p + abs((d.uy - oy) / shh) ** p <= 1]
        best = self._pick_best(candidates)

        if best is not None:
            # Path 1: 窗口内有匹配 → 更新锁定目标
            _, _, _, ux, uy, r = best
            w, h = abs(r[2] - r[0]), abs(r[3] - r[1])        # 计算新宽高
            self.lock_target = (ux, uy, w, h)                 # 更新锁定
            self.lock_miss_count = 0                          # 清零丢失计数
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
            # 从所有检测中选取最近的目标
            best = self._pick_best([(d.dis, d.conf, d.cls, d.ux, d.uy, d.r) for d in detections])
            _, _, _, ux, uy, r = best
            w, h = abs(r[2] - r[0]), abs(r[3] - r[1])
            self.lock_target = (ux, uy, w, h)                 # 重新锁定
            _, _, _, _, nshw, nshh, narea = search_rect(ux, uy, w, h)
            yellow_log(f"解锁-最近桶：({ux},{uy}) 目标面积={w*h}")
            return TrackResult(TrackAction.DETECT, ux, uy, True)
        else:
            # Path 2: 丢失但未超限 → 沿用旧位置进行预测
            logger.warning(f"锁定-历史：({ox},{oy}) miss={self.lock_miss_count} ")
            return TrackResult(TrackAction.PREDICT, ox, oy, True)

    # ---- LOCKED + no detections (Paths 4, 5) ----

    def _locked_no_dets(self) -> TrackResult:
        """锁定状态下无任何检测时的处理"""
        ox, oy, ow, oh = self.lock_target
        sx1, sy1, sx2, sy2, shw, shh, sarea = search_rect(ox, oy, ow, oh)

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
        best = self._pick_best([(d.dis, d.conf, d.cls, d.ux, d.uy, d.r) for d in detections])
        _, _, _, ux, uy, r = best
        w, h = abs(r[2] - r[0]), abs(r[3] - r[1])
        self.lock_target = (ux, uy, w, h)                    # 进入锁定状态
        self.lock_miss_count = 0
        self.lock_frame_count = 0
        _, _, _, _, shw, shh, sarea = search_rect(ux, uy, w, h)
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
    双模型YOLO检测器（按模式区分：2m/1m，不分锁定/未锁定）
    detect_full / detect_crop 都根据 mode 选择对应模型。
    """

    def __init__(self) -> None:
        # self.model_full = YOLO(MODEL_FULL_PATH)
        # self.model_crop = YOLO(MODEL_CROP_PATH)
        self.model = YOLO(MODEL_PATH)
        self._cx = D435_WIDTH // 2
        self._cy = D435_HEIGHT // 2

    def _select_model(self, mode: str):
        """根据模式返回对应的模型和参数 (conf, full_imgsz， iou)"""
        if mode == '1m':
            return MODEL_1M_CONF, MODEL_1M_IMGSZ, MODEL_1M_IOU
        return MODEL_2M_CONF, MODEL_2M_IMGSZ, MODEL_2M_IOU

    def detect_full(self, frame: np.ndarray, mode: str) -> List[Detection]:
        conf, imgsz, iou = self._select_model(mode)
        t0 = time.time()
        results = self.model.predict(
            source=frame, device=MODEL_DEVICE, show=False,
            stream=False, verbose=False, iou=iou, conf=conf, imgsz=imgsz)
        t1 = time.time()
        infer_time = int((t1-t0) * 1000)
        yellow_log(f"全图推理：{infer_time}ms")
        return self._parse(results, 0, 0)

    def detect_crop(self, crop: np.ndarray, ox: int, oy: int,
                    mode: str) -> List[Detection]:
        h, w = crop.shape[:2]
        imgsz = ((max(w, h) + MODEL_IMGSZ_STEP - 1) // MODEL_IMGSZ_STEP) * MODEL_IMGSZ_STEP
        conf, _, iou = self._select_model(mode)
        t0 = time.time()
        results = self.model.predict(
            source=crop, device=MODEL_DEVICE, show=False,
            stream=False, verbose=False, iou=iou, conf=conf, imgsz=imgsz)
        t1 = time.time()
        infer_time = int((t1-t0) * 1000)
        logger.info(f"裁减推理：{infer_time}ms")
        return self._parse(results, ox, oy)

    def _parse(self, results: Any, ox: int, oy: int) -> List[Detection]:
        """
        将YOLO推理结果解析为 Detection 列表
        ox, oy: 当检测区域为裁剪图时，代表该区域左上角在全图中的偏移
        """
        detections: List[Detection] = []
        for result in results:                           # 遍历每张图的推理结果（通常只有一张）
            boxes = result.boxes                         # 检测框信息
            names = result.names                         # 类别名称字典
            if boxes is None:                            # 无检测则跳过
                continue
            for box in boxes:
                r = box.xyxy[0].cpu().numpy().astype(int) # 获取 [x1, y1, x2, y2] 坐标
                # 加上裁剪偏移，转换为全图坐标
                r[0] += ox; r[1] += oy; r[2] += ox; r[3] += oy
                ux = int((r[0] + r[2]) / 2)              # 中心x
                uy = int((r[1] + r[3]) / 2)              # 中心y
                w = abs(int(r[2] - r[0]))                # 宽度
                h = abs(int(r[3] - r[1]))                # 高度
                area = w * h                             # 面积
                if area > MAX_AREA:                      # 面积过大则过滤
                    continue
                # 计算框中心到图像中心的欧氏距离
                dis = ((ux - self._cx) ** 2 + (uy - self._cy) ** 2) ** 0.5
                detections.append(Detection(
                    ux=ux, uy=uy, cls=int(box.cls[0]), conf=float(box.conf[0]),
                    r=r.tolist(), w=w, h=h, area=area, dis=dis, box=box, names=names))
        return detections


# ======================================================================
# 9. CameraSource — D435 camera with internal capture thread
# ======================================================================

class CameraSource:
    """D435相机封装，内部使用独立线程持续采集帧放入 FrameBuffer"""

    def __init__(self) -> None:
        self._pipeline: Optional[rs.pipeline] = None       # RealSense pipeline
        self._buf = FrameBuffer()                          # 帧缓冲
        self._stop = threading.Event()                     # 停止采集信号
        self._thread: Optional[threading.Thread] = None    # 采集线程

    def start(self) -> None:
        """启动RealSense管道和采集线程"""
        self._pipeline = rs.pipeline()                     # 创建管道
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, D435_WIDTH, D435_HEIGHT, rs.format.bgr8, 30) # 配置彩色流
        self._pipeline.start(cfg)                          # 启动管道
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture, daemon=True) # 创建守护线程
        self._thread.start()
        logger.info("D435 capture thread started")

    def stop(self) -> None:
        """停止采集线程并释放相机资源"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)                  # 等待线程退出
        if self._pipeline is not None:
            self._pipeline.stop()                         # 停止管道
            logger.info("D435摄像头已释放")

    def get_frame(self, seen_seq: int) -> Tuple[bool, Optional[np.ndarray], int]:
        """从缓冲区获取最新帧"""
        return self._buf.get_latest(seen_seq)
    
    @staticmethod
    def wait_for_camera(timeout: float = 30.0, poll_interval: float = 2.0) -> bool:
        """
        阻塞等待 D435 相机连接。
        timeout: 最大等待秒数，None 表示无限等待
        poll_interval: 检测间隔
        return: 是否成功连接
        """
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
            frames = self._pipeline.wait_for_frames()     # 等待一组帧
            color = frames.get_color_frame()              # 获取彩色帧
            if color:
                self._buf.put(np.asanyarray(color.get_data())) # 将图像数据放入缓冲


# ======================================================================
# 10. DetectionPipeline — main processing loop
# ======================================================================

class DetectionPipeline:
    """主处理管道：采集 → 推理 → 跟踪 → 输出 → 显示"""

    MODEL_DELAY = 3.0  # 2m→1m 切换时延用2m模型的秒数

    def __init__(self, camera: CameraSource, detector: YOLODetector,
                 tracker: LockTracker, writer: OutputWriter,
                 monitor: FileMonitor) -> None:
        self.camera = camera             # 相机源
        self.detector = detector         # YOLO检测器
        self.tracker = tracker           # 跟踪器
        self.writer = writer             # 输出写入器
        self.monitor = monitor           # 文件监视器（仅用于读取当前模式）
        self._model_mode: Optional[str] = None   # 当前实际使用的模型模式
        self._delay_until: float = 0.0           # 2m→1m 延时到期时间戳
        self._used: bool = False                 # 是否用过 1m/2m（用于区分 0 模式）
        self._pause_start: float = 0.0           # D435 暂停计时起点

    @staticmethod
    def _wait_for_clear() -> None:
        """轮询 data.txt 直到为空（下位机清空=可以开始新一轮）"""
        while True:
            try:
                with open('data.txt', 'r') as f:
                    if f.read().strip() == '':
                        break
            except Exception:
                pass
            time.sleep(0.25)
        yellow_log("data.txt 已清空，准备下一轮")

    def run(self, stop_event: threading.Event,
            display_queue: queue.Queue) -> None:
        """主循环，运行在独立线程中（camera 已在 main 启动）"""
        last_mode: Optional[str] = None  # 上一次的模式
        last_seq: int = 0                # 已处理的帧序号

        try:
            while not stop_event.is_set():
                mode = self.monitor.read_command()  # 获取当前命令（1m/2m/0/其他）
                delay_time = 0.0

                # ---- 预热预览 / 暂停预览 ----
                if mode not in ('1m', '2m'):
                    if last_mode in ('1m', '2m'):
                        # 刚从检测切到暂停：写 '0'，启动计时
                        with open('gaozhi.txt', 'w') as f:
                            f.write('0')
                        if not self._used:
                            self._used = True
                        self._pause_start = time.time()
                    last_mode = mode

                    # 暂停超时检查（仅在用过 1m/2m 后）
                    if self._used and time.time() - self._pause_start > D435_PAUSE_TIMEOUT:
                        yellow_log(f"D435 暂停超时 {D435_PAUSE_TIMEOUT}s，释放相机")
                        self.camera.stop()
                        cv2.destroyAllWindows()
                        self._wait_for_clear()
                        # 重建相机，下一轮
                        self.camera = CameraSource()
                        self.camera.start()
                        self.tracker.reset()
                        self.writer.reset(cold_start=True)
                        self._used = False
                        self._model_mode = None
                        last_seq = 0
                        yellow_log("D435 相机已重启（预热预览中，等待指令）")
                        continue

                    # 推原始帧预览（画面不冻）
                    got, frame, last_seq = self.camera.get_frame(last_seq)
                    if got:
                        try:
                            display_queue.put((frame, DISPLAY_WIN_NAME), block=False)
                        except queue.Full:
                            pass
                    time.sleep(0.01)
                    continue

                # 模式切换处理
                if mode != last_mode:
                    logger.info(f"模式切换: {last_mode} -> {mode}")
                    cold = last_mode not in ('1m', '2m')  # 是否冷启动
                    self.writer.reset(cold_start=cold)    # 重置输出状态
                    self.tracker.reset()                  # 重置跟踪器
                    if cold:
                        logger.info("冷启动：历史坐标已清空")
                    # 2m→1m 延时3s仍用2m模型，其余切换立即生效
                    if last_mode == '2m' and mode == '1m':
                        self._delay_until = time.time() + self.MODEL_DELAY
                        logger.info(f"2m→1m: 保持2m模型 {self.MODEL_DELAY}s")
                    else:
                        self._model_mode = mode
                        self._delay_until = 0
                    last_mode = mode


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
                    time.sleep(0.001)                     # 无新帧则短暂休眠
                    continue

                canvas = frame.copy()                 # 创建画布副本用于绘制
                self._process_frame(frame, canvas, effective_mode)  # 核心处理

                # 在画布上叠加模式信息
                if effective_mode == mode:
                    cv2.putText(canvas, f"{mode}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(canvas, f"{effective_mode}->{mode}...{round(delay_time,1)}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # 将画布放入显示队列（非阻塞）
                try:
                    display_queue.put((canvas, DISPLAY_WIN_NAME), block=False)
                except queue.Full:
                    pass

        finally:
            self.camera.stop()  # 确保释放相机

    def _process_frame(self, frame: np.ndarray, canvas: np.ndarray,
                       mode: str) -> None:
        """单帧核心处理：推理、跟踪、绘制、输出"""
        # 预推理：检查是否需要强制解锁 (Path 8)
        self.tracker.increment_frame_count()               # 锁定帧数+1
        if self.tracker.is_locked and self.tracker.lock_frame_count >= LOCK_MAX_HIT:
            self.tracker.force_unlock()                    # 达到上限强制解锁

        was_locked = self.tracker.is_locked                # 记录推理前的锁定状态（用于绘制）

        # YOLO 推理
        if was_locked:
            # 锁定状态：根据锁定目标计算搜索区域并进行裁剪检测
            ox, oy, ow, oh = self.tracker.lock_target
            sx1, sy1, sx2, sy2, shw, shh, _ = search_rect(ox, oy, ow, oh)
            # 裁剪坐标限制在图像尺寸内
            csx1 = max(0, int(sx1)); csy1 = max(0, int(sy1))
            csx2 = min(D435_WIDTH, int(sx2)); csy2 = min(D435_HEIGHT, int(sy2))
            crop = frame[csy1:csy2, csx1:csx2]            # 裁剪区域
            detections = self.detector.detect_crop(crop, csx1, csy1, mode)  # 裁剪检测
            if detections:
                # 绘制黄色搜索框
                draw_lock_rect(canvas, ox, oy, shw, shh, (0, 255, 255))
            else :
                # 如果锁定状态下无检测，提前绘制橙色搜索框（因为tracker可能马上解锁）
                draw_lock_rect(canvas, ox, oy, shw, shh, (0, 165, 255))
        else:
            # 未锁定状态：全图检测
            detections = self.detector.detect_full(frame, mode)

        # 绘制所有检测框和中心点
        for d in detections:
            ux, uy = draw_square(canvas, d.box, d.names, d.r)   # 画框和标签
            cv2.circle(canvas, (d.ux, d.uy), 4, (255, 255, 255), 5)  # 加粗中心点
            cv2.putText(canvas, str([d.ux, d.uy]), (d.ux + 20, d.uy + 10),
                        0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)  # 显示坐标
            

        # 运行跟踪状态机，得到跟踪结果
        result = self.tracker.update(detections)

        # logger.info(f"模式：{mode}{'锁定' if result.is_locked else ''} "
        #              f"检测数={len(detections)} "
        #              f"锁帧数={self.tracker.lock_frame_count} "
        #              f"丢帧数={self.tracker.lock_miss_count}")

        # 根据跟踪结果写入输出文件
        if result.action == TrackAction.DETECT:
            self.writer.write_detection(mode, result.x, result.y)
            # 如果刚从非锁定变为锁定，绘制绿色搜索框表示新锁定
            if not was_locked and result.is_locked:
                ox, oy, ow, oh = self.tracker.lock_target
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
    """程序主入口"""
    logger.info("Yolo目标检测-程序启动")

    clear_files(FILES_TO_CLEAR)                  # 清空通讯文件

    # 初始化各个模块
    detector = YOLODetector()                    # YOLO检测器
    camera = CameraSource()                      # 相机源
    tracker = LockTracker()                      # 跟踪器
    writer = OutputWriter('gaozhi.txt')          # 输出写入器
    monitor = FileMonitor('data.txt')            # 文件监视器
    pipeline = DetectionPipeline(camera, detector, tracker, writer, monitor)  # 主管道

    # 检查D435相机是否连接
    if not CameraSource.wait_for_camera(timeout=60):
        logger.error("相机不可用，程序终止。")
        return
    logger.info("D435 相机已连接。")

    camera.start()  # 提前启动相机预热
    yellow_log("D435 相机已开启（预热预览中，等待指令）")

    monitor.start()                              # 启动文件监视线程

    main_stop = threading.Event()                # 主程序停止信号
    display_queue: queue.Queue = queue.Queue(maxsize=DISPLAY_QUEUE_SIZE)  # 显示帧队列

    # 管道持续运行，mode='0' 时内部 idle 写 '0'，不再启停
    pipeline_thread = threading.Thread(
        target=pipeline.run, args=(main_stop, display_queue), daemon=True)
    pipeline_thread.start()

    # logger.info("完成Yolo26n模型加载")
    logger.info("系统初始化完成，等待指令...")

    # 显示循环（主线程）
    window_created = False                       # 窗口是否已创建
    try:
        while True:
            try:
                frame, win_name = display_queue.get(timeout=0.05)  # 从队列获取帧
            except queue.Empty:
                if main_stop.is_set():
                    break
                continue

            if not window_created:
                # 创建可调整大小的显示窗口
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL |
                                cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                cv2.resizeWindow(win_name, D435_WIDTH, D435_HEIGHT)
                cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
                window_created = True

            cv2.imshow(win_name, frame)           # 显示画面
            if cv2.waitKey(1) & 0xFF == ord('q'): # 按 'q' 键退出
                main_stop.set()
                break

    except KeyboardInterrupt:
        logger.info("键盘中断，正在退出...")
    finally:
        main_stop.set()                           # 设置停止信号
        monitor.stop()                            # 停止文件监视器
        pipeline_thread.join(timeout=3)           # 等待管道线程退出
        cv2.destroyAllWindows()
        logger.info("程序已退出")


if __name__ == '__main__':
    main()          # 脚本直接运行时执行 main()