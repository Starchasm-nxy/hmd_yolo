import cv2
import time
import queue
import threading
import logging
from collections import deque
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ==================== 配置 ====================
FILES_TO_CLEAR = ['data.txt', 'gaozhi.txt']
display_queue = queue.Queue(maxsize=3)

D435_WIDTH = 848
D435_HEIGHT = 480
MODEL_LOCKED_PATH = "/home/fu/weights/tong_blue_v0.pt"    # 锁定期间（裁剪推理）
MODEL_UNLOCKED_PATH = "/home/fu/weights/tong_blue_v0.pt"  # 重判期间（全图推理）

# ==================== 历史坐标超时清空 ====================
HISTORY_CLEAR_ENABLED = 1            # 0=禁用, 1=超时后清空历史坐标
HISTORY_CLEAR_TIMEOUT = 10.0         # 超时时长（秒）

# ==================== 跳帧控制 ====================
FRAME_SKIP_ENABLED = 0          # 0=每帧推理, 1=跳帧推理
FRAME_SKIP_N = 2                 # 每隔N帧推理一次

# ==================== 锁定追踪参数 ====================
LOCK_MAX_HIT = 15               # 连续命中超过此数则强制重判
LOCK_MAX_MISS = 7               # 连续丢帧超过此数则解除锁定
LOCK_SEARCH_RATIO = 5         # 搜索窗口 = 目标尺寸 * 此比率
LOCK_MIN_SEARCH_RADIUS = 130     # 搜索半径下限（像素，远目标）
LOCK_MAX_SEARCH_RADIUS = 260    # 搜索半径上限（像素，近目标）

# ==================== YOLO 推理参数（锁定 / 非锁定） ====================
# 锁定期间（裁剪推理）—— imgsz 根据裁剪尺寸动态计算
MODEL_LOCKED_CONF = 0.5
# 非锁定期间（全图推理）—— 按模式
MODEL_UNLOCKED_2M_IMGSZ = 640;    MODEL_UNLOCKED_1M_IMGSZ = 640
MODEL_UNLOCKED_2M_CONF = 0.5;     MODEL_UNLOCKED_1M_CONF = 0.5

# ==================== 模型加载 ====================
model_locked = YOLO(MODEL_LOCKED_PATH)       # 锁定期间用（裁剪推理）
# model_unlocked = YOLO(MODEL_UNLOCKED_PATH)   # 重判期间用（全图推理）

# ==================== 线程间共享状态 ====================
class SharedState:
    def __init__(self):
        self.last_value = None      # 上一次写入的有效检测值
        self.last_detection_time = 0  # 上一次检测到目标的时间戳
        self.b_zero = True          # 是否首次进入对桶模式（2m）
        # 锁定追踪状态
        self.lock_target = None     # (x, y, w, h) 被锁定目标（丢帧时冻结不动）
        self.lock_miss_count = 0    # 连续丢帧计数
        self.lock_frame_count = 0   # 锁定持续帧数，达 LOCK_MAX_HIT 时强制重判

state = SharedState()

# ==================== 工具函数 ====================
def clear_files(files):
    for file_name in files:
        with open(file_name, 'w'):
            pass
    logger.info("通讯txt文件已建立并清空")

def read_data_file():
    try:
        with open('data.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def write_detection(processed_list, mode):
    """根据模式将检测结果写入 gaozhi.txt"""
    # 2m 首次进入对桶模式时打印提示
    if mode == '2m' and state.b_zero:    
        logger.info("我方即将进入对桶程序※※")
        state.b_zero = False

    if not processed_list:
        # 无检测结果：2m 回落 last_value，1m 写 0
        if mode == '2m' and state.last_value is not None:
            with open('gaozhi.txt', 'w') as f:
                f.write(state.last_value + '\n')
        else:
            with open('gaozhi.txt', 'w') as f:
                f.write('0')
        return

    content = ' '.join(processed_list)
    with open('gaozhi.txt', 'w') as f:
        f.write(content)

    # 更新 last_value（2m 下 content='0' 时不覆盖）
    if mode == '2m' and content.strip() == '0':
        return
    state.last_value = content.strip()

def draw_square(image, box, names, r):
    ux = int((r[0] + r[2]) / 2)
    uy = int((r[1] + r[3]) / 2)
    cls = int(box.cls[0])
    conf = box.conf[0]
    label = f"{names[cls]} {conf:.2f}"
    cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (221, 185, 193), 2)
    cv2.putText(image, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (176, 196, 222), 2)
    cv2.circle(image, (ux, uy), 5, (240, 240, 240), -1)
    return ux, uy

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def search_rect(x, y, w, h):
    """返回 (sx1, sy1, sx2, sy2, shw, shh, area)"""
    shw = clamp(w * LOCK_SEARCH_RATIO / 2, LOCK_MIN_SEARCH_RADIUS, LOCK_MAX_SEARCH_RADIUS)
    shh = clamp(h * LOCK_SEARCH_RATIO / 2, LOCK_MIN_SEARCH_RADIUS, LOCK_MAX_SEARCH_RADIUS)
    sx1, sy1 = int(x - shw), int(y - shh)
    sx2, sy2 = int(x + shw), int(y + shh)
    return sx1, sy1, sx2, sy2, int(shw), int(shh), (sx2 - sx1) * (sy2 - sy1)

def draw_lock_rect(canvas, x, y, shw, shh, color):
    cv2.rectangle(canvas, (x - shw, y - shh), (x + shw, y + shh), color, 2)
    cv2.circle(canvas, (x, y), 4, color, -1)

def pick_nearest(items):
    """从 tong_list 中选 dis 最小的，若无则返回 None"""
    if not items:
        return None
    return min(items, key=lambda x: x[0])

# ==================== 文件监控线程 ====================
def file_monitor_thread(stop_event, cmd_event):
    """监控 data.txt 指令变化，通过 cmd_event 通知检测管理线程"""
    last_content = None

    while not stop_event.is_set():
        try:
            with open('data.txt', 'r') as f:
                content = f.read().strip()
        except:
            time.sleep(0.05)
            continue

        if content != last_content:
            if content in ['2m', '1m']:
                logger.info(f"检测到{content}模式")
                cmd_event.set()
            elif content == '0':
                logger.info("检测到0模式，关闭摄像头")
                with open('gaozhi.txt', 'w') as f:
                    f.write('0')
                cmd_event.clear()
            else:
                logger.info("未检测到有效内容，等待...")
            last_content = content

        time.sleep(0.05)

# ==================== 线程安全帧缓冲区 ====================
class FrameBuffer:
    """单生产者-单消费者帧缓冲区，只保留最新帧"""
    def __init__(self):
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()

    def put(self, frame):
        with self._lock:
            self._frame = frame
            self._seq += 1

    def get_latest(self, seen_seq=0):
        """获取最新帧（带序列号去重），返回 (has_new, frame, current_seq)"""
        with self._lock:
            if self._seq <= seen_seq or self._frame is None:
                return False, None, self._seq
            return True, self._frame.copy(), self._seq

def camera_capture_loop(pipeline, frame_buf, stop_event):
    """独立采集线程：D435 color frames → FrameBuffer"""
    while not stop_event.is_set():
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            frame = np.asanyarray(color_frame.get_data())
            frame_buf.put(frame)

# ==================== 相机检测线程 ====================
#
# 视觉判断逻辑（锁定追踪状态机）：
#
#   整体策略：锁定一个目标后持续追踪，避免多目标跳变。
#
#   UNLOCKED（无锁定）
#     → 全图 YOLO 推理（model_unlocked）
#     → 从所有检测中选 dis 最小（离画面中心最近）的目标
#     → 建立锁定，进入 LOCKED 状态
#
#   LOCKED（有锁定）
#     → 裁剪推理（model_locked）：以锁定目标为中心，search_rect 确定裁剪区域
#     → 在搜索矩形内筛选候选，选 dis 最小的
#     → 命中：更新 lock_target，lock_miss_count 清零
#     → 丢帧：lock_miss_count++，仍输出上次锁定坐标
#       → miss > LOCK_MAX_MISS：解除锁定，从全图 tong_list 重新选目标建立锁定
#     → lock_frame_count >= LOCK_MAX_HIT：强制重判（全图推理刷新锁定）
#
#   无检测时：
#     - 有锁 → 丢帧计数，超限后解除，回落 last_value 或写 0
#     - 无锁 → 回落 last_value，超时清空则写 0
#
#   写入 gaozhi.txt 格式：[1, ux, uy] 或 [0]
#
def camera_detection_thread(stop_event):
    """D435 相机检测：独立采集 + YOLO 检测 + 结果写入"""
    # ---------- 打开 D435 摄像头 ----------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, D435_WIDTH, D435_HEIGHT, rs.format.bgr8, 30)
    pipeline.start(config)

    logger.debug("d435_detect线程已启动")

    # ---------- 启动独立采集线程 ----------
    frame_buf = FrameBuffer()
    cap_stop_event = threading.Event()
    capture_thread = threading.Thread(
        target=camera_capture_loop,
        args=(pipeline, frame_buf, cap_stop_event),
        daemon=True
    )
    capture_thread.start()

    processed_list = []
    last_mode = None
    last_seq = 0
    frame_count = 0
    last_canvas = None
    fps_times = deque(maxlen=30)
    fps_print_count = 0

    try:
        while not stop_event.is_set():
            mode = read_data_file()

            if mode not in ['1m', '2m']:
                # data变为'0'时清空历史，下次切回有效模式时强制重置
                state.last_value = None
                state.last_detection_time = 0
                state.b_zero = True
                last_mode = mode
                time.sleep(0.01)
                continue

            # 模式切换时重置
            if mode != last_mode:
                logger.info(f"模式切换: {last_mode} -> {mode}")
                if last_mode == '1m' and mode != '1m':
                    with open('gaozhi.txt', 'w') as f:
                        f.write('0')
                    state.last_value = None
                processed_list = []
                last_mode = mode
                frame_count = 0
                fps_print_count = 5
                # 重置锁定追踪状态
                state.lock_target = None
                state.lock_miss_count = 0
                state.lock_frame_count = 0

            # ---------- 从缓冲区取最新帧（非阻塞）----------
            got, frame, last_seq = frame_buf.get_latest(last_seq)
            if not got:
                time.sleep(0.001)
                continue

            now = time.time()
            fps_times.append(now)
            fps = len(fps_times) / (fps_times[-1] - fps_times[0]) if len(fps_times) > 1 else 0
            if fps_print_count > 0:
                logger.info(f"实时帧率: {fps:.1f} FPS")
                fps_print_count -= 1

            frame_count += 1
            do_inference = (not FRAME_SKIP_ENABLED or frame_count % FRAME_SKIP_N == 0)

            if do_inference:
                canvas = frame.copy()
    
                # 定期解锁检查（推理前，触发则本帧用全图重判）
                if state.lock_target is not None:
                    state.lock_frame_count += 1
                    if state.lock_frame_count >= LOCK_MAX_HIT:
                        logger.info(f"锁定满{LOCK_MAX_HIT}帧，强制重判")
                        state.lock_target = None
                        state.lock_miss_count = 0
                        state.lock_frame_count = 0
    
                is_locked = (state.lock_target is not None)
    
                # --- YOLO 推理 ---
                if is_locked:
                    lock_ox, lock_oy, lock_w, lock_h = state.lock_target
                    sx1, sy1, sx2, sy2, shw, shh, search_area = search_rect(lock_ox, lock_oy, lock_w, lock_h)
                    # 裁剪区域 clamp 到画面边界
                    csx1, csy1 = max(0, int(sx1)), max(0, int(sy1))
                    csx2, csy2 = min(D435_WIDTH, int(sx2)), min(D435_HEIGHT, int(sy2))
                    crop = frame[csy1:csy2, csx1:csx2]
                    # imgsz 根据裁剪尺寸动态计算（向上取整到 32 的倍数）
                    crop_max_dim = max(csx2 - csx1, csy2 - csy1)
                    locked_imgsz = ((crop_max_dim + 31) // 32) * 32
                    t0 = time.time()
                    results = model_locked.predict(source=crop, device='cpu', show=False,
                                                    stream=False, verbose=False, iou=0.45,
                                                    conf=MODEL_LOCKED_CONF, imgsz=locked_imgsz)
                    t1 = time.time()
                    infer_time = int((t1 - t0) * 1000)
                else:
                    unlocked_imgsz = MODEL_UNLOCKED_2M_IMGSZ if mode == '2m' else MODEL_UNLOCKED_1M_IMGSZ
                    unlocked_conf = MODEL_UNLOCKED_2M_CONF if mode == '2m' else MODEL_UNLOCKED_1M_CONF
                    t0 = time.time()
                    results = model_locked.predict(source=frame, device='cpu', show=False,
                                                      stream=False, verbose=False, iou=0.45,
                                                      conf=unlocked_conf, imgsz=unlocked_imgsz)
                    t1 = time.time()
                    infer_time = int((t1 - t0) * 1000)
    
                # --- 解析检测结果 ---
                tong_list = []
                for result in results:
                    boxes = result.boxes
                    names = result.names
                    if boxes is not None:
                        for box in boxes:
                            r = box.xyxy[0].cpu().numpy().astype(int)
                            if is_locked:
                                # 裁剪坐标转全图坐标
                                r[0] += csx1; r[1] += csy1
                                r[2] += csx1; r[3] += csy1
                            ux, uy = draw_square(canvas, box, names, r)
    
                            length = abs(int(r[0] - r[2]))
                            width = abs(int(r[1] - r[3]))
                            area = length * width
    
                            cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 5)
                            cv2.putText(canvas, str([ux, uy]), (ux + 20, uy + 10), 0, 1,
                                        [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    
                            dis = ((ux - D435_WIDTH // 2) ** 2 + (uy - D435_HEIGHT // 2) ** 2) ** 0.5
    
                            if area <= 100000:
                                tong_list.append((dis, int(box.cls[0]), ux, uy, r))
    
                # ---------- 锁定追踪筛选 ----------
                cur_det_count = len(tong_list)
                lock_status = "锁定" if state.lock_target else ""
                logger.debug(f"模式：{mode}{lock_status} 检测数={cur_det_count} 锁帧数={state.lock_frame_count} 丢帧数={state.lock_miss_count}")
    
                if tong_list:
                    if state.lock_target is not None:
                        # --- 有锁定：在搜索矩形内选最近中心 ---
                        lock_ox, lock_oy, lock_w, lock_h = state.lock_target
                        sx1, sy1, sx2, sy2, shw, shh, search_area = search_rect(lock_ox, lock_oy, lock_w, lock_h)
                        draw_lock_rect(canvas, lock_ox, lock_oy, shw, shh, (0, 255, 255))
    
                        candidates = [item for item in tong_list
                                      if sx1 <= item[2] <= sx2 and sy1 <= item[3] <= sy2]
                        best_match = pick_nearest(candidates)
    
                        if best_match is not None:
                            _, _, ux, uy, r = best_match
                            w, h = abs(r[2] - r[0]), abs(r[3] - r[1])
                            state.lock_target = (ux, uy, w, h)
                            state.lock_miss_count = 0
                            processed_list = [str(1), str(ux), str(uy)]
                            state.last_detection_time = time.time()
                            logger.info(f"锁定桶：({ux},{uy}) 目标面积={w*h} 搜索框=({sx1},{sy1},{sx2},{sy2}) 搜索面积={search_area}")
                        else:
                            state.lock_miss_count += 1
                            if state.lock_miss_count > LOCK_MAX_MISS:
                                logger.info(f"丢帧满{LOCK_MAX_MISS}帧，强制重判")
                                state.lock_target = None
                                state.lock_miss_count = 0
                                state.lock_frame_count = 0
                                best = pick_nearest(tong_list)
                                _, _, ux, uy, r = best
                                w, h = abs(r[2] - r[0]), abs(r[3] - r[1])
                                state.lock_target = (ux, uy, w, h)
                                processed_list = [str(1), str(ux), str(uy)]
                                state.last_detection_time = time.time()
                                _, _, _, _, nshw, nshh, narea = search_rect(ux, uy, w, h)
                                logger.info(f"解锁-最近桶：({ux},{uy}) 目标面积={w*h} 搜索框=({ux-nshw},{uy-nshh},{ux+nshw},{uy+nshh}) 搜索面积={narea}")
                            else:
                                processed_list = [str(1), str(lock_ox), str(lock_oy)]
                                logger.info(f"丢帧：({lock_ox},{lock_oy}) miss={state.lock_miss_count} 搜索框=({sx1},{sy1},{sx2},{sy2}) 搜索面积={search_area}")
                    else:
                        # --- 无锁定：选最近中心，建立锁定 ---
                        best = pick_nearest(tong_list)
                        _, _, ux, uy, r = best
                        w, h = abs(r[2] - r[0]), abs(r[3] - r[1])
                        state.lock_target = (ux, uy, w, h)
                        state.lock_miss_count = 0
                        state.lock_frame_count = 0
                        processed_list = [str(1), str(ux), str(uy)]
                        state.last_detection_time = time.time()
                        _, _, _, _, shw, shh, sarea = search_rect(ux, uy, w, h)
                        logger.info(f"最近桶：({ux},{uy}) 目标面积={w*h} 搜索框=({ux-shw},{uy-shh},{ux+shw},{uy+shh}) 搜索面积={sarea}")
                        draw_lock_rect(canvas, ux, uy, shw, shh, (0, 255, 0))
                else:
                    # --- 无任何检测 ---
                    if state.lock_target is not None:
                        lock_ox, lock_oy, lock_w, lock_h = state.lock_target
                        sx1, sy1, sx2, sy2, shw, shh, search_area = search_rect(lock_ox, lock_oy, lock_w, lock_h)
                        draw_lock_rect(canvas, lock_ox, lock_oy, shw, shh, (0, 165, 255))
    
                        state.lock_miss_count += 1
                        if state.lock_miss_count > LOCK_MAX_MISS:
                            logger.info(f"丢帧满{LOCK_MAX_MISS}帧，强制重判")
                            state.lock_target = None
                            state.lock_miss_count = 0
                            state.lock_frame_count = 0
                            if HISTORY_CLEAR_ENABLED and state.last_value is not None \
                                    and time.time() - state.last_detection_time > HISTORY_CLEAR_TIMEOUT:
                                state.last_value = None
                            if state.last_value is not None:
                                processed_list = state.last_value.split()
                            else:
                                processed_list = [str(0)]
                        else:
                            processed_list = [str(1), str(lock_ox), str(lock_oy)]
                            logger.info(f"丢帧(无检测)：({lock_ox},{lock_oy}) miss={state.lock_miss_count} 搜索框=({sx1},{sy1},{sx2},{sy2}) 搜索面积={search_area}")
                    else:
                        if HISTORY_CLEAR_ENABLED and state.last_value is not None \
                                and time.time() - state.last_detection_time > HISTORY_CLEAR_TIMEOUT:
                            state.last_value = None
                            logger.info(f"超过{HISTORY_CLEAR_TIMEOUT}秒未检测到目标，清空历史坐标")
                        if state.last_value is not None:
                            processed_list = state.last_value.split()
                            logger.info(f"上一桶：{processed_list}")
                        else:
                            processed_list = [str(0)]
    
                write_detection(processed_list, mode)
                last_canvas = canvas.copy()

            else:
                if last_canvas is not None:
                    canvas = last_canvas.copy()
                else:
                    canvas = frame.copy()

            # ---------- 显示模式文字和帧率 ----------
            cv2.putText(canvas, f"{mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"FPS:{fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # ---------- 送入显示队列 ----------
            try:
                display_queue.put((canvas, "d435"), block=False)
            except queue.Full:
                pass

    finally:
        cap_stop_event.set()
        capture_thread.join(timeout=1)
        pipeline.stop()
        logger.info("D435摄像头已释放")

# ==================== 检测生命周期管理线程 ====================
def detection_manager_thread(stop_event, cmd_event):
    """根据 cmd_event 状态管理检测线程的启停"""
    detect_stop_event = None
    detect_thread = None

    # 启动前检查 D435 摄像头
    ctx = rs.context()
    if len(ctx.devices) == 0:
        for _ in range(5):
            logger.warning("未检测到D435相机，请检查连接！")
    else:
        logger.info("D435相机已连接。")

    while not stop_event.is_set():
        # 等待启动信号
        cmd_event.wait()
        if stop_event.is_set():
            break

        # 启动检测线程
        if detect_thread is None or not detect_thread.is_alive():
            detect_stop_event = threading.Event()
            detect_thread = threading.Thread(
                target=camera_detection_thread,
                args=(detect_stop_event,)
            )
            detect_thread.start()

        # 等待停止信号
        while cmd_event.is_set() and not stop_event.is_set():
            time.sleep(0.1)

        # 停止检测线程
        if detect_stop_event:
            detect_stop_event.set()
        if detect_thread and detect_thread.is_alive():
            detect_thread.join()
        # 统一清理共享状态，确保无论竞态都执行
        state.last_value = None
        state.last_detection_time = 0
        state.b_zero = True
        state.lock_target = None
        state.lock_miss_count = 0
        state.lock_frame_count = 0
        with open('gaozhi.txt', 'w') as f:
            f.write('0')
        cv2.destroyAllWindows()
        detect_thread = None

    # 程序退出时的最终清理
    if detect_stop_event:
        detect_stop_event.set()
    if detect_thread and detect_thread.is_alive():
        detect_thread.join()
    # 统一清理共享状态，确保无论竞态都执行
    state.last_value = None
    state.last_detection_time = 0
    state.b_zero = True
    state.lock_target = None
    state.lock_miss_count = 0
    state.lock_frame_count = 0
    with open('gaozhi.txt', 'w') as f:
        f.write('0')
    cv2.destroyAllWindows()

# ==================== 主程序入口 ====================
if __name__ == '__main__':
    try:
        logger.info("Yolo26n目标检测-程序启动")
        logger.info("开始Yolo26n模型加载")

        clear_files(FILES_TO_CLEAR)

        main_stop_event = threading.Event()
        cmd_event = threading.Event()

        # 文件监控线程
        t_monitor = threading.Thread(
            target=file_monitor_thread,
            args=(main_stop_event, cmd_event)
        )
        t_monitor.start()

        # 检测管理线程
        t_manager = threading.Thread(
            target=detection_manager_thread,
            args=(main_stop_event, cmd_event)
        )
        t_manager.start()

        logger.info("完成Yolo26n模型加载")
        logger.info("系统初始化完成，等待指令...")

        # ---------- 显示循环 ----------
        window_created = False
        win_name = "d435"

        while True:
            if not display_queue.empty():
                frame, _ = display_queue.get()
                if not window_created:
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL |
                                    cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                    cv2.resizeWindow(win_name, D435_WIDTH, D435_HEIGHT)
                    window_created = True
                cv2.imshow(win_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    main_stop_event.set()
                    break
            else:
                time.sleep(0.01)

        cv2.destroyAllWindows()
        t_manager.join()
        t_monitor.join()

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()