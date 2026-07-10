import os
import cv2
import time
import queue
import threading
from ultralytics import YOLO
import numpy as np

# ==================== 配置 ====================
FILES_TO_CLEAR = ['data.txt', 'gaozhi.txt']
display_queue = queue.Queue(maxsize=3)

USB_CAM_PATH = '/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0'
USB_WIDTH = 640
USB_HEIGHT = 480
CALIB_FILE = 'calib_resultA.npz'
MODEL_LOCKED_PATH = "/home/fu/weights/tongv3.pt"
MODEL_UNLOCKED_PATH = "/home/fu/weights/tongv3.pt"
USB_MAX_FPS = 60

# ==================== 相机曝光参数 ====================
AUTO_EXPOSURE = 1
EXPOSURE_VALUE = 500

# ==================== 历史坐标超时清空 ====================
HISTORY_CLEAR_ENABLED = 1
HISTORY_CLEAR_TIMEOUT = 10.0

# ==================== 锁定追踪参数 ====================
LOCK_MAX_HIT = 15
LOCK_MAX_MISS = 7
LOCK_SEARCH_RATIO = 2.5
LOCK_MIN_SEARCH_RADIUS = 110
LOCK_MAX_SEARCH_RADIUS = 180

# ==================== YOLO 推理参数 ====================
MODEL_LOCKED_CONF = 0.5
MODEL_UNLOCKED_2M_IMGSZ = 640; MODEL_UNLOCKED_1M_IMGSZ = 640
MODEL_UNLOCKED_2M_CONF = 0.5; MODEL_UNLOCKED_1M_CONF = 0.5

# ==================== 模型加载 ====================
model_locked = YOLO(MODEL_LOCKED_PATH)
model_unlocked = YOLO(MODEL_UNLOCKED_PATH)

# ==================== 线程间共享状态 ====================
class SharedState:
    def __init__(self):
        self.last_value = None
        self.last_detection_time = 0
        self.b_zero = True
        self.lock_target = None
        self.lock_miss_count = 0
        self.lock_frame_count = 0
        self.last_detection_count = 0
        self.last_search_rect = None
        self.last_processed_list = None
        self.last_mode = None
        self.last_boxes_info = []

state = SharedState()

# ==================== 工具函数 ====================
def clear_files(files):
    for file_name in files:
        with open(file_name, 'w'):
            pass
    print("[INFO] 通讯txt文件已建立并清空")

def read_data_file():
    try:
        with open('data.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def write_detection(processed_list, mode):
    if mode == '2m' and state.b_zero:
        for _ in range(8):
            print("[INFO]我方即将进入对桶程序※※")
        state.b_zero = False

    if not processed_list:
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

    if mode == '2m' and content.strip() == '0':
        return
    state.last_value = content.strip()

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def search_rect(ox, oy, w, h):
    shw = clamp(w * LOCK_SEARCH_RATIO / 2, LOCK_MIN_SEARCH_RADIUS, LOCK_MAX_SEARCH_RADIUS)
    shh = clamp(h * LOCK_SEARCH_RATIO / 2, LOCK_MIN_SEARCH_RADIUS, LOCK_MAX_SEARCH_RADIUS)
    sx1, sy1 = int(ox - shw), int(oy - shh)
    sx2, sy2 = int(ox + shw), int(oy + shh)
    return sx1, sy1, sx2, sy2, int(shw), int(shh), (sx2 - sx1) * (sy2 - sy1)

def pick_nearest(items):
    if not items:
        return None
    return min(items, key=lambda x: x[0])

def draw_overlay(frame, state, processed_list, is_locked, lock_ox, lock_oy, lock_w, lock_h,
                 search_rect_coords, mode, boxes_info):
    canvas = frame.copy()

    # 绘制所有目标框
    if boxes_info:
        for (r, cls, conf, names) in boxes_info:
            x1, y1, x2, y2 = r
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (221, 185, 193), 2)
            label = f"{names[cls]} {conf:.2f}"
            cv2.putText(canvas, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (176, 196, 222), 2)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(canvas, (cx, cy), 5, (240, 240, 240), -1)

    # 锁定图形
    if is_locked:
        if search_rect_coords is not None:
            sx1, sy1, sx2, sy2 = search_rect_coords
            shw = (sx2 - sx1) // 2
            shh = (sy2 - sy1) // 2
            cv2.rectangle(canvas, (int(lock_ox - shw), int(lock_oy - shh)),
                          (int(lock_ox + shw), int(lock_oy + shh)), (0, 255, 255), 2)
        cv2.circle(canvas, (int(lock_ox), int(lock_oy)), 4, (0, 255, 255), -1)

    # 输出坐标
    if processed_list and len(processed_list) >= 3 and processed_list[0] == '1':
        try:
            ux = int(processed_list[1])
            uy = int(processed_list[2])
            ox = int(ux / 1.325)
            oy = uy
            cv2.circle(canvas, (ox, oy), 5, (0, 255, 0), -1)
            cv2.putText(canvas, f"[{ux}, {uy}]", (ox + 20, oy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (225, 255, 255), 2, cv2.LINE_AA)
        except:
            pass

    cv2.putText(canvas, mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return canvas

# ==================== 文件监控线程 ====================
def file_monitor_thread(stop_event, cmd_event):
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
                print(f"[INFO] 检测到{content}模式")
                cmd_event.set()
            elif content == '0':
                print("[INFO] 检测到0模式，关闭摄像头")
                cmd_event.clear()
            else:
                print("[INFO] 未检测到有效内容，等待...")
            last_content = content
        time.sleep(0.05)

# ==================== 帧缓冲区 ====================
class FrameBuffer:
    def __init__(self):
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()

    def put(self, frame):
        with self._lock:
            self._frame = frame
            self._seq += 1

    def get_latest(self, seen_seq=0):
        with self._lock:
            if self._seq <= seen_seq or self._frame is None:
                return False, None, self._seq
            return True, self._frame.copy(), self._seq

def camera_capture_loop(cap, frame_buf, stop_event, mapx=None, mapy=None):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            if mapx is not None and mapy is not None:
                frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            frame_buf.put(frame)

# ==================== 检测线程 ====================
def camera_detection_thread(stop_event, calib_file=None):
    cap = cv2.VideoCapture(USB_CAM_PATH, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, USB_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USB_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE)
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_VALUE)
    cap.set(cv2.CAP_PROP_FPS, USB_MAX_FPS)

    if not cap.isOpened():
        print("[ERROR] 无法打开USB摄像头，请检查设备路径！")
        return

    mapx, mapy = None, None
    if calib_file and os.path.exists(calib_file):
        try:
            data = np.load(calib_file)
            mtx = data['mtx']
            dist = data['dist']
            newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
                mtx, dist, (USB_WIDTH, USB_HEIGHT), 0, (USB_WIDTH, USB_HEIGHT))
            mapx, mapy = cv2.initUndistortRectifyMap(
                mtx, dist, None, newcameramtx, (USB_WIDTH, USB_HEIGHT), 5)
            print(f"[INFO] 畸变校正已启用")
        except Exception as e:
            print(f"[ERROR] 加载标定文件失败: {e}")

    frame_buf = FrameBuffer()
    cap_stop_event = threading.Event()
    capture_thread = threading.Thread(
        target=camera_capture_loop,
        args=(cap, frame_buf, cap_stop_event, mapx, mapy),
        daemon=True
    )
    capture_thread.start()

    print("[DEBUG] usb_detect线程已启动")

    processed_list = []
    last_mode = None
    last_seq = 0
    frame_count = 0
    frame_skip = 2

    fps_last_time = time.time()
    fps_frame_cnt = 0

    try:
        while not stop_event.is_set():
            mode = read_data_file()

            if mode not in ['1m', '2m']:
                state.last_value = None
                state.last_detection_time = 0
                state.b_zero = True
                last_mode = mode
                time.sleep(0.01)
                continue

            if mode != last_mode:
                print(f"[INFO] 模式切换: {last_mode} -> {mode}")
                if last_mode == '1m' and mode != '1m':
                    with open('gaozhi.txt', 'w') as f:
                        f.write('0')
                    state.last_value = None
                processed_list = []
                last_mode = mode
                frame_count = 0
                state.lock_target = None
                state.lock_miss_count = 0
                state.lock_frame_count = 0
                state.last_detection_count = 0
                state.last_search_rect = None
                state.last_processed_list = None
                state.last_boxes_info = []

            got, frame, last_seq = frame_buf.get_latest(last_seq)
            if not got:
                time.sleep(0.001)
                continue

            frame_count += 1
            do_inference = (frame_count % frame_skip == 0)

            fps_frame_cnt += 1
            if time.time() - fps_last_time >= 1.0:
                fps_val = fps_frame_cnt / (time.time() - fps_last_time)
                print(f"帧率={fps_val:.1f} fps")
                fps_last_time = time.time()
                fps_frame_cnt = 0

            if do_inference:
                # 解锁条件
                if state.lock_target is not None:
                    state.lock_frame_count += 1
                    if state.lock_frame_count >= LOCK_MAX_HIT:
                        print(f"[INFO] 锁定满{LOCK_MAX_HIT}帧，强制重判")
                        state.lock_target = None
                        state.lock_miss_count = 0
                        state.lock_frame_count = 0

                is_locked = (state.lock_target is not None)

                if is_locked:
                    lock_ox, lock_oy, lock_w, lock_h = state.lock_target
                    sx1, sy1, sx2, sy2, shw, shh, _ = search_rect(lock_ox, lock_oy, lock_w, lock_h)
                    csx1 = max(0, int(sx1)); csy1 = max(0, int(sy1))
                    csx2 = min(USB_WIDTH, int(sx2)); csy2 = min(USB_HEIGHT, int(sy2))
                    crop = frame[csy1:csy2, csx1:csx2]
                    crop_max_dim = max(csx2 - csx1, csy2 - csy1)
                    locked_imgsz = ((crop_max_dim + 31) // 32) * 32
                    t0 = time.time()
                    results = model_locked.predict(source=crop, device='cpu', show=False,
                                                   stream=False, verbose=False, iou=0.45,
                                                   conf=MODEL_LOCKED_CONF, imgsz=locked_imgsz)
                    print(f"锁定推理-裁剪({csx1},{csy1},{csx2},{csy2}) imgsz={locked_imgsz} 耗时={int((time.time()-t0)*1000)}ms")
                else:
                    unlocked_imgsz = MODEL_UNLOCKED_2M_IMGSZ if mode == '2m' else MODEL_UNLOCKED_1M_IMGSZ
                    unlocked_conf = MODEL_UNLOCKED_2M_CONF if mode == '2m' else MODEL_UNLOCKED_1M_CONF
                    t0 = time.time()
                    results = model_unlocked.predict(source=frame, device='cpu', show=False,
                                                     stream=False, verbose=False, iou=0.45,
                                                     conf=unlocked_conf, imgsz=unlocked_imgsz)
                    print(f"重判推理-全图 imgsz={unlocked_imgsz} 耗时={int((time.time()-t0)*1000)}ms")

                tong_list = []
                boxes_info_list = []

                for result in results:
                    boxes = result.boxes
                    names = result.names
                    if boxes is not None:
                        for box in boxes:
                            r = box.xyxy[0].cpu().numpy().astype(int)
                            if is_locked:
                                r[0] += csx1; r[1] += csy1
                                r[2] += csx1; r[3] += csy1

                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            boxes_info_list.append((r.copy(), cls, conf, names))

                            length = abs(r[2] - r[0])
                            width = abs(r[3] - r[1])
                            area = length * width

                            ox = (r[0] + r[2]) // 2
                            oy = (r[1] + r[3]) // 2
                            ux = int(ox * 1.325)
                            uy = oy   # ✅ 修复：定义 uy

                            dis = ((ox - 320)**2 + (oy - 240)**2)**0.5
                            if area <= 100000:
                                tong_list.append((dis, cls, ux, uy, ox, oy, r))

                cur_det_count = len(tong_list)
                print(f"模式：{mode} 检测数={cur_det_count}")

                # ---------- 锁定追踪 ----------
                if tong_list:
                    if state.lock_target is not None:
                        lock_ox, lock_oy, lock_w, lock_h = state.lock_target
                        sx1, sy1, sx2, sy2, _, _, _ = search_rect(lock_ox, lock_oy, lock_w, lock_h)
                        candidates = [item for item in tong_list
                                      if sx1 <= item[4] <= sx2 and sy1 <= item[5] <= sy2]
                        best_match = pick_nearest(candidates)
                        if best_match:
                            _, _, ux, uy, ox, oy, r = best_match
                            w = abs(r[2] - r[0]); h = abs(r[3] - r[1])
                            state.lock_target = (ox, oy, w, h)
                            state.lock_miss_count = 0
                            processed_list = [str(1), str(ux), str(uy)]
                            state.last_detection_time = time.time()
                        else:
                            state.lock_miss_count += 1
                            lock_ux = int(lock_ox * 1.325)
                            if state.lock_miss_count > LOCK_MAX_MISS:
                                state.lock_target = None
                                state.lock_miss_count = 0
                                state.lock_frame_count = 0
                                best = pick_nearest(tong_list)
                                _, _, ux, uy, ox, oy, r = best
                                w = abs(r[2] - r[0]); h = abs(r[3] - r[1])
                                state.lock_target = (ox, oy, w, h)
                                processed_list = [str(1), str(ux), str(uy)]
                                state.last_detection_time = time.time()
                            else:
                                processed_list = [str(1), str(lock_ux), str(lock_oy)]
                    else:
                        best = pick_nearest(tong_list)
                        _, _, ux, uy, ox, oy, r = best
                        w = abs(r[2] - r[0]); h = abs(r[3] - r[1])
                        state.lock_target = (ox, oy, w, h)
                        state.lock_miss_count = 0
                        state.lock_frame_count = 0
                        processed_list = [str(1), str(ux), str(uy)]
                        state.last_detection_time = time.time()
                else:
                    if state.lock_target is not None:
                        lock_ox, lock_oy, lock_w, lock_h = state.lock_target
                        _, _, _, _, _, _, _ = search_rect(lock_ox, lock_oy, lock_w, lock_h)
                        state.lock_miss_count += 1
                        lock_ux = int(lock_ox * 1.325)
                        if state.lock_miss_count > LOCK_MAX_MISS:
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
                            processed_list = [str(1), str(lock_ux), str(lock_oy)]
                    else:
                        if HISTORY_CLEAR_ENABLED and state.last_value is not None \
                                and time.time() - state.last_detection_time > HISTORY_CLEAR_TIMEOUT:
                            state.last_value = None
                        if state.last_value is not None:
                            processed_list = state.last_value.split()
                        else:
                            processed_list = [str(0)]

                state.last_detection_count = cur_det_count
                if state.lock_target is not None:
                    sx1, sy1, sx2, sy2, _, _, _ = search_rect(*state.lock_target)
                    state.last_search_rect = (sx1, sy1, sx2, sy2)
                else:
                    state.last_search_rect = None
                state.last_processed_list = processed_list.copy()
                state.last_mode = mode
                state.last_boxes_info = boxes_info_list

                is_locked_now = (state.lock_target is not None)
                if is_locked_now:
                    lock_ox, lock_oy, lock_w, lock_h = state.lock_target
                else:
                    lock_ox = lock_oy = lock_w = lock_h = 0
                canvas = draw_overlay(frame, state, processed_list, is_locked_now,
                                      lock_ox, lock_oy, lock_w, lock_h,
                                      state.last_search_rect, mode, boxes_info_list)
                write_detection(processed_list, mode)
                try:
                    display_queue.put((canvas, "usb"), block=False)
                except queue.Full:
                    pass
            else:
                # 跳帧重绘
                if state.last_processed_list is not None:
                    is_locked_now = (state.lock_target is not None)
                    if is_locked_now:
                        lock_ox, lock_oy, lock_w, lock_h = state.lock_target
                    else:
                        lock_ox = lock_oy = lock_w = lock_h = 0
                    canvas = draw_overlay(frame, state, state.last_processed_list,
                                          is_locked_now, lock_ox, lock_oy, lock_w, lock_h,
                                          state.last_search_rect,
                                          state.last_mode if state.last_mode else mode,
                                          state.last_boxes_info)
                else:
                    canvas = frame.copy()
                    cv2.putText(canvas, mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2, cv2.LINE_AA)
                try:
                    display_queue.put((canvas, "usb"), block=False)
                except queue.Full:
                    pass

    finally:
        cap_stop_event.set()
        capture_thread.join(timeout=1)
        cap.release()
        print("[INFO] USB摄像头已释放")

# ==================== 管理线程 ====================
def detection_manager_thread(stop_event, cmd_event):
    detect_stop_event = None
    detect_thread = None

    test_cap = cv2.VideoCapture(USB_CAM_PATH, cv2.CAP_V4L2)
    if test_cap.isOpened():
        print(f"\033[32m[INFO] USB摄像头 {USB_CAM_PATH} 已连接。\033[0m")
        test_cap.release()
    else:
        for _ in range(5):
            print(f"\033[31m[WARN] 未检测到USB摄像头，请检查连接！\033[0m")

    while not stop_event.is_set():
        cmd_event.wait()
        if stop_event.is_set():
            break

        if detect_thread is None or not detect_thread.is_alive():
            detect_stop_event = threading.Event()
            detect_thread = threading.Thread(
                target=camera_detection_thread,
                args=(detect_stop_event, CALIB_FILE)
            )
            detect_thread.start()

        while cmd_event.is_set() and not stop_event.is_set():
            time.sleep(0.1)

        if detect_stop_event:
            detect_stop_event.set()
        if detect_thread and detect_thread.is_alive():
            detect_thread.join()
        cv2.destroyAllWindows()
        detect_thread = None

    if detect_stop_event:
        detect_stop_event.set()
    if detect_thread and detect_thread.is_alive():
        detect_thread.join()
    cv2.destroyAllWindows()

# ==================== 主函数 ====================
if __name__ == '__main__':
    try:
        print("[INFO] Yolo26n目标检测-程序启动")
        clear_files(FILES_TO_CLEAR)

        main_stop_event = threading.Event()
        cmd_event = threading.Event()

        t_monitor = threading.Thread(target=file_monitor_thread, args=(main_stop_event, cmd_event))
        t_monitor.start()

        t_manager = threading.Thread(target=detection_manager_thread, args=(main_stop_event, cmd_event))
        t_manager.start()

        print("[INFO] 系统初始化完成，等待指令...")

        window_created = False
        win_name = "usb"
        while True:
            if not display_queue.empty():
                frame, _ = display_queue.get()
                if not window_created:
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                    cv2.resizeWindow(win_name, 640, 480)
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
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()