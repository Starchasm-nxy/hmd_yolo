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
MODEL_PATH = "/home/fu/weights/tongv3.pt"

# ==================== 相机曝光参数 ====================
AUTO_EXPOSURE = 1       # 0 = 自动曝光, 1 = 手动曝光
EXPOSURE_VALUE = 300    # 手动曝光值（数值越小曝光越短）

# ==================== 模型加载 ====================
model = YOLO(MODEL_PATH)

# ==================== 线程间共享状态 ====================
class SharedState:
    def __init__(self):
        self.last_value = None      # 上一次写入的有效检测值
        self.b_zero = True          # 是否首次进入对桶模式（2m）

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

def write_detection(processed_list):
    """根据当前模式将检测结果写入 gaozhi.txt"""
    mode = read_data_file()

    if mode == '2m':
        _write_duitong_mode(processed_list)
    elif mode == '1m':
        _write_normal_mode(processed_list)

def _write_duitong_mode(processed_list):
    """2m 模式（对桶程序）"""
    if state.b_zero:
        msg = "[INFO]我方即将进入对桶程序※※"
        for _ in range(8):
            print(msg)
        state.b_zero = False

    if processed_list:
        content = ' '.join(processed_list)
        with open('gaozhi.txt', 'w') as f:
            f.write(content)
        if content.strip() != '0':
            state.last_value = content.strip()
    elif state.last_value is not None:
        with open('gaozhi.txt', 'w') as f:
            f.write(state.last_value + '\n')
        print(state.last_value)

def _write_normal_mode(processed_list):
    """1m 模式"""
    if processed_list:
        content = ' '.join(processed_list)
        with open('gaozhi.txt', 'w') as f:
            f.write(content)
        state.last_value = content.strip()
    else:
        with open('gaozhi.txt', 'w') as f:
            f.write('0')

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
                for _ in range(7):
                    print(f"[INFO] 检测到{content}模式")
                cmd_event.set()
            elif content == '0':
                print("[INFO] 检测到0模式，关闭摄像头")
                cmd_event.clear()
            else:
                print("[INFO] 未检测到有效内容，等待...")
            last_content = content

        time.sleep(0.05)

# ==================== 相机检测线程 ====================
def camera_detection_thread(stop_event, calib_file=None):
    """USB 相机检测：畸变校正 + YOLO 检测 + 结果写入"""
    # ---------- 打开摄像头 ----------
    cap = cv2.VideoCapture(USB_CAM_PATH, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, USB_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USB_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE)  # 0=自动, 1=手动
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_VALUE)

    if not cap.isOpened():
        print("[ERROR] 无法打开USB摄像头，请检查设备路径！")
        return

    # ---------- 加载畸变校正 ----------
    mapx, mapy = None, None
    do_undistort = False
    if calib_file and os.path.exists(calib_file):
        try:
            data = np.load(calib_file)
            mtx = data['mtx']
            dist = data['dist']
            newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
                mtx, dist, (USB_WIDTH, USB_HEIGHT), 0, (USB_WIDTH, USB_HEIGHT))
            mapx, mapy = cv2.initUndistortRectifyMap(
                mtx, dist, None, newcameramtx, (USB_WIDTH, USB_HEIGHT), 5)
            do_undistort = True
            print(f"[INFO] 畸变校正已启用（来自 {calib_file}）")
        except Exception as e:
            print(f"[ERROR] 加载标定文件失败: {e}")
    else:
        print(f"[WARN] 标定文件 {calib_file} 不存在，跳过畸变校正。")

    print("[DEBUG] usb_detect线程已启动")

    processed_list = []
    last_mode = None

    try:
        while not stop_event.is_set():
            mode = read_data_file()

            if mode not in ['1m', '2m']:
                time.sleep(0.01)
                continue

            # 模式切换时重置
            if mode != last_mode:
                print(f"[INFO] 模式切换: {last_mode} -> {mode}")
                if last_mode == '1m' and mode != '1m':
                    with open('gaozhi.txt', 'w') as f:
                        f.write('0')
                    state.last_value = None
                processed_list = []
                last_mode = mode

            # ---------- 读取一帧 ----------
            ret, frame = cap.read()
            if not ret:
                print("[WARN] USB摄像头读取帧失败")
                time.sleep(0.01)
                continue

            if do_undistort:
                frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

            # ---------- YOLO 检测 ----------
            results = model.predict(source=frame, device='cpu', show=False,
                                    stream=False, verbose=False, iou=0.45, conf=0.6)

            tong_list = []
            canvas = frame.copy()

            for result in results:
                boxes = result.boxes
                names = result.names
                if boxes is not None:
                    for box in boxes:
                        r = box.xyxy[0].cpu().numpy().astype(int)
                        ux, uy = draw_square(canvas, box, names, r)

                        length = abs(int(r[0] - r[2]))
                        width = abs(int(r[1] - r[3]))
                        area = length * width

                        ox = ux
                        oy = uy
                        ux = int(ux * 1.325)

                        cv2.circle(canvas, (ox, oy), 4, (255, 255, 255), 5)
                        cv2.putText(canvas, str([ux, uy]), (ox + 20, oy + 10), 0, 1,
                                    [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                        dis = ((ox - 320) ** 2 + (oy - 240) ** 2) ** 0.5

                        if area <= 70000:
                            tong_list.append((dis, int(box.cls[0]), ux, uy))

            cv2.putText(canvas, f"{mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # ---------- 筛选最近桶 ----------
            print(f"模式：{mode}")
            if tong_list:
                min_dis = min(item[0] for item in tong_list)
                for dis, _, ux, uy in tong_list:
                    if dis == min_dis:
                        print(f"最近桶：({ux},{uy})")
                        processed_list = [str(1), str(ux), str(uy)]
                        break
            else:
                if state.last_value is not None:
                    processed_list = state.last_value.split()
                    print(f"上一桶：{processed_list}")
                else:
                    processed_list = [str(0)]

            write_detection(processed_list)

            # ---------- 送入显示队列 ----------
            try:
                display_queue.put((canvas, "chaoqian"), block=False)
            except queue.Full:
                pass

    finally:
        cap.release()
        print("[INFO] USB摄像头已释放")

# ==================== 检测生命周期管理线程 ====================
def detection_manager_thread(stop_event, cmd_event):
    """根据 cmd_event 状态管理检测线程的启停"""
    detect_stop_event = None
    detect_thread = None

    # 启动前检查 USB 摄像头
    test_cap = cv2.VideoCapture(USB_CAM_PATH, cv2.CAP_V4L2)
    if test_cap.isOpened():
        print(f"\033[32m[INFO] USB摄像头 {USB_CAM_PATH} 已连接。\033[0m")
        test_cap.release()
    else:
        for _ in range(5):
            print(f"\033[31m[WARN] 未检测到USB摄像头，请检查连接！\033[0m")

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
                args=(detect_stop_event, CALIB_FILE)
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
        cv2.destroyAllWindows()
        detect_thread = None

    # 程序退出时的最终清理
    if detect_stop_event:
        detect_stop_event.set()
    if detect_thread and detect_thread.is_alive():
        detect_thread.join()
    cv2.destroyAllWindows()

# ==================== 主程序入口 ====================
if __name__ == '__main__':
    try:
        print("[INFO] Yolo26n目标检测-程序启动")
        print("[INFO] 开始Yolo26n模型加载")

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

        print("[INFO] 完成Yolo26n模型加载")
        print("[INFO] 系统初始化完成，等待指令...")

        # ---------- 显示循环 ----------
        window_created = False
        win_name = "chaoqian"

        while True:
            if not display_queue.empty():
                frame, _ = display_queue.get()
                if not window_created:
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL |
                                    cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
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
