import os
import cv2
import time
import queue
import threading
from ultralytics import YOLO
import numpy as np

# ==================== 参数设置 ====================
files_to_clear = ['data.txt', 'gaozhi.txt']
display_queue = queue.Queue(maxsize=3)

# ==================== USB摄像头参数 ====================
USB_CAM_PATH = '/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0'   # 可根据实际情况修改
USB_WIDTH = 640
USB_HEIGHT = 480
USB_FPS = 25
CALIB_FILE = 'calib_resultA.npz'   # 畸变校正文件
UX,UY = 323, 243 #usb相机中心点

# ==================== 全局变量 ====================
last_value = None
b_zero = True
last_processed_list = []

# ==================== 模型权重加载 ====================
model_1m = YOLO("/home/fu/weights/tongv3.pt")
# model_2m = YOLO("/home/fu/权重/tongv2.pt")

# ==================== 工具函数（与原程序完全相同） ====================
def clear_files(files):
    for file_name in files:
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, 'w') as f:
            print("[INFO]通讯txt文件已建立并清空")
            pass

def read_data_file():
    try:
        with open('data.txt', 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return None

def write_to_file(processed_list, filename='gaozhi.txt'):
    global last_value, b_zero
    b = None
    b_value = read_data_file()

    if b_value == '2m':
        b = 1
    elif b_value == '1m':
        b = 2

    if b_zero and b == 1:
        message = "[INFO]我方即将进入对桶程序※※"
        for _ in range(8):
            print(message)
        b_zero = False

    if b == 2:
        if processed_list:
            content = ' '.join(processed_list)
            with open(filename, 'w') as f:
                f.write(content)
            last_value = content.strip()
        else:
            processed_list = [str(0)]
            content = ' '.join(processed_list)
            with open(filename, 'w') as f:
                f.write(content)
            return last_value

    if b == 1:
        if processed_list:
            content = ' '.join(processed_list)
            with open(filename, 'w') as f:
                f.write(content)
            if content.strip() != '0':
                last_value = content.strip()
        else:
            if last_value is not None:
                with open(filename, 'w') as f:
                    f.write(last_value + '\n')
                print(last_value)
            if last_value is not None and not os.path.getsize(filename):
                return last_value
        return None

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

# def monitor_file():
#     print("开始监控文件...")
#     while True:
#         try:
#             with open('data.txt', 'r') as file:
#                 print(f"data.txt 内容:{file.read().strip()}", flush=True)
#         except:
#             pass
#         time.sleep(1)

def read_file_content(shuru_file_path, start_event, stop_event):
    global c
    last_result = None
    while not stop_event.is_set():
        try:
            with open(shuru_file_path, 'r') as file:
                content = file.read().strip()
                if content != last_result:
                    if content in ['2m', '1m']:
                        for i in range(7):
                            print(f"[INFO] 检测到{content}模式")
                        start_event.set()
                        c = 1
                    elif content == '0':
                        print("[INFO] 检测到0模式，关闭摄像头")
                        c = 0
                    else:
                        print("[INFO] 未检测到有效内容，等待...")
                    last_result = content
        except:
            pass
        time.sleep(0.05)


# ==================== USB相机检测线程（含畸变校正 + 相机状态显示 + 视频保存） ====================
def usb_detect(shuchu_file_path, stop_event, start_event, calib_file=None):
    """核心检测线程：使用USB相机 + 可选畸变校正，并实时录制分段视频"""
    global processed_list, last_processed_list, last_value

    # ---------- 打开USB摄像头 ----------
    cap = cv2.VideoCapture(USB_CAM_PATH, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, USB_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USB_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, USB_FPS)

    if not cap.isOpened():
        print("[ERROR] 无法打开USB摄像头，请检查设备路径！")
        start_event.wait()
        return

    print(f"[INFO] USB摄像头已打开: {USB_WIDTH}x{USB_HEIGHT} @ {USB_FPS}fps")

    # ---------- 加载畸变校正 ----------
    mapx, mapy = None, None
    do_undistort = False
    if calib_file and os.path.exists(calib_file):
        try:
            data = np.load(calib_file)
            mtx = data['mtx']
            dist = data['dist']
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (USB_WIDTH, USB_HEIGHT), 0, (USB_WIDTH, USB_HEIGHT))
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (USB_WIDTH, USB_HEIGHT), 5)
            do_undistort = True
            print(f"[INFO] 畸变校正已启用（来自 {calib_file}）")
        except Exception as e:
            print(f"[ERROR] 加载标定文件失败: {e}")
    else:
        print(f"[WARN] 标定文件 {calib_file} 不存在，跳过畸变校正。")

    start_event.wait()
    print("[DEBUG] usb_detect线程已启动")


    # 变量初始化
    processed_list = []
    last_processed_list = []
    last_mode = None

    try:
        while not stop_event.is_set():
            data_content = read_data_file()

            if data_content in ['1m','2m']:
                model = model_1m
            else:
                continue

            # 模式切换时的重置操作
            if data_content != last_mode:
                print(f"[INFO] 模式切换: {last_mode} -> {data_content}")
                if last_mode == '1m' and data_content != '1m':
                    with open('gaozhi.txt', 'w') as f:
                        f.write('0')
                    last_value = None
                processed_list = []
                last_processed_list = []
                last_mode = data_content

            # ---------- 读取一帧并校正 ----------
            ret, frame = cap.read()
            if not ret:
                print("[WARN] USB摄像头读取帧失败")
                continue

            if do_undistort:
                frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

            
            results = model.predict(source=frame, device='cpu', show=False,
                                    stream=False, verbose=False, iou=0.45, conf=0.5)

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
                        ux =  int(ux*1.325)
                        # uy = 

                        cv2.circle(canvas, (ox, oy), 4, (255, 255, 255), 5)
                        cv2.putText(canvas, str([ux, uy]), (ox + 20, oy + 10), 0, 1,
                                    [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                        dis = ((ox - 320) ** 2 + (oy - 240) ** 2) ** 0.5

                        class_id = int(box.cls[0])
                        if area <= 70000:
                            tong_list.append((dis, class_id, ux, uy))

            cv2.putText(canvas, f"{data_content}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # 筛选策略
            print(f"模式：{data_content}")
            if data_content in ['1m','2m']:
                
                if tong_list:
                    min_dis = min(item[0] for item in tong_list)
                    for dis, cls_id, ux, uy in tong_list:
                        if dis == min_dis:
                            print(f"最近桶：({ux},{uy})")
                            processed_list = [str(1), str(ux), str(uy)]
                            break
                else:
                    if last_value is not None:
                        processed_list = last_value.split()
                        print(f"上一桶：{processed_list}")
                    else:
                        processed_list = [str(0)]
                write_to_file(processed_list)

            # 放入显示队列
            try:
                display_queue.put((canvas, "chaoqian"), block=False)
            except queue.Full:
                pass

    finally:
        cap.release()
        print("[INFO] USB摄像头已释放")

# ==================== 主控制线程 ====================
def main_control():
    global c, start_event
    last_c = None
    running = False
    stop_event = None
    t_detect = None

    # 检测USB摄像头
    test_cap = cv2.VideoCapture(USB_CAM_PATH, cv2.CAP_V4L2)
    if test_cap.isOpened():
        print(f"\033[32m[INFO] USB摄像头 {USB_CAM_PATH} 已连接。\033[0m")
        test_cap.release()
    else:
        for _ in range(5):
            print(f"\033[31m[WARN] 未检测到USB摄像头，请检查连接！\033[0m")

    while True:
        if c != last_c:
            print(f"[DEBUG] main_control 检测到 c 变化: {last_c} -> {c}")

            if running:
                stop_event.set()
                if t_detect:
                    t_detect.join()
                cv2.destroyAllWindows()
                time.sleep(0.2)
                running = False

            stop_event = threading.Event()
            if c == 1:
                # 传入标定文件路径，启用畸变校正
                t_detect = threading.Thread(target=usb_detect, args=('gaozhi.txt', stop_event, start_event, CALIB_FILE))
                t_detect.start()
                running = True

            last_c = c
        time.sleep(0.1)

# ==================== 主程序入口 ====================
if __name__ == '__main__':
    try:
        print("[INFO] Yolo26n目标检测-程序启动")
        print("[INFO] 开始Yolo26n模型加载")

        c = 0
        clear_files(files_to_clear)

        shuru_file_path = 'data.txt'
        shuchu_file_path = 'gaozhi.txt'
        start_event = threading.Event()
        stop_event = threading.Event()

        # monitor_thread = threading.Thread(target=monitor_file)
        # monitor_thread.daemon = True
        # monitor_thread.start()

        t1 = threading.Thread(target=read_file_content, args=(shuru_file_path, start_event, stop_event))
        t1.start()

        main_thread = threading.Thread(target=main_control)
        main_thread.start()

        print("[INFO] 完成Yolo26n模型加载")
        print("[INFO] 系统初始化完成，等待指令...")

        window_created = False
        while True:
            if not display_queue.empty():
                frame, window_name = display_queue.get()
                name = "chaoqian"
                if not window_created:
                    cv2.namedWindow(name, cv2.WINDOW_NORMAL |
                                    cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                    cv2.resizeWindow(name, 640, 480)
                    window_created = True
                cv2.imshow(name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break
            else:
                time.sleep(0.01)

        cv2.destroyAllWindows()
        main_thread.join()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()