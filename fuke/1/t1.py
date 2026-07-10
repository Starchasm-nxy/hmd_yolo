import os
import cv2
import time
import queue
import threading
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs

## ==================== 参数设置 ====================
files_to_clear = ['data.txt', 'gaozhi.txt']
display_queue = queue.Queue(maxsize=3)

# ==================== 全局变量 ====================
last_value = None
b_zero = True
last_processed_list = []

# ==================== 模型权重加载 ====================
model_1m = YOLO("/home/luck/yolov5_d435i_detection-main/2m.pt")
model_2m = YOLO("/home/luck/yolov5_d435i_detection-main/2m.pt")

# ==================== 工具函数 ====================
def clear_files(files):
    """清空指定的文件，确保通信文件初始为空"""
    for file_name in files:
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, 'w') as f:
            print("[INFO]通讯txt文件已建立并清空")
            pass

def read_data_file():
    """读取 data.txt 文件内容（去除首尾空白），若文件不存在返回 None"""
    try:
        with open('data.txt', 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return None

def write_to_file(processed_list, filename='gaozhi.txt'):
    """
    根据当前 data.txt 中的模式（b_value），决定如何写入坐标。
    processed_list: 包含坐标信息的列表，如 ['1', 'ux', 'uy'] 或 ['0']
    """
    global last_value, b_zero
    b = None
    b_value = read_data_file()

    # 将 data.txt 中的命令映射为内部模式编号
    if b_value in ['r', 'm', 'l']:
        b = 1          # 2米选桶模式
    elif b_value == '1m':
        b = 2          # 1米锁定模式

    # 首次进入选桶模式时打印提示（仅一次）
    if b_zero and b == 1:
        message = "[INFO]我方即将进入对桶程序※※"
        for _ in range(8):
            print(message)
        b_zero = False

    # 1米模式（锁定）：直接写入当前检测结果
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

    # 选桶模式（r/m/l）
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

# ==================== 绘图函数 ====================
def draw_square(image, box, names, r):
    """在图像上绘制目标框、类别标签和中心点，返回中心像素坐标"""
    ux = int((r[0] + r[2]) / 2)
    uy = int((r[1] + r[3]) / 2)
    cls = int(box.cls[0])
    conf = box.conf[0]
    label = f"{names[cls]} {conf:.2f}"
    cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (221, 185, 193), 2)
    cv2.putText(image, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (176, 196, 222), 2)
    cv2.circle(image, (ux, uy), 5, (240, 240, 240), -1)
    return ux, uy

# ==================== 监控 data.txt 文件内容（调试用） ====================
def monitor_data_file():
    """持续打印 data.txt 的内容，用于调试观察"""
    print("开始监控data.txt文件...")
    while True:
        try:
            with open('data.txt', 'r') as file:
                print(f"data.txt 内容: {file.read().strip()}", flush=True)
        except:
            pass
        time.sleep(0.25)

# ==================== 文件内容监听线程 ====================
def read_file_content(shuru_file_path, start_event, stop_event):
    """
    监听 data.txt 的变化，当出现 'r','m','l','1m','0' 时，
    设置全局变量 c = 1 并触发 start_event，启动检测线程。
    """
    global c
    last_result = None

    while not stop_event.is_set():
        try:
            with open(shuru_file_path, 'r') as file:
                content = file.read().strip()
                if content != last_result:
                    if content in ['r', 'm', 'l', '1m', '0']:
                        for i in range(8):
                            print(f"[INFO] 检测到{content}模式")
                        start_event.set()
                        c = 1
                    else:
                        print("[INFO] 未检测到有效内容，等待...")
                    last_result = content
        except:
            pass
        time.sleep(0.05)

# [整理] 提取公共选桶逻辑，避免 r/m/l 分支重复代码
def apply_mode_selection(data_content, tong_list):
    """
    根据模式 data_content 对 tong_list 进行过滤，并选择最近桶。
    返回 selected_list (如 ['1','ux','uy'] 或 ['0'])。
    """
    # 定义过滤函数字典
    filter_map = {
        'r': lambda item: item[2] < 848,          # 右侧（原条件，未实际筛选）
        'm': lambda item: item[2] > 80 and item[2] < 768,
        'l': lambda item: item[2] > 0,            # 左侧（原条件，未实际筛选）
    }

    if data_content not in filter_map:
        return [str(0)]

    # 过滤
    filtered = [item for item in tong_list if filter_map[data_content](item)]

    if filtered:
        # 找最近桶
        min_dis = min(item[0] for item in filtered)
        for dis, lab, ux, uy in filtered:
            if dis == min_dis:
                with open('gaozhi.txt', 'r') as file:
                    print(f"gaozhi.txt 内容: {file.read().strip()}", flush=True)
                return [str(1), str(ux), str(uy)]
    return [str(0)]


# ==================== D435i 相机检测线程 ====================
def d435_detect(shuchu_file_path, stop_event, start_event):
    """核心检测线程：根据 data.txt 中的命令切换不同的目标筛选策略。"""
    global processed_list, last_processed_list, last_value

    # 初始化 RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    align_to = rs.stream.color
    align = rs.align(align_to)

    start_event.wait()
    print("[DEBUG] d435_detect线程已启动")
    pipeline.start(config)

    # 变量初始化
    processed_list = []
    last_processed_list = []
    last_mode = None
    v = 0                     # [整理] v 定义后未实际使用，保留原样
    count1 = 0                # [整理] count1 定义后未实际使用，保留原样

    try:
        while not stop_event.is_set():
            data_content = read_data_file()

            # 根据指令选择模型
            if data_content in ['r', 'm', 'l', '0']:
                model = model_2m
            elif data_content == '1m':
                model = model_1m
            else:
                continue

            # 模式切换时的重置操作
            if data_content != last_mode:
                print(f"[INFO] 模式切换: {last_mode} -> {data_content}")
                # 从1米模式退出时，清空输出文件和记忆值
                if last_mode == '1m' and data_content != '1m':
                    with open('gaozhi.txt', 'w') as f:
                        f.write('0')
                    last_value = None
                # 重置相关变量
                v = 0
                count1 = 0
                processed_list = []
                last_processed_list = []
                last_mode = data_content

            # 获取对齐后的图像帧
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            color_image = np.asanyarray(color_frame.get_data())

            # YOLOv8 推理
            results = model.predict(source=color_image, device='cpu', show=False,
                                    stream=False, verbose=False, iou=0.45, conf=0.6)

            tong_list = []       # 候选桶列表，元素为 (距离, 类别ID, ux, uy)
            canvas = color_image.copy()

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

                        dis = aligned_depth_frame.get_distance(ux, uy)
                        dis = np.round(np.array(dis), 3)

                        camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)
                        camera_xyz = np.round(np.array(camera_xyz), 3)
                        camera_xyz = camera_xyz.tolist()

                        cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 5)
                        cv2.putText(canvas, str(camera_xyz), (ux + 20, uy + 10), 0, 1,
                                    [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                        class_id = int(box.cls[0])

                        if area <= 70000:
                            tong_list.append((dis, class_id, ux, uy))

            # 显示当前模式
            cv2.putText(canvas, f"{data_content}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # ========== 根据指令选择不同的筛选策略 ==========
            if data_content in ['r', 'm', 'l']:
                # [整理] 使用提取的公共函数替代三个重复分支
                processed_list = apply_mode_selection(data_content, tong_list)
                write_to_file(processed_list)

            elif data_content == '1m':   # 1米锁定模式
                if tong_list:
                    min_dis = min(dis for dis, lab, ux, uy in tong_list)
                    for dis, lab, ux, uy in tong_list:
                        if dis == min_dis:
                            print(f"找到最近的桶（dis值最近）：ux={ux}, uy={uy}")
                            processed_list = [str(1), str(ux), str(uy)]
                            break
                else:
                    if last_value is not None:
                        processed_list = last_value.split()
                        print(f"[INFO] 1m模式未检测到桶，返回上一帧: {processed_list}")
                    else:
                        processed_list = [str(0)]
                write_to_file(processed_list)

            # （原代码中此处存在一大段被注释掉的卡尔曼/预测逻辑，整理时已删除）

            # 将绘制好的图像放入显示队列
            if not display_queue.full():
                display_queue.put((canvas, "chaoqian"))

    finally:
        pipeline.stop()

# ==================== 主控制线程 ====================
def main_control():
    """监听全局变量 c 的变化，当 c == 1 时启动 D435 检测线程。"""
    global c, start_event
    last_c = None
    running = False
    stop_event = None
    t_d435 = None

    ctx = rs.context()
    if len(ctx.devices) == 0:
        for _ in range(5):
            print(f"\033[31m[WARN] 未检测到D435相机，请检查连接！\033[0m")
    else:
        print(f"\033[32m[INFO] D435相机已连接。\033[0m")

    while True:
        if c != last_c:
            print(f"[DEBUG] main_control 检测到 c 变化: {last_c} -> {c}")

            if running:
                stop_event.set()
                if t_d435:
                    t_d435.join()
                cv2.destroyAllWindows()
                time.sleep(0.2)
                running = False

            stop_event = threading.Event()
            if c == 1:
                t_d435 = threading.Thread(target=d435_detect, args=('gaozhi.txt', stop_event, start_event))
                t_d435.start()
                running = True

            last_c = c
        time.sleep(0.1)

# ==================== 主程序入口 ====================
if __name__ == '__main__':
    try:
        print("[INFO] YoloV8目标检测-程序启动")
        print("[INFO] 开始YoloV8模型加载")

        c = 0
        clear_files(files_to_clear)

        shuru_file_path = 'data.txt'
        shuchu_file_path = 'gaozhi.txt'
        start_event = threading.Event()
        stop_event = threading.Event()

        monitor_thread = threading.Thread(target=monitor_data_file)
        monitor_thread.daemon = True
        monitor_thread.start()

        t1 = threading.Thread(target=read_file_content, args=(shuru_file_path, start_event, stop_event))
        t1.start()

        main_thread = threading.Thread(target=main_control)
        main_thread.start()

        print("[INFO] 完成YoloV8模型加载")
        print("[INFO] 系统初始化完成，等待指令...")

        window_created = False
        while True:
            if not display_queue.empty():
                frame, window_name = display_queue.get()
                name = "chaoqian"
                if not window_created:
                    cv2.namedWindow(name, cv2.WINDOW_NORMAL |
                                    cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                    cv2.resizeWindow(name, 1696, 960)
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