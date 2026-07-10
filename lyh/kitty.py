'''
by lyh
手动选桶程序 - YOLOv8版本
增加 2m 模式支持 - 2026.04.30
'''
import os
import cv2
import time
import queue
import threading
import traceback
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs

##参数设置
files_to_clear = ['data.txt', 'gaozhi.txt']
display_queue = queue.Queue(maxsize=3)

# 全局变量
last_value = None
b_zero = True
last_processed_list = []
c = 0                     # 新增：全局变量c
start_event = None        # 新增：全局事件占位

# 模型权重声明lyhbes
model_1m = YOLO("/home/fu/weights/tongv3.pt")
model_2m = YOLO("/home/fu/weights/tongv3.pt")  # 可替换为单独训练的2m权重


def clear_files(files):
    for file_name in files:
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, 'w') as f:
            print("[INFO]通讯txt文件已建立并清空", flush=True)
            pass


def read_data_file():
    try:
        with open('data.txt', 'r') as file:
            return file.read().strip()
    except FileNotFoundError:  #如果找不到data文件就报错，避免系统崩溃
        return None


def write_to_file(processed_list, filename='gaozhi.txt'):
    global last_value, b_zero
    b = None
    b_value = read_data_file()

    # [2m] 增加对 '2m' 的识别
    if b_value in ['r', 'm', 'l']:
        b = 1
    elif b_value in ['1m', '2m']:
        b = 2

    # 检查b是否变为1
    if b_zero and b == 1:
        message = "[INFO]我方即将进入对桶程序※※"
        for _ in range(8):
            print(message, flush=True)
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
                print(last_value, flush=True)
            if last_value is not None and not os.path.getsize(filename):
                return last_value
        return None


# 绘制方框和标签
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


# 监控data.txt文件内容的函数
def monitor_data_file():
    print("开始监控data.txt文件...", flush=True)
    while True:
        try:
            with open('data.txt', 'r') as file:
                print(f"data.txt 内容: {file.read().strip()}", flush=True)
        except:
            pass
        time.sleep(0.25)


# 读取文件内容并根据内容设置事件和变量
def read_file_content(shuru_file_path, start_event, stop_event):
    global c
    last_result = None

    while not stop_event.is_set():
        try:
            with open(shuru_file_path, 'r') as file:
                content = file.read().strip()
                if content != last_result:
                    # [2m] 增加 '2m' 有效模式
                    if content in ['r', 'm', 'l', '1m', '2m', '0']:
                        for i in range(8):
                            print(f"[INFO] 检测到{content}模式", flush=True)
                        start_event.set()
                        c = 1
                    else:
                        print(f"[INFO] 未检测到有效内容 '{content}'，等待...", flush=True)
                    last_result = content
        except Exception as e:
            print(f"[WARN] 读取 data.txt 出错: {e}", flush=True)
        time.sleep(0.05)


# D435i相机检测线程
def d435_detect(shuchu_file_path, stop_event, start_event):
    global processed_list, last_processed_list, last_value

    print("[DEBUG] d435_detect 线程已创建，等待 start_event...", flush=True)
    start_event.wait()
    print("[DEBUG] d435_detect start_event 已收到，开始初始化相机...", flush=True)

    # 初始化RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        print("[DEBUG] 正在启动 RealSense 管线...", flush=True)
        pipeline.start(config)
        print("[DEBUG] RealSense 管线启动成功", flush=True)
    except Exception as e:
        print(f"[ERROR] RealSense 启动失败: {e}", flush=True)
        traceback.print_exc()
        return

    # 初始化变量
    processed_list = []
    last_processed_list = []
    last_mode = None
    frame_count = 0

    try:
        while not stop_event.is_set():
            data_content = read_data_file()

            # [2m] 2m模式使用 model_2m
            if data_content in ['r', 'm', 'l', '0', '2m']:
                model = model_2m
            elif data_content == '1m':
                model = model_1m
            else:
                time.sleep(0.01)
                continue

            # 检测模式切换，重置前视相关变量
            if data_content != last_mode:
                print(f"[INFO] 模式切换: {last_mode} -> {data_content}", flush=True)
                # [2m] 从1m或2m切换到其他模式时清空输出
                if last_mode in ['1m', '2m'] and data_content not in ['1m', '2m']:
                    with open('gaozhi.txt', 'w') as f:
                        f.write('0')
                    last_value = None
                v = 0
                count1 = 0
                processed_list = []
                last_processed_list = []
                last_mode = data_content

            # 获取图像帧
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            color_image = np.asanyarray(color_frame.get_data())

            # YOLOv8推理
            results = model.predict(source=color_image, device='cpu', show=False,
                                    stream=False, verbose=False, iou=0.45, conf=0.6)

            tong_list = []
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

                        if area <= 70000:  # 桶类别
                            tong_list.append((dis, class_id, ux, uy))

            # 在画面上显示当前使用的模型
            cv2.putText(canvas, f"{data_content}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 处理不同模式的逻辑
            if data_content == 'r':
                tong_list = [item for item in tong_list if item[2] < 848]
                if tong_list:
                    min_dis = min(dis for dis, lab, ux, uy in tong_list)
                    for dis, lab, ux, uy in tong_list:
                        if dis == min_dis:
                            with open('gaozhi.txt', 'r') as file:
                                print(f"gaozhi.txt 内容: {file.read().strip()}", flush=True)
                            processed_list = [str(1), str(ux), str(uy)]
                            break
                else:
                    processed_list = [str(0)]
                write_to_file(processed_list)
            elif data_content == 'l':
                tong_list = [item for item in tong_list if item[2] > 0]
                if tong_list:
                    min_dis = min(dis for dis, lab, ux, uy in tong_list)
                    for dis, lab, ux, uy in tong_list:
                        if dis == min_dis:
                            with open('gaozhi.txt', 'r') as file:
                                print(f"gaozhi.txt 内容: {file.read().strip()}", flush=True)
                            processed_list = [str(1), str(ux), str(uy)]
                            break
                else:
                    processed_list = [str(0)]
                write_to_file(processed_list)
            elif data_content == 'm':
                tong_list = [item for item in tong_list if item[2] > 80 and item[2] < 768]
                if tong_list:
                    min_dis = min(dis for dis, lab, ux, uy in tong_list)
                    for dis, lab, ux, uy in tong_list:
                        if dis == min_dis:
                            with open('gaozhi.txt', 'r') as file:
                                print(f"gaozhi.txt 内容: {file.read().strip()}", flush=True)
                            processed_list = [str(1), str(ux), str(uy)]
                            break
                else:
                    processed_list = [str(0)]
                write_to_file(processed_list)
            # [2m] 增加2m分支，行为同1m
            elif data_content in ['1m', '2m']:
                if tong_list:
                    min_dis = min(dis for dis, lab, ux, uy in tong_list)
                    for dis, lab, ux, uy in tong_list:
                        if dis == min_dis:
                            print(f"找到最近的桶（dis值最近）：ux={ux}, uy={uy}", flush=True)
                            processed_list = [str(1), str(ux), str(uy)]
                            break
                else:
                    if last_value is not None:
                        processed_list = last_value.split()
                        print(f"[INFO] {data_content}模式未检测到桶，返回上一帧: {processed_list}", flush=True)
                    else:
                        processed_list = [str(0)]
                write_to_file(processed_list)

            # 显示画面
            if not display_queue.full():
                display_queue.put((canvas, "chaoqian"))
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"[DEBUG] 已推送 {frame_count} 帧到显示队列", flush=True)
            else:
                if frame_count % 30 == 0:
                    print(f"[WARN] 显示队列已满！当前帧数: {frame_count}", flush=True)

    except Exception as e:
        print(f"[ERROR] d435_detect 线程异常: {e}", flush=True)
        traceback.print_exc()
    finally:
        print("[DEBUG] d435_detect 线程退出，停止管线", flush=True)
        pipeline.stop()


# 主控制函数
def main_control():
    global c, start_event
    last_c = None
    running = False
    stop_event = None
    t_d435 = None

    # 检查D435相机连接
    ctx = rs.context()
    if len(ctx.devices) == 0:
        for _ in range(5):
            print(f"\033[31m[WARN] 未检测到D435相机，请检查连接！\033[0m", flush=True)
    else:
        print(f"\033[32m[INFO] D435相机已连接。\033[0m", flush=True)

    while True:
        if c != last_c:
            print(f"[DEBUG] main_control 检测到 c 变化: {last_c} -> {c}", flush=True)

            # 关闭旧线程
            if running:
                stop_event.set()
                if t_d435:
                    t_d435.join()
                cv2.destroyAllWindows()
                time.sleep(0.2)
                running = False

            stop_event = threading.Event()
            # 启动新线程
            if c == 1:
                t_d435 = threading.Thread(target=d435_detect, args=('gaozhi.txt', stop_event, start_event))
                t_d435.start()
                running = True

            last_c = c
        time.sleep(0.1)


if __name__ == '__main__':
    try:
        print("[INFO] YoloV8目标检测-程序启动", flush=True)
        print("[INFO] 开始YoloV8模型加载", flush=True)

        # 初始化全局变量
        c = 0
        clear_files(files_to_clear)

        shuru_file_path = 'data.txt'
        shuchu_file_path = 'gaozhi.txt'
        start_event = threading.Event()
        stop_event = threading.Event()

        # 启动监控线程
        monitor_thread = threading.Thread(target=monitor_data_file)
        monitor_thread.daemon = True
        monitor_thread.start()

        # 启动文件读取线程
        t1 = threading.Thread(target=read_file_content, args=(shuru_file_path, start_event, stop_event))
        t1.start()

        # 启动主控制线程
        main_thread = threading.Thread(target=main_control)
        main_thread.start()

        print("[INFO] 完成YoloV8模型加载", flush=True)
        print("[INFO] 系统初始化完成，等待指令...", flush=True)

        # 主线程负责显示窗口
        window_created = False
        display_checked = 0
        while True:
            if not display_queue.empty():
                frame, window_name = display_queue.get()
                name = "chaoqian"
                if not window_created:
                    print("[DEBUG] 正在创建显示窗口...", flush=True)
                    cv2.namedWindow(name, cv2.WINDOW_NORMAL |
                                    cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                    cv2.resizeWindow(name, 1696, 960)
                    window_created = True
                    print("[DEBUG] 显示窗口已创建", flush=True)
                cv2.imshow(name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break
            else:
                display_checked += 1
                if display_checked % 300 == 1:
                    print(f"[DEBUG] 等待显示帧... (已检查{display_checked}次, 队列大小={display_queue.qsize()})", flush=True)
                time.sleep(0.01)

        cv2.destroyAllWindows()
        main_thread.join()

    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
