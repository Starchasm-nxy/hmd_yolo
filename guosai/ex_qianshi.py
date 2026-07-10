'''
by hmy 2025.8.14——2025.8.21
last by hmy 2025.8月底
手动选桶程序 - YOLOv8版本
功能：根据 data.txt 中的指令（r, m, l, 1m）选择画面中不同区域或距离的桶，
     并输出目标像素坐标到 gaozhi.txt。
'''
import os
import cv2
import time
import queue
import threading
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs

## ==================== 参数设置 ====================
# [说明] 需要清空的通信文件列表
files_to_clear = ['data.txt', 'gaozhi.txt']
# [说明] 显示队列，用于线程间传递绘制好的图像
display_queue = queue.Queue(maxsize=3)

# ==================== 全局变量 ====================
# [说明] last_value: 用于1m模式（锁定阶段）记忆上一帧的目标坐标字符串
last_value = None
# [说明] b_zero: 标志位，用于在首次进入 'r'/'m'/'l' 模式时打印提示信息（仅一次）
b_zero = True
# [说明] last_processed_list: 保存上一次处理后的坐标列表，用于平滑或记忆（本程序中未充分使用）
last_processed_list = []

# ==================== 模型权重加载 ====================
# [说明] 两个模型实际是同一个文件（2m.pt），可根据需要替换为不同模型
model_1m = YOLO("/home/luck/yolov5_d435i_detection-main/2m.pt")
model_2m = YOLO("/home/luck/yolov5_d435i_detection-main/2m.pt")

# ==================== 工具函数：清空通信文件 ====================
def clear_files(files):
    """清空指定的文件，确保通信文件初始为空"""
    for file_name in files:
        file_path = os.path.join(os.getcwd(), file_name)   # 构建绝对路径
        with open(file_path, 'w') as f:                    # 写模式打开（覆盖）
            print("[INFO]通讯txt文件已建立并清空")
            pass

# ==================== 读取 data.txt 内容 ====================
def read_data_file():
    """读取 data.txt 文件内容（去除首尾空白），若文件不存在返回 None"""
    try:
        with open('data.txt', 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return None

# ==================== 写入坐标到 gaozhi.txt ====================
def write_to_file(processed_list, filename='gaozhi.txt'):
    """
    根据当前 data.txt 中的模式（b_value），决定如何写入坐标。
    processed_list: 包含坐标信息的列表，如 ['1', 'ux', 'uy'] 或 ['0']
    """
    global last_value, b_zero
    b = None
    b_value = read_data_file()

    # [说明] 将 data.txt 中的命令映射为内部模式编号 b
    if b_value in ['r', 'm', 'l']:
        b = 1          # 2米选桶模式（左/中/右）
    elif b_value == '1m':
        b = 2          # 1米锁定模式

    # [说明] 首次进入选桶模式时打印提示（仅一次）
    if b_zero and b == 1:
        message = "[INFO]我方即将进入对桶程序※※"
        for _ in range(8):
            print(message)
        b_zero = False

    # [说明] 1米模式（锁定）：直接写入当前检测结果
    if b == 2:
        if processed_list:
            content = ' '.join(processed_list)    # 列表元素用空格连接成字符串
            with open(filename, 'w') as f:
                f.write(content)
            last_value = content.strip()          # 更新记忆值
        else:
            processed_list = [str(0)]
            content = ' '.join(processed_list)
            with open(filename, 'w') as f:
                f.write(content)
            return last_value

    # [说明] 选桶模式（r/m/l）：若当前有检测结果则写入，否则若记忆值存在则写入记忆值（保持输出）
    if b == 1:
        if processed_list:
            content = ' '.join(processed_list)
            with open(filename, 'w') as f:
                f.write(content)
            if content.strip() != '0':
                last_value = content.strip()      # 非零结果才更新记忆值
        else:
            if last_value is not None:
                with open(filename, 'w') as f:
                    f.write(last_value + '\n')
                print(last_value)
            if last_value is not None and not os.path.getsize(filename):
                return last_value
        return None

# ==================== 绘制检测框和标签 ====================
def draw_square(image, box, names, r):
    """在图像上绘制目标框、类别标签和中心点，返回中心像素坐标"""
    ux = int((r[0] + r[2]) / 2)   # 中心x
    uy = int((r[1] + r[3]) / 2)   # 中心y
    cls = int(box.cls[0])         # 类别索引
    conf = box.conf[0]            # 置信度
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
                        start_event.set()   # 通知检测线程开始工作
                        c = 1               # 设置模式标志（main_control 会据此启动线程）
                    else:
                        print("[INFO] 未检测到有效内容，等待...")
                    last_result = content
        except:
            pass
        time.sleep(0.05)

# ==================== D435i 相机检测线程 ====================
def d435_detect(shuchu_file_path, stop_event, start_event):
    """
    核心检测线程：根据 data.txt 中的命令切换不同的目标筛选策略。
    - 'r': 只选画面右侧区域的目标（x坐标较大）
    - 'm': 只选画面中间区域的目标（x坐标在 80~768 之间）
    - 'l': 只选画面左侧区域的目标（x坐标较小）
    - '1m': 锁定模式，选择距离最近的目标，无目标时保持上一帧结果
    """
    global processed_list, last_processed_list, last_value

    # 初始化 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    align_to = rs.stream.color
    align = rs.align(align_to)      # 深度对齐到彩色

    start_event.wait()              # 等待启动信号
    print("[DEBUG] d435_detect线程已启动")
    pipeline.start(config)

    # 初始化变量
    processed_list = []
    last_processed_list = []
    last_mode = None                # 记录上一次的模式，用于检测模式切换

    try:
        while not stop_event.is_set():
            data_content = read_data_file()   # 读取当前指令

            # 根据指令选择对应的模型（此处两个模型相同，可扩展）
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

            # 获取深度内参（用于像素转相机坐标）
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

                        # 获取该像素点的深度值（米）
                        dis = aligned_depth_frame.get_distance(ux, uy)
                        dis = np.round(np.array(dis), 3)

                        # 将像素坐标转换为相机坐标系下的三维坐标
                        camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)
                        camera_xyz = np.round(np.array(camera_xyz), 3)
                        camera_xyz = camera_xyz.tolist()

                        # 在画面上显示三维坐标
                        cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 5)
                        cv2.putText(canvas, str(camera_xyz), (ux + 20, uy + 10), 0, 1,
                                    [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                        class_id = int(box.cls[0])

                        # 面积过滤（70000 为经验阈值，排除过近的大面积干扰）
                        if area <= 70000:
                            tong_list.append((dis, class_id, ux, uy))

            # 在画面上显示当前模式（r/m/l/1m）
            cv2.putText(canvas, f"{data_content}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # ========== 根据指令选择不同的筛选策略 ==========
            if data_content == 'r':   # 右侧桶（x > 424 半区，此处简单用 x < 848 代表右侧）
                tong_list = [item for item in tong_list if item[2] < 848]   # 实际上所有 x 都小于848，此条件无实际过滤作用（可能是代码未完善）
                if tong_list:
                    # 选择距离最近的桶
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

            elif data_content == 'l':   # 左侧桶（x > 0，同样条件过宽，可能未精确实现左右划分）
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

            elif data_content == 'm':   # 中间桶（x 在 80~768 范围内，大致中间区域）
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

            elif data_content == '1m':   # 1米锁定模式
                if tong_list:
                    # 选择距离最近的桶
                    min_dis = min(dis for dis, lab, ux, uy in tong_list)
                    for dis, lab, ux, uy in tong_list:
                        if dis == min_dis:
                            print(f"找到最近的桶（dis值最近）：ux={ux}, uy={uy}")
                            processed_list = [str(1), str(ux), str(uy)]
                            break
                else:
                    # 无目标时使用上一帧记忆的坐标
                    if last_value is not None:
                        processed_list = last_value.split()
                        print(f"[INFO] 1m模式未检测到桶，返回上一帧: {processed_list}")
                    else:
                        processed_list = [str(0)]
                write_to_file(processed_list)

            # （以下为被注释的复杂逻辑，原本用于处理多个桶的平滑和预测，已废弃）

            # 将绘制好的图像放入显示队列
            if not display_queue.full():
                display_queue.put((canvas, "chaoqian"))

    finally:
        pipeline.stop()

# ==================== 主控制线程 ====================
def main_control():
    """
    监听全局变量 c 的变化，当 c == 1 时启动 D435 检测线程。
    提供线程生命周期管理（停止旧线程、启动新线程）。
    """
    global c, start_event
    last_c = None
    running = False
    stop_event = None
    t_d435 = None

    # 检查 D435 相机连接
    ctx = rs.context()
    if len(ctx.devices) == 0:
        for _ in range(5):
            print(f"\033[31m[WARN] 未检测到D435相机，请检查连接！\033[0m")
    else:
        print(f"\033[32m[INFO] D435相机已连接。\033[0m")

    while True:
        if c != last_c:
            print(f"[DEBUG] main_control 检测到 c 变化: {last_c} -> {c}")

            # 关闭旧线程
            if running:
                stop_event.set()
                if t_d435:
                    t_d435.join()
                cv2.destroyAllWindows()
                time.sleep(0.2)
                running = False

            stop_event = threading.Event()
            # 启动新线程（D435检测线程）
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

        # 初始化全局变量
        c = 0
        clear_files(files_to_clear)

        shuru_file_path = 'data.txt'
        shuchu_file_path = 'gaozhi.txt'
        start_event = threading.Event()
        stop_event = threading.Event()

        # 启动监控线程（调试用，持续打印 data.txt 内容）
        monitor_thread = threading.Thread(target=monitor_data_file)
        monitor_thread.daemon = True   # 守护线程，主程序退出时自动结束
        monitor_thread.start()

        # 启动文件读取线程（监听命令）
        t1 = threading.Thread(target=read_file_content, args=(shuru_file_path, start_event, stop_event))
        t1.start()

        # 启动主控制线程（管理检测线程）
        main_thread = threading.Thread(target=main_control)
        main_thread.start()

        print("[INFO] 完成YoloV8模型加载")
        print("[INFO] 系统初始化完成，等待指令...")

        # 主线程负责图像显示
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