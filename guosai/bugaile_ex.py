### power by wjh 2025-09-22
### 4m解算国赛最终版
# #已上机实测
# 速度提升---调参及任务规划
# #模型特点：
# 2m.pt：D435i相机模型，2m以下检测精度高
# 3.5m.pt：USB相机模型，2m-4m检测精度高

# ========== 导入必要的库 ==========
import os                         # 操作系统接口，用于路径操作
import cv2                        # OpenCV，图像处理和显示
import time                       # 时间函数，用于延时和计时
import queue                      # 线程安全队列，用于线程间数据传递
import threading                  # 多线程支持，实现并发
from ultralytics import YOLO      # YOLOv8/v5 模型加载与推理
import numpy as np                # 数值计算，数组处理
import pyrealsense2 as rs         # Intel RealSense D435i 深度相机SDK
from sklearn.cluster import DBSCAN  # DBSCAN聚类算法，用于目标点聚类

# ========== 全局参数设置 ==========
# [说明] 相机高度（米），用于单目测距（假设地面为平面）
Z = 4.31
# [说明] DBSCAN聚类半径（像素），距离小于此值的点视为同一簇
eps = 50
# [说明] DBSCAN最小样本数，至少收集这么多点才进行聚类
min_samples = 15
# [说明] 最大等待时间（秒），超过此时间即使点数不足也强制聚类
max_time = 5.0
# [说明] 需要清空的通信文件列表（用于与主控程序交互）
files_to_clear = ['data.txt', 'gaozhi.txt']
# [说明] 相机畸变校正参数文件（由相机标定生成）
files_for_pixel = 'calib_result.npz'
# [说明] USB摄像头设备路径（通过ID固定，避免插拔变化）
cam_num = '/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0'
# [说明] 加载YOLO模型：USB模型（3-4m远距离），D435模型（2m内近距离）
model_usb = YOLO("/home/son/yolov5_d435i_detection-main/gs4mcz.pt")
model_d435 = YOLO("/home/son/yolov5_d435i_detection-main/2m.pt")

# [说明] 显示队列：用于将绘制好的图像从检测线程传递到主线程显示，避免OpenCV多线程冲突
display_queue = queue.Queue(maxsize=3)
# [说明] 程序启动时间记录（用于性能计时，未实际使用）
start_time = None

# ========== 工具函数：清空通信文件 ==========
def clear_files(files):
    """
    清空指定的文件，确保每次启动时通信文件为空。
    文件用于与主控系统（如树莓派、上位机）交换指令和坐标数据。
    """
    for file_name in files:
        file_path = os.path.join(os.getcwd(), file_name)  # 构建绝对路径
        with open(file_path, 'w') as f:                   # 以写模式打开（覆盖）
            print("[INFO]通讯txt文件已建立并清空")
            pass   # 空操作，仅清空文件

# ========== 检查摄像头连接状态 ==========
def check_camera(cam_num):
    """
    检查USB摄像头和RealSense D435i是否连接正常。
    输出彩色提示信息（绿色成功，红色警告）。
    """
    # 检查USB摄像头
    cap = cv2.VideoCapture(cam_num)
    if not cap.isOpened():
        for _ in range(5):   # 重复提示5次，确保用户注意到
            print(f"\033[31m[WARN] 无法打开usb摄像头，请检查连接或权限。\033[0m")
        cap.release()
    else:
        print(f"\033[32m[INFO] usb摄像头已成功连接\033[0m")
        cap.release()

    # 检查D435相机（通过RealSense上下文获取设备列表）
    ctx = rs.context()
    if len(ctx.devices) == 0:
        for _ in range(5):
            print(f"\033[31m[WARN] 未检测到D435相机，请检查连接！\033[0m")
    else:
        print(f"\033[32m[INFO] D435相机已连接。\033[0m")

# ========== 主控制线程：根据c值切换相机与模型 ==========
# [说明] 以下全局变量用于线程间协调：
#   c: 当前工作模式 (0=等待, 2=D435模式, 3=USB模式)
#   b: D435模式下的子状态 (2=寻找目标, 1=保持目标)
#   d: USB模式下的目标选择模式 (1~6，对应不同桶组合)
#   start_event: 通知所有检测线程可以开始工作
#   stop_event:  通知所有线程停止工作
global c, b, d, start_event, shuchu_file_path
global t_test, t_detect, t_d435, camera

def main_control():
    """
    监听全局变量 c 的变化，动态创建/销毁相机和检测线程。
    实现不同模式（USB远距离 / D435近距离）之间的无缝切换。
    """
    global c, cam_num, model, shuchu_file_path, start_event
    global t_test, t_detect, t_d435, camera

    last_c = None           # 记录上一次的 c 值，用于检测变化
    running = False         # 是否有线程正在运行
    camera = 0              # 当前相机对象（cv2.VideoCapture）
    stop_event = None       # 停止事件，用于通知线程退出
    frame_queue = None      # USB模式下的帧队列

    check_camera(cam_num)   # 启动时检查相机连接

    while True:
        # 当 c 发生变化时（由 read_file_content 线程修改）
        if c != last_c:
            print(f"[DEBUG] main_control 检测到 c 变化: {last_c} -> {c}")

            # 如果之前有线程在运行，先优雅地关闭它们
            if running:
                stop_event.set()                 # 通知所有线程停止
                if t_test: t_test.join()         # 等待test线程结束
                if t_detect: t_detect.join()     # 等待usb_detect线程结束
                if camera: camera.release()      # 释放USB摄像头
                if t_d435: t_d435.join()         # 等待d435_detect线程结束
                cv2.destroyAllWindows()          # 关闭所有OpenCV窗口
                time.sleep(0.2)                  # 等待资源完全释放
                running = False

            # 创建新的停止事件（用于新线程）
            stop_event = threading.Event()

            # 根据 c 值启动相应模式
            if c == 3:   # USB远距离模式（3-4米）
                camera = cv2.VideoCapture(cam_num)          # 打开USB摄像头
                model = model_usb                           # 使用USB模型
                frame_queue = queue.Queue(maxsize=3)        # 帧队列（传递校正后图像）
                # 创建两个线程：test（采集+校正）、usb_detect（检测+聚类+通信）
                t_test = threading.Thread(target=test, args=(camera, frame_queue, start_event, stop_event))
                t_detect = threading.Thread(target=usb_detect, args=(model, frame_queue, shuchu_file_path, stop_event, start_event))
                t_test.start()
                t_detect.start()
                t_d435 = None   # D435线程不使用
                running = True

            elif c == 2:   # D435近距离模式（2米内）
                model = model_d435                          # 使用D435模型
                # 创建一个线程：d435_detect（深度相机检测+通信）
                t_d435 = threading.Thread(target=d435_detect, args=(model, shuchu_file_path, stop_event, start_event))
                t_d435.start()
                t_test = None
                t_detect = None
                camera = None
                running = True

            last_c = c   # 更新记录值
        time.sleep(0.1)  # 降低CPU占用

# ========== 像素坐标转相机坐标系（世界坐标） ==========
def pixel_to_camera(ux, uy, mtx, w, h):
    """
    根据小孔成像模型和已知相机高度 Z，将像素坐标转换为相机坐标系下的 X,Y（单位：米）。
    前提：目标位于地平面上，且相机光轴与地面垂直（或近似垂直）。
    """
    fx = mtx[0, 0]          # x方向焦距（像素单位）
    fy = mtx[1, 1]          # y方向焦距
    cx = w / 2              # 图像中心x坐标（假设光心在图像中心）
    cy = h / 2              # 图像中心y坐标
    X = (ux - cx) * Z / fx  # 相似三角形计算X
    Y = (uy - cy) * Z / fy  # 相似三角形计算Y
    return X, Y

# ========== USB畸变校正线程 ==========
def test(camera, frame_queue, start_event, stop_event):
    """
    持续从USB摄像头读取原始图像，进行畸变校正，并将校正后的图像放入队列。
    使用预先标定的参数文件。
    """
    start_event.wait()   # 等待主程序发出开始信号
    print("[DEBUG] test线程已启动")

    # 加载畸变校正参数
    calib = np.load(files_for_pixel)
    mtx = calib['mtx']      # 内参矩阵
    dist = calib['dist']    # 畸变系数
    h, w = 480, 640         # 图像分辨率（固定）
    # 计算最优相机矩阵和有效区域（ROI）
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    # 计算映射表，用于 remap 快速校正（比 undistort 快）
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)

    while not stop_event.is_set():   # 只要未收到停止信号就循环
        ret, frame = camera.read()   # 读取一帧
        if ret:
            if c == 3:   # 仅在USB模式下处理（防止模式切换时的残余帧）
                # 使用 remap 进行畸变校正
                undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
                # 如果队列未满，放入校正后的图像（非阻塞）
                if not frame_queue.full():
                    frame_queue.put(undistorted)
        else:
            print("无法读取摄像头画面，请检查摄像头连接或权限设置。")
            stop_event.set()   # 读取失败则设置停止事件，退出循环
        time.sleep(0.01)       # 约100fps，防止CPU过载

# ========== 绘制检测框和标签 ==========
def draw_square(image, box, names, r):
    """
    在图像上绘制目标框、类别标签和中心点，并返回中心像素坐标。
    r: 边界框坐标 [x1, y1, x2, y2]
    """
    ux = int((r[0] + r[2]) / 2)   # 中心x
    uy = int((r[1] + r[3]) / 2)   # 中心y
    cls = int(box.cls[0])          # 类别索引
    conf = box.conf[0]             # 置信度
    label = f"{names[cls]} {conf:.2f}"   # 标签文字
    cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (221, 185, 193), 2)   # 画框
    cv2.putText(image, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (176, 196, 222), 2)   # 写文字
    cv2.circle(image, (ux, uy), 5, (240, 240, 240), -1)   # 画中心点（实心圆）
    return ux, uy

# ========== USB检测线程（含聚类和通信） ==========
def usb_detect(model, frame_queue, shuchu_file_path, stop_event, start_event):
    """
    从队列获取校正后的图像，运行YOLO检测。
    收集目标中心点，达到条件后使用DBSCAN聚类，选择两个桶输出其世界坐标到文件。
    模式 d 决定选择哪两个桶（左右组合）。
    """
    global start_time
    start_event.wait()
    print("[DEBUG] usb_detect线程已启动")

    # 初始化输出文件，写入占位符（无目标）
    with open(shuchu_file_path, 'w') as file:
        file.write("0 0 0 0")
    result_str = "0 0 0 0"
    last_result = None

    # 加载内参，用于坐标转换
    calib = np.load(files_for_pixel)
    mtx = calib['mtx']

    points = []                   # 收集待聚类的像素点列表
    last_cluster_time = time.time()   # 上次聚类的时间
    clustered_once = False        # 是否已完成本次模式的聚类（只聚一次）
    final_result_str = None       # 最终输出字符串（聚类后固定输出）

    while not stop_event.is_set():
        # 如果已经聚类过，则持续输出相同的结果（不再处理新帧，节省计算）
        if clustered_once:
            if final_result_str is not None:
                with open(shuchu_file_path, 'w') as file:
                    file.write(final_result_str)
            time.sleep(0.1)
            continue

        # 从队列取出一帧（阻塞等待）
        if not frame_queue.empty():
            frame = frame_queue.get()
            h, w = frame.shape[:2]

            # YOLO推理（设备CPU，关闭冗余输出）
            results = model.predict(source=frame, device='cpu', show=False, stream=False, verbose=False, iou=0.45, conf=0.6)

            for result in results:
                image = result.orig_img
                names = result.names
                boxes = result.boxes
                for box in boxes:
                    r = box.xyxy[0].cpu().numpy().astype(int)   # 获取边界框坐标
                    ux, uy = draw_square(image, box, names, r)  # 绘制并获取中心点
                    x, y = pixel_to_camera(ux, uy, mtx, w, h)   # 转换为世界坐标（用于显示）
                    cv2.putText(image, f"({x:.2f}m, {y:.2f}m)", (ux + 10, uy), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 240, 240), 3)
                    points.append((ux, uy))   # 收集像素坐标

            current_time = time.time()
            last_time = current_time - last_cluster_time

            # 触发聚类的条件：点数足够 或 超时且有点
            keydoor = False
            if len(points) >= min_samples:
                keydoor = True
            elif last_time >= max_time and len(points) > 0:
                keydoor = True

            if keydoor:
                # 执行DBSCAN聚类，返回聚类中心列表（像素坐标）
                centers = dbscan_cluster_and_draw(image, points, eps, min_samples)
                camera_centers = []
                for center in centers:
                    x_cam, y_cam = pixel_to_camera(center[0], center[1], mtx, w, h)
                    camera_centers.append((x_cam, y_cam))

                # 按x坐标从小到大排序（从左到右）
                camera_centers.sort(key=lambda c: c[0])

                # 如果有至少3个桶，根据模式 d 选择两个桶输出
                if len(camera_centers) >= 3:
                    if len(camera_centers) > 2:
                        # d的含义由外部命令决定（如mlkai, rmkai等）
                        if d == 6:
                            center1 = camera_centers[1]   # 中间
                            center2 = camera_centers[0]   # 左边
                        elif d == 5:
                            center1 = camera_centers[2]   # 右边
                            center2 = camera_centers[1]   # 中间
                        elif d == 4:
                            center1 = camera_centers[2]   # 右边
                            center2 = camera_centers[0]   # 左边
                        elif d == 3:
                            center1 = camera_centers[0]   # 左边
                            center2 = camera_centers[1]   # 中间
                        elif d == 2:
                            center1 = camera_centers[0]   # 左边
                            center2 = camera_centers[2]   # 右边
                        elif d == 1:
                            center1 = camera_centers[1]   # 中间
                            center2 = camera_centers[2]   # 右边

                        # 检查坐标是否在合理范围内（防止误检）
                        if -4 <= center1[0] <= 4 and -4 <= center2[0] <= 4 and -2.5 <= center1[1] <= 2.5 and -2.5 <= center2[1] <= 2.5:
                            result_str = f"{center1[0]:.1f} {center1[1]:.1f} {center2[0]:.1f} {center2[1]:.1f}"
                        elif -4 <= center1[0] <= 4 and -2.5 <= center1[1] <= 2.5:
                            result_str = f"{center1[0]:.1f} {center1[1]:.1f} 0 0"
                        elif -4 <= center2[0] <= 4 and -2.5 <= center2[1] <= 2.5:
                            result_str = f"0 0 {center2[0]:.1f} {center2[1]:.1f}"
                        else:
                            result_str = "0 0 0 0"

                    # 如果结果有变化，写入文件
                    if result_str != last_result:
                        with open(shuchu_file_path, 'w') as file:
                            file.write(str(result_str))
                        last_result = result_str

                    # 标记已聚类，后续循环直接输出结果，不再处理图像
                    clustered_once = True
                    final_result_str = result_str
                else:
                    # 桶的数量不足，输出全零
                    if last_result != "0 0 0 0":
                        with open(shuchu_file_path, 'w') as file:
                            file.write("0 0 0 0")
                        last_result = "0 0 0 0"

            last_cluster_time = current_time   # 更新聚类时间

            # 将绘制好的图像放入显示队列（供主线程显示）
            if 'image' in locals():
                if not display_queue.full():
                    display_queue.put((image, "atx"))

            # 打印输出文件内容（调试用）
            with open('gaozhi.txt', 'r') as file3:
                content = file3.read().strip()
                print(f"gaozhi.txt 内容: {content}", flush=True)
        else:
            time.sleep(0.01)   # 队列空，稍作等待

# ========== 通信文件监听线程 ==========
def read_file_content(shuru_file_path, start_event, stop_event):
    """
    监听 data.txt 文件的变化，根据内容设置全局变量 c, b, d，
    并触发 start_event 启动对应的检测线程。
    该线程是整个程序的“指令入口”。
    """
    global c, b, start_time, d
    last_result = None   # 记录上次文件内容

    # 初次读取并显示内容
    with open(shuru_file_path, 'r') as file2:
        print(f"data.txt 内容: {file2.read().strip()}", flush=True)
    time.sleep(0.25)

    while not stop_event.is_set():
        with open(shuru_file_path, 'r') as file2:
            content = file2.read().strip()
            # 仅当内容变化时才处理
            if content != last_result:
                # 各命令对应的含义：
                # mlkai, rmkai...  -> USB模式，d值不同，选择不同桶组合
                # 2m -> D435模式，b=2（寻找目标）
                # 1m -> D435模式，b=1（锁定目标，用于投放）
                if content == 'mlkai':
                    for i in range(10):
                        print("[INFO] 已经到达4m，即将打开usb相机ml模式")
                    start_event.set()   # 通知所有线程开始
                    c = 3
                    d = 6
                    start_time = time.time()
                elif content == 'rmkai':
                    for i in range(10):
                        print("[INFO] 已经到达4m，即将打开usb相机rm模式")
                    start_event.set()
                    c = 3
                    d = 5
                    start_time = time.time()
                elif content == 'rlkai':
                    for i in range(10):
                        print("[INFO] 已经到达4m，即将打开usb相机rl模式")
                    start_event.set()
                    c = 3
                    d = 4
                    start_time = time.time()
                elif content == 'lmkai':
                    for i in range(10):
                        print("[INFO] 已经到达4m，即将打开usb相机lm模式")
                    start_event.set()
                    c = 3
                    d = 3
                    start_time = time.time()
                elif content == 'lrkai':
                    for i in range(10):
                        print("[INFO] 已经到达4m，即将打开usb相机lr模式")
                    start_event.set()
                    c = 3
                    d = 2
                    start_time = time.time()
                elif content == 'mrkai':
                    for i in range(10):
                        print("[INFO] 已经到达4m，即将打开usb相机mr模式")
                    start_event.set()
                    c = 3
                    d = 1
                    start_time = time.time()
                elif content == '2m':
                    for i in range(10):
                        print("[INFO] 即将切换D435i相机，开始锁定目标桶")
                    start_event.set()
                    c = 2
                    b = 2
                elif content == '1m':
                    for i in range(10):
                        print("[INFO] 目标桶已锁定，即将投放")
                    start_event.set()
                    c = 2
                    b = 1
                else:
                    print("[INFO] 未检测到内容，等待...")
                last_result = content

# ========== D435i检测线程 ==========
def d435_detect(model, shuchu_file_path, stop_event, start_event):
    """
    使用RealSense D435i深度相机进行近距离目标检测。
    根据 b 值决定工作模式：
      b=2: 寻找目标，输出 "1 x y"（目标中心像素坐标）
      b=1: 保持上次目标，即使暂时丢失也不变
    """
    # 初始化RealSense管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    align_to = rs.stream.color
    align = rs.align(align_to)   # 深度对齐到彩色

    start_event.wait()
    print("[DEBUG] d435_detect线程已启动")

    # 初始输出 "0"
    with open(shuchu_file_path, 'w') as file:
        file.write("0")
    pipeline.start(config)

    result_str = "0"
    last_result = None
    last_value = "0"   # 用于b=1时保持上一次输出

    while not stop_event.is_set():
        tong_list = []   # 候选桶列表，每个元素为 (偏移量, (ux, uy))
        # 获取对齐后的帧
        image = pipeline.wait_for_frames()
        aligned_frames = align.process(image)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())

        # YOLO推理
        results = model.predict(source=frame, device='cpu', show=False, stream=False, verbose=False, iou=0.45, conf=0.6)

        for result in results:
            image = result.orig_img
            names = result.names
            boxes = result.boxes
            for box in boxes:
                r = box.xyxy[0].cpu().numpy().astype(int)
                ux, uy = draw_square(frame, box, names, r)
                cv2.putText(image, f"( {ux},  {uy})", (ux + 10, uy), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 240, 240), 3)
                area = (r[2] - r[0]) * (r[3] - r[1])
                # dis = aligned_depth_frame.get_distance(ux, uy)  # 可获取深度，此处未使用
                if b == 2:
                    # 面积限制，防止近距离干扰（如地面反光）
                    if area <= 30000:
                        # 偏移量 = 目标x坐标与图像中心（424）的绝对差值，越小越靠近中心
                        tong_list.append((abs(ux - 424), (ux, uy)))
                elif b == 1:
                    # 不限制面积，直接加入
                    tong_list.append(((abs(ux - 424)), (ux, uy)))

                # 按偏移量从小到大排序，取前2个（最多保留两个候选）
                tong_list.sort(key=lambda x: x[0])
                if len(tong_list) > 2:
                    tong_list = tong_list[:2]

        if c == 2:
            if len(tong_list) >= 1:
                center = tong_list[0][1]   # 选择偏移量最小的目标
                result_str = f"1 {center[0]} {center[1]}"
                if b == 1:
                    last_value = result_str   # 更新记忆值
                else:
                    last_value = "0"
            else:
                if b == 2:
                    result_str = "0"
                    with open(shuchu_file_path, 'w') as file:
                        file.write(str(result_str))
                    last_result = result_str
                elif b == 1:
                    result_str = last_value   # 没有目标时使用上次记忆的目标

        # 仅当结果变化时写入文件（避免频繁IO）
        if not (b == 2 and len(tong_list) == 0):
            if result_str != last_result:
                with open(shuchu_file_path, 'w') as file:
                    file.write(str(result_str))
                last_result = result_str

        # 放入显示队列
        if 'image' in locals():
            if not display_queue.full():
                display_queue.put((image, "atx"))

        # 调试输出
        with open('gaozhi.txt', 'r') as file3:
            content = file3.read().strip()
            print(f"gaozhi.txt 内容: {content}", flush=True)

# ========== DBSCAN聚类函数 ==========
def dbscan_cluster_and_draw(image, points, eps, min_samples):
    """
    对像素点列表进行DBSCAN聚类，并在图像上绘制聚类结果。
    返回各聚类中心的像素坐标列表。
    """
    if len(points) == 0:
        print(f"没有桶")
        return []

    X = np.array(points)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.fit_predict(X)   # 每个点的簇标签（-1为噪声）
    cluster_centers = []
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(max(labels) + 2)]

    # 遍历每个簇（跳过噪声簇-1）
    for i in range(max(labels) + 1):
        cluster_points = X[labels == i]
        cluster_center = np.mean(cluster_points, axis=0)   # 计算中心（均值）
        cluster_centers.append(cluster_center)

        if image is not None:
            # 绘制簇内所有点
            for pt in cluster_points:
                cv2.circle(image, (int(pt[0]), int(pt[1])), 8, colors[i], -1)
            # 绘制簇中心（大圆）
            cv2.circle(image, (int(cluster_center[0]), int(cluster_center[1])), 15, colors[i], 3)
            cv2.putText(image, f"cluster{i+1}", (int(cluster_center[0]), int(cluster_center[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)

    # 保存聚类结果图像（用于复盘分析）
    cv2.imwrite(f"聚类结果图/cluster_{int(time.time())}.jpg", image)
    return cluster_centers

# ========== 主程序入口 ==========
if __name__ == '__main__':
    try:
        # 初始化变量
        t_test = None
        t_detect = None
        t_d435 = None
        camera = None
        c = 0   # 0=等待模式
        b = 0   # D435子模式（2或1）
        clear_files(files_to_clear)

        shuru_file_path = 'data.txt'      # 输入指令文件
        shuchu_file_path = 'gaozhi.txt'   # 输出坐标文件

        start_event = threading.Event()   # 启动事件（初始为未触发）
        stop_event = threading.Event()    # 停止事件（目前未实际用于停止，保留）

        # 启动通信监听线程（负责读取 data.txt 并设置模式）
        t1 = threading.Thread(target=read_file_content, args=(shuru_file_path, start_event, stop_event))
        t1.start()

        # 启动主控制线程（负责根据 c 值管理相机和检测线程）
        main_thread = threading.Thread(target=main_control)
        main_thread.start()

        # 主线程负责图像显示
        window_created = False
        frame_count = 0
        start_time = time.time()

        while True:
            # 从显示队列取图像（非阻塞）
            if not display_queue.empty():
                frame, window_name = display_queue.get()

                # 创建窗口（仅一次）
                if not window_created:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 900, 800)
                    window_created = True

                # USB模式下，在图像四角显示世界坐标（用于调试）
                if c == 3:
                    h, w = frame.shape[:2]
                    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
                    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
                    calib = np.load(files_for_pixel)
                    mtx = calib['mtx']
                    for i, (x, y) in enumerate(corners):
                        X, Y = pixel_to_camera(x, y, mtx, w, h)
                        cv2.circle(frame, (x, y), 8, colors[i], -1)
                        # 根据角落位置调整文字偏移，避免重叠
                        if x < w // 2 and y < h // 2:
                            tx, ty = x + 10, y + 30
                        elif x >= w // 2 and y < h // 2:
                            tx, ty = x - 220, y + 30
                        elif x < w // 2 and y >= h // 2:
                            tx, ty = x + 10, y - 10
                        else:
                            tx, ty = x - 220, y - 10
                        cv2.putText(
                            frame,
                            f"({x},{y}) ({X:.2f},{Y:.2f})m",
                            (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2
                        )

                # 显示图像
                cv2.imshow(window_name, frame)
                # 按 'q' 键退出程序
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)

        # 清理资源
        cv2.destroyAllWindows()
        main_thread.join()

    except Exception as e:
        print(f"An error occurred: {e}")

### wjh start in 2025-04-20
### wjh 2025-04-20
### wjh 2025-04-20