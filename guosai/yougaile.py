#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
基于 YOLO + RealSense D435i + USB 广角相机的目标检测与定位程序（优化整合版）
================================================================================
原版作者：wjh  2025-09-22
优化整合：保留全部原始参数，改进线程架构、消除重复代码、增强资源管理

功能说明：
1. 通过读取 data.txt 文件接收外部指令，切换工作模式。
2. USB 远距离模式（c=3）：使用 USB 广角相机 + YOLO 检测，通过 DBSCAN 聚类
   稳定输出两个目标桶的世界坐标（X,Y），写入 gaozhi.txt。
3. D435i 近距离模式（c=2）：使用深度相机 + YOLO 检测，输出中心目标像素坐标，
   支持寻找目标（b=2）和锁定目标（b=1）两种子模式。

优化要点：
- 主控线程使用队列阻塞等待模式切换，替代轮询全局变量。
- 命令处理改用字典映射，消除冗长的 if-elif 分支。
- 公共绘制、坐标转换、文件写入等函数统一抽取，避免重复代码。
- RealSense 相机封装为上下文管理器，确保资源自动释放。
- 文件写入采用“临时文件+原子重命名”防止读写冲突。
- 标定数据全局加载一次，减少重复 I/O。
- 引入 logging 模块替代 print，方便调试和日志记录。
================================================================================
"""

import os
import cv2
import time
import queue
import threading
import logging
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

# ---------------------------- 日志配置 ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---------------------------- 参数设置（全部保持原值不变） ----------------------------
Z = 4.31                                 # 相机距离地面的假设高度（米），用于单目测距
eps = 50                                 # DBSCAN 聚类半径（像素）
min_samples = 15                         # DBSCAN 最小聚类点数
max_time = 5.0                           # 最大等待聚类时间（秒）
files_to_clear = ['data.txt', 'gaozhi.txt']   # 启动时需要清空的通信文件
files_for_pixel = 'calib_result.npz'     # 相机畸变校正参数文件
cam_num = '/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0'  # USB 设备路径

# 模型权重路径（原样保留）
MODEL_USB_PATH = "/home/son/yolov5_d435i_detection-main/gs4mcz.pt"
MODEL_D435_PATH = "/home/son/yolov5_d435i_detection-main/2m.pt"

# ---------------------------- 全局共享资源 ----------------------------
# 一次性加载畸变校正参数，避免每次读取文件
CALIB_DATA = np.load(files_for_pixel)
MTX = CALIB_DATA['mtx']          # 相机内参矩阵
DIST = CALIB_DATA['dist']        # 畸变系数

# 显示队列：将绘制好的图像传递给主线程显示
display_queue = queue.Queue(maxsize=3)

# 模式切换队列：命令监听线程将新指令放入此队列，主控线程阻塞等待
mode_queue = queue.Queue()

# 命令映射字典：将 data.txt 中的字符串映射为内部模式参数
# 结构：{命令字符串: {'c': 主模式, 'b': 子模式(可选), 'd': 组合模式(可选)}}
COMMAND_MAP = {
    'mlkai': {'c': 3, 'd': 6},
    'rmkai': {'c': 3, 'd': 5},
    'rlkai': {'c': 3, 'd': 4},
    'lmkai': {'c': 3, 'd': 3},
    'lrkai': {'c': 3, 'd': 2},
    'mrkai': {'c': 3, 'd': 1},
    '2m':    {'c': 2, 'b': 2},
    '1m':    {'c': 2, 'b': 1},
}

# ---------------------------- 通用工具函数 ----------------------------
def clear_files(files):
    """清空指定的文件列表（用于初始化通信文件）"""
    for file_name in files:
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, 'w') as f:
            pass
    logger.info("通讯文件已清空: %s", files)

def safe_write(filepath, content):
    """
    原子写入文件：先写入临时文件，再重命名为目标文件。
    避免读写冲突，防止其他程序读到不完整的内容。
    """
    tmp_path = filepath + '.tmp'
    with open(tmp_path, 'w') as f:
        f.write(content)
    os.replace(tmp_path, filepath)   # 原子操作（Unix/Linux 下）

def pixel_to_camera(ux, uy, w, h):
    """
    将像素坐标转换为相机坐标系下的 X, Y 坐标（单位：米）。
    假设地面为平面，相机光轴垂直于地面，且已知相机高度 Z。
    """
    fx = MTX[0, 0]
    fy = MTX[1, 1]
    cx = w / 2.0
    cy = h / 2.0
    X = (ux - cx) * Z / fx
    Y = (uy - cy) * Z / fy
    return X, Y

def draw_square(image, box, names, r):
    """
    在图像上绘制检测框、类别标签、置信度和中心点。
    返回中心点的像素坐标 (ux, uy)。
    """
    ux = int((r[0] + r[2]) / 2)
    uy = int((r[1] + r[3]) / 2)
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    label = f"{names[cls]} {conf:.2f}"
    cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (221, 185, 193), 2)
    cv2.putText(image, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (176, 196, 222), 2)
    cv2.circle(image, (ux, uy), 5, (240, 240, 240), -1)
    return ux, uy

def draw_detections(image, results, convert_to_world=False, w=None, h=None):
    """
    统一处理 YOLO 检测结果的可视化，并可选择是否叠加世界坐标。
    返回检测到的所有目标的中心像素坐标列表。
    """
    points = []
    for result in results:
        names = result.names
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            r = box.xyxy[0].cpu().numpy().astype(int)
            ux, uy = draw_square(image, box, names, r)
            if convert_to_world and w is not None and h is not None:
                X, Y = pixel_to_camera(ux, uy, w, h)
                cv2.putText(image, f"({X:.2f}m, {Y:.2f}m)", (ux + 10, uy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 240, 240), 3)
            points.append((ux, uy))
    return points

def dbscan_cluster_and_draw(image, points, eps, min_samples):
    """DBSCAN 聚类并绘制结果，返回聚类中心列表（像素坐标）"""
    if len(points) == 0:
        logger.warning("没有桶可供聚类")
        return []
    X = np.array(points)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    cluster_centers = []
    # 为每个簇生成随机颜色
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(max(labels) + 2)]
    for i in range(max(labels) + 1):
        cluster_pts = X[labels == i]
        center = np.mean(cluster_pts, axis=0)
        cluster_centers.append(center)
        # 绘制簇内点及中心
        if image is not None:
            for pt in cluster_pts:
                cv2.circle(image, (int(pt[0]), int(pt[1])), 8, colors[i], -1)
            cv2.circle(image, (int(center[0]), int(center[1])), 15, colors[i], 3)
            cv2.putText(image, f"cluster{i+1}", (int(center[0]), int(center[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)
    # 保存聚类结果图像（便于复盘）
    cv2.imwrite(f"聚类结果图/cluster_{int(time.time())}.jpg", image)
    return cluster_centers

# ---------------------------- RealSense 上下文管理器 ----------------------------
class RealSenseCamera:
    """RealSense D435i 相机的上下文管理器，自动处理启动与停止"""
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)

    def __enter__(self):
        self.pipeline.start(self.config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pipeline.stop()

    def get_frames(self):
        """获取对齐后的深度帧和彩色帧"""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        return depth_frame, color_frame

# ---------------------------- USB 采集与校正线程 ----------------------------
def usb_capture(camera, frame_queue, start_event, stop_event):
    """
    从 USB 摄像头持续读取图像，进行畸变校正，并放入队列供检测线程使用。
    """
    start_event.wait()   # 等待外部启动信号
    logger.info("USB 采集线程启动")

    # 计算畸变校正映射表（只计算一次）
    h, w = 480, 640
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(MTX, DIST, (w, h), 0, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(MTX, DIST, None, newcameramtx, (w, h), 5)

    while not stop_event.is_set():
        ret, frame = camera.read()
        if not ret:
            logger.error("无法读取 USB 摄像头画面")
            stop_event.set()
            break
        # 畸变校正
        undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        # 非阻塞放入队列
        try:
            frame_queue.put(undistorted, timeout=0.1)
        except queue.Full:
            pass
        time.sleep(0.01)   # 约100fps
    camera.release()
    logger.info("USB 采集线程退出")

# ---------------------------- USB 检测线程 ----------------------------
def usb_detect(model, frame_queue, start_event, stop_event, d):
    """
    USB 模式检测线程：
    - 从队列获取校正后的图像
    - YOLO 推理
    - 收集点进行 DBSCAN 聚类
    - 根据参数 d 选择两个桶输出世界坐标
    """
    start_event.wait()
    logger.info("USB 检测线程启动，d=%d", d)

    # 初始输出占位符
    safe_write('gaozhi.txt', "0 0 0 0")
    last_result = None
    points = []                     # 累积的检测点
    last_cluster_time = time.time()
    clustered_once = False
    final_result = None

    while not stop_event.is_set():
        if clustered_once:
            # 聚类完成后持续输出相同结果，不再处理新帧（降低 CPU 占用）
            if final_result is not None:
                safe_write('gaozhi.txt', final_result)
            time.sleep(0.5)
            continue

        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        h, w = frame.shape[:2]
        results = model.predict(source=frame, device='cpu', show=False,
                                stream=False, verbose=False, iou=0.45, conf=0.6)

        # 绘制检测结果并收集像素点
        image = frame.copy() if len(results) > 0 else frame
        pts = draw_detections(image, results, convert_to_world=True, w=w, h=h)
        points.extend(pts)

        current_time = time.time()
        trigger_cluster = False
        if len(points) >= min_samples:
            trigger_cluster = True
        elif (current_time - last_cluster_time) >= max_time and len(points) > 0:
            trigger_cluster = True

        if trigger_cluster:
            # 执行聚类
            centers = dbscan_cluster_and_draw(image, points, eps, min_samples)
            camera_centers = [pixel_to_camera(c[0], c[1], w, h) for c in centers]
            camera_centers.sort(key=lambda x: x[0])   # 按 X 坐标升序

            result_str = "0 0 0 0"
            if len(camera_centers) >= 3:
                # 根据 d 选择两个桶的组合（原逻辑完全保留）
                if d == 6:
                    c1, c2 = camera_centers[1], camera_centers[0]
                elif d == 5:
                    c1, c2 = camera_centers[2], camera_centers[1]
                elif d == 4:
                    c1, c2 = camera_centers[2], camera_centers[0]
                elif d == 3:
                    c1, c2 = camera_centers[0], camera_centers[1]
                elif d == 2:
                    c1, c2 = camera_centers[0], camera_centers[2]
                else:  # d == 1
                    c1, c2 = camera_centers[1], camera_centers[2]

                # 范围检查（原逻辑）
                in_range = lambda x, y: -4 <= x <= 4 and -2.5 <= y <= 2.5
                c1_ok = in_range(c1[0], c1[1])
                c2_ok = in_range(c2[0], c2[1])
                if c1_ok and c2_ok:
                    result_str = f"{c1[0]:.1f} {c1[1]:.1f} {c2[0]:.1f} {c2[1]:.1f}"
                elif c1_ok:
                    result_str = f"{c1[0]:.1f} {c1[1]:.1f} 0 0"
                elif c2_ok:
                    result_str = f"0 0 {c2[0]:.1f} {c2[1]:.1f}"
                else:
                    result_str = "0 0 0 0"

            if result_str != last_result:
                safe_write('gaozhi.txt', result_str)
                last_result = result_str

            clustered_once = True
            final_result = result_str
        else:
            last_cluster_time = current_time

        # 将图像放入显示队列
        try:
            display_queue.put((image, "USB_View"), timeout=0.1)
        except queue.Full:
            pass

    logger.info("USB 检测线程退出")

# ---------------------------- D435i 检测线程 ----------------------------
def d435_detect(model, start_event, stop_event, b):
    """
    D435i 近距离检测线程：
    - b=2：寻找目标，输出最靠近图像中心的桶的像素坐标
    - b=1：锁定目标，若无目标则保持上一次输出
    """
    start_event.wait()
    logger.info("D435 检测线程启动，b=%d", b)

    safe_write('gaozhi.txt', "0")
    last_result = "0"
    last_value = "0"

    with RealSenseCamera() as cam:
        while not stop_event.is_set():
            depth_frame, color_frame = cam.get_frames()
            if depth_frame is None:
                continue
            color_image = np.asanyarray(color_frame.get_data())

            results = model.predict(source=color_image, device='cpu', show=False,
                                    stream=False, verbose=False, iou=0.45, conf=0.6)

            tong_list = []   # 候选桶列表：(偏移量, (ux, uy))
            image = color_image.copy()
            for result in results:
                for box in result.boxes:
                    r = box.xyxy[0].cpu().numpy().astype(int)
                    ux, uy = draw_square(image, box, result.names, r)
                    area = (r[2] - r[0]) * (r[3] - r[1])
                    if b == 2 and area > 30000:
                        continue   # 面积过滤
                    # 偏移量 = 距图像中心（424）的水平距离
                    offset = abs(ux - 424)
                    tong_list.append((offset, (ux, uy)))

            result_str = "0"
            if tong_list:
                # 选择偏移量最小的目标
                best = min(tong_list, key=lambda x: x[0])
                center = best[1]
                result_str = f"1 {center[0]} {center[1]}"
                if b == 1:
                    last_value = result_str
            else:
                if b == 2:
                    result_str = "0"
                else:  # b == 1 且无目标，使用记忆值
                    result_str = last_value

            if result_str != last_result:
                safe_write('gaozhi.txt', result_str)
                last_result = result_str

            try:
                display_queue.put((image, "D435_View"), timeout=0.1)
            except queue.Full:
                pass

    logger.info("D435 检测线程退出")

# ---------------------------- 命令监听线程 ----------------------------
def read_file_content(shuru_file_path, start_event, stop_event):
    """
    监听 data.txt 文件内容变化。
    当出现有效命令时，将对应参数放入 mode_queue 通知主控线程，
    并设置 start_event 告知检测线程可以开始工作。
    """
    last_content = None
    logger.info("命令监听线程启动，监控文件: %s", shuru_file_path)
    while not stop_event.is_set():
        try:
            with open(shuru_file_path, 'r') as f:
                content = f.read().strip()
        except Exception:
            time.sleep(0.05)
            continue

        if content != last_content and content in COMMAND_MAP:
            config = COMMAND_MAP[content].copy()
            logger.info("收到新指令: %s -> %s", content, config)
            start_event.set()                 # 启动检测线程
            mode_queue.put(config)            # 通知主控线程切换模式
            last_content = content
        time.sleep(0.05)

# ---------------------------- 主控线程 ----------------------------
def main_control():
    """
    主控线程：等待 mode_queue 中的模式切换指令，动态创建/销毁检测线程。
    实现了 USB 和 D435 两种模式之间的无缝切换。
    """
    global c
    current_threads = []
    stop_event = None
    start_event = threading.Event()
    camera = None

    # 检查摄像头连接（仅打印提示）
    cap_test = cv2.VideoCapture(cam_num)
    if cap_test.isOpened():
        logger.info("USB 摄像头连接正常")
        cap_test.release()
    else:
        logger.warning("无法打开 USB 摄像头")

    ctx = rs.context()
    if len(ctx.devices) > 0:
        logger.info("D435 相机连接正常")
    else:
        logger.warning("未检测到 D435 相机")

    while True:
        # 阻塞等待新模式指令
        config = mode_queue.get()
        logger.debug("主控线程收到模式切换: %s", config)

        # 停止并清理旧线程
        if stop_event:
            stop_event.set()
        for t in current_threads:
            t.join(timeout=2.0)
        if camera:
            camera.release()
            camera = None
        cv2.destroyAllWindows()
        current_threads.clear()

        # 创建新的停止事件和启动事件
        stop_event = threading.Event()
        start_event = threading.Event()
        c = config['c']

        # 根据主模式启动对应线程
        if c == 3:   # USB 模式
            d = config['d']
            camera = cv2.VideoCapture(cam_num)
            frame_queue = queue.Queue(maxsize=3)
            t_cap = threading.Thread(target=usb_capture, args=(camera, frame_queue, start_event, stop_event))
            t_det = threading.Thread(target=usb_detect, args=(model_usb, frame_queue, start_event, stop_event, d))
            t_cap.start()
            t_det.start()
            current_threads = [t_cap, t_det]
            logger.info("已切换到 USB 模式 (d=%d)", d)

        elif c == 2:   # D435 模式
            b = config['b']
            t_det = threading.Thread(target=d435_detect, args=(model_d435, start_event, stop_event, b))
            t_det.start()
            current_threads = [t_det]
            logger.info("已切换到 D435 模式 (b=%d)", b)

        else:
            logger.warning("未知模式 c=%d", c)

# ---------------------------- 主程序入口 ----------------------------
if __name__ == '__main__':
    logger.info("程序启动，初始化中...")

    # 清空通信文件
    clear_files(files_to_clear)

    # 加载 YOLO 模型
    logger.info("加载 YOLO 模型...")
    model_usb = YOLO(MODEL_USB_PATH)
    model_d435 = YOLO(MODEL_D435_PATH)
    logger.info("模型加载完成")

    # 创建全局停止事件（用于程序退出）
    global_stop = threading.Event()

    # 启动命令监听线程
    t_listener = threading.Thread(target=read_file_content, args=('data.txt', threading.Event(), global_stop))
    t_listener.daemon = True
    t_listener.start()

    # 启动主控线程
    t_main = threading.Thread(target=main_control)
    t_main.daemon = True
    t_main.start()

    # 主线程负责图像显示
    logger.info("系统初始化完成，等待指令...")
    window_created = False
    try:
        while True:
            try:
                frame, win_name = display_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not window_created:
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_name, 900, 800)
                window_created = True

            # 如果是 USB 模式，可在图像四角显示世界坐标（原功能保留）
            if c == 3 and frame is not None:
                h, w = frame.shape[:2]
                corners = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
                for i, (x, y) in enumerate(corners):
                    X, Y = pixel_to_camera(x, y, w, h)
                    cv2.circle(frame, (x, y), 8, colors[i], -1)
                    offset_x = 10 if x < w//2 else -220
                    offset_y = 30 if y < h//2 else -10
                    cv2.putText(frame, f"({x},{y}) ({X:.2f},{Y:.2f})m",
                                (x + offset_x, y + offset_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)

            cv2.imshow(win_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("用户按下 'q'，程序退出")
                break

    except KeyboardInterrupt:
        logger.info("收到中断信号，程序退出")
    finally:
        global_stop.set()
        cv2.destroyAllWindows()
        logger.info("程序已退出")