import cv2
import pyrealsense2 as rs
import numpy as np
import queue
import threading
import time
from sklearn.cluster import DBSCAN

# ==================== 可调参数 ====================
MAX_DISTANCE_M = 2.0            # 只考虑2米内的点
DBSCAN_EPS = 0.1               # 10厘米聚类半径（三维，单位：米）
DBSCAN_MIN_SAMPLES = 50        # 最小聚类点数
DEPTH_SCALE = 0.001            # 默认深度比例（校准后可能微调）
DOWNSAMPLE_STEP = 2            # 降采样步长
# =================================================

class RealSenseCapture(threading.Thread):
    """负责从RealSense获取彩色帧、深度帧，并传递相机内参"""
    def __init__(self, color_queue, depth_queue, intrinsics_queue, stop_event):
        super().__init__()
        self.color_queue = color_queue
        self.depth_queue = depth_queue
        self.intrinsics_queue = intrinsics_queue
        self.stop_event = stop_event

    def run(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)

        # 获取相机内参（彩色流与深度对齐后，使用彩色流内参即可）
        profile = pipeline.get_active_profile()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics_queue.put((intr.fx, intr.fy, intr.ppx, intr.ppy))

        try:
            while not self.stop_event.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color_img = np.asanyarray(color_frame.get_data())
                depth_img = np.asanyarray(depth_frame.get_data())

                # 满队列时丢弃旧帧，保证实时性
                if self.color_queue.full():
                    self.color_queue.get()
                if self.depth_queue.full():
                    self.depth_queue.get()
                self.color_queue.put(color_img)
                self.depth_queue.put(depth_img)
        except Exception as e:
            print(f"[Capture Error] {e}")
        finally:
            pipeline.stop()


class NearestClusterProcessor(threading.Thread):
    """从深度队列取帧，用DBSCAN找出最近的物体簇，返回其边界框"""
    def __init__(self, depth_queue, result_queue, intrinsics_queue, stop_event):
        super().__init__()
        self.depth_queue = depth_queue
        self.result_queue = result_queue
        self.intrinsics_queue = intrinsics_queue
        self.stop_event = stop_event
        self.fx = self.fy = self.cx = self.cy = None
        self.width = self.height = None

    def _wait_for_intrinsics(self):
        """等待获取相机内参"""
        fx, fy, cx, cy = self.intrinsics_queue.get()
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.width, self.height = 640, 480   # 与配置一致

    def process(self, depth_img):
        """基于DBSCAN找出最近物体的矩形框，返回 (x1,y1,x2,y2,ave_cm) 或 None"""
        if self.fx is None:
            return None

        h, w = depth_img.shape
        # 生成像素网格坐标
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        # 以毫米为单位的深度值
        z_mm = depth_img.astype(np.float32)
        # 有效深度掩码
        valid_mask = (z_mm > 0) & (z_mm < MAX_DISTANCE_M * 1000)

        # 只考虑距离最近的前景区域（自适应阈值）
        if not np.any(valid_mask):
            return None
        min_z = z_mm[valid_mask].min()
        # 保留深度 <= min_z + 0.5m 的点（假设最近物体整体深度跨度不超过0.5m）
        near_mask = valid_mask & (z_mm <= min_z + 500)  # 500 mm

        # 提取近处点的像素坐标和深度
        u_near = u[near_mask]
        v_near = v[near_mask]
        z_near = z_mm[near_mask]

        # 降采样（每 ds 个像素取一个）
        ds = DOWNSAMPLE_STEP
        if len(u_near) > 10000:  # 点数太多时才降采样
            sample_indices = np.arange(0, len(u_near), ds)
            u_near = u_near[sample_indices]
            v_near = v_near[sample_indices]
            z_near = z_near[sample_indices]

        if len(u_near) < DBSCAN_MIN_SAMPLES:
            return None

        # 将深度从毫米转为米
        z = z_near * 0.001
        # 反投影到相机坐标系
        X = (u_near - self.cx) * z / self.fx
        Y = (v_near - self.cy) * z / self.fy

        # 三维 DBSCAN 聚类
        points = np.column_stack((X, Y, z))
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(points)
        labels = db.labels_

        # 过滤掉噪声（label = -1）
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        if len(unique_labels) == 0:
            return None

        # 为每个簇计算图像矩形和平均深度
        clusters = []
        for label in unique_labels:
            mask = labels == label
            cluster_u = u_near[mask]
            cluster_v = v_near[mask]
            cluster_z = z_near[mask]
            if len(cluster_u) < DBSCAN_MIN_SAMPLES:
                continue
            x1, y1 = cluster_u.min(), cluster_v.min()
            x2, y2 = cluster_u.max(), cluster_v.max()
            ave_cm = cluster_z.mean() / 10.0   # 平均厘米
            clusters.append((x1, y1, x2, y2, ave_cm))

        if not clusters:
            return None

        # 选择平均深度最小的簇（即最近的物体）
        clusters.sort(key=lambda c: c[4])   # 按距离排序
        return clusters[0]   # 返回 (x1,y1,x2,y2,dist_cm)

    def run(self):
        self._wait_for_intrinsics()
        last_process_time = 0  # 上次聚类时间戳

        while not self.stop_event.is_set():
            # 总是从深度队列取出一帧，避免队列阻塞
            try:
                depth_img = self.depth_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if depth_img is None:
                continue

            # 检查是否到达1秒间隔
            now = time.time()
            if now - last_process_time >= 1.0:
                result = self.process(depth_img)
                last_process_time = now

                # 放入结果队列（非阻塞，最新结果覆盖旧结果）
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.result_queue.put(result, timeout=0.1)
            # 如果时间未到，直接丢弃该帧，不进行处理


def display_loop(color_queue, result_queue, stop_event):
    """主线程：显示图像并绘制矩形框"""
    print("系统启动。按 'q' 退出。")
    while not stop_event.is_set():
        try:
            color_img = color_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        result = None
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            pass

        if result is not None:
            x1, y1, x2, y2, dist_cm = result
            cv2.rectangle(color_img, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            cv2.putText(color_img, f"{dist_cm:.1f} cm",
                        (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(color_img, "No object",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Nearest Object (DBSCAN)', color_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 线程间通信队列
    color_queue = queue.Queue(maxsize=2)
    depth_queue = queue.Queue(maxsize=2)
    result_queue = queue.Queue(maxsize=2)
    intrinsics_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    # 启动捕获线程
    capture_thread = RealSenseCapture(color_queue, depth_queue, intrinsics_queue, stop_event)
    # 启动处理线程
    proc_thread = NearestClusterProcessor(depth_queue, result_queue, intrinsics_queue, stop_event)

    capture_thread.start()
    proc_thread.start()

    try:
        display_loop(color_queue, result_queue, stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        capture_thread.join(timeout=2)
        proc_thread.join(timeout=2)
        print("程序退出。")