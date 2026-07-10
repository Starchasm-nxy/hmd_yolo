import cv2
import pyrealsense2 as rs
import numpy as np
import queue
import threading
import time
from sklearn.cluster import DBSCAN

# ==================== 可调参数 ====================
MAX_DISTANCE_M = 4.0           # 只考虑4米内的点
DBSCAN_EPS = 0.1               # 5厘米聚类半径（三维，单位：米）
DBSCAN_MIN_SAMPLES = 50        # 最小聚类点数
DEPTH_SCALE = 0.001            # 默认深度比例（校准后可能微调）
GRID_STEP = 10                 # 采样步长，得到 64x48 个点
MIN_AREA = 100                 # 最小面积（1格：10x10 px）
MAX_AREA = 102400             # 最大面积（画面 1/3）
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
    """从深度队列取帧，用64x48采样点进行DBSCAN，过滤面积后框出最近物体"""
    def __init__(self, depth_queue, result_queue, intrinsics_queue, stop_event):
        super().__init__()
        self.depth_queue = depth_queue
        self.result_queue = result_queue
        self.intrinsics_queue = intrinsics_queue
        self.stop_event = stop_event
        self.fx = self.fy = self.cx = self.cy = None
        self.grid_u = None
        self.grid_v = None

    def _wait_for_intrinsics(self):
        fx, fy, cx, cy = self.intrinsics_queue.get()
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        # 预先生成 64x48 采样网格坐标
        w, h = 640, 480
        u, v = np.meshgrid(np.arange(0, w, GRID_STEP), np.arange(0, h, GRID_STEP))
        self.grid_u = u.flatten()
        self.grid_v = v.flatten()

    def process(self, depth_img):
        if self.fx is None or self.grid_u is None:
            return None

        # 获取采样点的深度值（毫米）
        z_mm = depth_img[self.grid_v, self.grid_u].astype(np.float32)
        valid = (z_mm > 500) & (z_mm < MAX_DISTANCE_M * 1000)
        u_val = self.grid_u[valid]
        v_val = self.grid_v[valid]
        z_val = z_mm[valid]

        if len(u_val) < DBSCAN_MIN_SAMPLES:
            return None

        # 转米并反投影
        z = z_val * 0.001
        X = (u_val - self.cx) * z / self.fx
        Y = (v_val - self.cy) * z / self.fy
        points = np.column_stack((X, Y, z))

        # DBSCAN 聚类
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(points)
        labels = db.labels_

        # 过滤噪声，收集簇
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        if len(unique_labels) == 0:
            return None

        clusters = []
        for label in unique_labels:
            mask = labels == label
            cluster_u = u_val[mask]
            cluster_v = v_val[mask]
            cluster_z = z_val[mask]

            # 计算最小外接矩形和面积
            x1, y1 = cluster_u.min(), cluster_v.min()
            x2, y2 = cluster_u.max(), cluster_v.max()
            area = (x2 - x1 + 1) * (y2 - y1 + 1)

            # 面积过滤：小于1格(100px)或大于画面1/3(102400px)丢弃
            if area < MIN_AREA or area > MAX_AREA:
                continue

            ave_cm = cluster_z.mean() / 10.0
            clusters.append((x1, y1, x2, y2, ave_cm))

        if not clusters:
            return None

        # 返回最近的物体（平均深度最小）
        clusters.sort(key=lambda c: c[4])
        return clusters[0]

    def run(self):
        self._wait_for_intrinsics()
        last_process_time = 0

        while not self.stop_event.is_set():
            try:
                depth_img = self.depth_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if depth_img is None:
                continue

            now = time.time()
            if now - last_process_time >= 0.1:
                result = self.process(depth_img)
                last_process_time = now

                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.result_queue.put(result, timeout=0.1)
            # 否则丢弃该帧，保持队列不堵塞


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
    color_queue = queue.Queue(maxsize=2)
    depth_queue = queue.Queue(maxsize=2)
    result_queue = queue.Queue(maxsize=2)
    intrinsics_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    capture_thread = RealSenseCapture(color_queue, depth_queue, intrinsics_queue, stop_event)
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