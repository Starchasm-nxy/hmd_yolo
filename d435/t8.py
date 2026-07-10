import cv2
import pyrealsense2 as rs
import numpy as np
import queue
import threading
import time

# ==================== 可调参数 ====================
MAX_DISTANCE_M = 2.0            # 只考虑2米内的点
GRID_STEP = 5                 # 采样步长，得到 64x48 个点
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
        while not self.stop_event.is_set():
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                pipeline.start(config)

                profile = pipeline.get_active_profile()
                intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                self.intrinsics_queue.put((intr.fx, intr.fy, intr.ppx, intr.ppy))

                while not self.stop_event.is_set():
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue

                    color_img = np.asanyarray(color_frame.get_data())
                    depth_img = np.asanyarray(depth_frame.get_data())

                    if self.color_queue.full():
                        self.color_queue.get()
                    if self.depth_queue.full():
                        self.depth_queue.get()
                    self.color_queue.put(color_img)
                    self.depth_queue.put(depth_img)

            except RuntimeError as e:
                print(f"[Capture] Frame timeout, restarting pipeline: {e}")
                pipeline.stop()
                time.sleep(0.5)
            except Exception as e:
                print(f"[Capture] Error: {e}")
                pipeline.stop()
                break
        # 清理
        try:
            pipeline.stop()
        except:
            pass


def display_loop(color_queue, depth_queue, stop_event):
    """主线程：显示彩色图像，并在采样点上绘制彩色圆点表示距离"""
    # 等待内参（但这里我们不需要反投影，只需像素坐标，所以无需内参）
    w, h = 640, 480
    # 生成采样网格坐标 (64x48)
    u, v = np.meshgrid(np.arange(0, w, GRID_STEP), np.arange(0, h, GRID_STEP))
    us = u.flatten()
    vs = v.flatten()

    print("系统启动。按 'q' 退出。")
    while not stop_event.is_set():
        try:
            color_img = color_queue.get(timeout=0.1)
            depth_img = depth_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # 在彩色图像上绘制每个采样点
        for i in range(len(us)):
            x, y = us[i], vs[i]
            z_mm = depth_img[y, x]
            # 滤除无效或过远点
            # if z_mm == 0 or z_mm > MAX_DISTANCE_M * 1000:
            #     continue
            dist_cm = z_mm / 10.0   # 转换为厘米

            # 颜色方案：近红(<50cm) -> 中黄(50-150cm) -> 远绿(>150cm)
            if dist_cm < 30:
                color = (0, 0, 255)      # 红色 (BGR)
            elif dist_cm < 60:
                color = (0, 255, 255)    # 黄色
            elif dist_cm < 90:
                color = (0, 255, 0)      # 绿色
            else:
                color = (255, 0, 0)
            cv2.circle(color_img, (int(x), int(y)), 1, color, -1)

        # 图例说明
        cv2.putText(color_img, "Red:<50cm  Yellow:50-150  Green:>150",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('64x48 Depth Sampling', color_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    color_queue = queue.Queue(maxsize=2)
    depth_queue = queue.Queue(maxsize=2)
    intrinsics_queue = queue.Queue(maxsize=1)  # 这里暂时不用，但保留接口
    stop_event = threading.Event()

    capture_thread = RealSenseCapture(color_queue, depth_queue, intrinsics_queue, stop_event)
    capture_thread.start()

    try:
        display_loop(color_queue, depth_queue, stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        capture_thread.join(timeout=2)
        print("程序退出。")