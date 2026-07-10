import cv2
import pyrealsense2 as rs
import numpy as np
import queue
import threading
import time

# -------------------- 配置参数 --------------------
MAX_DISTANCE_M = 5.0
EXCLUDE_RADIUS = 20          # 屏蔽半径（像素）
MORPH_KERNEL_SIZE = 3        # 形态学开运算核大小

# -------------------- 摄像头帧捕获类 --------------------
class RealSenseCapture(threading.Thread):
    """独立线程捕获彩色+深度帧，放入队列"""
    def __init__(self, color_queue, depth_queue, stop_event):
        super().__init__()
        self.color_queue = color_queue
        self.depth_queue = depth_queue
        self.stop_event = stop_event
        self.pipeline = None
        self.config = None

    def run(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)
        
        try:
            while not self.stop_event.is_set():
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                
                # 转为 numpy 数组并放入队列（非阻塞）
                color_img = np.asanyarray(color_frame.get_data())
                depth_img = np.asanyarray(depth_frame.get_data())
                
                # 如果队列满了就丢弃旧帧
                if self.color_queue.full():
                    self.color_queue.get()
                if self.depth_queue.full():
                    self.depth_queue.get()
                self.color_queue.put(color_img)
                self.depth_queue.put(depth_img)
        except Exception as e:
            print(f"Capture thread error: {e}")
        finally:
            self.pipeline.stop()

# -------------------- 深度处理类（最近三点检测）--------------------
class NearestPointsProcessor(threading.Thread):
    """独立线程：从深度队列取帧，计算最近三个点，结果放入结果队列"""
    def __init__(self, depth_queue, result_queue, stop_event):
        super().__init__()
        self.depth_queue = depth_queue
        self.result_queue = result_queue
        self.stop_event = stop_event

    def process(self, depth_img):
        """找出最近三个不重叠的点，返回列表 [(x,y,dist_cm), ...]"""
        depth_scale = 0.001  # 通常 RealSense 深度比例约 0.001，可动态获取但这里简化
        max_raw = int(MAX_DISTANCE_M / depth_scale)
        
        # 过滤无效点和过远点
        mask = (depth_img > 0) & (depth_img < max_raw)
        filtered = np.where(mask, depth_img, np.iinfo(depth_img.dtype).max)
        
        # 开运算去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                           (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
        binary = (filtered < np.iinfo(depth_img.dtype).max).astype(np.uint8) * 255
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        filtered[opened == 0] = np.iinfo(depth_img.dtype).max
        
        points = []
        work = filtered.copy()
        for _ in range(3):
            min_val = np.min(work)
            if min_val == np.iinfo(depth_img.dtype).max:
                break
            min_idx = np.argmin(work)
            y, x = np.unravel_index(min_idx, work.shape)
            dist_cm = min_val * depth_scale * 100
            points.append((x, y, dist_cm))
            # 屏蔽周围区域
            cv2.circle(work, (x, y), EXCLUDE_RADIUS, 
                       np.iinfo(depth_img.dtype).max, -1)
        return points

    def run(self):
        while not self.stop_event.is_set():
            try:
                depth_img = self.depth_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if depth_img is None:
                continue
            points = self.process(depth_img)
            # 非阻塞放入结果队列，如果满了就替换最新的等待结果
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            self.result_queue.put(points, timeout=0.1)

# -------------------- 主显示线程（在主线程中运行）--------------------
def display_loop(color_queue, result_queue, stop_event):
    """主函数运行显示逻辑，从两个队列取数据，绘制并显示"""
    print("系统启动。按 'q' 退出。")
    while not stop_event.is_set():
        try:
            color_img = color_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        # 获取最新的处理结果（不阻塞）
        points = None
        try:
            points = result_queue.get_nowait()
        except queue.Empty:
            pass
        
        # 绘制
        if points is not None:
            for i, (x, y, dist) in enumerate(points):
                cv2.circle(color_img, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(color_img, f"{i+1}: {dist:.1f}cm",
                            (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(color_img, f"Top {len(points)} nearest",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        else:
            cv2.putText(color_img, "No valid points",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        
        cv2.imshow('Nearest 3 Points', color_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    cv2.destroyAllWindows()

# -------------------- 主入口 --------------------
if __name__ == '__main__':
    # 队列大小限制，避免内存暴涨
    color_queue = queue.Queue(maxsize=2)
    depth_queue = queue.Queue(maxsize=2)
    result_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    # 启动捕获线程和处理器线程
    capture_thread = RealSenseCapture(color_queue, depth_queue, stop_event)
    processor_thread = NearestPointsProcessor(depth_queue, result_queue, stop_event)
    
    capture_thread.start()
    processor_thread.start()
    
    # 主线程负责显示
    try:
        display_loop(color_queue, result_queue, stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        capture_thread.join(timeout=2)
        processor_thread.join(timeout=2)
        print("程序已退出。")