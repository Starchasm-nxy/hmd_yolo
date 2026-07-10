import cv2
import threading
import queue
import time

# ---------- 摄像头配置 ----------
CAM_PATH = '/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0'          # 或设备索引如 4
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 15
BUFFER_SIZE = 2                  # 缓冲队列大小，避免堆积

# ---------- 帧读取线程 ----------
class CameraCapture(threading.Thread):
    def __init__(self, cam_path, frame_queue):
        super().__init__()
        self.cam_path = cam_path
        self.frame_queue = frame_queue
        self.stop_event = threading.Event()

    def run(self):
        cap = cv2.VideoCapture(self.cam_path)
        if not cap.isOpened():
            print("无法打开摄像头")
            return

        # 尝试设置 MJPG 和高帧率
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("读取帧失败")
                break
            # 非阻塞放入队列，队列满则丢弃旧帧
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)
        cap.release()

    def stop(self):
        self.stop_event.set()

# ---------- 主函数 ----------
if __name__ == '__main__':
    frame_queue = queue.Queue(maxsize=BUFFER_SIZE)
    cam_thread = CameraCapture(CAM_PATH, frame_queue)
    cam_thread.start()

    # 等待第一帧
    time.sleep(1)

    print("USB 摄像头已启动，按 'q' 键退出...")
    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            print("未收到画面")
            continue

        cv2.imshow('USB Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam_thread.stop()
            break

    cam_thread.join()
    cv2.destroyAllWindows()