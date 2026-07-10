import cv2
import threading
import queue

CAM_PATH = '/dev/video4'
WIDTH, HEIGHT = 1920, 1080
FPS = 30
BUFFER_SIZE = 2  # 只保留最新一帧，丢弃旧帧保证实时性

class CameraThread(threading.Thread):
    def __init__(self, path, queue):
        super().__init__()
        self.path = path
        self.queue = queue
        self._stop = threading.Event()

    def run(self):
        cap = cv2.VideoCapture(self.path, cv2.CAP_V4L2)
        # 强制使用 MJPG 格式
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        # 确认实际使用的格式
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        print(f"实际格式: {codec}, 分辨率: {int(cap.get(3))}x{int(cap.get(4))}")

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            # 队列满则丢弃最旧的一帧
            if self.queue.full():
                self.queue.get_nowait()
            self.queue.put(frame)
        cap.release()

    def stop(self):
        self._stop.set()

if __name__ == '__main__':
    frame_queue = queue.Queue(maxsize=2)
    cam_thread = CameraThread(CAM_PATH, frame_queue)
    cam_thread.start()

    print("按 'q' 或 Ctrl+C 退出...")
    try:
        while True:
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            cv2.imshow('USB Camera', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n正在退出...")
    finally:
        cam_thread.stop()
        cam_thread.join()
        cv2.destroyAllWindows()
        print("已安全退出。")