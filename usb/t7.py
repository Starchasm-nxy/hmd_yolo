import cv2
import threading
import queue
import numpy as np
import os

# ========== 可调参数 ==========
CAM_PATH = '/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0'
WIDTH, HEIGHT = 640, 480       # 根据摄像头支持的分辨率设定（MJPG）
FPS = 30
BUFFER_SIZE = 2
CALIB_FILE = 'fisheye_calib.npz'    # 标定文件路径（如果不存在则自动跳过校正）
# =============================

class CameraThread(threading.Thread):
    def __init__(self, path, queue, calib_file=None):
        super().__init__()
        self.path = path
        self.queue = queue
        self.calib_file = calib_file
        self._stop = threading.Event()

        # 畸变矫正相关变量
        self.mapx = None
        self.mapy = None
        self.do_undistort = False

    def load_calibration(self, filepath):
        """尝试加载标定文件，成功则生成 remap 映射表"""
        if not os.path.exists(filepath):
            print(f"[WARN] 标定文件 {filepath} 不存在，将跳过畸变校正。")
            return False
        try:
            data = np.load(filepath)
            mtx = data['mtx']
            dist = data['dist']
            # 计算最优新内参和映射矩阵
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (WIDTH, HEIGHT), 0, (WIDTH, HEIGHT))
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (WIDTH, HEIGHT), 5)
            print(f"[INFO] 畸变校正已启用（来自 {filepath}）")
            return True
        except Exception as e:
            print(f"[ERROR] 加载标定文件失败: {e}")
            return False

    def run(self):
        # 加载畸变校正
        if self.calib_file:
            self.do_undistort = self.load_calibration(self.calib_file)

        cap = cv2.VideoCapture(self.path, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        # 确认实际参数
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        print(f"实际格式: {codec}, 分辨率: {int(cap.get(3))}x{int(cap.get(4))}")

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                continue

            # 畸变矫正（如果启用）
            if self.do_undistort:
                frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)

            if self.queue.full():
                self.queue.get_nowait()
            self.queue.put(frame)
        cap.release()

    def stop(self):
        self._stop.set()

if __name__ == '__main__':
    frame_queue = queue.Queue(maxsize=BUFFER_SIZE)
    cam_thread = CameraThread(CAM_PATH, frame_queue, calib_file=CALIB_FILE)
    cam_thread.start()

    print("按 'q' 或 Ctrl+C 退出...")
    try:
        while True:
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            cv2.imshow('USB Camera (Undistorted)', frame)
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