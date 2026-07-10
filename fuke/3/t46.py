import os
import cv2
import time
import queue
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs
from sklearn.cluster import DBSCAN

##参数设置
Z = 4.31  # 相机高度
eps=50 # DBSCAN聚类半径（像素）
min_samples=15 # DBSCAN最小聚类点数
max_time = 5.0 # 最大等待时间（秒）
files_to_clear = ['data.txt','gaozhi.txt']
files_for_pixel = 'calib_resultA.npz' # 畸变校正文件(根据相机改变)
# USB摄像头编号
cam_num =  '/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0' 
# 所需权重  
model_usb = YOLO("/home/fu/weights/tong_blue_v0.pt")
# ---- D435 配置 ----
D435_WIDTH = 848
D435_HEIGHT = 480
MODEL_2M_PATH = "/home/fu/weights/tong_blue_v0.pt"
MODEL_1M_PATH = "/home/fu/weights/tong_blue_v0.pt"
LOCK_MAX_HIT = 15
LOCK_MAX_MISS = 7
LOCK_SEARCH_RATIO = 2.5
LOCK_MIN_SEARCH_RADIUS = 130
LOCK_MAX_SEARCH_RADIUS = 270
MODEL_2M_CONF = 0.5
MODEL_2M_IMGSZ = 640
MODEL_1M_CONF = 0.5
MODEL_1M_IMGSZ = 640
MODEL_IOU = 0.45
MODEL_IMGSZ_STEP = 32
MODEL_DEVICE = 'cpu'
MAX_AREA = 223300
FRAME_SKIP_ENABLED = False
FRAME_SKIP_N = 2
HISTORY_CLEAR_ENABLED = True
HISTORY_CLEAR_TIMEOUT = 10.0

display_queue = queue.Queue(maxsize=3)
start_time = None

#清空文件
def clear_files(files):
    for file_name in files:
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, 'w') as f:
            print("[INFO]通讯txt文件已建立并清空")
            pass
        
# 检查摄像头连接
def check_camera(cam_num):
    cap = cv2.VideoCapture(cam_num)
    if not cap.isOpened():
        for _ in range(5):
            print(f"\033[31m[WARN] 无法打开usb摄像头，请检查连接或权限。\033[0m")
        cap.release()
    else:
        print(f"\033[32m[INFO] usb摄像头已成功连接\033[0m")
        cap.release()
    ctx = rs.context()
    if len(ctx.devices) == 0:
        for _ in range(5):
            print(f"\033[31m[WARN] 未检测到D435相机，请检查连接！\033[0m")
    else:
        print(f"\033[32m[INFO] D435相机已连接。\033[0m")
    
#相机切换及权重切换
def main_control(): 
    global c, cam_num, model, shuchu_file_path, start_event
    global t_test, t_detect, t_d435, camera
    last_c = None
    running = False
    camera = 0
    stop_event = None
    frame_queue = None
    check_camera(cam_num)
    while True:
        if c != last_c:
            print(f"[DEBUG] main_control 检测到 c 变化: {last_c} -> {c}")
            # 关闭旧线程和相机
            if running:
                stop_event.set()
                if t_test: t_test.join()
                if t_detect: t_detect.join()
                if camera: camera.release()
                if t_d435: t_d435.join()
                cv2.destroyAllWindows()
                time.sleep(0.2)
                running = False
            stop_event = threading.Event()
            # 启动新线程和相机
            if c == 3:
                camera = cv2.VideoCapture(cam_num)
                model = model_usb
                frame_queue = queue.Queue(maxsize=3)
                t_test = threading.Thread(target=test, args=(camera, frame_queue, start_event, stop_event))
                t_detect = threading.Thread(target=usb_detect, args=(model, frame_queue, shuchu_file_path, stop_event, start_event))
                t_test.start()
                t_detect.start()
                t_d435 = None
                running = True
            elif c == 2:
                t_d435 = threading.Thread(target=d435_detect, args=(shuchu_file_path, stop_event, start_event))
                t_d435.start()
                t_test = None
                t_detect = None
                camera = None
                running = True
            if c == 0:
                with open('gaozhi.txt', 'w') as f:
                    f.write('0')
            last_c = c
        time.sleep(0.1)
        
#坐标转换
def pixel_to_camera(ux, uy, mtx, w, h):  
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = w / 2
    cy = h / 2
    X = (ux - cx) * Z / fx
    Y = (uy - cy) * Z / fy
    return X, Y

class TrackAction(Enum):
    DETECT = auto()
    PREDICT = auto()
    LOST = auto()

@dataclass
class TrackResult:
    action: TrackAction
    x: int = 0
    y: int = 0
    is_locked: bool = False

@dataclass
class Detection:
    ux: int
    uy: int
    cls: int
    conf: float
    r: list
    w: int
    h: int
    area: int
    dis: float
    box: object
    names: dict

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def search_rect(x, y, w, h):
    shw = clamp(w * LOCK_SEARCH_RATIO / 2, LOCK_MIN_SEARCH_RADIUS, LOCK_MAX_SEARCH_RADIUS)
    shh = clamp(h * LOCK_SEARCH_RATIO / 2, LOCK_MIN_SEARCH_RADIUS, LOCK_MAX_SEARCH_RADIUS)
    sx1, sy1 = int(x - shw), int(y - shh)
    sx2, sy2 = int(x + shw), int(y + shh)
    return sx1, sy1, sx2, sy2, int(shw), int(shh), (sx2 - sx1) * (sy2 - sy1)

def pick_nearest(items):
    if not items:
        return None
    return min(items, key=lambda x: x[0])

def draw_lock_rect(canvas, x, y, shw, shh, color):
    cv2.rectangle(canvas, (x - shw, y - shh), (x + shw, y + shh), color, 2)
    cv2.circle(canvas, (x, y), 4, color, -1)

# USB畸变校正
def test(camera, frame_queue, start_event, stop_event):
    start_event.wait()
    print("[DEBUG] test线程已启动")
    # 畸变校正参数
    calib = np.load(files_for_pixel)
    mtx = calib['mtx']
    dist = calib['dist']
    h, w = 480, 640
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    while not stop_event.is_set():
        ret, frame = camera.read()
        if ret:
            if c == 3:
                undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
                if not frame_queue.full():
                    frame_queue.put(undistorted)
        else:
            print("无法读取摄像头画面，请检查摄像头连接或权限设置。")
            stop_event.set()
        time.sleep(0.01)
    
# 绘制方框和标签
def draw_square(image, box, names, r):
    ux = int((r[0] + r[2]) / 2)
    uy = int((r[1] + r[3]) / 2)
    cls = int(box.cls[0])
    conf = box.conf[0]
    label = f"{names[cls]} {conf:.2f}"
    cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (221, 185, 193), 2)
    cv2.putText(image, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (176, 196, 222), 2)
    cv2.circle(image, (ux, uy), 5, ( 240, 240, 240), -1)
    return ux,uy

# USB相机检测
def usb_detect(model, frame_queue, shuchu_file_path, stop_event, start_event):
    global start_time
    start_event.wait()
    print("[DEBUG] usb_detect线程已启动")
    # 启动时立即写一次
    with open(shuchu_file_path, 'w') as file:
        file.write("0 0 0 0")
    result_str = "0 0 0 0"
    last_result = None
    centers = []
    camera_centers = []  
    calib = np.load(files_for_pixel)
    mtx = calib['mtx']
    points = []  # 用于存储像素坐标
    last_cluster_time = time.time()  
    clustered_once = False 
    final_result_str = None 
    while not stop_event.is_set():
        if clustered_once:
            if final_result_str is not None:
                with open(shuchu_file_path, 'w') as file:
                    file.write(final_result_str)
            time.sleep(0.1)
            continue
        if not frame_queue.empty():
            frame = frame_queue.get()
            h, w = frame.shape[:2]
            results = model.predict(source=frame, device='cpu', show=False, stream=False, verbose=False, iou=0.45, conf=0.6)
            for result in results:
                image = result.orig_img
                names = result.names
                boxes = result.boxes
                for box in boxes:
                    r = box.xyxy[0].cpu().numpy().astype(int)
                    ux, uy = draw_square(image, box, names, r)
                    x, y = pixel_to_camera(ux, uy, mtx, w, h)
                    cv2.putText(image, f"({x:.2f}m, {y:.2f}m)", (ux + 10, uy), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 240, 240), 3)
                    points.append((ux, uy)) 
            current_time = time.time()
            last_time= current_time - last_cluster_time
            keydoor = False 
            if len(points) >= min_samples:
                keydoor = True
            elif last_time >= max_time and len(points) > 0:  # 如果超过最大等待时间
                keydoor = True
            if keydoor:
                centers = dbscan_cluster_and_draw(image, points, eps, min_samples)
                camera_centers = []
                for center in centers:
                    x_cam, y_cam = pixel_to_camera(center[0], center[1], mtx, w, h)
                    camera_centers.append((x_cam, y_cam))
                # 按x轴从小到大排序
                camera_centers.sort(key=lambda c: c[0])
                #多个目标
                if len(camera_centers) >=3:
                    if len(camera_centers) > 2:
                        if d == 6:
                            center1 = camera_centers[1]
                            center2 = camera_centers[0]
                        elif d == 5:
                            center1 = camera_centers[2]
                            center2 = camera_centers[1]
                        elif d == 4:
                            center1 = camera_centers[2]
                            center2 = camera_centers[0]
                        elif d == 3:
                            center1 = camera_centers[0]
                            center2 = camera_centers[1]
                        elif d == 2:
                            center1 = camera_centers[0]
                            center2 = camera_centers[2]
                        elif d == 1:
                            center1 = camera_centers[1]
                            center2 = camera_centers[2]
                        if -4<=center1[0]<=4 and -4<=center2[0]<=4 and -2.5<=center1[1]<=2.5 and -2.5<=center2[1] <=2.5:
                            result_str = f"{center1[0]:.1f} {center1[1]:.1f} {center2[0]:.1f} {center2[1]:.1f}"
                        elif -4<=center1[0]<=4 and -2.5<=center1[1]<=2.5:
                            result_str = f"{center1[0]:.1f} {center1[1]:.1f} 0 0"
                        elif -4<=center2[0]<=4 and -2.5<=center2[1]<=2.5:
                            result_str = f"0 0 {center2[0]:.1f} {center2[1]:.1f}"
                        else:
                            result_str = "0 0 0 0"
                    if result_str !=last_result:
                        with open(shuchu_file_path, 'w') as file:
                            file.write(str(result_str))
                        last_result = result_str
                    clustered_once = True
                    final_result_str = result_str
                else:
                    if last_result != "0 0 0 0":
                        with open(shuchu_file_path, 'w') as file:
                            file.write("0 0 0 0")
                        last_result = "0 0 0 0"
            last_cluster_time = current_time  # 更新上次聚类时间
            if 'image' in locals():
                if not display_queue.full():
                    display_queue.put((image, "usb"))
            with open('gaozhi.txt', 'r') as file3:
                content = file3.read().strip()
                print(f"gaozhi.txt 内容: {content}", flush=True)
        else:
            time.sleep(0.01)  
# 读取文件内容并根据内容设置事件和变量
def read_file_content(shuru_file_path, start_event, stop_event):
    global c, b, start_time, d
    last_result = None
    with open(shuru_file_path, 'r') as file2:
        print(f"data.txt 内容: {file2.read().strip()}", flush=True)
    time.sleep(0.25)
    while not stop_event.is_set():
        with open(shuru_file_path, 'r') as file2:
            content = file2.read().strip()
            if content != last_result:
                if content == 'ml':
                    for i in range(10):
                        print("[INFO] 已经到达4m，即将打开usb相机ml模式")
                    start_event.set()
                    c = 3
                    d = 6
                    start_time = time.time()
                elif content == 'rm':
                    for i in range(10):
                        print("[INFO] 已经到达4m，即将打开usb相机rm模式")
                    start_event.set()
                    c = 3
                    d = 5
                    start_time = time.time()
                elif content == 'rl':
                    for i in range(10):
                        print("[INFO] 已经到达4m，即将打开usb相机rl模式")
                    start_event.set()
                    c = 3
                    d = 4
                    start_time = time.time()
                elif content == 'lm':
                    for i in range(10):
                        print("[INFO] 已经到达4m，即将打开usb相机lm模式")
                    start_event.set()
                    c = 3
                    d = 3
                    start_time = time.time()
                elif content == 'lr':
                    for i in range(10):
                        print("[INFO] 已经到达4m，即将打开usb相机lr模式")
                    start_event.set()
                    c = 3
                    d = 2
                    start_time = time.time()
                elif content == 'mr':
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
                elif content == '0':
                    for i in range(5):
                        print("[INFO] 检测到0模式，关闭摄像头")
                    c = 0
                    b = 0
                else:
                    print("[INFO] 未检测到内容，等待...")
                last_result = content

class FrameBuffer:
    def __init__(self):
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
    def put(self, frame):
        with self._lock:
            self._frame = frame
            self._seq += 1
    def get_latest(self, seen_seq=0):
        with self._lock:
            if self._seq <= seen_seq or self._frame is None:
                return False, None, self._seq
            return True, self._frame.copy(), self._seq

class OutputWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        self._lock = threading.Lock()
        self.last_value = None
        self.last_detection_time = 0.0
        self.b_zero = True
        self.a_zero = True
    def write_detection(self, mode, ux, uy):
        with self._lock:
            if mode == '2m' and self.b_zero:
                print("[D435] 2m对桶开始※※※"); self.b_zero = False
            if mode == '1m' and self.a_zero:
                print("[D435] 1m对桶开始※※※"); self.a_zero = False
            self._write(f"1 {ux} {uy}")
            self.last_value = f"{ux} {uy}"
            self.last_detection_time = time.time()
    def write_prediction(self, mode, ux, uy):
        with self._lock:
            if mode == '2m' and self.b_zero:
                print("[D435] 2m对桶开始※※※"); self.b_zero = False
            if mode == '1m' and self.a_zero:
                print("[D435] 1m对桶开始※※※"); self.a_zero = False
            self._write(f"2 {ux} {uy}")
            self.last_value = f"{ux} {uy}"
    def write_fallback(self, mode):
        with self._lock:
            self._check_timeout()
            if mode == '2m' and self.b_zero:
                print("[D435] 2m对桶开始※※※"); self.b_zero = False
            if mode == '1m' and self.a_zero:
                print("[D435] 1m对桶开始※※※"); self.a_zero = False
            if mode != '0' and self.last_value is not None:
                content = "0" if self.last_value == '0' else "2 " + self.last_value
            else:
                content = "0"
            self._write(content)
            if mode == '2m' and content.strip() == '0':
                return
            parts = content.strip().split()
            self.last_value = '0' if parts[0] == '0' else ' '.join(parts[1:])
    def reset(self, cold_start=False):
        with self._lock:
            if cold_start:
                self.last_value = None
                self.last_detection_time = time.time()
                self.b_zero = True
                self.a_zero = True
                self._write("0")
    def _check_timeout(self):
        if HISTORY_CLEAR_ENABLED and self.last_value is not None \
                and time.time() - self.last_detection_time > HISTORY_CLEAR_TIMEOUT:
            self.last_value = None
            print(f"[D435] 超过{HISTORY_CLEAR_TIMEOUT}s未检测到目标，清空历史坐标")
    def _write(self, content):
        with open(self.file_path, 'w') as f:
            f.write(content)

class LockTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.lock_target = None
        self.lock_miss_count = 0
        self.lock_frame_count = 0
    @property
    def is_locked(self):
        return self.lock_target is not None
    def reset(self):
        with self._lock:
            self.lock_target = None
            self.lock_miss_count = 0
            self.lock_frame_count = 0
    def increment_frame_count(self):
        with self._lock:
            if self.is_locked:
                self.lock_frame_count += 1
    def force_unlock(self):
        with self._lock:
            print(f"[D435] 锁定满{LOCK_MAX_HIT}帧，强制重判")
            self.lock_target = None
            self.lock_miss_count = 0
            self.lock_frame_count = 0
    def update(self, detections):
        with self._lock:
            if self.is_locked:
                return self._locked_with_dets(detections) if detections else self._locked_no_dets()
            else:
                return self._unlocked_with_dets(detections) if detections else self._unlocked_no_dets()
    def _locked_with_dets(self, detections):
        ox, oy, ow, oh = self.lock_target
        sx1, sy1, sx2, sy2, shw, shh, sarea = search_rect(ox, oy, ow, oh)
        candidates = [(d.dis, d.cls, d.ux, d.uy, d.r) for d in detections
                      if sx1 <= d.ux <= sx2 and sy1 <= d.uy <= sy2]
        best = pick_nearest(candidates)
        if best is not None:
            _, _, ux, uy, r = best
            w, h = abs(r[2]-r[0]), abs(r[3]-r[1])
            self.lock_target = (ux, uy, w, h)
            self.lock_miss_count = 0
            print(f"[D435] 锁定桶：({ux},{uy}) 目标面积={w*h}")
            return TrackResult(TrackAction.DETECT, ux, uy, True)
        self.lock_miss_count += 1
        if self.lock_miss_count > LOCK_MAX_MISS:
            print(f"[D435] 丢帧满{LOCK_MAX_MISS}帧，强制重判")
            self.lock_target = None; self.lock_miss_count = 0; self.lock_frame_count = 0
            best = pick_nearest([(d.dis, d.cls, d.ux, d.uy, d.r) for d in detections])
            _, _, ux, uy, r = best
            w, h = abs(r[2]-r[0]), abs(r[3]-r[1])
            self.lock_target = (ux, uy, w, h)
            print(f"[D435] 解锁-最近桶：({ux},{uy}) 目标面积={w*h}")
            return TrackResult(TrackAction.DETECT, ux, uy, True)
        else:
            print(f"[D435] 锁定-历史：({ox},{oy}) miss={self.lock_miss_count}")
            return TrackResult(TrackAction.PREDICT, ox, oy, True)
    def _locked_no_dets(self):
        ox, oy, ow, oh = self.lock_target
        self.lock_miss_count += 1
        if self.lock_miss_count > LOCK_MAX_MISS:
            print(f"[D435] 丢帧满{LOCK_MAX_MISS}帧，强制重判")
            self.lock_target = None; self.lock_miss_count = 0; self.lock_frame_count = 0
            return TrackResult(TrackAction.LOST, 0, 0, False)
        else:
            print(f"[D435] 锁定-历史：({ox},{oy}) miss={self.lock_miss_count}")
            return TrackResult(TrackAction.PREDICT, ox, oy, True)
    def _unlocked_with_dets(self, detections):
        best = pick_nearest([(d.dis, d.cls, d.ux, d.uy, d.r) for d in detections])
        _, _, ux, uy, r = best
        w, h = abs(r[2]-r[0]), abs(r[3]-r[1])
        self.lock_target = (ux, uy, w, h)
        self.lock_miss_count = 0; self.lock_frame_count = 0
        print(f"[D435] 锁定-最近桶：({ux},{uy}) 目标面积={w*h}")
        return TrackResult(TrackAction.DETECT, ux, uy, True)
    def _unlocked_no_dets(self):
        return TrackResult(TrackAction.LOST, 0, 0, False)

class YOLODetector:
    def __init__(self):
        self.model_2m = YOLO(MODEL_2M_PATH)
        self.model_1m = YOLO(MODEL_1M_PATH)
        self._cx = D435_WIDTH // 2
        self._cy = D435_HEIGHT // 2
    def _select_model(self, mode):
        if mode == '1m':
            return self.model_1m, MODEL_1M_CONF, MODEL_1M_IMGSZ
        return self.model_2m, MODEL_2M_CONF, MODEL_2M_IMGSZ
    def detect_full(self, frame, mode):
        model, conf, imgsz = self._select_model(mode)
        t0 = time.time()
        results = model.predict(source=frame, device=MODEL_DEVICE, show=False,
                                stream=False, verbose=False, iou=MODEL_IOU, conf=conf, imgsz=imgsz)
        print(f"[D435] 全图推理：{int((time.time()-t0)*1000)}ms")
        return self._parse(results, 0, 0)
    def detect_crop(self, crop, ox, oy, mode):
        h, w = crop.shape[:2]
        imgsz = ((max(w, h) + MODEL_IMGSZ_STEP - 1) // MODEL_IMGSZ_STEP) * MODEL_IMGSZ_STEP
        model, conf, _ = self._select_model(mode)
        t0 = time.time()
        results = model.predict(source=crop, device=MODEL_DEVICE, show=False,
                                stream=False, verbose=False, iou=MODEL_IOU, conf=conf, imgsz=imgsz)
        print(f"[D435] 裁减推理：{int((time.time()-t0)*1000)}ms")
        return self._parse(results, ox, oy)
    def _parse(self, results, ox, oy):
        detections = []
        for result in results:
            boxes = result.boxes
            names = result.names
            if boxes is None:
                continue
            for box in boxes:
                r = box.xyxy[0].cpu().numpy().astype(int)
                r[0] += ox; r[1] += oy; r[2] += ox; r[3] += oy
                ux = int((r[0] + r[2]) / 2)
                uy = int((r[1] + r[3]) / 2)
                w = abs(int(r[2] - r[0]))
                h = abs(int(r[3] - r[1]))
                area = w * h
                if area > MAX_AREA:
                    continue
                dis = ((ux - self._cx) ** 2 + (uy - self._cy) ** 2) ** 0.5
                detections.append(Detection(ux=ux, uy=uy, cls=int(box.cls[0]),
                    conf=float(box.conf[0]), r=r.tolist(), w=w, h=h,
                    area=area, dis=dis, box=box, names=names))
        return detections

class CameraSource:
    def __init__(self):
        self._pipeline = None
        self._buf = FrameBuffer()
        self._stop = threading.Event()
        self._thread = None
    def start(self):
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, D435_WIDTH, D435_HEIGHT, rs.format.bgr8, 30)
        self._pipeline.start(cfg)
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture, daemon=True)
        self._thread.start()
        print("[D435] capture thread started")
    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._pipeline is not None:
            self._pipeline.stop()
            print("[D435] 摄像头已释放")
    def get_frame(self, seen_seq):
        return self._buf.get_latest(seen_seq)
    def _capture(self):
        while not self._stop.is_set():
            frames = self._pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if color:
                self._buf.put(np.asanyarray(color.get_data()))

# 定义D435i相机检测线程 (LockTracker pipeline)
def d435_detect(shuchu_file_path, stop_event, start_event):
    start_event.wait()
    print("[DEBUG] d435_detect线程已启动 (LockTracker pipeline)")
    with open(shuchu_file_path, 'w') as f:
        f.write("0")
    detector = YOLODetector()
    camera = CameraSource()
    tracker = LockTracker()
    writer = OutputWriter(shuchu_file_path)
    camera.start()
    last_b = None
    last_mode = None
    delay_until = 0.0
    last_seq = 0
    frame_count = 0
    last_canvas = None
    try:
        while not stop_event.is_set():
            current_b = b
            if current_b not in (1, 2):
                if last_b in (1, 2):
                    with open(shuchu_file_path, 'w') as f:
                        f.write('0')
                last_b = current_b
                time.sleep(0.01)
                continue
            mode = '2m' if current_b == 2 else '1m'
            if mode != last_mode:
                print(f"[D435] 模式切换: {last_mode} -> {mode}")
                cold = last_mode is None
                writer.reset(cold_start=cold)
                tracker.reset()
                if last_mode == '2m' and mode == '1m':
                    delay_until = time.time() + 3.0
                    print("[D435] 2m->1m: 保持2m模型3s")
                else:
                    delay_until = 0.0
                last_mode = mode
                frame_count = 0
            last_b = current_b
            if delay_until > time.time() and mode == '1m':
                effective_mode = '2m'
            else:
                effective_mode = mode
            got, frame, last_seq = camera.get_frame(last_seq)
            if not got:
                time.sleep(0.001)
                continue
            frame_count += 1
            do_inference = (not FRAME_SKIP_ENABLED or frame_count % FRAME_SKIP_N == 0)
            if do_inference:
                canvas = frame.copy()
                tracker.increment_frame_count()
                if tracker.is_locked and tracker.lock_frame_count >= LOCK_MAX_HIT:
                    tracker.force_unlock()
                was_locked = tracker.is_locked
                if was_locked:
                    ox, oy, ow, oh = tracker.lock_target
                    sx1, sy1, sx2, sy2, shw, shh, _ = search_rect(ox, oy, ow, oh)
                    csx1 = max(0, int(sx1)); csy1 = max(0, int(sy1))
                    csx2 = min(D435_WIDTH, int(sx2)); csy2 = min(D435_HEIGHT, int(sy2))
                    crop = frame[csy1:csy2, csx1:csx2]
                    detections = detector.detect_crop(crop, csx1, csy1, effective_mode)
                    if detections:
                        draw_lock_rect(canvas, ox, oy, shw, shh, (0, 255, 255))
                    else:
                        draw_lock_rect(canvas, ox, oy, shw, shh, (0, 165, 255))
                else:
                    detections = detector.detect_full(frame, effective_mode)
                for d in detections:
                    draw_square(canvas, d.box, d.names, d.r)
                    cv2.circle(canvas, (d.ux, d.uy), 4, (255, 255, 255), 5)
                    cv2.putText(canvas, f"[{d.ux},{d.uy}]", (d.ux+20, d.uy+10),
                                0, 1, (225, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                result = tracker.update(detections)
                if result.action == TrackAction.DETECT:
                    writer.write_detection(mode, result.x, result.y)
                    if not was_locked and result.is_locked:
                        ox, oy, ow, oh = tracker.lock_target
                        _, _, _, _, shw, shh, _ = search_rect(ox, oy, ow, oh)
                        draw_lock_rect(canvas, ox, oy, shw, shh, (0, 255, 0))
                elif result.action == TrackAction.PREDICT:
                    writer.write_prediction(mode, result.x, result.y)
                else:
                    writer.write_fallback(mode)
                last_canvas = canvas.copy()
            else:
                canvas = last_canvas.copy() if last_canvas is not None else frame.copy()
            cv2.putText(canvas, f"D435:{effective_mode}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            if not display_queue.full():
                display_queue.put((canvas, "d435"))
            if os.path.exists('gaozhi.txt'):
                with open('gaozhi.txt', 'r') as f3:
                    print(f"gaozhi.txt 内容: {f3.read().strip()}", flush=True)
    finally:
        camera.stop()

# DBSCAN聚类并绘制
def dbscan_cluster_and_draw(image, points, eps, min_samples):
    if len(points) == 0:
        print(f"没有桶")
        return []
    X = np.array(points)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.fit_predict(X)
    cluster_centers = []
    colors = [tuple(np.random.randint(0,255,3).tolist()) for _ in range(max(labels)+2)]
    for i in range(max(labels)+1):
        cluster_points = X[labels==i]
        cluster_center = np.mean(cluster_points, axis=0)
        cluster_centers.append(cluster_center)
        # 可视化聚类点(复盘)
        if image is not None:
            for pt in cluster_points:
                cv2.circle(image, (int(pt[0]), int(pt[1])), 8, colors[i], -1)
            cv2.circle(image, (int(cluster_center[0]), int(cluster_center[1])), 15, colors[i], 3)
            cv2.putText(image, f"cluster{i+1}", (int(cluster_center[0]), int(cluster_center[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)
    cv2.imwrite(f"聚类结果图/cluster_{int(time.time())}.jpg", image)
    return cluster_centers

if __name__ == '__main__':
    try:
        t_test = None
        t_detect = None
        t_d435 = None
        camera = None
        c = 0
        b = 0
        clear_files(files_to_clear)
        shuru_file_path = 'data.txt'
        shuchu_file_path = 'gaozhi.txt'
        start_event = threading.Event()
        stop_event = threading.Event()
        t1 = threading.Thread(target=read_file_content, args=(shuru_file_path, start_event, stop_event))
        t1.start()
        main_thread = threading.Thread(target=main_control)
        main_thread.start()
        # 主线程负责显示窗口,fps显示
        window_created = False
        frame_count = 0
        start_time = time.time()
        while True:
            if not display_queue.empty():
                frame, window_name = display_queue.get()
                # frame 是聚类后带标注的 image
                if not window_created:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 900, 800)
                    window_created = True
                if c == 3:
                # 画面四角坐标
                    h, w = frame.shape[:2]
                    corners = [ (0,0), (w-1,0), (0,h-1), (w-1,h-1) ]
                    colors = [ (0,0,255), (0,255,0), (255,0,0), (0,255,255) ]
                    calib = np.load(files_for_pixel)
                    mtx = calib['mtx']
                    for i, (x, y) in enumerate(corners):
                        X, Y = pixel_to_camera(x, y, mtx, w, h)
                        cv2.circle(frame, (x, y), 8, colors[i], -1)
                        if x < w // 2 and y < h // 2:      # 左上
                            tx, ty = x + 10, y + 30
                        elif x >= w // 2 and y < h // 2:   # 右上
                            tx, ty = x - 220, y + 30
                        elif x < w // 2 and y >= h // 2:   # 左下
                            tx, ty = x + 10, y - 10
                        else:                              # 右下
                            tx, ty = x - 220, y - 10
                        cv2.putText(
                            frame,
                            f"({x},{y}) ({X:.2f},{Y:.2f})m",
                            (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2
                        )
                cv2.imshow(window_name, frame) 
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)
        cv2.destroyAllWindows()
        main_thread.join()
    except Exception as e:
        print(f"An error occurred: {e}")