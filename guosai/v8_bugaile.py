###power by wjh 2025-09-22
### 4m解算国赛最终版
# 已上机实测
# 速度提升---调参及任务规划
# #模型特点：
# 2m.pt：D435i相机模型，2m以下检测精度高
# 3.5m.pt：USB相机模型，2m-4m检测精度高

import os
import cv2
import time
import queue
import threading
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs
import time
from sklearn.cluster import DBSCAN

##参数设置
Z = 4.31  # 相机高度
eps=50 # DBSCAN聚类半径（像素）
min_samples=15 # DBSCAN最小聚类点数
max_time = 5.0 # 最大等待时间（秒）
files_to_clear = ['data.txt','gaozhi.txt']
files_for_pixel = 'calib_result.npz' # 畸变校正文件(根据相机改变)
# USB摄像头编号
cam_num =  '/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0' 
# 所需权重  
model_usb = YOLO("/home/son/yolov5_d435i_detection-main/gs4mcz.pt")
model_d435 = YOLO("/home/son/yolov5_d435i_detection-main/2m.pt")

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
                model = model_d435
                t_d435 = threading.Thread(target=d435_detect, args=(model, shuchu_file_path, stop_event, start_event))
                t_d435.start()
                t_test = None
                t_detect = None
                camera = None
                running = True
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
                    display_queue.put((image, "atx"))
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
                if content == 'mlkai':
                    for i in range(10):
                        print("[INFO] 已经到达4m，即将打开usb相机ml模式")
                    start_event.set()
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
# 定义D435i相机检测线程
def d435_detect(model, shuchu_file_path, stop_event, start_event):
    pipeline = rs.pipeline()  # 定义流程pipeline
    config = rs.config()  # 定义配置config
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    align_to = rs.stream.color  # 与color流对齐
    align = rs.align(align_to)
    start_event.wait()
    print("[DEBUG] d435_detect线程已启动")
    # 启动时立即写一次
    with open(shuchu_file_path, 'w') as file:
        file.write("0")
    pipeline.start(config)  # 流程开始
    result_str = "0"
    last_result = None
    last_value = "0"
    while not stop_event.is_set():
        tong_list = []
        image = pipeline.wait_for_frames()
        aligned_frames = align.process(image)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        results = model.predict(source=frame, device='cpu', show=False, stream=False, verbose=False, iou=0.45, conf=0.6)
        for result in results:
            image = result.orig_img
            names = result.names
            boxes = result.boxes
            for box in boxes:
                r = box.xyxy[0].cpu().numpy().astype(int)
                ux, uy = draw_square(frame, box, names, r)
                cv2.putText(image, f"( {ux},  {uy})", (ux + 10, uy), cv2.FONT_HERSHEY_SIMPLEX, 1, ( 240, 240, 240), 3)
                area = (r[2]-r[0])*(r[3]-r[1])
                dis = aligned_depth_frame.get_distance(ux, uy)
                if b == 2:
                    if area <= 30000:#面积限制，防止近距离误判
                        tong_list.append((abs(ux-424), (ux, uy)))
                elif b == 1:
                    tong_list.append(((abs(ux-424)), (ux, uy)))
                tong_list.sort(key=lambda x: x[0])
                if len(tong_list) > 2:
                    tong_list = tong_list[:2]#防止进程堵塞
        if c == 2:
            if len(tong_list) >= 1:
                center = tong_list[0][1]
                result_str = f"1 {center[0]} {center[1]}"
                if b == 1:
                    last_value = result_str
                else:
                    last_value = "0"
            else:
                if b == 2:
                    result_str = "0"
                    with open(shuchu_file_path, 'w') as file:
                        file.write(str(result_str))
                    last_result = result_str
                elif b == 1:
                    result_str = last_value  # 直接返回上一次目标字符串
        # 只有不是b==2无目标时才用变化判断
        if not (b == 2 and len(tong_list) == 0):
            if result_str != last_result:
                with open(shuchu_file_path, 'w') as file:
                    file.write(str(result_str))
                last_result = result_str
        if 'image' in locals():
            if not display_queue.full():
                display_queue.put((image, "atx"))
        with open('gaozhi.txt', 'r') as file3:
            content = file3.read().strip()
            print(f"gaozhi.txt 内容: {content}", flush=True)

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
###wjh start in 2025-04-20  
####wjh 2025-04-20
####wjh 2025-04-20
