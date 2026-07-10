import os
import cv2
import time
import queue
import threading
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs
from sklearn.cluster import DBSCAN

Z = 4.31
eps = 50
min_samples = 15
max_time = 5.0
files_to_clear = ['data.txt','gaozhi.txt']
files_for_pixel = 'calib_rseult.npz'

cam_num = '/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0'

model_usb = YOLO("/home/son/yolov5_d435i_detection-main/gs4mcz.pt")
model_d435 = YOLO("/home/son/yolov5_d435i_detection-main/2m.pt")

display_queue = queue.Queue(maxsize = 3)
start_time = None

def clear_files(files):
    for file_name in files:
        file_path = os.path.join(os.getcwd(),file_name)
        with open(file_path,'w') as f:
            print("[INFO]通讯txt文件已建立并清空")
            
def check_camera(cam_num):
    cap = cv2.VideoCapture(cam_num)
    if not cap.isOpened():
        for _ in range():
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


global c, b, d, start_event, shuchu_file_path
global t_test, t_detect, t_d435, camera
    
def main_control():

    last_c = None
    running = False
    camera = 0
    stop_event = None
    frame_queue = None
    check_camera(cam_num)

    while True:
        if c != last_c:
            print(f"[DEBUG] main_control 检测到 c 变化: {last_c} -> {c}")
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

        if

