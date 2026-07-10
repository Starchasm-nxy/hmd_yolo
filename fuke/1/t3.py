'''
by hmy 2025.8.14——2025.8.21
last by hmy 2025.8月底
手动选桶程序 - YOLOv8版本
'''
import os
import cv2
import time
import queue
import threading
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs

##参数设置
files_to_clear = ['data.txt','gaozhi.txt']
display_queue = queue.Queue(maxsize=3)

# 全局变量
last_value = None 
b_zero = True   
last_processed_list = []
#prev_ux_global = None
#prev_delta_ux = 0
#count2 = 0

# 模型权重声明
model_1m = YOLO("/home/luck/yolov5_d435i_detection-main/2m.pt")
model_2m = YOLO("/home/luck/yolov5_d435i_detection-main/2m.pt")


def clear_files(files):
    for file_name in files:
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, 'w') as f:
            print("[INFO]通讯txt文件已建立并清空")
            pass
      




def read_data_file():
    try:
        with open('data.txt', 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return None

def write_to_file(processed_list, filename='gaozhi.txt'):
    global last_value, b_zero
    b = None
    b_value = read_data_file()
    
    if b_value in ['r', 'm', 'l']:
        b = 1
    elif b_value == '1m':
        b = 2
  
        
    # 检查b是否变为1
    if b_zero and b == 1:
        message = "[INFO]我方即将进入对桶程序※※"
        for _ in range(8):
            print(message)
        b_zero = False
        
    if b == 2:
        if processed_list:
            content = ' '.join(processed_list)
            with open(filename, 'w') as f:
                f.write(content)
            last_value = content.strip()
        else:
            processed_list = [str(0)]
            content = ' '.join(processed_list) 
            with open(filename, 'w') as f:
                f.write(content)
            return last_value
        
    if b == 1:
        if processed_list:
            content = ' '.join(processed_list)
            with open(filename, 'w') as f:
                f.write(content)
            if content.strip() != '0':
                last_value = content.strip()
        else:
            if last_value is not None:
                with open(filename, 'w') as f:
                    f.write(last_value + '\n')
                print(last_value)
            if last_value is not None and not os.path.getsize(filename):
                return last_value
        return None
 
def draw_square(image, box, names, r):
    ux = int((r[0] + r[2]) / 2)
    uy = int((r[1] + r[3]) / 2)
    cls = int(box.cls[0])
    conf = box.conf[0]
    label = f"{names[cls]} {conf:.2f}"
    cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (221, 185, 193), 2)
    cv2.putText(image, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (176, 196, 222), 2)
    cv2.circle(image, (ux, uy), 5, (240, 240, 240), -1)
    return ux, uy

# 监控data.txt文件内容的函数
def monitor_data_file():
    print("开始监控data.txt文件...")
    while True:
        try:
            with open('data.txt', 'r') as file:
                print(f"data.txt 内容: {file.read().strip()}", flush=True)
        except:
            pass
        time.sleep(0.25)

# 读取文件内容并根据内容设置事件和变量
def read_file_content(shuru_file_path, start_event, stop_event):
    global c
    last_result = None
    
    while not stop_event.is_set():
        try:
            with open(shuru_file_path, 'r') as file:
                content = file.read().strip()
                if content != last_result:
                    if content in ['r','m' ,'l' , '1m','0']:
                        for i in range(8):
                            print(f"[INFO] 检测到{content}模式")
                        start_event.set()
                        c = 1
                    else:
                        print("[INFO] 未检测到有效内容，等待...")
                    last_result = content
        except:
            pass
        time.sleep(0.05)

# D435i相机检测线程
def d435_detect(shuchu_file_path, stop_event, start_event):
    global processed_list, last_processed_list,last_value
    
    # 初始化RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    start_event.wait()
    print("[DEBUG] d435_detect线程已启动")
    pipeline.start(config)
    
    # 初始化变量
    processed_list = []
    last_processed_list = []
    last_mode = None
    
    try:
        while not stop_event.is_set():
            data_content = read_data_file()
            
            # 根据模式选择模型
            if data_content in ['r', 'm', 'l','0']:
                model = model_2m
            elif data_content == '1m':
                model = model_1m
            else:
                continue
            
            # 检测模式切换，重置前视相关变量
            if data_content != last_mode:
                print(f"[INFO] 模式切换: {last_mode} -> {data_content}")
                if last_mode == '1m' and data_content != '1m':
                    with open('gaozhi.txt', 'w') as f:
                        f.write('0')
                    last_value = None     
                v = 0
                count1 = 0
                processed_list = []
                last_processed_list = []
                last_mode = data_content
            
            # 获取图像帧
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                continue
                
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            color_image = np.asanyarray(color_frame.get_data())
            
            # YOLOv8推理
            results = model.predict(source=color_image, device='cpu', show=False, 
                                  stream=False, verbose=False, iou=0.45, conf=0.6)
            
            tong_list = []
            canvas = color_image.copy()
            
            for result in results:
                boxes = result.boxes
                names = result.names
                if boxes is not None:
                    for box in boxes:
                        r = box.xyxy[0].cpu().numpy().astype(int)
                        ux, uy = draw_square(canvas, box, names, r)
                        
                        length = abs(int(r[0] - r[2]))
                        width = abs(int(r[1] - r[3]))
                        area = length * width
                        
                        dis = aligned_depth_frame.get_distance(ux, uy)
                        dis = np.round(np.array(dis), 3)
                        
                        camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)
                        camera_xyz = np.round(np.array(camera_xyz), 3)
                        camera_xyz = camera_xyz.tolist()
                        
                        cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 5)
                        cv2.putText(canvas, str(camera_xyz), (ux + 20, uy + 10), 0, 1,
                                   [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                        
                        class_id = int(box.cls[0])

                        if area <= 70000:  # 桶类别
                            tong_list.append((dis, class_id, ux, uy))

            # 在画面上显示当前使用的模型
            cv2.putText(canvas, f"{data_content}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 处理不同模式的逻辑
            if data_content == 'r':
                tong_list = [item for item in tong_list if item[2] < 848]
                if tong_list:
                    min_dis = min(dis for dis, lab, ux, uy in tong_list)
                    for dis, lab, ux, uy in tong_list:
                        if dis == min_dis:
                            with open('gaozhi.txt', 'r') as file:
                                print(f"gaozhi.txt 内容: {file.read().strip()}", flush=True)
                            processed_list = [str(1), str(ux), str(uy)]
                            break
                else:
                    processed_list = [str(0)]
                write_to_file(processed_list)
            elif data_content == 'l':
                tong_list = [item for item in tong_list if item[2] > 0]
                if tong_list:
                    min_dis = min(dis for dis, lab, ux, uy in tong_list)
                    for dis, lab, ux, uy in tong_list:
                        if dis == min_dis:
                            with open('gaozhi.txt', 'r') as file:
                                print(f"gaozhi.txt 内容: {file.read().strip()}", flush=True)
                            processed_list = [str(1), str(ux), str(uy)]
                            break
                else:
                    processed_list = [str(0)]
                write_to_file(processed_list)
            elif data_content == 'm':
                tong_list = [item for item in tong_list if item[2] > 80 and item[2] < 768]
                if tong_list:
                    min_dis = min(dis for dis, lab, ux, uy in tong_list)
                    for dis, lab, ux, uy in tong_list:
                        if dis == min_dis:
                            with open('gaozhi.txt', 'r') as file:
                                print(f"gaozhi.txt 内容: {file.read().strip()}", flush=True)
                            processed_list = [str(1), str(ux), str(uy)]
                            break
                else:
                    processed_list = [str(0)]
                write_to_file(processed_list)      
                
            if data_content == '1m':
                if tong_list:
                    min_dis = min(dis for dis, lab, ux, uy in tong_list)
                    for dis, lab, ux, uy in tong_list:
                        if dis == min_dis:
                            print(f"找到最近的桶（dis值最近）：ux={ux}, uy={uy}")
                            processed_list = [str(1), str(ux), str(uy)]
                            break
                else:
                    if last_value is not None:
                        processed_list = last_value.split()
                        print(f"[INFO] 1m模式未检测到桶，返回上一帧: {processed_list}")
                    else:
                        processed_list = [str(0)]
                write_to_file(processed_list)
                    
            # 显示画面
            if not display_queue.full():
                display_queue.put((canvas, "chaoqian"))
                
    finally:
        pipeline.stop()

# 主控制函数
def main_control():
    global c, start_event
    last_c = None
    running = False
    stop_event = None
    t_d435 = None
    
    # 检查D435相机连接
    ctx = rs.context()
    if len(ctx.devices) == 0:
        for _ in range(5):
            print(f"\033[31m[WARN] 未检测到D435相机，请检查连接！\033[0m")
    else:
        print(f"\033[32m[INFO] D435相机已连接。\033[0m")
    
    while True:
        if c != last_c:
            print(f"[DEBUG] main_control 检测到 c 变化: {last_c} -> {c}")
            
            # 关闭旧线程
            if running:
                stop_event.set()
                if t_d435:
                    t_d435.join()
                cv2.destroyAllWindows()
                time.sleep(0.2)
                running = False
                
            stop_event = threading.Event()
            #time.sleep(3)  # 从输入到开启摄像头亲测5秒，防止飞机到位置没停稳就开启摄像头
            # 启动新线程
            if c == 1:
                t_d435 = threading.Thread(target=d435_detect,args=('gaozhi.txt', stop_event, start_event))
                t_d435.start()
                running = True
                
            last_c = c
        time.sleep(0.1)

if __name__ == '__main__':
    try:
        print("[INFO] YoloV8目标检测-程序启动")
        print("[INFO] 开始YoloV8模型加载")
        
        # 初始化全局变量
        c = 0
        clear_files(files_to_clear)
        
        shuru_file_path = 'data.txt'
        shuchu_file_path = 'gaozhi.txt'
        start_event = threading.Event()
        stop_event = threading.Event()
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=monitor_data_file)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # 启动文件读取线程
        t1 = threading.Thread(target=read_file_content, args=(shuru_file_path, start_event, stop_event))
        t1.start()
        
        # 启动主控制线程
        main_thread = threading.Thread(target=main_control)
        main_thread.start()
        
        print("[INFO] 完成YoloV8模型加载")
        print("[INFO] 系统初始化完成，等待指令...")
        
        # 主线程负责显示窗口
        window_created = False
        while True:
            if not display_queue.empty():
                frame, window_name = display_queue.get()
                name = "chaoqian"
                if not window_created:
                    cv2.namedWindow(name, cv2.WINDOW_NORMAL | 
                                   cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                    cv2.resizeWindow(name, 1696, 960)
                    window_created = True
                cv2.imshow(name, frame)  
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break
            else:
                time.sleep(0.01)
                
        cv2.destroyAllWindows()
        main_thread.join()
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
