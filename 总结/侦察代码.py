import cv2
import numpy as np
import sys
import time
import os
# 导入跨系统获取USB设备信息的库
try:
    # Linux系统：用pyudev获取USB设备信息
    import pyudev
except ImportError:
    pass


# -------------------------- 配置参数（关键：修改为你的相机VID和PID） --------------------------
RECORD_TIMEOUT = 15  # 最大录制时长(秒)
DARKNESS_CHECK_DELAY = 0.6  # 第二次亮度检测延迟(秒)
output_video_path = 'recorded_video.mp4' 
BRIGHTNESS_THRESHOLD = 100  # 亮度阈值
WAIT_DURATION = 40  # 等待时间(秒)
#你的相机USB硬件ID（VID=厂商ID，PID=产品ID），格式：(VID, PID)
TARGET_CAMERA_USB_ID = (0x1b80,0xe13e)  # 相机的VID/PID
# -------------------------------------------------------------------------------------------

# 初始化状态变量
recording = False
recording_start_time = None
video_writer = None
darkness_detected = False
darkness_start_time = None
cv2.namedWindow('Recording', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Recording', 2560, 1600)


def get_camera_usb_id_linux(cam_path):
    """Linux系统：根据相机设备路径（/dev/videoX）获取USB VID和PID"""
    context = pyudev.Context()
    device = pyudev.Device.from_device_file(context, cam_path)
    # 从设备属性中提取VID和PID（格式：ID_VENDOR_ID=1234，ID_MODEL_ID=5678）
    vid = int(device.get('ID_VENDOR_ID', '0'), 16) if device.get('ID_VENDOR_ID') else 0
    pid = int(device.get('ID_MODEL_ID', '0'), 16) if device.get('ID_MODEL_ID') else 0
    return (vid, pid)





def find_target_camera():
    """跨系统查找匹配目标USB ID（VID/PID）的相机，返回设备路径（Linux)"""
    available_cameras = []
    # 遍历前10个可能的相机设备（覆盖绝大多数场景）
    for i in range(10):
        # 1. 尝试Linux设备路径
        if sys.platform.startswith('linux'):
            cam_path = f"/dev/video{i}"
            if os.path.exists(cam_path):
                cap = cv2.VideoCapture(cam_path)
                if cap.isOpened():
                    # 获取该相机的USB ID
                    cam_usb_id = get_camera_usb_id_linux(cam_path)
                    available_cameras.append((cam_path, cam_usb_id))
                    cap.release()

    # 筛选匹配目标USB ID的相机
    for cam_source, cam_usb_id in available_cameras:
        if cam_usb_id == TARGET_CAMERA_USB_ID:
            print(f"[INFO] 找到目标相机！USB ID: {hex(cam_usb_id[0])}:{hex(cam_usb_id[1])}，设备: {cam_source}")
            return cam_source
    # 未找到目标相机
    print(f"[ERROR] 未找到匹配USB ID（{hex(TARGET_CAMERA_USB_ID[0])}:{hex(TARGET_CAMERA_USB_ID[1])}）的相机，请检查连接！")
    return None


# ----------------------------------------------------
def get_frame_brightness(frame):
    if frame is None:
        return 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def check_brightness_and_start_recording(frame):
    global recording, recording_start_time, video_writer, darkness_detected
    brightness = get_frame_brightness(frame)
    if brightness > BRIGHTNESS_THRESHOLD and not recording:
        recording = True
        recording_start_time = cv2.getTickCount()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        print(f"[INFO] 亮度足够（{brightness:.2f}），开始录制...")
        darkness_detected = False


def detect_and_record(frame):
    global recording, recording_start_time, video_writer, darkness_detected, darkness_start_time
    if frame is None or frame.size == 0:
        return False, frame

    # 图像饱和度增强
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation_scale = 1.5
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255).astype(hsv.dtype)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    check_brightness_and_start_recording(frame)

    if recording:
        current_time = cv2.getTickCount()
        elapsed_time = (current_time - recording_start_time) / cv2.getTickFrequency()
        current_brightness = get_frame_brightness(frame)

        video_writer.write(frame)

        # 暗态检测逻辑
        if current_brightness <= BRIGHTNESS_THRESHOLD:
            if not darkness_detected:
                darkness_detected = True
                darkness_start_time = current_time
                print(f"[INFO] 第一次检测到暗态（亮度{current_brightness:.2f}），{DARKNESS_CHECK_DELAY}秒后确认")
            else:
                darkness_elapsed = (current_time - darkness_start_time) / cv2.getTickFrequency()
                if darkness_elapsed >= DARKNESS_CHECK_DELAY:
                    recording = False
                    video_writer.release()
                    print(f"[INFO] 暗态确认，结束录制")
                    return True, frame
        else:
            if darkness_detected:
                darkness_detected = False
                print(f"[INFO] 亮度恢复，重置暗态检测")

        # 超时强制结束
        if elapsed_time >= RECORD_TIMEOUT:
            recording = False
            video_writer.release()
            print(f"[INFO] 超过最大时长（{RECORD_TIMEOUT}s），强制结束录制")
            return True, frame

    return False, frame


if __name__ == "__main__":
    # 关键：通过USB ID查找目标相机，而非用固定索引
    camera_source = find_target_camera()
    if camera_source is None:
        sys.exit(1)

    # 打开目标相机
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开目标相机：{camera_source}")
        sys.exit(1)

    fourcc = cv2.VideoWriter_fourcc(*'YUYV')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    print("[INFO] 开始实时检测...")

    # 等待阶段
    start_time = time.time()
    waiting_complete = False
    while not waiting_complete:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 无法读取帧，退出")
            break
        elapsed_wait = time.time() - start_time
        remaining = WAIT_DURATION - elapsed_wait
        cv2.imshow('Recording', frame)
        if elapsed_wait >= WAIT_DURATION:
            waiting_complete = True
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

    # 录制检测阶段
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 无法读取帧，退出")
            break
        recording_complete, processed_frame = detect_and_record(frame)
        cv2.imshow('Recording', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if recording_complete:
            break

    # 回放视频
    if recording_complete:
        for _ in range(3):
            cap_playback = cv2.VideoCapture(output_video_path)
            while cap_playback.isOpened():
                success, frame = cap_playback.read()
                if success:
                    cv2.imshow('Recording', frame)
                else:
                    break
                if cv2.waitKey(55) & 0xFF == ord('q'):
                    break
            cap_playback.release()

    # 释放资源
    if video_writer is not None and video_writer.isOpened():
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

