import os
import cv2
import time
import numpy as np

# ==================== 参数设置 ====================
USB_CAM_PATH = '/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0'   # USB摄像头路径
USB_WIDTH = 640
USB_HEIGHT = 480
CALIB_FILE = 'calib_resultA.npz'          # 畸变校正文件（若不存在则跳过校正）
VIDEO_FPS = 60                            # 保存视频的目标帧率
SEGMENT_SEC = 10                          # 每段视频时长（秒）
FRAMES_PER_SEGMENT = VIDEO_FPS * SEGMENT_SEC  # 每段视频的帧数

BASE_VIDEO_DIR = 'videos'                 # 一级文件夹

def create_next_video_dir():
    """在videos文件夹下创建并返回下一个顺序的二级文件夹路径"""
    if not os.path.exists(BASE_VIDEO_DIR):
        os.makedirs(BASE_VIDEO_DIR)

    existing_dirs = [d for d in os.listdir(BASE_VIDEO_DIR)
                     if os.path.isdir(os.path.join(BASE_VIDEO_DIR, d)) and d.isdigit()]
    max_num = 0
    for d in existing_dirs:
        num = int(d)
        if num > max_num:
            max_num = num

    next_num = max_num + 1
    new_dir = os.path.join(BASE_VIDEO_DIR, str(next_num))
    os.makedirs(new_dir)
    return new_dir

def load_calibration(calib_file, width, height):
    """加载畸变校正映射，返回(mapx, mapy, do_undistort)"""
    if not os.path.exists(calib_file):
        print(f"[WARN] 标定文件 {calib_file} 不存在，跳过畸变校正。")
        return None, None, False
    try:
        data = np.load(calib_file)
        mtx = data['mtx']
        dist = data['dist']
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 0, (width, height))
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (width, height), 5)
        print(f"[INFO] 畸变校正已启用（来自 {calib_file}）")
        return mapx, mapy, True
    except Exception as e:
        print(f"[ERROR] 加载标定文件失败: {e}")
        return None, None, False

def main():
    # 1. 等待20秒
    print("[INFO] 程序启动，等待20秒...")
    time.sleep(2)

    # 2. 打开USB摄像头
    cap = cv2.VideoCapture(USB_CAM_PATH, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, USB_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USB_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)   # 尽可能设置到60帧

    if not cap.isOpened():
        print(f"[ERROR] 无法打开USB摄像头: {USB_CAM_PATH}")
        return

    # 检查曝光值（如果支持）
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)  # 返回 ms，-1 表示不支持
    if exposure == -1:
        exp_str = "N/A"
    else:
        exp_str = f"{exposure:.2f} ms"

    print(f"[INFO] USB摄像头已打开: {USB_WIDTH}x{USB_HEIGHT} | 曝光: {exp_str}")

    # 畸变校正加载
    mapx, mapy, do_undistort = load_calibration(CALIB_FILE, USB_WIDTH, USB_HEIGHT)

    # 3. 创建保存目录
    save_dir = create_next_video_dir()
    print(f"[INFO] 视频将保存至: {save_dir}")

    # 视频录制准备
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_index = 1
    out = None
    frame_count = 0

    # 帧率统计变量
    start_time = time.time()        # 用于计算实际帧率
    fps_frame_count = 0
    last_print_time = start_time

    print("[INFO] 开始录制，按 'q' 键退出...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 摄像头读取帧失败")
            continue

        # 畸变校正
        if do_undistort:
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        # 录制控制（分段）
        if out is None or frame_count >= FRAMES_PER_SEGMENT:
            if out is not None:
                out.release()
                video_index += 1
                frame_count = 0

            video_name = os.path.join(save_dir, f"{video_index}.mp4")
            out = cv2.VideoWriter(video_name, fourcc, VIDEO_FPS, (USB_WIDTH, USB_HEIGHT))
            if not out.isOpened():
                print(f"[ERROR] 无法创建视频文件: {video_name}")
                break
            print(f"[INFO] 开始录制视频段: {video_name}")

        out.write(frame)
        frame_count += 1
        fps_frame_count += 1

        # 实时信息打印（大约每秒一次）
        now = time.time()
        elapsed = now - last_print_time
        if elapsed >= 1.0:   # 每秒打印一次
            # 计算实际帧率
            if elapsed > 0:
                actual_fps = fps_frame_count / elapsed
            else:
                actual_fps = 0.0
            # 再次读取曝光信息（如果支持）
            exposure_val = cap.get(cv2.CAP_PROP_EXPOSURE)
            if exposure_val == -1:
                exp_str = "N/A"
            else:
                exp_str = f"{exposure_val:.2f} ms"
            print(f"[INFO] 实时 | FPS: {actual_fps:5.1f} | 分辨率: {USB_WIDTH}x{USB_HEIGHT} | 曝光: {exp_str} | 已录制: {video_index}段")
            fps_frame_count = 0
            last_print_time = now

        # 显示画面
        cv2.imshow("USB Camera (60fps recording)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 清理
    if out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 程序结束")

if __name__ == '__main__':
    main()