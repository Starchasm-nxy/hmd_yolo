import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs

# ==================== 参数设置 ====================
VIDEO_FPS = 30                            # 录制帧率
SEGMENT_SEC = 10                          # 每段视频时长（秒）
FRAMES_PER_SEGMENT = VIDEO_FPS * SEGMENT_SEC  # 每段帧数（300帧）

BASE_COLOR_DIR = 'videos_color'           # 彩色视频一级文件夹
BASE_DEPTH_DIR = 'videos_depth'           # 深度视频一级文件夹
DISPLAY_WINDOW_COLOR = 'RealSense Color'
DISPLAY_WINDOW_DEPTH = 'RealSense Depth'

# 深度伪彩色映射参数（D435 深度单位为毫米）
DEPTH_ALPHA = 0.03
DEPTH_COLORMAP = cv2.COLORMAP_JET

def main():
    # ---------- 1. 等待 20 秒 ----------
    print("[INFO] 程序启动，等待 20 秒...")
    time.sleep(20)

    # ---------- 2. 配置并启动 RealSense 管道 ----------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, VIDEO_FPS)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, VIDEO_FPS)

    try:
        profile = pipeline.start(config)
        print("[INFO] RealSense 已启动 (color + depth 640x480 @ 30fps)")

        # 从配置文件中获取实际的帧率（可靠方法）
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        actual_fps = color_stream.fps()
        print(f"[INFO] 实际帧率: {actual_fps}")

        # ---------- 3. 初始化视频写入与段计数 ----------
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        color_writer = None
        depth_writer = None
        segment_number = 0
        frame_count = 0

        # 帧率统计
        start_ticks = time.time()
        last_print_time = start_ticks
        total_frames = 0

        print("[INFO] 开始录制，按 'q' 键退出...")

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=DEPTH_ALPHA),
                DEPTH_COLORMAP
            )

            cv2.imshow(DISPLAY_WINDOW_COLOR, color_image)
            cv2.imshow(DISPLAY_WINDOW_DEPTH, depth_colormap)

            # 视频分段写入
            if color_writer is None or frame_count >= FRAMES_PER_SEGMENT:
                if color_writer is not None:
                    color_writer.release()
                    depth_writer.release()
                    print(f"[INFO] 段 {segment_number} 录制完成")

                segment_number += 1
                color_dir = os.path.join(BASE_COLOR_DIR, str(segment_number))
                depth_dir = os.path.join(BASE_DEPTH_DIR, str(segment_number))
                os.makedirs(color_dir, exist_ok=True)
                os.makedirs(depth_dir, exist_ok=True)

                color_video_path = os.path.join(color_dir, f"{segment_number}.mp4")
                depth_video_path = os.path.join(depth_dir, f"{segment_number}.mp4")

                color_writer = cv2.VideoWriter(color_video_path, fourcc, VIDEO_FPS, (640, 480))
                depth_writer = cv2.VideoWriter(depth_video_path, fourcc, VIDEO_FPS, (640, 480))

                if not color_writer.isOpened() or not depth_writer.isOpened():
                    print("[ERROR] 无法创建视频文件，请检查磁盘空间和权限")
                    break

                print(f"[INFO] 开始录制段 {segment_number}: {color_video_path}, {depth_video_path}")
                frame_count = 0

            color_writer.write(color_image)
            depth_writer.write(depth_colormap)
            frame_count += 1
            total_frames += 1

            # 实时信息打印（每秒一次）
            now = time.time()
            if now - last_print_time >= 1.0:
                elapsed = now - start_ticks
                actual_fps_val = total_frames / elapsed if elapsed > 0 else 0.0
                print(f"[INFO] 实际 FPS: {actual_fps_val:5.1f} | 当前段: {segment_number} | "
                      f"已录帧数: {frame_count}/{FRAMES_PER_SEGMENT}")
                last_print_time = now

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 用户按下 'q'，退出录制")
                break

    finally:
        if 'color_writer' in locals() and color_writer is not None:
            color_writer.release()
        if 'depth_writer' in locals() and depth_writer is not None:
            depth_writer.release()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] 程序结束，所有视频已保存")

if __name__ == '__main__':
    main()