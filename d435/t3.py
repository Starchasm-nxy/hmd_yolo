import cv2
import pyrealsense2 as rs
import numpy as np
from collections import deque

if __name__ == '__main__':
    # ---------- 初始化 RealSense ----------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    # ---------- 可移动矩形参数 ----------
    rect_w, rect_h = 10, 10        # 矩形宽高（像素）
    img_w, img_h = 640, 480        # 图像尺寸（需与彩色流一致）
    rect_x = img_w // 2 - rect_w // 2   # 起始位置：中心
    rect_y = img_h // 2 - rect_h // 2
    step = 10                      # 每次移动的像素数

    # ---------- 距离历史缓冲 ----------
    N_FRAMES = 15                  # 滑动窗口帧数
    TRIM = 2                       # 去掉的最高/最低个数
    dist_history: deque = deque(maxlen=N_FRAMES)

    print("控制说明：W A S D 移动矩形框，按 'q' 退出")
    print(f"当前测距区域大小：{rect_w}x{rect_h} 像素")
    print(f"显示 {N_FRAMES} 帧内平均距离（去掉 {TRIM} 个最高和最低）")

    try:
        while True:
            # 等待帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # ---------- 裁剪矩形区域并计算平均深度 ----------
            x1, y1 = rect_x, rect_y
            x2, y2 = rect_x + rect_w, rect_y + rect_h
            roi = depth_image[y1:y2, x1:x2]
            valid = roi[roi > 0]   # 过滤无效深度（0 表示无数据）

            if len(valid) > 0:
                avg_mm = np.mean(valid)
                avg_cm = avg_mm / 10.0

                # 加入历史缓冲
                dist_history.append(avg_cm)

                # 计算去头尾平均
                if len(dist_history) >= N_FRAMES - TRIM * 2:
                    sorted_dist = sorted(dist_history)
                    trimmed = sorted_dist[TRIM:-TRIM]
                    trimmed_avg = np.mean(trimmed)
                    dist_text = f"{avg_cm:.1f} cm  |  avg: {trimmed_avg:.1f} cm"
                else:
                    dist_text = f"{avg_cm:.1f} cm  |  avg: ..."
            else:
                dist_text = "Out of range"

            # ---------- 绘制矩形和文字 ----------
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 文字位置限制在窗口内：上方够就放上方，否则放矩形下方
            (tw, th), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_x = max(0, min(x1, img_w - tw))
            if y1 - 10 - th > 0:
                text_y = y1 - 10                     # 放矩形上方
            else:
                text_y = y2 + th + 10                # 放矩形下方
            cv2.putText(color_image, dist_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 窗口顶部显示帧数
            cv2.putText(color_image, f"buffer: {len(dist_history)}/{N_FRAMES}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('RealSense - Move ROI with WASD', color_image)

            # ---------- 键盘控制（WASD） ----------
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('w'):   # 上
                rect_y = max(0, rect_y - step)
            elif key == ord('s'):   # 下
                rect_y = min(img_h - rect_h, rect_y + step)
            elif key == ord('a'):   # 左
                rect_x = max(0, rect_x - step)
            elif key == ord('d'):   # 右
                rect_x = min(img_w - rect_w, rect_x + step)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()