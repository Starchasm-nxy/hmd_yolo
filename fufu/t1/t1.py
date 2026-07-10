"""
D435 + YOLO 目标检测 + 距离测量 + 视频录制
- 848×480 图像上绘制 16×10 网格
- YOLO 实时检测目标，输出距离、像素坐标、置信度
- 距离 = 目标框面积缩小7倍的中心区域平均深度
- 保存带标记的彩色视频和深度伪彩色视频到文件夹
"""

import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# ==================== 参数设置 ====================

# D435 分辨率
D435_WIDTH = 848
D435_HEIGHT = 480

# 网格参数：16列 × 10行
GRID_COLS = 16
GRID_ROWS = 10
CELL_W = D435_WIDTH // GRID_COLS    # 53 像素/格
CELL_H = D435_HEIGHT // GRID_ROWS   # 48 像素/格

# YOLO 模型参数
MODEL_PATH = "/home/fu/weights/qianshitest.pt"
MODEL_CONF = 0.5
MODEL_IOU = 0.45
MODEL_IMGSZ = 640
MODEL_DEVICE = 'cpu'

# 视频录制参数
VIDEO_FPS = 30
SEGMENT_SEC = 20                          # 每段视频时长（秒）
FRAMES_PER_SEGMENT = VIDEO_FPS * SEGMENT_SEC  # 每段帧数

# 输出目录
BASE_COLOR_DIR = 'videos_color'
BASE_DEPTH_DIR = 'videos_depth'

# 深度伪彩色映射（动态归一化，无距离截断）
DEPTH_COLORMAP = cv2.COLORMAP_JET

# 显示窗口名称
DISPLAY_WIN_COLOR = 'Color Detection'
DISPLAY_WIN_DEPTH = 'Depth Detection'


# ==================== 工具函数 ====================

def draw_grid(image: np.ndarray, color=(80, 80, 80), thickness=1) -> np.ndarray:
    """在图像上绘制 16×10 的细线网格"""
    h, w = image.shape[:2]
    # 竖线（17条列边界，只画内部15条）
    for i in range(1, GRID_COLS):
        x = i * CELL_W
        cv2.line(image, (x, 0), (x, h), color, thickness)
    # 横线（11条行边界，只画内部9条）
    for j in range(1, GRID_ROWS):
        y = j * CELL_H
        cv2.line(image, (0, y), (w, y), color, thickness)
    return image


def get_center_depth(depth_image: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """
    计算目标矩形框面积7倍缩小的中心点的平均距离。

    原理：以检测框中心为基准，取面积 = 框面积 / 7 的矩形区域，
    计算该区域内有效深度值的平均距离（单位：毫米）。

    参数:
        depth_image: D435 深度图（单位 mm）
        x1, y1, x2, y2: 目标边界框
    返回:
        (平均距离_mm, (cx1, cy1, cx2, cy2))
        若无有效深度则距离为 None
    """
    W = x2 - x1
    H = y2 - y1
    area = W * H
    if area <= 0:
        return None, (0, 0, 0, 0)

    # 中心区域尺寸：保持宽高比，面积缩小 7 倍 → 每边缩小 sqrt(7) 倍
    scale = 1.0 / np.sqrt(9)
    cw = int(W * scale)
    ch = int(H * scale)

    # 至少 1×1 像素
    cw = max(1, cw)
    ch = max(1, ch)

    # 边界框中心点
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # 中心区域边界（限制在图像内）
    cx1 = max(0, cx - cw // 2)
    cy1 = max(0, cy - ch // 2)
    cx2 = min(D435_WIDTH, cx1 + cw)
    cy2 = min(D435_HEIGHT, cy1 + ch)
    # 修正：确保不超出图像
    cx1 = max(0, cx2 - cw)
    cy1 = max(0, cy2 - ch)

    # 裁剪 ROI 并计算平均深度
    roi = depth_image[cy1:cy2, cx1:cx2]
    valid = roi[roi > 0]   # 过滤无效深度（0 表示无数据）

    if len(valid) > 0:
        return float(np.mean(valid)), (cx1, cy1, cx2, cy2)
    return None, (cx1, cy1, cx2, cy2)


# ==================== 主程序 ====================

def main():
    print("[INFO] Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("[INFO] YOLO model loaded.")

    # ---------- 配置并启动 RealSense ----------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, D435_WIDTH, D435_HEIGHT, rs.format.bgr8, VIDEO_FPS)
    config.enable_stream(rs.stream.depth, D435_WIDTH, D435_HEIGHT, rs.format.z16, VIDEO_FPS)
    pipeline.start(config)
    print(f"[INFO] RealSense started: color + depth {D435_WIDTH}x{D435_HEIGHT} @ {VIDEO_FPS}fps")

    # ---------- 视频写入初始化 ----------
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    color_writer = None
    depth_writer = None
    segment_number = 0
    frame_count = 0

    # FPS 统计
    start_ticks = time.time()
    last_print_time = start_ticks
    total_frames = 0
    fps_value = 0.0

    print("[INFO] 开始检测与录制，按 'q' 键退出...")

    frame_idx = 0            # 帧序号（用于隔帧测距）
    last_valid_dist = None   # 最近一次有效距离（mm），用作 Out of range 时的回退

    # ---------- 创建 1920×1080 显示窗口 ----------
    cv2.namedWindow(DISPLAY_WIN_COLOR, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow(DISPLAY_WIN_DEPTH, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(DISPLAY_WIN_COLOR, 1920, 1080)
    cv2.resizeWindow(DISPLAY_WIN_DEPTH, 1920, 1080)

    try:
        while True:
            # ---------- 获取帧 ----------
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            frame_idx += 1

            # 创建彩色画布和深度伪彩色画布
            color_canvas = color_image.copy()

            # 深度伪彩色：动态归一化，无距离限制
            depth_norm = np.zeros_like(depth_image, dtype=np.uint8)
            valid = depth_image > 0
            if valid.any():
                vmin = int(depth_image[valid].min())
                vmax = int(depth_image[valid].max())
                if vmax > vmin:
                    depth_norm[valid] = ((depth_image[valid] - vmin)
                                         / (vmax - vmin) * 255).astype(np.uint8)
                else:
                    depth_norm[valid] = 128
            depth_canvas = cv2.applyColorMap(depth_norm, DEPTH_COLORMAP)

            # ---------- 绘制 16×10 网格 ----------
            draw_grid(color_canvas, color=(80, 80, 80), thickness=1)
            draw_grid(depth_canvas, color=(80, 80, 80), thickness=1)

            # ---------- YOLO 推理 ----------
            t0 = time.time()
            results = model.predict(
                source=color_image, device=MODEL_DEVICE, show=False,
                stream=False, verbose=False, iou=MODEL_IOU,
                conf=MODEL_CONF, imgsz=MODEL_IMGSZ
            )
            infer_ms = int((time.time() - t0) * 1000)

            # ---------- 处理检测结果 ----------
            for result in results:
                boxes = result.boxes
                names = result.names
                if boxes is None:
                    continue
                for box in boxes:
                    # 解析检测框
                    r = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = r
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # 中心点坐标
                    ux = (x1 + x2) // 2
                    uy = (y1 + y2) // 2

                    # ---------- 隔帧测距 + 回退（每2帧更新一次） ----------
                    raw_mm, (cx1, cy1, cx2, cy2) = get_center_depth(depth_image, x1, y1, x2, y2)

                    if frame_idx % 2 == 0:                     # 偶数帧：更新距离
                        if raw_mm is not None:
                            last_valid_dist = raw_mm
                        avg_mm = raw_mm if raw_mm is not None else last_valid_dist
                    else:                                      # 奇数帧：复用上次距离
                        avg_mm = last_valid_dist

                    if avg_mm is not None:
                        dist_text = f"{avg_mm / 10:.1f} cm"   # mm → cm
                    else:
                        dist_text = "Out of range"

                    # 测距矩形颜色（有数据=青色，无数据=灰色）
                    measure_color = (255, 255, 0) if avg_mm is not None else (150, 150, 150)

                    # --- 文字行：全部标在测距矩形旁边 ---
                    lines = [
                        f"{names[cls]} {conf:.2f}",
                        f"dist: {dist_text}",
                        f"pos: ({ux},{uy})",
                    ]
                    line_h = 15          # 行高
                    gap = 4              # 与测距矩形的间距

                    # 优先放右侧；若右侧空间不足则放左侧
                    text_x = cx2 + gap
                    if text_x > D435_WIDTH - 135:
                        text_x = cx1 - 135 - gap
                    text_x = max(2, min(text_x, D435_WIDTH - 135))

                    # 文字垂直居中于测距矩形
                    text_start_y = cy1 + (cy2 - cy1) // 2 - (len(lines) * line_h) // 2 + 10

                    # ------ 在两幅画布上绘制（抽取为 inline 循环） ------
                    for canvas in (color_canvas, depth_canvas):
                        # 大框（绿色）
                        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # 中心测距矩形（青色/灰色，线宽1）
                        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), measure_color, 1)
                        # 中心点
                        cv2.circle(canvas, (ux, uy), 4, (0, 0, 255), -1)
                        # 文字逐行排列
                        for i, line in enumerate(lines):
                            ty = text_start_y + i * line_h
                            ty = max(12, min(D435_HEIGHT - 4, ty))  # 钳制不越界
                            cv2.putText(canvas, line, (text_x, ty),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2,
                                        cv2.LINE_AA)

                    # 控制台输出
                    print(f"[DET] {names[cls]:<10} conf={conf:.2f}  "
                          f"pos=({ux:>3d},{uy:>3d})  dist={dist_text}")

            # ---------- 叠加 FPS 和推理时间 ----------
            overlay = f"FPS:{fps_value:.1f} | infer:{infer_ms}ms | grid:{GRID_COLS}x{GRID_ROWS}"
            cv2.putText(color_canvas, overlay, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(depth_canvas, overlay, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # ---------- 显示 ----------
            cv2.imshow(DISPLAY_WIN_COLOR, color_canvas)
            cv2.imshow(DISPLAY_WIN_DEPTH, depth_canvas)

            # ---------- 视频分段写入 ----------
            if color_writer is None or frame_count >= FRAMES_PER_SEGMENT:
                # 关闭上一段
                if color_writer is not None:
                    color_writer.release()
                    depth_writer.release()
                    print(f"[INFO] 段 {segment_number} 录制完成")

                # 创建新段
                segment_number += 1
                color_dir = os.path.join(BASE_COLOR_DIR, str(segment_number))
                depth_dir = os.path.join(BASE_DEPTH_DIR, str(segment_number))
                os.makedirs(color_dir, exist_ok=True)
                os.makedirs(depth_dir, exist_ok=True)

                color_video_path = os.path.join(color_dir, f"{segment_number}.mp4")
                depth_video_path = os.path.join(depth_dir, f"{segment_number}.mp4")

                color_writer = cv2.VideoWriter(
                    color_video_path, fourcc, VIDEO_FPS, (D435_WIDTH, D435_HEIGHT))
                depth_writer = cv2.VideoWriter(
                    depth_video_path, fourcc, VIDEO_FPS, (D435_WIDTH, D435_HEIGHT))

                if not color_writer.isOpened() or not depth_writer.isOpened():
                    print("[ERROR] 无法创建视频文件，请检查磁盘空间和权限")
                    break

                print(f"[INFO] 开始录制段 {segment_number}: "
                      f"{color_video_path}, {depth_video_path}")
                frame_count = 0

            # 写入当前帧
            color_writer.write(color_canvas)
            depth_writer.write(depth_canvas)
            frame_count += 1
            total_frames += 1

            # ---------- 实时状态打印（每秒一次） ----------
            now = time.time()
            if now - last_print_time >= 1.0:
                elapsed = now - start_ticks
                fps_value = total_frames / elapsed if elapsed > 0 else 0.0
                print(f"[INFO] 实际 FPS: {fps_value:5.1f} | 当前段: {segment_number} | "
                      f"已录帧数: {frame_count}/{FRAMES_PER_SEGMENT}")
                last_print_time = now

            # ---------- 按键退出 ----------
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 用户按下 'q'，退出录制")
                break

    finally:
        # 释放资源
        if 'color_writer' in locals() and color_writer is not None:
            color_writer.release()
        if 'depth_writer' in locals() and depth_writer is not None:
            depth_writer.release()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] 程序结束，所有视频已保存")


if __name__ == '__main__':
    main()
