import cv2
import pyrealsense2 as rs
import numpy as np

if __name__ == '__main__':
    # 1. 构建管道和配置
    pipeline = rs.pipeline()
    config = rs.config()

    # 👇 在启动前一次性启用彩色流和深度流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # 2. 启动管道
    pipeline.start(config)

    print("RealSense 已启动，按 'q' 键退出")
    try:
        while True:
            # 等待帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            # 彩色帧与深度帧都可能为空（极少见），做防御检查
            if not color_frame or not depth_frame:
                continue

            # 彩色图像
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow('RealSense Color', color_image)

            # 深度图像
            depth_image = np.asanyarray(depth_frame.get_data())
            # 归一化到 0-255 以便 jet 映射（alpha 根据深度范围调整，单位是毫米）
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            cv2.imshow('RealSense Depth', depth_colormap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()