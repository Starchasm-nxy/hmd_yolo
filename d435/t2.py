import cv2
import pyrealsense2 as rs
import numpy as np

if __name__ == '__main__':
    # 1. 初始化管道和配置
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    # 2. 获取深度传感器的深度比例（将像素值转换为米）
    depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    print("RealSense 模拟超声波测距已启动，按 'q' 键退出")
    try:
        while True:
            # 3. 等待并获取帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 将彩色和深度帧转换为 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 4. 计算“超声波”距离：获取中心区域的平均深度
            height, width = depth_image.shape
            center_y, center_x = height // 2, width // 2
            size = 10  # 取中心10x10像素区域计算平均值
            center_region = depth_image[center_y - size:center_y + size, center_x - size:center_x + size]
            
            # 过滤掉无效深度值（0表示无数据）
            valid_depths = center_region[center_region > 0]
            
            if len(valid_depths) > 0:
                avg_distance_mm = np.mean(valid_depths)  # 原始单位是毫米
                avg_distance_cm = avg_distance_mm / 10.0
                distance_text = f"Distance: {avg_distance_cm:.1f} cm"
                print(f"中心区域平均距离: {avg_distance_cm:.1f} cm")
            else:
                distance_text = "Distance: Out of range"

            # 5. 在彩色图像上绘制距离信息
            cv2.putText(color_image, distance_text, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # 绘制中心参考区域（红色方框）
            cv2.rectangle(color_image, 
                          (center_x - size, center_y - size), 
                          (center_x + size, center_y + size), 
                          (0, 0, 255), 2)

            # 6. 显示结果
            cv2.imshow('RealSense Distance Sensor (Ultrasonic Simulation)', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()