import cv2
import pyrealsense2 as rs
import numpy as np

# --- 可调参数 ---
MAX_DISTANCE_M = 5.0        # 只考虑 5 米内的点 (单位: 米)
EXCLUDE_RADIUS = 20         # 找到一个最近点后，排除其周围半径（像素）
                           # 防止同个物体上选中多个点
MORPH_KERNEL_SIZE = 3       # 开运算内核大小（奇数），用于降噪

if __name__ == '__main__':
    # 初始化 RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    # 获取深度比例（将像素值转为米）
    depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"深度比例: {depth_scale:.4f} m/unit")
    print("按 'q' 退出程序")

    # 形态学核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # 转 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())   # 单位: 原始值

            # ---------- 深度图预处理 ----------
            # 转成米，并滤除无效/过远点
            depth_m = depth_image.astype(np.float32) * depth_scale
            max_val = MAX_DISTANCE_M / depth_scale  # 对应原始值上限
            mask = (depth_image > 0) & (depth_image < max_val)
            # 掩膜有效区域，其余置为最大值
            depth_filtered = np.where(mask, depth_image, np.iinfo(depth_image.dtype).max)

            # 开运算去噪（只对有效范围操作）
            # 先把不可用区域设为 0，开运算后再还原
            binary_mask = mask.astype(np.uint8) * 255
            opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            # 用开运算后的掩膜过滤
            depth_filtered[opened_mask == 0] = np.iinfo(depth_image.dtype).max

            # ---------- 找最近 3 个点（不重复区域） ----------
            nearest_points = []  # 存储 (x, y, 距离cm)
            work_depth = depth_filtered.copy()  # 用于逐步屏蔽

            for _ in range(3):
                # 找到当前有效区域内最小值的坐标
                min_val = np.min(work_depth)
                if min_val == np.iinfo(depth_image.dtype).max:
                    break   # 没有更多有效点

                min_idx = np.argmin(work_depth)
                y, x = np.unravel_index(min_idx, work_depth.shape)  # 注意顺序
                dist_cm = min_val * depth_scale * 100  # 转厘米

                nearest_points.append((x, y, dist_cm))

                # 屏蔽周围区域：画一个圆，填充为最大值
                cv2.circle(work_depth, (x, y), EXCLUDE_RADIUS,
                           np.iinfo(depth_image.dtype).max, -1)

            # ---------- 可视化 ----------
            # 在彩色图上绘制
            for i, (x, y, d) in enumerate(nearest_points):
                color = (0, 0, 255)   # 红色
                cv2.circle(color_image, (x, y), 6, color, -1)
                cv2.putText(color_image, f"{i+1}: {d:.1f}cm",
                            (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 加入提示信息
            cv2.putText(color_image, f"Found {len(nearest_points)} targets",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('Top 3 Nearest Points', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()