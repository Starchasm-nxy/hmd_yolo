#!/usr/bin/env python3
"""
D435 RGB-D SLAM — Python 实现
==============================
使用 Intel RealSense D435 深度相机 + Open3D 进行实时 SLAM 建模。

流程:
  1. D435 采集对齐的彩色+深度帧 (640x480 @30fps)
  2. RGB-D 帧间里程计 (frame-to-frame odometry)
  3. TSDF 体积融合 (volumetric fusion)
  4. 实时 3D 地图可视化

依赖:
  pip install open3d pyrealsense2 opencv-python numpy

用法:
  python d435_slam.py          # 启动 SLAM，每帧都处理
  python d435_slam.py --skip 3 # 每3帧处理一帧 (跳帧省计算)
  python d435_slam.py --noodom # 不使用里程计，直接累积 (调试用)

操作:
  - 慢速移动相机构建地图
  - 按 'q' 退出，按 's' 保存地图为 map.ply
  - 按 'r' 重置地图
"""

import os
import sys
import time
import threading
import argparse
from collections import deque
from pathlib import Path

# 修复 Conda 环境下 OpenCV Qt 字体警告
os.environ.setdefault("QT_QPA_FONTSDIR", "/usr/share/fonts")

import numpy as np
import cv2
import pyrealsense2 as rs

import open3d as o3d
# 使用传统 API (更稳定，文档完善)
from open3d.pipelines import odometry as o3d_odometry
from open3d.pipelines import integration as o3d_integration


# ============================================================
# 配置
# ============================================================
WIDTH, HEIGHT = 640, 480
DEPTH_SCALE = 1000.0          # D435 深度单位: mm → m
VOXEL_SIZE = 0.005            # TSDF 体素尺寸 (5mm)
TSDF_TRUNC = 0.04             # TSDF 截断距离 (4cm)
MAX_DEPTH = 4.0               # 最大深度 (m)
MIN_DEPTH = 0.3               # 最小深度 (m)
FPS_WINDOW = 60               # FPS 滑动平均窗口


class D435Camera:
    """D435 相机封装 — 后台线程采集对齐的彩色+深度帧"""

    def __init__(self, width=WIDTH, height=HEIGHT, fps=30):
        self.width = width
        self.height = height
        self.fps = fps

        self._pipeline = rs.pipeline()
        self._cfg = rs.config()
        self._cfg.enable_stream(rs.stream.depth,  width, height, rs.format.z16,  fps)
        self._cfg.enable_stream(rs.stream.color,  width, height, rs.format.rgb8, fps)
        self._align = rs.align(rs.stream.color)

        self._stop = threading.Event()
        self._thread = None

        # 帧缓冲 (单槽，只保留最新帧)
        self._lock = threading.Lock()
        self._color_frame = None
        self._depth_aligned_mm = None   # 毫米单位的对齐深度
        self._intrinsics = None          # (fx, fy, ppx, ppy)
        self._seq = 0

    @property
    def intrinsics(self):
        """返回 (fx, fy, ppx, ppy)"""
        return self._intrinsics

    def start(self):
        """启动相机管道和采集线程"""
        profile = self._pipeline.start(self._cfg)
        # 获取颜色相机内参
        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        self._intrinsics = (intr.fx, intr.fy, intr.ppx, intr.ppy)
        print(f"[D435] 相机内参: fx={intr.fx:.1f} fy={intr.fy:.1f} "
              f"ppx={intr.ppx:.1f} ppy={intr.ppy:.1f}")

        self._stop.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止采集"""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._pipeline.stop()
        print("[D435] 相机已释放")

    def get_latest(self, seen_seq=0):
        """
        获取最新帧对。返回 (new, color_rgb, depth_aligned_mm, seq)。
        new=False 表示没有新帧。
        """
        with self._lock:
            if self._seq <= seen_seq or self._color_frame is None:
                return False, None, None, self._seq
            return True, self._color_frame.copy(), self._depth_aligned_mm.copy(), self._seq

    def _capture_loop(self):
        """后台采集线程"""
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                aligned = self._align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color_rgb = np.asanyarray(color_frame.get_data())  # HxWx3, RGB8
                depth_mm = np.asanyarray(depth_frame.get_data())   # HxW, uint16 (mm)

                with self._lock:
                    self._color_frame = color_rgb
                    self._depth_aligned_mm = depth_mm
                    self._seq += 1
            except Exception as e:
                print(f"[D435] 采集错误: {e}")
                time.sleep(0.01)


# ============================================================
# SLAM 核心
# ============================================================

class RGBDSLAM:
    """RGB-D SLAM: 里程计 + TSDF 体积融合"""

    def __init__(self, intrinsics, voxel_size=VOXEL_SIZE):
        self.voxel_size = voxel_size

        # 相机内参 (Open3D 格式)
        fx, fy, ppx, ppy = intrinsics
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            WIDTH, HEIGHT, fx, fy, ppx, ppy)

        # TSDF 体积
        self._reset_volume()

        # 相机位姿 (4x4 变换矩阵，初始为单位阵)
        self.pose = np.eye(4)

        # 里程计参数
        self.odom_option = o3d_odometry.OdometryOption()
        self.odom_option.depth_min = MIN_DEPTH
        self.odom_option.depth_max = MAX_DEPTH
        self.odom_option.depth_diff_max = 0.07

        # 轨迹 (用于显示)
        self.trajectory = []  # [(x, y, z), ...]

        # 统计
        self.frame_count = 0
        self.odom_success = 0

    def _reset_volume(self):
        """重置 TSDF 体积"""
        self.volume = o3d_integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=TSDF_TRUNC,
            color_type=o3d_integration.TSDFVolumeColorType.RGB8)

    def reset(self):
        """重置地图和位姿"""
        self._reset_volume()
        self.pose = np.eye(4)
        self.trajectory = []
        self.frame_count = 0
        self.odom_success = 0
        print("[SLAM] 地图已重置")

    def process_frame(self, color_rgb, depth_mm):
        """
        处理一帧: 里程计 → 积分 → 返回点云
        返回: (map_pcd, current_pose) 或 (None, None) 如果跳帧
        """
        self.frame_count += 1

        # 1. 构建 Open3D RGBDImage
        depth_m = depth_mm.astype(np.float32) / DEPTH_SCALE  # 转换为米
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_rgb),
            o3d.geometry.Image(depth_m),
            depth_scale=1.0,       # 已经转换为米
            depth_trunc=MAX_DEPTH,
            convert_rgb_to_intensity=False)

        # 2. RGB-D 里程计 (frame-to-frame)
        if self.frame_count == 1:
            success = True
            trans = np.eye(4)
        else:
            # 使用上一帧的 RGB-D 图像进行匹配
            success, trans, _ = o3d_odometry.compute_rgbd_odometry(
                rgbd, self._last_rgbd,
                self.intrinsic, self.pose,
                o3d_odometry.RGBDOdometryJacobianFromHybridTerm(),
                self.odom_option)

        if success and self.frame_count > 1:
            self.odom_success += 1
            self.pose = self.pose @ trans  # 累积位姿
            self.trajectory.append(self.pose[:3, 3].copy())
        elif self.frame_count == 1:
            self.trajectory.append(np.zeros(3))

        # 3. 积分到 TSDF 体积
        self.volume.integrate(rgbd, self.intrinsic, np.linalg.inv(self.pose))

        # 4. 提取点云用于可视化
        map_pcd = self.volume.extract_point_cloud()
        map_pcd.transform(self.pose)  # 变换到世界坐标系

        # 保存当前帧用于下一次里程计
        self._last_rgbd = rgbd

        return map_pcd, self.pose.copy()

    def get_map_mesh(self):
        """从 TSDF 体积提取 mesh (用于高质量可视化)"""
        mesh = self.volume.extract_triangle_mesh()
        mesh.transform(self.pose)
        mesh.compute_vertex_normals()
        return mesh


# ============================================================
# 可视化
# ============================================================

class SLAMVisualizer:
    """Open3D 可视化窗口, 显示 3D 地图 + 相机轨迹"""

    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="D435 RGB-D SLAM",
                              width=1280, height=720)
        # 背景深灰色
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.15])
        opt.point_size = 2.0

        # 视角控制
        vc = self.vis.get_view_control()
        vc.set_zoom(0.5)
        vc.set_front([0, 0, -1])
        vc.set_up([0, -1, 0])

        self._map_added = False
        self._pcd_cloud = None

    def update(self, map_pcd, pose, trajectory):
        """更新 3D 视图"""
        # 清除旧点云
        if self._map_added and self._pcd_cloud is not None:
            self.vis.remove_geometry(self._pcd_cloud, reset_bounding_box=False)

        # 添加当前地图点云
        if map_pcd is not None and len(map_pcd.points) > 0:
            # 体素下采样 (限制点云大小)
            map_pcd = map_pcd.voxel_down_sample(voxel_size=0.01)
            self.vis.add_geometry(map_pcd, reset_bounding_box=False)
            self._pcd_cloud = map_pcd
            self._map_added = True

        # 更新轨迹线
        if len(trajectory) >= 2 and len(trajectory) % 5 == 0:
            traj_points = np.array(trajectory)
            line_pcd = o3d.geometry.PointCloud()
            line_pcd.points = o3d.utility.Vector3dVector(traj_points)
            colors = np.zeros((len(traj_points), 3))
            # 渐变色: 早期 → 蓝色, 近期 → 绿色
            t = np.linspace(0, 1, len(traj_points))
            colors[:, 0] = 0.2 * (1 - t)
            colors[:, 1] = t
            colors[:, 2] = (1 - t)
            line_pcd.colors = o3d.utility.Vector3dVector(colors)
            # 清理旧轨迹 (通过重建)
            self.vis.add_geometry(line_pcd, reset_bounding_box=False)

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()


# ============================================================
# 2D 预览窗口 (OpenCV, 显示相机画面 + 深度热力图)
# ============================================================

class PreviewWindow:
    """OpenCV 窗口: 彩色画面 + 深度热力图 + FPS + 状态信息"""

    def __init__(self):
        self.fps_times = deque(maxlen=FPS_WINDOW)

    def show(self, color_rgb, depth_mm, pose, frame_count, odom_ok):
        """显示预览画面"""
        now = time.time()
        self.fps_times.append(now)

        # FPS
        if len(self.fps_times) > 1:
            fps = len(self.fps_times) / (self.fps_times[-1] - self.fps_times[0])
        else:
            fps = 0

        # 深度热力图 (彩色映射)
        depth_clipped = np.clip(depth_mm.astype(np.float32), 0, 4000)
        depth_vis = (depth_clipped / 4000.0 * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # 彩色图 (RGB → BGR for OpenCV)
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

        # 状态叠加
        x, y, z = pose[:3, 3]
        status_lines = [
            f"FPS: {fps:.1f}",
            f"Frames: {frame_count}",
            f"Pos: ({x:.2f}, {y:.2f}, {z:.2f})",
            f"Odometry: {'OK' if odom_ok else 'LOST'}",
            "q=quit s=save r=reset",
        ]
        for i, line in enumerate(status_lines):
            cv2.putText(color_bgr, line, (10, 25 + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                       (0, 255, 0) if i < 4 else (180, 180, 180),
                       2, cv2.LINE_AA)

        # 左右并排
        combined = np.hstack([color_bgr, depth_color])
        cv2.imshow("D435 SLAM - Preview", combined)
        return cv2.waitKey(1) & 0xFF


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="D435 RGB-D SLAM")
    parser.add_argument("--skip", type=int, default=1,
                       help="跳帧间隔 (每N帧处理1帧, 默认1)")
    parser.add_argument("--noodom", action="store_true",
                       help="不使用里程计, 直接累积")
    parser.add_argument("--no3d", action="store_true",
                       help="不显示 3D 窗口, 仅显示 2D 预览")
    args = parser.parse_args()

    print("=" * 50)
    print("  D435 RGB-D SLAM — Open3D + pyrealsense2")
    print("=" * 50)
    print(f"  分辨率: {WIDTH}x{HEIGHT}")
    print(f"  跳帧: 每 {args.skip} 帧处理1帧")
    print(f"  里程计: {'禁用' if args.noodom else '启用'}")
    print(f"  3D视图: {'禁用' if args.no3d else '启用'}")
    print()

    # 1. 启动 D435
    cam = D435Camera()
    cam.start()
    time.sleep(0.5)  # 等待第一帧到达

    if cam.intrinsics is None:
        print("[ERROR] 无法获取相机内参，退出")
        cam.stop()
        return

    # 2. 初始化 SLAM
    slam = RGBDSLAM(cam.intrinsics)
    skip_frame_count = 0

    # 3. 可视化
    if not args.no3d:
        viz = SLAMVisualizer()
    else:
        viz = None
    preview = PreviewWindow()

    print("\n[SLAM] 开始建图... 慢速移动相机。按 'q' 退出。\n")

    last_seq = 0
    try:
        while True:
            # 获取最新帧
            got, color_rgb, depth_mm, last_seq = cam.get_latest(last_seq)
            if not got:
                time.sleep(0.001)
                continue

            # 跳帧
            skip_frame_count += 1
            if skip_frame_count % args.skip != 0:
                continue

            # 处理帧
            map_pcd, pose = slam.process_frame(color_rgb, depth_mm)
            odom_ok = slam.odom_success >= slam.frame_count - 3  # 允许少量丢失

            # 更新 3D 视图
            if viz is not None:
                try:
                    viz.update(map_pcd, pose, slam.trajectory)
                except Exception:
                    pass  # 忽略 GUI 临时错误

            # 更新 2D 预览
            key = preview.show(color_rgb, depth_mm, pose, slam.frame_count, odom_ok)

            # 按键处理
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = "map.ply"
                mesh = slam.get_map_mesh()
                if len(mesh.vertices) > 0:
                    o3d.io.write_triangle_mesh(fname, mesh)
                    print(f"[SAVE] 地图已保存到 {fname}")
                else:
                    print("[SAVE] 地图为空, 跳过保存")
            elif key == ord('r'):
                slam.reset()

    except KeyboardInterrupt:
        print("\n[SLAM] 中断")
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        if viz is not None:
            viz.close()
        print("[SLAM] 退出.")


if __name__ == "__main__":
    main()
