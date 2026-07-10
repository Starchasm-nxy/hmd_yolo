#!/usr/bin/env python3
"""
D435 简化 SLAM — ICP 点云累积
==============================
比 d435_slam.py 更轻量的实现:
  - D435 → 对齐的彩色+深度帧
  - 深度 → 3D 点云 (利用相机内参)
  - 帧间 ICP 配准 (point-to-plane)
  - 点云累积 (无 TSDF 体积, 更轻量)
  - OpenCV 窗口实时预览 + 简单 3D 视角

用法:
  python d435_slam_simple.py
  python d435_slam_simple.py --skip 3  # 跳帧

依赖: pip install open3d pyrealsense2 opencv-python numpy
"""

import os
import sys
import time
import threading
import argparse

import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d

os.environ.setdefault("QT_QPA_FONTSDIR", "/usr/share/fonts")

# ============================================================
# 配置
# ============================================================
WIDTH, HEIGHT = 640, 480
VOXEL_SIZE = 0.01           # 降采样体素尺寸 (1cm)
ICP_THRESHOLD = 0.05        # ICP 对应点最大距离 (5cm)
MAX_DEPTH_M = 4.0


# ============================================================
# D435 相机 (与 d435_slam.py 相同)
# ============================================================
class D435Camera:
    """D435 相机, 后台采集对齐的彩色+深度帧"""

    def __init__(self, width=WIDTH, height=HEIGHT, fps=30):
        self._pipeline = rs.pipeline()
        self._cfg = rs.config()
        self._cfg.enable_stream(rs.stream.depth,  width, height, rs.format.z16,  fps)
        self._cfg.enable_stream(rs.stream.color,  width, height, rs.format.rgb8, fps)
        self._align = rs.align(rs.stream.color)

        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._color_frame = None
        self._depth_mm = None
        self._intrinsics = None
        self._seq = 0

    @property
    def intrinsics(self):
        return self._intrinsics

    def start(self):
        profile = self._pipeline.start(self._cfg)
        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        self._intrinsics = (intr.fx, intr.fy, intr.ppx, intr.ppy)

        self._stop.clear()
        self._thread = threading.Thread(target=self._capture, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._pipeline.stop()

    def get_latest(self, seen_seq=0):
        with self._lock:
            if self._seq <= seen_seq or self._color_frame is None:
                return False, None, None, self._seq
            return True, self._color_frame.copy(), self._depth_mm.copy(), self._seq

    def _capture(self):
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                aligned = self._align.process(frames)
                depth = np.asanyarray(aligned.get_depth_frame().get_data())
                color = np.asanyarray(aligned.get_color_frame().get_data())
                with self._lock:
                    self._color_frame = color
                    self._depth_mm = depth
                    self._seq += 1
            except Exception:
                time.sleep(0.005)


# ============================================================
# 深度图 → 点云
# ============================================================
def depth_to_pointcloud(depth_mm, color_rgb, fx, fy, ppx, ppy,
                         max_depth_m=MAX_DEPTH_M, stride=2):
    """
    将深度图转换为 Open3D 彩色点云。
    stride=2 表示隔点采样 (减少 75% 点数)
    """
    h, w = depth_mm.shape
    depth_m = depth_mm.astype(np.float32) / 1000.0  # → 米

    # 生成像素坐标网格 (隔点采样)
    vv, uu = np.mgrid[stride//2:h:stride, stride//2:w:stride]
    zz = depth_m[vv, uu]
    valid = (zz > 0.2) & (zz < max_depth_m)
    vv, uu, zz = vv[valid], uu[valid], zz[valid]

    # 投影到 3D
    xx = (uu - ppx) * zz / fx
    yy = (vv - ppy) * zz / fy

    # 颜色
    colors = color_rgb[vv, uu, :].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.column_stack([xx, yy, zz]))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="D435 Simple ICP SLAM")
    parser.add_argument("--skip", type=int, default=1,
                       help="跳帧间隔 (默认1)")
    args = parser.parse_args()

    print("=" * 50)
    print("  D435 Simple ICP SLAM")
    print("=" * 50)

    # 1. 启动相机
    cam = D435Camera()
    cam.start()
    time.sleep(0.5)

    fx, fy, ppx, ppy = cam.intrinsics
    print(f"  内参: fx={fx:.1f} fy={fy:.1f} ppx={ppx:.1f} ppy={ppy:.1f}")

    # 2. SLAM 状态
    world_pcd = o3d.geometry.PointCloud()  # 全局地图
    pose = np.eye(4)                        # 当前位姿
    prev_pcd = None                         # 上一帧点云

    # 可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="D435 ICP-SLAM 3D", width=800, height=600)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.05, 0.05, 0.1])

    print("\n[ICP SLAM] 开始建图... 按 'q' 退出\n")

    last_seq = 0
    frame_n = 0
    fps_times = []

    try:
        while True:
            got, color, depth_mm, last_seq = cam.get_latest(last_seq)
            if not got:
                time.sleep(0.001)
                continue

            frame_n += 1
            if frame_n % args.skip != 0:
                continue

            # 深度 → 当前帧点云 (stride=2 隔点采样提速)
            t0 = time.time()
            cur_pcd = depth_to_pointcloud(depth_mm, color, fx, fy, ppx, ppy, stride=2)

            if len(cur_pcd.points) < 100:
                continue

            # 体素降采样
            cur_pcd = cur_pcd.voxel_down_sample(VOXEL_SIZE)

            # ICP 配准 (point-to-plane, 第一帧跳过)
            if prev_pcd is not None and len(prev_pcd.points) > 500:
                # 初步位姿估计
                init_guess = np.eye(4)
                reg = o3d.pipelines.registration.registration_icp(
                    cur_pcd, prev_pcd, ICP_THRESHOLD, init_guess,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=30))
                delta_pose = reg.transformation
                pose = pose @ delta_pose

            # 当前帧变换到世界坐标系后加入全局地图
            cur_world = o3d.geometry.PointCloud(cur_pcd)
            cur_world.transform(pose)

            # 全局地图下采样 (控制点数)
            world_pcd += cur_world
            world_pcd = world_pcd.voxel_down_sample(VOXEL_SIZE)

            prev_pcd = cur_pcd
            dt = time.time() - t0

            # FPS
            fps_times.append(time.time())
            if len(fps_times) > 30:
                fps_times.pop(0)
            fps = len(fps_times) / (fps_times[-1] - fps_times[0]) if len(fps_times) > 1 else 0

            # 更新 3D 视图
            vis.clear_geometries()
            vis.add_geometry(o3d.geometry.PointCloud(world_pcd), reset_bounding_box=False)

            # 画当前相机位置
            cam_point = o3d.geometry.PointCloud()
            cam_point.points = o3d.utility.Vector3dVector(pose[:3, 3].reshape(1, 3))
            cam_point.paint_uniform_color([1, 0, 0])
            vis.add_geometry(cam_point, reset_bounding_box=False)

            vis.poll_events()
            vis.update_renderer()

            # 终端状态
            x, y, z = pose[:3, 3]
            print(f"\r[#{frame_n}] fps={fps:.1f} pos=({x:.2f},{y:.2f},{z:.2f}) "
                  f"pts={len(world_pcd.points)}  ", end="", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()

        # 保存地图
        if len(world_pcd.points) > 0:
            o3d.io.write_point_cloud("map_simple.ply", world_pcd)
            print(f"\n地图已保存: map_simple.ply ({len(world_pcd.points)} 点)")
        print("退出.")


if __name__ == "__main__":
    main()
