import cv2
import numpy as np
import glob
import os

# ============ 参数设置 ============
CHESSBOARD_SIZE = (11, 8)     # 内角点数量 (列, 行)，12x9棋盘 → 11x8
SQUARE_SIZE = 15              # 每个方格的边长（毫米）‼️请用尺子实际测量‼️
IMAGE_FOLDER = "calib_imgs/*.jpg"   # 图片存放路径
OUTPUT_FILE = "usb_calib.npz"
# =================================

# 1. 准备世界坐标系中的三维点（棋盘格所有内角点的物理坐标）
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []   # 三维点
imgpoints = []   # 二维像点
image_size = None

# 2. 读取所有图片，检测棋盘格
images = glob.glob(IMAGE_FOLDER)
if len(images) == 0:
    print(f"❌ 在 '{IMAGE_FOLDER}' 中没有找到图片，请检查路径。")
    exit()

print(f"找到 {len(images)} 张图片，开始检测棋盘格...")
for fname in sorted(images):
    img = cv2.imread(fname)
    if img is None:
        print(f"  无法读取 {fname}，跳过")
        continue
    if image_size is None:
        image_size = (img.shape[1], img.shape[0])  # (宽, 高)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        objpoints.append(objp)
        # 亚像素精度优化
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners)
        print(f"  ✔️ 成功：{fname}")
    else:
        print(f"  ❌ 未检测到完整棋盘格：{fname}")

if len(objpoints) < 5:
    print("\n❌ 成功检测的图片太少（至少需要5张）。请检查：")
    print("  - 棋盘格是否完整出现在画面中")
    print("  - 内角点数量是否正确（应为11x8）")
    print("  - 光照是否均匀，无强烈反光")
    exit()

# 3. 标定
print(f"\n使用 {len(objpoints)} 张图片进行标定...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

# 4. 保存结果
np.savez(OUTPUT_FILE, mtx=mtx, dist=dist)
print(f"\n✅ 标定完成！结果已保存至 {OUTPUT_FILE}")
print("重投影误差:", ret)
print("内参矩阵:\n", mtx)
print("畸变系数:\n", dist.ravel())