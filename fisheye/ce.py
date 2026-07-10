import cv2
import numpy as np
import glob

# 棋盘格参数
PATTERN_SIZE = (11, 8)
SQUARE_SIZE  = 20

objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

objpoints = []   # 世界坐标（每个元素将被转为 (N,1,3) 以满足 CV_32FC3）
imgpoints = []   # 图像坐标（已是 (N,1,2) CV_32FC2）

images = glob.glob('./calib_images/img_*.png')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)

    if ret:
        corners_sub = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        # ★ 关键修改：重塑为 (N,1,3) 并确保 float32
        objpoints.append(objp.reshape(-1, 1, 3).astype(np.float32))
        imgpoints.append(corners_sub.astype(np.float32))  # 也显式转换保证安全
    else:
        print(f'角点检测失败：{fname}')

if len(objpoints) == 0:
    print('未找到有效棋盘格，请重新采集图像。')
    exit()

# 初始化相机参数
K = np.eye(3, dtype=np.float64)
D = np.zeros((4, 1), dtype=np.float64)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
flags = cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

# 执行标定
ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints, imgpoints, gray.shape[::-1],
    K, D, None, None,
    flags, criteria
)

print('✓ 标定完成')
print('内参矩阵 K:\n', K)
print('畸变系数 D:\n', D)

# 保存参数
np.savez('fisheye_calib.npz', K=K, D=D)