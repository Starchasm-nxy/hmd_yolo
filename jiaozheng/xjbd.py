import cv2
import numpy as np
import glob

# 设置棋盘格模板规格
chessboard_size = (11, 8)  # 9x6内角点
square_size = 20  # 每个格子的实际大小（单位：mm，可自定义）

# 世界坐标系中的棋盘格点
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3d点
imgpoints = []  # 2d点

# 读取所有标定图片
images = glob.glob('calib_imgs/*.jpg')  # 放你拍的图片路径

gray = None  # 先定义 gray
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"无法读取图片: {fname}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    print(f"正在处理: {fname}, 检测到角点: {ret}")
    if ret:
        # 亚像素精确化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        # 可视化
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)
cv2.destroyAllWindows()

if objpoints and imgpoints and gray is not None:
    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("相机内参矩阵:\n", mtx)
    print("畸变系数:\n", dist)
    # 保存到文件
    np.savez('calib_resultF.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print("标定结果已保存到 calib_resultE.npz")
else:
    print("没有检测到有效的棋盘格角点，无法标定。")
