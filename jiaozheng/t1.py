import cv2
import os

save_dir = 'calib_imgs'
os.makedirs(save_dir, exist_ok=True)

# 直接使用设备路径，不指定 CAP_V4L2
cap = cv2.VideoCapture('/dev/video4')
if not cap.isOpened():
    # 备用：尝试索引 4
    cap = cv2.VideoCapture(4)

if not cap.isOpened():
    print("无法打开摄像头，请检查连接或权限。")
    exit()

# 设置格式与分辨率
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
cap.set(cv2.CAP_PROP_FPS, 30)

# 验证
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"实际分辨率: {actual_w} x {actual_h}")

img_id = 0
print("按 s 保存图片，按 q 退出。")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        cv2.imshow('capture', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            img_path = os.path.join(save_dir, f'{img_id:02d}.jpg')
            cv2.imwrite(img_path, frame)
            print(f"已保存: {img_path}")
            img_id += 1
        elif key == ord('q'):
            break
except KeyboardInterrupt:
    print("\n用户退出")
finally:
    cap.release()
    cv2.destroyAllWindows()