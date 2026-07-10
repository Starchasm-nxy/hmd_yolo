import cv2
import os

save_dir = 'calib_imgs'
os.makedirs(save_dir, exist_ok=True)
cap = cv2.VideoCapture('/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0')  # 根据实际摄像头编号修改
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

img_id = 0
print("按 s 保存图片，按 q 退出。")
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break
    if ret:
        print("实际分辨率：", frame.shape[1], "x", frame.shape[0])
    cv2.imshow('capture', frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        img_path = os.path.join(save_dir, f'{img_id:02d}.jpg')
        cv2.imwrite(img_path, frame)
        print(f"已保存: {img_path}")
        img_id += 1
    elif key & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
