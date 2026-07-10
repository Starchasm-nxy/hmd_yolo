import cv2, os

cam_path = '/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0'
cap = cv2.VideoCapture(cam_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 根据相机调整
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

img_dir = './calib_images'
os.makedirs(img_dir, exist_ok=True)
count = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.imshow('Capture - Press S to save, Q to quit', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite(f'{img_dir}/img_{count:03d}.png', frame)
        print(f'Saved image {count}')
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()