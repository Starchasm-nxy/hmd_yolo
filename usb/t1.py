import cv2

# 方法一：直接用设备路径（推荐，稳定）
cap = cv2.VideoCapture('/dev/video4')

# 方法二：用索引（4 表示 /dev/video4）
# cap = cv2.VideoCapture(4)

if not cap.isOpened():
    print("无法打开 USB 摄像头，请检查连接或设备路径。")
    exit()

print("USB 摄像头已启动，按 'q' 键退出...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取画面。")
        break
    cv2.imshow('USB Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()