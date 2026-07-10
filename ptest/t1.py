import cv2
import sys
from ultralytics import YOLO

# ========= 参数设置 =========
MODEL_PATH = "/home/fu/权重/tongv2.pt"  # 你的模型路径
VIDEO_PATH = "123.mp4"    # 视频文件，与脚本同目录可直接用文件名
CONF_THRESH = 0.6
IOU_THRESH = 0.45
# ===========================

def main():
    # 加载模型
    print("[INFO] 加载模型...")
    model = YOLO(MODEL_PATH)

    # 打开视频
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频文件: {VIDEO_PATH}")
        sys.exit(1)

    print("[INFO] 开始处理视频，按 'q' 退出，按空格暂停/继续")
    pause = False

    while True:
        if not pause:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] 视频播放结束")
                break

            # YOLO 推理
            results = model.predict(frame, device='cpu', show=False,
                                    stream=False, verbose=False,
                                    iou=IOU_THRESH, conf=CONF_THRESH)

            # 可视化
            annotated = frame.copy()
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0]
                        cls = int(box.cls[0])
                        name = result.names[cls]
                        label = f"{name} {conf:.2f}"

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("YOLO Video Test", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            pause = not pause
            if pause:
                print("暂停中，按空格继续...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
