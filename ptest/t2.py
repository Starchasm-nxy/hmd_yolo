import cv2
import sys
from ultralytics import YOLO

# ========= 参数设置 =========
MODEL_PATH = "/home/fu/权重/tongs.pt"  # 你的模型路径
VIDEO_PATH = "123.mp4"                 # 视频文件，与脚本同目录可直接用文件名
OUTPUT_PATH = "output.mp4"            # 输出视频文件名
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

    # 获取原视频的帧率、尺寸等属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or fps > 120:
        fps = 25  # 若获取失败则使用默认帧率
    print(f"[INFO] 原视频: {width}x{height}, {fps:.2f} fps")

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者用 'avc1' 得到更小的文件
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not out.isOpened():
        print("[ERROR] 无法创建输出视频文件，请检查编码器")
        sys.exit(1)

    print(f"[INFO] 开始处理视频，按 'q' 退出，按空格暂停/继续")
    print(f"[INFO] 结果将保存至: {OUTPUT_PATH}")
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
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0]
                        cls = int(box.cls[0])
                        name = result.names[cls]
                        label = f"{name} {conf:.2f}"

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 写入输出视频
            out.write(annotated)

            # 显示窗口
            cv2.imshow("YOLO Video Test", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            pause = not pause
            if pause:
                print("暂停中，按空格继续...")

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] 输出视频已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()