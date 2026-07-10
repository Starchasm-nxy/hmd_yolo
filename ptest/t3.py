import cv2
import sys
import os
from pathlib import Path
from ultralytics import YOLO

# ========= 参数设置 =========
MODEL_PATH = "/home/fu/weights/4m8kd.pt"          # 你的模型路径
VIDEO_DIR = "/media/fu/RAN/4m"        # 视频文件夹（绝对路径）
CONF_THRESH = 0.6
IOU_THRESH = 0.45
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
# ===========================


def get_video_files(directory: str) -> list:
    """扫描文件夹，返回所有视频文件的路径列表"""
    video_files = []
    for f in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, f)
        if os.path.isfile(fpath) and Path(f).suffix.lower() in VIDEO_EXTENSIONS:
            video_files.append(fpath)
    return video_files


def process_video(model, video_path: str) -> bool:
    """
    处理单个视频，在窗口中实时显示结果。
    返回 True 表示正常结束，False 表示用户按 'q' 退出。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return True  # 跳过这个，继续下一个

    video_name = os.path.basename(video_path)
    print(f"[INFO] 正在处理: {video_name}  (q=退出全部, 空格=暂停, n=下一个视频)")

    pause = False

    while True:
        if not pause:
            ret, frame = cap.read()
            if not ret:
                print(f"[INFO] {video_name} 播放结束")
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

            # 在窗口标题上显示当前视频名
            cv2.imshow(f"YOLO - {video_name}", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            return False  # 退出全部
        elif key == ord(' '):
            pause = not pause
            if pause:
                print(f"  [{video_name}] 暂停中，按空格继续...")
        elif key == ord('n'):
            print(f"  [{video_name}] 跳到下一个视频")
            break

    cap.release()
    return True  # 继续下一个


def main():
    # 加载模型
    print("[INFO] 加载模型...")
    model = YOLO(MODEL_PATH)

    # 扫描视频文件
    video_files = get_video_files(VIDEO_DIR)
    if not video_files:
        print(f"[ERROR] 在 {VIDEO_DIR} 中没有找到视频文件")
        sys.exit(1)

    print(f"[INFO] 找到 {len(video_files)} 个视频文件:")
    for vf in video_files:
        print(f"  - {os.path.basename(vf)}")
    print("[INFO] 按键说明: q=退出 空格=暂停 n=跳到下一个视频")

    for video_path in video_files:
        should_continue = process_video(model, video_path)
        if not should_continue:
            print("[INFO] 用户退出")
            break
        cv2.destroyAllWindows()

    cv2.destroyAllWindows()
    print("[INFO] 处理完毕")


if __name__ == "__main__":
    main()
