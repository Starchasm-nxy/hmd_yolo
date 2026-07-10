"""
Gazebo Pan-Tilt Camera YOLO Tracking (screen-capture version)
- Captures Gazebo GUI window for YOLO input
- Controls pan/tilt joints via ign topic -p
"""

import os
import time
import signal
import subprocess
import cv2
import numpy as np
from ultralytics import YOLO

# ==================== 参数 ====================

CAMERA_WIDTH  = 848
CAMERA_HEIGHT = 480
CENTER_X      = CAMERA_WIDTH // 2    # 424
CENTER_Y      = CAMERA_HEIGHT // 2

# 关节角度范围 (rad)
PAN_MIN   = -1.57
PAN_MAX   =  1.57
TILT_MIN  = -0.78
TILT_MAX  =  0.78

# 死区
DEAD_X = 30
DEAD_Y = 40

# 平滑 + 限速
SMOOTH_ALPHA  = 0.35
PAN_STEP_MAX  = 0.08    # rad/frame (~4.5 deg)
TILT_STEP_MAX = 0.05

# YOLO
MODEL_PATH  = "/home/fu/weights/qianshitest.pt"
MODEL_CONF  = 0.3
MODEL_IOU   = 0.45
MODEL_IMGSZ = 640
MODEL_DEVICE = 'cpu'

# Gazebo
WORLD_PATH        = "/home/fu/stm32demo_ws/gazebo/worlds/tracking.world"
MODEL_PATH_GAZEBO = "/home/fu/stm32demo_ws/gazebo/models"

# ==================== Gazebo 管理 ====================

_gazebo_proc = None


def gazebo_start():
    global _gazebo_proc
    env = os.environ.copy()
    env["GAZEBO_MODEL_PATH"] = f"{MODEL_PATH_GAZEBO}:{env.get('GAZEBO_MODEL_PATH', '')}"

    _gazebo_proc = subprocess.Popen(
        ["gazebo", WORLD_PATH],
        env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    time.sleep(8)
    print(f"[INFO] Gazebo started (PID {_gazebo_proc.pid})")


def gazebo_stop():
    global _gazebo_proc
    if _gazebo_proc:
        try:
            os.killpg(os.getpgid(_gazebo_proc.pid), signal.SIGTERM)
            _gazebo_proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(_gazebo_proc.pid), signal.SIGKILL)
            except Exception:
                pass
    subprocess.run(["pkill", "-9", "gzserver"], capture_output=True)
    subprocess.run(["pkill", "-9", "gzclient"], capture_output=True)
    print("[INFO] Gazebo stopped")


def joint_cmd(joint_name: str, target_rad: float, p_gain=30.0, d_gain=3.0):
    """Send JointCmd position target via ign topic"""
    try:
        msg = (f"name: 'pan_tilt_camera::{joint_name}', "
               f"position: {{target: {target_rad:.4f}, "
               f"p_gain: {p_gain}, i_gain: 0.0, d_gain: {d_gain}}}")
        subprocess.run(
            ["ign", "topic", "-t",
             f"/gazebo/tracking_world/pan_tilt_camera/{joint_name}/joint_cmd",
             "-m", "ignition.msgs.JointCmd", "-p", msg],
            timeout=3, capture_output=True,
        )
        return True
    except Exception:
        return False


# ==================== 屏幕截图 ====================

def capture_gazebo_window():
    """用 xdotool + import 截 Gazebo 窗口"""
    try:
        # 找到 Gazebo 窗口 ID
        r = subprocess.run(
            ["xdotool", "search", "--name", "Gazebo"],
            capture_output=True, text=True, timeout=3
        )
        if not r.stdout.strip():
            return None
        win_id = r.stdout.strip().split('\n')[0]

        # 激活窗口并截屏
        subprocess.run(["xdotool", "windowactivate", win_id], timeout=2)
        time.sleep(0.05)
        subprocess.run(
            ["import", "-window", win_id, "/tmp/gazebo_screenshot.png"],
            timeout=2
        )

        frame = cv2.imread("/tmp/gazebo_screenshot.png")
        return frame
    except Exception:
        return None


# ==================== 主程序 ====================

def main():
    import atexit
    atexit.register(gazebo_stop)

    # ── 启动 Gazebo ──
    gazebo_start()

    # ── 归中 ──
    for _ in range(5):
        joint_cmd("pan_joint", 0.0)
        joint_cmd("tilt_joint", 0.0)
        time.sleep(0.2)
    print("[INFO] Pan/Tilt centered")

    # ── 加载 YOLO ──
    print("[INFO] Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("[INFO] YOLO loaded")

    # ── 状态 ──
    prev_pan  = 0.0
    prev_tilt = 0.0
    last_pan  = 0.0
    last_tilt = 0.0
    fps_value = 0.0
    total_frames = 0
    start_ticks = time.time()
    last_print = start_ticks

    cv2.namedWindow("Gazebo Tracking", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Gazebo Tracking", 1280, 720)

    print("[INFO] Tracking started — press 'q' to quit")

    try:
        while True:
            # ── 截取 Gazebo 窗口 ──
            frame = capture_gazebo_window()
            if frame is None:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            if h < 100 or w < 100:
                time.sleep(0.05)
                continue

            canvas = frame.copy()

            # ── YOLO 推理 ──
            t0 = time.time()
            results = model.predict(
                source=frame, device=MODEL_DEVICE, show=False,
                stream=False, verbose=False, iou=MODEL_IOU,
                conf=MODEL_CONF, imgsz=MODEL_IMGSZ,
            )
            infer_ms = int((time.time() - t0) * 1000)

            target_detected = False
            target_cx = w // 2
            target_cy = h // 2

            # ── 取最高置信度目标 ──
            best_conf = 0.0
            best_box  = None
            for result in results:
                boxes = result.boxes
                names = result.names
                if boxes is None:
                    continue
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box

            if best_box is not None:
                r = best_box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = r
                cls = int(best_box.cls[0])
                conf = best_conf
                target_cx = (x1 + x2) // 2
                target_cy = (y1 + y2) // 2
                target_detected = True

                # 绘制
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(canvas, (target_cx, target_cy), 4, (0, 0, 255), -1)
                label = f"{names[cls]} {conf:.2f} ({target_cx},{target_cy})"
                cv2.putText(canvas, label, (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2,
                            cv2.LINE_AA)

                # ── 计算 pan/tilt ──
                center_x = w // 2
                center_y = h // 2

                if abs(target_cx - center_x) > DEAD_X:
                    raw_pan = (target_cx / center_x - 1.0) * PAN_MAX
                    raw_pan = np.clip(raw_pan, PAN_MIN, PAN_MAX)
                else:
                    raw_pan = last_pan

                if abs(target_cy - center_y) > DEAD_Y:
                    raw_tilt = -(target_cy / center_y - 1.0) * TILT_MAX
                    raw_tilt = np.clip(raw_tilt, TILT_MIN, TILT_MAX)
                else:
                    raw_tilt = last_tilt

                # 平滑 + 限速
                smooth_pan  = SMOOTH_ALPHA * raw_pan  + (1 - SMOOTH_ALPHA) * prev_pan
                smooth_tilt = SMOOTH_ALPHA * raw_tilt + (1 - SMOOTH_ALPHA) * prev_tilt
                prev_pan  = smooth_pan
                prev_tilt = smooth_tilt

                def limit(v, last, step):
                    if v - last > step: return last + step
                    if last - v > step: return last - step
                    return v

                target_pan  = limit(smooth_pan,  last_pan,  PAN_STEP_MAX)
                target_tilt = limit(smooth_tilt, last_tilt, TILT_STEP_MAX)

                joint_cmd("pan_joint",  target_pan)
                joint_cmd("tilt_joint", target_tilt)
                last_pan  = target_pan
                last_tilt = target_tilt

                info = (f"Pan:{np.degrees(target_pan):+.0f}deg  "
                        f"Tilt:{np.degrees(target_tilt):+.0f}deg")
                cv2.putText(canvas, info, (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # 画十字线
            cv2.line(canvas, (w//2, 0), (w//2, h), (255, 0, 0), 2)
            cv2.line(canvas, (0, h//2), (w, h//2), (255, 0, 0), 2)

            # FPS
            total_frames += 1
            now = time.time()
            elapsed = now - start_ticks
            fps_value = total_frames / elapsed if elapsed > 0 else 0.0
            overlay = f"FPS:{fps_value:.1f} infer:{infer_ms}ms"
            cv2.putText(canvas, overlay, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if now - last_print >= 1.0:
                det = "YES" if target_detected else "NO"
                print(f"[INFO] FPS:{fps_value:.1f} target:{det} "
                      f"pan:{np.degrees(last_pan):+.0f} tilt:{np.degrees(last_tilt):+.0f}")
                last_print = now

            cv2.imshow("Gazebo Tracking", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] User quit")
                break

    finally:
        cv2.destroyAllWindows()
        gazebo_stop()
        print("[INFO] Program ended")


if __name__ == "__main__":
    main()
