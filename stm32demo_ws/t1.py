"""
D435 + YOLO 目标检测 → STM32 舵机追踪 (via OpenOCD telnet)
- 848×480 彩色图像, YOLO 实时检测
- 目标在画面左边(x<424) → 舵机角度减小(左转)
- 目标在画面右边(x>=424) → 舵机角度增大(右转)
- 通过 ST-Link V2 + OpenOCD telnet 写 STM32 SRAM (0x20000000)
"""

import os
import time
import signal
import subprocess
import telnetlib
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# ==================== 参数 ====================

D435_WIDTH  = 848
D435_HEIGHT = 480
CENTER_X    = D435_WIDTH // 2      # 424 — 画面中轴线

# YOLO
MODEL_PATH  = "/home/fu/weights/qianshitest.pt"
MODEL_CONF  = 0.3
MODEL_IOU   = 0.45
MODEL_IMGSZ = 640
MODEL_DEVICE = 'cpu'

# 舵机控制
SRAM_ADDR     = 0x20000000         # remote_angle 在 STM32 的固定地址
ANGLE_MIN     = 0
ANGLE_MAX     = 180
SMOOTH_ALPHA  = 0.35               # 指数平滑系数 (越小越平滑)
DEAD_ZONE     = 10                  # 中心死区 ±30px 不调整
MAX_STEP      = 4                   # 每帧最大角度变化 (防抖+限速)
FAIL_RESET    = 10                  # 连续失败 N 次后重启 OpenOCD

# OpenOCD
OPENOCD_CFG_INTERFACE = "/usr/share/openocd/scripts/interface/stlink-v2.cfg"
OPENOCD_CFG_TARGET    = "/usr/share/openocd/scripts/target/stm32f1x.cfg"
TELNET_HOST = "localhost"
TELNET_PORT = 4444

# ==================== OpenOCD 管理 ====================

_openocd_proc = None
_telnet_conn  = None


def openocd_start():
    """后台启动 OpenOCD daemon"""
    global _openocd_proc
    cmd = [
        "openocd",
        "-f", OPENOCD_CFG_INTERFACE,
        "-f", OPENOCD_CFG_TARGET,
    ]
    _openocd_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    time.sleep(2)  # 等 OpenOCD 初始化完成
    print(f"[INFO] OpenOCD started (PID {_openocd_proc.pid})")


def openocd_stop():
    """关闭 OpenOCD"""
    global _openocd_proc, _telnet_conn
    if _telnet_conn:
        try:
            _telnet_conn.close()
        except Exception:
            pass
        _telnet_conn = None
    if _openocd_proc:
        try:
            os.killpg(os.getpgid(_openocd_proc.pid), signal.SIGTERM)
            _openocd_proc.wait(timeout=3)
        except Exception:
            _openocd_proc.kill()
        _openocd_proc = None
        print("[INFO] OpenOCD stopped")


def telnet_connect():
    """连接 OpenOCD telnet"""
    global _telnet_conn
    for attempt in range(10):
        try:
            _telnet_conn = telnetlib.Telnet(TELNET_HOST, TELNET_PORT, timeout=3)
            # 吃掉欢迎信息
            _telnet_conn.read_until(b">", timeout=2)
            print(f"[INFO] Telnet connected to OpenOCD (attempt {attempt+1})")
            return True
        except Exception:
            time.sleep(0.5)
    print("[ERROR] Cannot connect to OpenOCD telnet")
    return False


_fail_count = 0

def servo_write_angle(angle: int):
    """通过 OpenOCD telnet 写舵机角度到 STM32 SRAM (fire-and-forget)"""
    global _telnet_conn, _openocd_proc, _fail_count
    angle = max(ANGLE_MIN, min(ANGLE_MAX, angle))
    try:
        cmd = f"mww {SRAM_ADDR:#x} {angle}\n"
        _telnet_conn.write(cmd.encode())
        # 不等待回显，避免 read_until 阻塞导致罢工
        _fail_count = 0
        return True
    except Exception:
        _fail_count += 1
        if _fail_count >= FAIL_RESET:
            print("[WARN] Telnet failed too many times, restarting OpenOCD...")
            # 先杀旧进程
            try:
                _telnet_conn.close()
            except Exception:
                pass
            try:
                os.killpg(os.getpgid(_openocd_proc.pid), signal.SIGTERM)
                _openocd_proc.wait(timeout=3)
            except Exception:
                pass
            # 重启
            openocd_start()
            telnet_connect()
            _fail_count = 0
            # 重试写入
            try:
                cmd = f"mww {SRAM_ADDR:#x} {angle}\n"
                _telnet_conn.write(cmd.encode())
                return True
            except Exception:
                return False
        return False


# ==================== 主程序 ====================

def main():
    # 注册退出清理
    import atexit
    atexit.register(openocd_stop)

    # ── 启动 OpenOCD + 连接 telnet ──
    openocd_start()
    if not telnet_connect():
        print("[FATAL] OpenOCD telnet unavailable, exiting")
        openocd_stop()
        return

    # ── 加载 YOLO ──
    print("[INFO] Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("[INFO] YOLO model loaded")

    # ── 启动 RealSense ──
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, D435_WIDTH, D435_HEIGHT,
                         rs.format.bgr8, 30)
    pipeline.start(config)
    print(f"[INFO] RealSense color {D435_WIDTH}x{D435_HEIGHT} @ 30fps")

    # ── 创建大窗口 (1920×1080) ──
    WIN_NAME = "YOLO Tracking → STM32 Servo"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WIN_NAME, 1920, 1080)

    # ── 状态变量 ──
    prev_angle = 90.0          # 指数平滑后的角度
    last_sent_angle = 90       # 实际发送的角度 (限速用)
    target_detected = False
    fps_value = 0.0
    total_frames = 0
    start_ticks = time.time()
    last_print = start_ticks

    # ── 舵机归中 ──
    servo_write_angle(90)
    last_sent_angle = 90
    print("[INFO] Servo reset to 90 deg")

    print("[INFO] Tracking started — target left → servo decreases, right → increases")
    print("[INFO] Press 'q' to quit")

    try:
        while True:
            # ── 获取彩色帧 ──
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            canvas = color_image.copy()

            # ── YOLO 推理 ──
            t0 = time.time()
            results = model.predict(
                source=color_image, device=MODEL_DEVICE, show=False,
                stream=False, verbose=False, iou=MODEL_IOU,
                conf=MODEL_CONF, imgsz=MODEL_IMGSZ,
            )
            infer_ms = int((time.time() - t0) * 1000)

            target_detected = False
            target_cx = CENTER_X

            # ── 处理检测结果（取置信度最高的目标） ──
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

                # ── 绘制检测框 ──
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(canvas, (target_cx, target_cy), 4, (0, 0, 255), -1)

                label = f"{names[cls]} {conf:.2f}  pos:({target_cx},{target_cy})"
                cv2.putText(canvas, label, (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2,
                            cv2.LINE_AA)

                # ── 死区判断：中心 ±20px 不调整 ──
                if abs(target_cx - CENTER_X) <= DEAD_ZONE:
                    pass  # keep current angle, do nothing
                else:
                    # ── 计算目标舵机角度 ──
                    # 映射: target_cx ∈ [0, 848] → raw_angle ∈ [180, 0]
                    raw_angle = 180.0 - (target_cx / D435_WIDTH) * 180.0

                    # 指数平滑
                    smooth_angle = (SMOOTH_ALPHA * raw_angle +
                                    (1.0 - SMOOTH_ALPHA) * prev_angle)
                    prev_angle = smooth_angle
                    target_angle = int(np.clip(smooth_angle, ANGLE_MIN, ANGLE_MAX))

                    # 限速：每帧变化不超过 MAX_STEP
                    delta = target_angle - last_sent_angle
                    if delta > MAX_STEP:
                        target_angle = last_sent_angle + MAX_STEP
                    elif delta < -MAX_STEP:
                        target_angle = last_sent_angle - MAX_STEP

                    # ── 发送到 STM32 ──
                    if servo_write_angle(target_angle):
                        last_sent_angle = target_angle

            # ── 画中轴线 x=424 ──
            cv2.line(canvas, (CENTER_X, 0), (CENTER_X, D435_HEIGHT),
                     (255, 0, 0), 2)
            cv2.putText(canvas, "< left", (CENTER_X - 80, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.putText(canvas, "right >", (CENTER_X + 10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # ── 方向指示 ──
            if target_detected:
                direction = "<<<" if target_cx < CENTER_X else ">>>"
                dir_color = (0, 165, 255) if target_cx < CENTER_X else (0, 255, 165)
                cv2.putText(canvas, direction,
                            (target_cx - 30, target_cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, dir_color, 3, cv2.LINE_AA)

                # 舵机角度标签
                try:
                    angle_label = f"Servo: {target_angle} deg"
                except UnboundLocalError:
                    angle_label = "Servo: -- deg"
                cv2.putText(canvas, angle_label, (10, D435_HEIGHT - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                            cv2.LINE_AA)

            # ── FPS 叠加 ──
            overlay = (f"FPS:{fps_value:.1f} | infer:{infer_ms}ms | "
                       f"target:{'YES' if target_detected else 'NO'}")
            cv2.putText(canvas, overlay, (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # ── 显示 ──
            cv2.imshow(WIN_NAME, canvas)

            # ── FPS 统计 ──
            total_frames += 1
            now = time.time()
            elapsed = now - start_ticks
            if elapsed > 0:
                fps_value = total_frames / elapsed
            if now - last_print >= 1.0:
                status = (f"[INFO] FPS:{fps_value:.1f}  "
                          f"infer:{infer_ms}ms  "
                          f"target:{'YES' if target_detected else 'NO':3s}")
                try:
                    status += f"  angle:{target_angle}"
                except UnboundLocalError:
                    status += "  angle:--"
                print(status)
                last_print = now

            # ── 退出 ──
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] User pressed 'q', exiting")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        openocd_stop()
        print("[INFO] Program ended")


if __name__ == "__main__":
    main()
