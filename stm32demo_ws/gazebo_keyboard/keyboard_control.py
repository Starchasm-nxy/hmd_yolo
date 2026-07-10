"""
Gazebo Pan-Tilt 键盘控制
终端方向键控制云台转动：
  左/右 → Pan 水平旋转 (±90°)
  上/下 → Tilt 俯仰旋转 (±60°)
  q    → 退出
  r    → 归中
"""

import os
import sys
import time
import tty
import termios
import select
import signal
import subprocess
import threading

# ==================== 参数 ====================

PAN_STEP   = 0.10    # 每次按键 pan 步长 (rad, ~5.7°)
TILT_STEP  = 0.05    # 每次按键 tilt 步长 (rad, ~2.9°)
PAN_MIN    = -1.57   # -90°
PAN_MAX    =  1.57   # +90°
TILT_MIN   = -1.05   # -60°
TILT_MAX   =  1.05   # +60°

GAZEBO_MODEL_PATH = "/home/fu/stm32demo_ws/gazebo_keyboard/models"
WORLD_PATH        = "/home/fu/stm32demo_ws/gazebo_keyboard/worlds/pan_tilt.world"

# ==================== 终端键盘 ====================

class TerminalKeyReader:
    """非阻塞读取终端按键（支持方向键）"""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)

    def restore(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_key(self, timeout=0.05):
        """返回按键字符串，超时返回 None"""
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            data = os.read(self.fd, 10)
            return data.decode('utf-8', errors='ignore')
        return None


# ==================== Gazebo 管理 ====================

_gazebo_proc = None


def gazebo_start():
    global _gazebo_proc
    env = os.environ.copy()
    env["GAZEBO_MODEL_PATH"] = f"{GAZEBO_MODEL_PATH}:{env.get('GAZEBO_MODEL_PATH', '')}"

    _gazebo_proc = subprocess.Popen(
        ["gazebo", WORLD_PATH],
        env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    time.sleep(6)
    print(f"[INFO] Gazebo started (PID {_gazebo_proc.pid})")


def gazebo_stop():
    global _gazebo_proc
    if _gazebo_proc:
        try:
            os.killpg(os.getpgid(_gazebo_proc.pid), signal.SIGTERM)
            _gazebo_proc.wait(timeout=3)
        except Exception:
            os.killpg(os.getpgid(_gazebo_proc.pid), signal.SIGKILL)
    subprocess.run(["pkill", "-9", "gzserver"], capture_output=True)
    subprocess.run(["pkill", "-9", "gzclient"], capture_output=True)
    print("\n[INFO] Gazebo stopped")


def joint_cmd(joint_name: str, target_rad: float, pgain=30.0, dgain=3.0):
    """发送 JointCmd 位置目标"""
    try:
        msg = (f"name: 'pan_tilt_unit::{joint_name}', "
               f"position: {{target: {target_rad:.4f}, "
               f"p_gain: {pgain}, i_gain: 0.0, d_gain: {dgain}}}")
        subprocess.run(
            ["ign", "topic", "-t",
             f"/gazebo/pan_tilt_world/pan_tilt_unit/{joint_name}/joint_cmd",
             "-m", "ignition.msgs.JointCmd", "-p", msg],
            timeout=3, capture_output=True,
        )
    except Exception:
        pass


# ==================== 主程序 ====================

def print_status(pan_rad, tilt_rad):
    pan_deg  = pan_rad * 180 / 3.14159
    tilt_deg = tilt_rad * 180 / 3.14159
    bar_w = 40
    pan_pos  = int((pan_rad - PAN_MIN) / (PAN_MAX - PAN_MIN) * bar_w)
    tilt_pos = int((tilt_rad - TILT_MIN) / (TILT_MAX - TILT_MIN) * bar_w)
    pan_bar  = " " * pan_pos + "█" + " " * (bar_w - pan_pos)
    tilt_bar = " " * tilt_pos + "█" + " " * (bar_w - tilt_pos)
    sys.stdout.write(
        f"\rPan :{pan_bar} {pan_deg:+6.1f}°  "
        f"Tilt:{tilt_bar} {tilt_deg:+6.1f}°"
    )
    sys.stdout.flush()


def main():
    import atexit
    atexit.register(gazebo_stop)

    # ── 启动 Gazebo ──
    gazebo_start()

    # ── 初始化变量 ──
    pan  = 0.0
    tilt = 0.0
    reader = TerminalKeyReader()

    # 归中
    joint_cmd("pan_joint", 0.0)
    joint_cmd("tilt_joint", 0.0)

    print("=" * 60)
    print("  Gazebo Pan-Tilt 键盘控制")
    print("  ← →  水平旋转 (Pan)    ↑ ↓  俯仰旋转 (Tilt)")
    print("  r    归中              q    退出")
    print("=" * 60)
    print_status(pan, tilt)

    try:
        while True:
            key = reader.get_key(timeout=0.08)
            if key is None:
                continue

            moved = False

            # ── 处理按键 ──
            if key == '\x1b':  # ESC 序列开始
                # 读取剩余部分
                seq = ''
                for _ in range(3):
                    ch = reader.get_key(timeout=0.02)
                    if ch:
                        seq += ch

                if seq == '[A':       # 上箭头 → Tilt 向上
                    tilt = min(tilt + TILT_STEP, TILT_MAX)
                    moved = True
                elif seq == '[B':     # 下箭头 → Tilt 向下
                    tilt = max(tilt - TILT_STEP, TILT_MIN)
                    moved = True
                elif seq == '[C':     # 右箭头 → Pan 右转
                    pan  = min(pan + PAN_STEP, PAN_MAX)
                    moved = True
                elif seq == '[D':     # 左箭头 → Pan 左转
                    pan  = max(pan - PAN_STEP, PAN_MIN)
                    moved = True

            elif key == 'q' or key == 'Q':
                print("\n[INFO] Quitting...")
                break
            elif key == 'r' or key == 'R':
                pan  = 0.0
                tilt = 0.0
                moved = True
            elif key == '\x03':  # Ctrl-C
                print("\n[INFO] Interrupted")
                break

            # ── 发送命令 ──
            if moved:
                joint_cmd("pan_joint", pan)
                joint_cmd("tilt_joint", tilt)
                print_status(pan, tilt)

    finally:
        reader.restore()
        gazebo_stop()
        print("\n[INFO] Done")


if __name__ == "__main__":
    main()
