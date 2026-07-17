"""
USB 相机参数实时调整工具
=/- 调值 | j/k 选参 | r 自动 | d 出厂 | q 退出
"""

import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==================== 相机路径 ====================
USB_CAM_PATH = '/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0'
USB_WIDTH = 640
USB_HEIGHT = 480

# ==================== 中文字体 ====================
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'

# ==================== 可调参数定义 ====================
# (key, prop_id, name, default, min, max, step, auto_key, description)
PARAMS_LIST = [
    ('e', cv2.CAP_PROP_AUTO_EXPOSURE,     '自动曝光',       1,    0,    3,   1,   None,   '切换: 手动(0)/自动(1)/光圈优先(3)'),
    ('X', cv2.CAP_PROP_EXPOSURE,          '曝光值',         0,  -13,    0,   1,   'e',    '手动曝光补偿(-13~0)，需先切到手动曝光'),
    ('b', cv2.CAP_PROP_BRIGHTNESS,        '亮度',           0,  -64,   64,   1,   None,   '画面亮度，越高越亮'),
    ('c', cv2.CAP_PROP_CONTRAST,          '对比度',         32,   0,   95,   1,   None,   '明暗差异，越高反差越大'),
    ('s', cv2.CAP_PROP_SATURATION,        '饱和度',         64,   0,  100,   1,   None,   '色彩浓度，越高越鲜艳'),
    ('g', cv2.CAP_PROP_GAIN,              '增益',           0,    0,  100,   1,   None,   '信号放大倍率，提亮但增噪点'),
    ('m', cv2.CAP_PROP_GAMMA,             'Gamma',        100,    1,  500,   1,   None,   '灰度曲线校正，调整中间调亮度'),
    ('w', cv2.CAP_PROP_AUTO_WB,           '自动白平衡',     1,    0,    1,   1,   None,   '切换: 手动(0)/自动(1)'),
    ('T', cv2.CAP_PROP_WB_TEMPERATURE,    '色温',         4600, 2800, 6500, 100,   'w',    '白平衡色温(K)，低=偏蓝 高=偏黄'),
    ('p', cv2.CAP_PROP_SHARPNESS,         '锐度',           2,    0,   10,   1,   None,   '边缘清晰度，越高越锐利'),
    ('B', cv2.CAP_PROP_BACKLIGHT,         '背光补偿',       0,    0,   10,   1,   None,   '逆光场景亮度补偿'),
    ('h', cv2.CAP_PROP_HUE,               '色调',           0,  -40,   40,   1,   None,   '色相偏移，负=偏红 正=偏蓝'),
]

KEY_MAP = {item[0]: i for i, item in enumerate(PARAMS_LIST)}
current = {}
selected_idx = 0


# ---------- 参数读写 ----------
def read_all_params(cap):
    for key, pid, *_ in PARAMS_LIST:
        val = cap.get(pid)
        current[key] = int(val) if val != -1 else None


def apply_param(cap, key):
    info = PARAMS_LIST[KEY_MAP[key]]
    pid, name = info[1], info[2]
    val = current[key]
    if val is not None:
        cap.set(pid, val)
        actual = int(cap.get(pid))
        if actual != val:
            print(f"  [WARN] {name} 写入 {val}，实际读回 {actual}")
            current[key] = actual


def disable_auto(cap, key):
    auto_key = PARAMS_LIST[KEY_MAP[key]][7]
    if auto_key is not None:
        auto_pid = PARAMS_LIST[KEY_MAP[auto_key]][1]
        cap.set(auto_pid, 0)
        current[auto_key] = 0


def adjust_param(cap, key, delta):
    _, _, name, _, vmin, vmax, step, *_ = PARAMS_LIST[KEY_MAP[key]]
    cur = current.get(key, 0)
    if cur is None:
        cur = 0
    new_val = max(vmin, min(vmax, cur + delta * step))
    disable_auto(cap, key)
    current[key] = new_val
    apply_param(cap, key)
    direction = "+" if delta > 0 else "-"
    print(f"  [{direction}] {name} -> {new_val}")


# ---------- 一键恢复 ----------
def reset_to_auto(cap):
    """切回自动模式，保留硬件当前值"""
    print("\n" + "=" * 50)
    print("  切回自动模式")
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1); current['e'] = 1
    cap.set(cv2.CAP_PROP_AUTO_WB, 1);      current['w'] = 1
    print("  AUTO_EXPOSURE -> 1 (自动) | AUTO_WB -> 1 (自动)")
    for key, pid, name, *_ in PARAMS_LIST:
        if key in ('e', 'w'):
            continue
        val = int(cap.get(pid))
        if val != -1:
            current[key] = val
    print("  其他参数保持硬件当前值")
    print("=" * 50 + "\n")


def reset_to_defaults(cap):
    """写入所有预设默认值"""
    print("\n" + "=" * 50)
    print("  恢复出厂默认值")
    for key, pid, name, default, _, _, _, auto_key, *_ in PARAMS_LIST:
        if key in ('e', 'w'):
            cap.set(pid, 1); current[key] = 1
            print(f"  {name} -> 自动")
        else:
            if auto_key and current.get(auto_key) == 1:
                disable_auto(cap, key)
            cap.set(pid, default); current[key] = default
            actual = int(cap.get(pid))
            ok = "v" if actual == default else f"(读回:{actual})"
            print(f"  {name} -> {default} {ok}")
    print("=" * 50 + "\n")


# ---------- 导入导出 ----------
EXPORT_FILE = "camera_params.json"

# t8内部key -> json字段名 + 注释
_JSON_MAP = [
    ('e', 'auto_exposure',       '0=手动 1=自动 3=光圈优先'),
    ('X', 'exposure',            '手动曝光值'),
    ('b', 'brightness',          '亮度'),
    ('c', 'contrast',            '对比度'),
    ('s', 'saturation',          '饱和度'),
    ('g', 'gain',                '增益'),
    ('m', 'gamma',               'Gamma'),
    ('w', 'auto_wb',             '0=手动 1=自动'),
    ('T', 'wb_temperature',      '色温 K'),
    ('p', 'sharpness',           '锐度'),
    ('B', 'backlight',           '背光补偿'),
    ('h', 'hue',                 '色调'),
]

# json字段名 -> (OpenCV prop_id, 默认值)
_JSON_TO_PROP = {
    'auto_exposure':    (cv2.CAP_PROP_AUTO_EXPOSURE,   1),
    'exposure':         (cv2.CAP_PROP_EXPOSURE,        0),
    'brightness':       (cv2.CAP_PROP_BRIGHTNESS,      0),
    'contrast':         (cv2.CAP_PROP_CONTRAST,       32),
    'saturation':       (cv2.CAP_PROP_SATURATION,     64),
    'gain':             (cv2.CAP_PROP_GAIN,            0),
    'gamma':            (cv2.CAP_PROP_GAMMA,         100),
    'auto_wb':          (cv2.CAP_PROP_AUTO_WB,         1),
    'wb_temperature':   (cv2.CAP_PROP_WB_TEMPERATURE,4600),
    'sharpness':        (cv2.CAP_PROP_SHARPNESS,       2),
    'backlight':        (cv2.CAP_PROP_BACKLIGHT,       0),
    'hue':              (cv2.CAP_PROP_HUE,             0),
}


def export_params(cap):
    """导出当前相机参数到 camera_params.json"""
    read_all_params(cap)
    data = {}
    for t8_key, json_key, _note in _JSON_MAP:
        val = current.get(t8_key)
        if val is not None:
            data[json_key] = val
    data['_meta'] = {
        'fourcc': 'MJPG',
        'width': 640,
        'height': 480,
        'fps': 30,
        'export_time': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(EXPORT_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print(f"  参数已导出到 {EXPORT_FILE}")
    print(f"{'='*55}")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"{'='*55}\n")


def import_params(cap):
    """从 camera_params.json 导入参数并写入相机"""
    try:
        with open(EXPORT_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"\n  [ERROR] {EXPORT_FILE} 不存在，请先按 s 导出")
        return
    except json.JSONDecodeError as e:
        print(f"\n  [ERROR] JSON 解析失败: {e}")
        return

    print(f"\n{'='*55}")
    print(f"  从 {EXPORT_FILE} 导入参数")
    print(f"{'='*55}")

    for json_key, (pid, default) in _JSON_TO_PROP.items():
        val = data.get(json_key, default)
        if json_key in ('auto_exposure', 'auto_wb'):
            # 自动/手动开关类
            label = {0: '手动', 1: '自动', 3: '光圈优先'}.get(val, str(val))
            cap.set(pid, val)
            # 更新 current
            mapped_t8 = {'auto_exposure': 'e', 'auto_wb': 'w'}[json_key]
            current[mapped_t8] = val
            print(f"  {json_key}: {val} ({label})")
        else:
            cap.set(pid, val)
            # 映射回 t8 key
            for t8_key, jk, _ in _JSON_MAP:
                if jk == json_key:
                    current[t8_key] = val
                    break
            actual = int(cap.get(pid))
            ok = "v" if actual == val else f"(读回:{actual})"
            print(f"  {json_key}: {val} {ok}")

    print(f"{'='*55}\n")


def toggle_auto_exposure(cap):
    vals = [0, 1, 3]; labels = {0: "手动", 1: "自动", 3: "光圈优先"}
    cur = current.get('e', 1)
    if cur not in vals: cur = 1
    new = vals[(vals.index(cur) + 1) % len(vals)]
    current['e'] = new
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, new)
    print(f"  自动曝光 -> {new} ({labels[new]})")


def toggle_auto_wb(cap):
    cur = current.get('w', 1)
    new = 1 if cur == 0 else 0
    current['w'] = new; cap.set(cv2.CAP_PROP_AUTO_WB, new)
    print(f"  自动白平衡 -> {'自动' if new else '手动'}")


# ==================== PIL 中文渲染 ====================
def _pil_put_text(img, xy, text, font, color):
    """用 PIL 在 OpenCV 图像上绘制文字（支持中文）"""
    h, w = img.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(xy, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _pil_put_multiline(img, lines, x, y_start, line_h, fn):
    """逐行调用 fn(line_img, x, y, text, ...)，返回 y_end"""
    y = y_start
    for text in lines:
        img = fn(img, (x, y), text)
        y += line_h
    return img, y


# ==================== UI 绘制 ====================
def draw_ui(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 加载字体
    try:
        font_cn = ImageFont.truetype(FONT_PATH, 16)
        font_sm = ImageFont.truetype(FONT_PATH, 13)
        font_tip = ImageFont.truetype(FONT_PATH, 14)
    except Exception:
        # 回退：无中文字体时用英文名
        font_cn = ImageFont.load_default()
        font_sm = ImageFont.load_default()
        font_tip = ImageFont.load_default()

    # --- 参数面板 ---
    n = len(PARAMS_LIST)
    panel_w, line_h = 390, 19
    panel_h = 12 + n * line_h

    # 半透明背景
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    overlay[5:5 + panel_h, 5:5 + panel_w] = cv2.addWeighted(
        overlay[5:5 + panel_h, 5:5 + panel_w], 0.40,
        panel, 0.60, 0)

    y = 10
    for i, (key, _, name, _, _, _, _, auto_key, desc) in enumerate(PARAMS_LIST):
        val = current.get(key)
        val_str = str(val) if val is not None else "N/A"
        is_sel = (i == selected_idx)

        # 颜色
        if key in ('e', 'w'):
            color = (0, 220, 80) if val == 1 else (255, 220, 40)
        elif auto_key and current.get(auto_key) == 1:
            color = (140, 140, 140)
        else:
            color = (255, 220, 40)

        # 选中行高亮背景
        if is_sel:
            hl = np.zeros((line_h, panel_w, 3), dtype=np.uint8)
            overlay[5 + y - 2:5 + y + line_h - 2, 5:5 + panel_w] = cv2.addWeighted(
                overlay[5 + y - 2:5 + y + line_h - 2, 5:5 + panel_w], 0.55,
                hl + 55, 0.45, 0)
            color = (255, 255, 255)

        prefix = ">" if is_sel else " "
        line = f"{prefix} [{key}] {name}: {val_str}"
        overlay = _pil_put_text(overlay, (12, y), line, font_cn,
                                (color[2], color[1], color[0]))
        y += line_h

    # --- 底部提示栏 ---
    if selected_idx < len(PARAMS_LIST):
        desc = PARAMS_LIST[selected_idx][8]
    else:
        desc = ""

    # 操作提示
    tips = "=/-:调值 | j/k:选参 | e/w:曝光白平衡 r:自动 d:出厂 | s:导出 l:导入 | q:退出"
    t_y = h - 30
    overlay = _pil_put_text(overlay, (10, t_y), tips, font_tip, (180, 180, 180))

    # 选中参数说明
    if desc:
        overlay = _pil_put_text(overlay, (10, t_y - 20), f"▸ {desc}", font_sm,
                                (120, 200, 255))

    return overlay


# ==================== 帮助 ====================
def print_help():
    print("""
  操作说明：
  ┌────────────┬──────────────────────────────────┐
  │   按键     │  功能                            │
  ├────────────┼──────────────────────────────────┤
  │  = / +     │  调大 当前选中的参数              │
  │  - / _     │  调小 当前选中的参数              │
  │  j / k     │  选中上/下一个参数               │
  │  Tab       │  选中下一个参数                  │
  │  e         │  切换自动曝光 (手动/自动/光圈)    │
  │  w         │  切换自动白平衡 (手动/自动)       │
  │  r         │  切回自动模式（保留硬件当前值）   │
  │  d         │  恢复出厂默认值（写入预设值）     │
  │  s         │  导出参数到 camera_params.json    │
  │  l         │  从 camera_params.json 导入参数   │
  │  q         │  退出程序                         │
  ├────────────┴──────────────────────────────────┤
  │  选中行底部会显示该参数的功能说明             │
  └───────────────────────────────────────────────┘
""")


# ==================== 主循环 ====================
def main():
    global selected_idx

    print("=" * 55)
    print("  USB 相机参数实时调整工具")
    print("=" * 55)

    cap = cv2.VideoCapture(USB_CAM_PATH, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开USB摄像头: {USB_CAM_PATH}")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, USB_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, USB_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    read_all_params(cap)
    print("[INFO] 摄像头已打开，当前参数：")
    print("-" * 40)
    for key, _, name, *_ in PARAMS_LIST:
        print(f"  {name:10s} = {current.get(key)}")
    print("-" * 40)
    print_help()

    win_name = "USB Camera Tuning"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(win_name, 640, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        overlay = draw_ui(frame)
        cv2.imshow(win_name, overlay)

        raw_key = cv2.waitKey(1) & 0xFF

        if raw_key == ord('q'):
            break

        elif raw_key == ord('r'):
            reset_to_auto(cap)
            read_all_params(cap)

        elif raw_key == ord('d'):
            reset_to_defaults(cap)
            read_all_params(cap)

        elif raw_key == ord('e'):
            toggle_auto_exposure(cap)

        elif raw_key == ord('w'):
            toggle_auto_wb(cap)

        elif raw_key == ord('s'):
            export_params(cap)

        elif raw_key == ord('l'):
            import_params(cap)

        elif raw_key in (ord('='), ord('+'), 43, 61):
            key = PARAMS_LIST[selected_idx][0]
            if key in ('e', 'w'):
                toggle_auto_exposure(cap) if key == 'e' else toggle_auto_wb(cap)
            else:
                adjust_param(cap, key, +1)

        elif raw_key in (ord('-'), ord('_'), 45, 95):
            key = PARAMS_LIST[selected_idx][0]
            if key in ('e', 'w'):
                toggle_auto_exposure(cap) if key == 'e' else toggle_auto_wb(cap)
            else:
                adjust_param(cap, key, -1)

        elif raw_key in (ord('j'), ord('J')):
            selected_idx = (selected_idx - 1) % len(PARAMS_LIST)

        elif raw_key in (ord('k'), ord('K'), 9):
            selected_idx = (selected_idx + 1) % len(PARAMS_LIST)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 程序结束")


if __name__ == '__main__':
    main()
