# ==================== USB相机参数（t8.py 导出） ====================
# 导出时间: 2026-07-15 10:06:41

USB_AUTO_EXPOSURE = 3  # 0=手动 1=自动 3=光圈优先
USB_EXPOSURE = 50  # 手动曝光值
USB_BRIGHTNESS = -37  # 亮度
USB_CONTRAST = 32  # 对比度
USB_SATURATION = 64  # 饱和度
USB_GAIN = 37  # 增益
USB_GAMMA = 100  # Gamma
USB_AUTO_WB = 1  # 0=手动 1=自动
USB_WB_TEMPERATURE = 4600  # 色温 K
USB_SHARPNESS = 2  # 锐度
USB_BACKLIGHT = 0  # 背光补偿
USB_HUE = 0  # 色调

# --- 可直接粘贴到 USBCameraSource.start() 的 cap.set() 块 ---
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
cap.set(cv2.CAP_PROP_AUTO_WB, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, 50)  # 手动曝光值
cap.set(cv2.CAP_PROP_BRIGHTNESS, -37)  # 亮度
cap.set(cv2.CAP_PROP_CONTRAST, 32)  # 对比度
cap.set(cv2.CAP_PROP_SATURATION, 64)  # 饱和度
cap.set(cv2.CAP_PROP_GAIN, 37)  # 增益
cap.set(cv2.CAP_PROP_GAMMA, 100)  # Gamma
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600)  # 色温 K
cap.set(cv2.CAP_PROP_SHARPNESS, 2)  # 锐度
cap.set(cv2.CAP_PROP_BACKLIGHT, 0)  # 背光补偿
cap.set(cv2.CAP_PROP_HUE, 0)  # 色调
