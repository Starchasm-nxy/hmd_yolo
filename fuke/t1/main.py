"""Entry point for the RealSense D435 + YOLOv8 detection and tracking program.

Initialises all modules, starts threads, and runs the display loop.
"""

import sys
import time
import queue
import threading
import logging
from typing import Optional

import cv2

from config_schema import Config, load_config
from utils import clear_files
from camera import CameraSource
from detector import YOLODetector
from tracker import LockTracker
from communication import FileMonitor, OutputWriter
from pipeline import DetectionPipeline
from display import create_window, show_frame, destroy_window

logger = logging.getLogger(__name__)

# Module-level display queue (shared between pipeline and main thread)
display_queue: queue.Queue = queue.Queue(maxsize=3)


def pipeline_manager_loop(
    main_stop: threading.Event,
    file_monitor: FileMonitor,
    pipeline: DetectionPipeline,
    window_name: str,
) -> None:
    """Manage pipeline lifecycle based on FileMonitor.command_active signal.

    Waits for the start command, spawns a pipeline thread, and stops it
    when the command is cleared. Preserves the original detection_manager_thread
    pattern.
    """
    pipeline_stop: Optional[threading.Event] = None
    pipeline_thread: Optional[threading.Thread] = None

    try:
        while not main_stop.is_set():
            # Wait for start command
            file_monitor.command_active.wait()
            if main_stop.is_set():
                break

            # Start pipeline if not already running
            if pipeline_thread is None or not pipeline_thread.is_alive():
                pipeline_stop = threading.Event()
                pipeline_thread = threading.Thread(
                    target=pipeline.run,
                    args=(pipeline_stop, display_queue),
                    daemon=True,
                )
                pipeline_thread.start()
                logger.info("检测管道已启动")

            # Wait for stop command
            while file_monitor.command_active.is_set() and not main_stop.is_set():
                time.sleep(0.1)

            # Stop pipeline
            if pipeline_stop is not None:
                pipeline_stop.set()
            if pipeline_thread is not None and pipeline_thread.is_alive():
                pipeline_thread.join(timeout=2)
            cv2.destroyAllWindows()
            pipeline_thread = None
            logger.info("检测管道已停止")

    finally:
        if pipeline_stop is not None:
            pipeline_stop.set()
        if pipeline_thread is not None and pipeline_thread.is_alive():
            pipeline_thread.join(timeout=2)
        cv2.destroyAllWindows()


def main() -> None:
    """Initialise and run the detection program."""
    # 1. Load configuration
    config = load_config("config.yaml")

    # 2. Setup logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
    )

    logger.info("Yolo26n目标检测-程序启动")
    logger.info("开始Yolo26n模型加载")

    # 3. Clear communication files
    clear_files(config.files.clear_on_start)
    logger.info("通讯txt文件已建立并清空")

    # 4. Initialise all modules
    camera = CameraSource(
        config.camera.width, config.camera.height, config.camera.fps,
    )
    detector = YOLODetector(
        model_path=config.model.path,
        device=config.model.device,
        iou=config.model.iou,
        locked_conf=config.inference.locked_conf,
        unlocked_1m_conf=config.inference.unlocked_1m_conf,
        unlocked_1m_imgsz=config.inference.unlocked_1m_imgsz,
        unlocked_2m_conf=config.inference.unlocked_2m_conf,
        unlocked_2m_imgsz=config.inference.unlocked_2m_imgsz,
        imgsz_step=config.model.imgsz_step,
        max_area=config.inference.max_area,
        frame_w=config.camera.width,
        frame_h=config.camera.height,
    )
    tracker = LockTracker(
        max_hit=config.lock_tracker.max_hit,
        max_miss=config.lock_tracker.max_miss,
        search_ratio=config.lock_tracker.search_ratio,
        min_search_radius=config.lock_tracker.min_search_radius,
        max_search_radius=config.lock_tracker.max_search_radius,
    )
    writer = OutputWriter(
        file_path=config.files.output,
        history_clear_enabled=config.history.clear_enabled,
        history_clear_timeout=config.history.clear_timeout,
    )
    file_monitor = FileMonitor(config.files.data)

    pipeline = DetectionPipeline(
        config, camera, detector, tracker, writer, file_monitor,
    )

    # 5. Check D435 connection
    if not CameraSource.check_device():
        for _ in range(5):
            logger.warning("未检测到D435相机，请检查连接！")
    else:
        logger.info("D435相机已连接。")

    # 6. Start file monitor
    file_monitor.start()

    # 7. Start pipeline manager thread
    main_stop = threading.Event()
    win_name = config.display.window_name

    manager_thread = threading.Thread(
        target=pipeline_manager_loop,
        args=(main_stop, file_monitor, pipeline, win_name),
        daemon=True,
    )
    manager_thread.start()

    logger.info("完成Yolo26n模型加载")
    logger.info("系统初始化完成，等待指令...")

    # 8. Main display loop
    window_created = False

    try:
        while True:
            try:
                frame, display_win_name = display_queue.get(timeout=0.05)
            except queue.Empty:
                if main_stop.is_set():
                    break
                continue

            if not window_created:
                create_window(
                    display_win_name,
                    config.camera.width,
                    config.camera.height,
                )
                window_created = True

            if show_frame(display_win_name, frame):
                main_stop.set()
                break

    except KeyboardInterrupt:
        logger.info("键盘中断，正在退出...")
    finally:
        main_stop.set()
        file_monitor.stop()
        manager_thread.join(timeout=3)
        destroy_window(win_name)
        logger.info("程序已退出")


if __name__ == '__main__':
    main()
