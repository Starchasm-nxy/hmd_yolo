"""
File monitoring and detection output writing.

FileMonitor: polls data.txt for command changes, signals via threading.Event.
OutputWriter: writes detection/prediction/fallback results to gaozhi.txt,
managing last_value, timeout, and the b_zero one-time log guard.
"""

import threading
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FileMonitor:
    """Polls a text file for command changes.

    Uses level-triggered signalling: command_active.set() for '1m'/'2m',
    command_active.clear() for '0'. Runs a daemon polling thread internally.
    """

    def __init__(self, file_path: str, poll_interval: float = 0.05) -> None:
        self.file_path = file_path
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.command_active = threading.Event()
        self._lock = threading.Lock()
        self.current_command: Optional[str] = None

    def start(self) -> None:
        """Start the background polling thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling thread to stop and join."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def read_command(self) -> Optional[str]:
        """Thread-safe read of the current command string."""
        with self._lock:
            return self.current_command

    def _run(self) -> None:
        last_content: Optional[str] = None
        while not self._stop_event.is_set():
            try:
                with open(self.file_path, 'r') as f:
                    content = f.read().strip()
            except Exception:
                time.sleep(self.poll_interval)
                continue

            if content != last_content:
                with self._lock:
                    self.current_command = content
                if content in ('1m', '2m'):
                    logger.info(f"检测到{content}模式")
                    self.command_active.set()
                elif content == '0':
                    logger.info("检测到0模式，关闭摄像头")
                    self.command_active.clear()
                else:
                    logger.info("未检测到有效内容，等待...")
                last_content = content

            time.sleep(self.poll_interval)


class OutputWriter:
    """Writes detection results to the output file.

    Manages last_value (for fallback), last_detection_time (for timeout),
    and b_zero (one-time 2m entry log). All state mutations are protected
    by a threading.Lock.

    Key behaviours preserved from original:
      - '1m' mode: no detection writes '0'
      - '2m' mode: no detection falls back to last_value (writes "2 <last>")
      - In '2m' mode, writing '0' does NOT overwrite last_value
      - b_zero controls the one-time log message on first 2m entry
      - History timeout clears last_value after configured seconds without detection
    """

    def __init__(
        self,
        file_path: str,
        history_clear_enabled: bool,
        history_clear_timeout: float,
    ) -> None:
        self.file_path = file_path
        self.history_clear_enabled = history_clear_enabled
        self.history_clear_timeout = history_clear_timeout
        self._lock = threading.Lock()
        self.last_value: Optional[str] = None
        self.last_detection_time: float = 0.0
        self.b_zero: bool = True

    # ------------------------------------------------------------------
    # Public write methods
    # ------------------------------------------------------------------

    def write_detection(self, mode: str, ux: int, uy: int) -> str:
        """Write a real detection: '1 ux uy'. Updates last_value and timestamp."""
        content = f"1 {ux} {uy}"
        with self._lock:
            self._ensure_2m_log(mode)
            self._write_file(content)
            self.last_value = f"{ux} {uy}"
            self.last_detection_time = time.time()
        return content

    def write_prediction(self, mode: str, ux: int, uy: int) -> str:
        """Write a position prediction: '2 ux uy'.

        Updates last_value (so subsequent fallbacks use the predicted position).
        Does NOT update last_detection_time (prediction is not a real detection).
        """
        content = f"2 {ux} {uy}"
        with self._lock:
            self._ensure_2m_log(mode)
            self._write_file(content)
            self.last_value = f"{ux} {uy}"
        return content

    def write_fallback(self, mode: str) -> str:
        """Write fallback output when the tracker reports LOST (no target).

        Steps (under lock):
          1. Check history timeout — clear last_value if expired.
          2. If 2m and b_zero: print one-time log, clear b_zero.
          3. In 2m mode with last_value: write "2 <last>" (or "0" if last_value is "0").
          4. Otherwise: write "0".
          5. In 2m mode, writing "0" does NOT overwrite last_value.
        """
        with self._lock:
            self._check_timeout()
            self._ensure_2m_log(mode)

            if mode == '2m' and self.last_value is not None:
                if self.last_value == '0':
                    content = "0"
                else:
                    content = "2 " + self.last_value
            else:
                content = "0"

            self._write_file(content)

            # In 2m mode, writing '0' preserves existing last_value
            if mode == '2m' and content.strip() == '0':
                return content

            # Update last_value from content
            parts = content.strip().split()
            if parts[0] == '0':
                self.last_value = '0'
            else:
                self.last_value = ' '.join(parts[1:])

        return content

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self, cold_start: bool = False) -> None:
        """Reset writer state on mode switch.

        cold_start=True (transition from non-1m/2m into 1m or 2m):
          Clears last_value, resets timestamp, sets b_zero=True, writes '0'.
        cold_start=False (1m <-> 2m switch):
          Preserves last_value and last_detection_time. Does NOT reset b_zero.
        """
        with self._lock:
            if cold_start:
                self.last_value = None
                self.last_detection_time = time.time()
                self.b_zero = True
                self._write_file("0")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_timeout(self) -> None:
        """Clear last_value if timeout has elapsed. Caller must hold self._lock."""
        if (
            self.history_clear_enabled
            and self.last_value is not None
            and time.time() - self.last_detection_time > self.history_clear_timeout
        ):
            self.last_value = None
            logger.info(f"超过{self.history_clear_timeout}秒未检测到目标，清空历史坐标")

    def _ensure_2m_log(self, mode: str) -> None:
        """Print the one-time 2m entry log. Caller must hold self._lock."""
        if mode == '2m' and self.b_zero:
            logger.info("我方即将进入对桶程序※※")
            self.b_zero = False

    def _write_file(self, content: str) -> None:
        """Write content string directly to self.file_path."""
        with open(self.file_path, 'w') as f:
            f.write(content)
