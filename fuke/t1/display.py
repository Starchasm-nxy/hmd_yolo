"""OpenCV display window management."""

import cv2
from typing import Optional


_window_states: dict = {}


def create_window(name: str, width: int, height: int) -> None:
    """Create a named OpenCV window. Idempotent — no-op if already exists."""
    if name in _window_states:
        return
    cv2.namedWindow(
        name,
        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED,
    )
    cv2.resizeWindow(name, width, height)
    _window_states[name] = True


def show_frame(name: str, frame) -> bool:
    """Display a frame in the named window. Returns True if 'q' was pressed."""
    cv2.imshow(name, frame)
    return (cv2.waitKey(1) & 0xFF) == ord('q')


def destroy_window(name: str) -> None:
    """Destroy the named window."""
    _window_states.pop(name, None)
    cv2.destroyWindow(name)
