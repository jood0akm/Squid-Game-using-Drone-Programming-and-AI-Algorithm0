"""Low-latency webcam capture utilities."""

import platform
import threading
import time

import cv2

from .config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CAMERA_FOURCC


class LatestFrameCapture:
    """Keep only the newest webcam frame to avoid latency buildup."""

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._running = cap.isOpened()
        self._thread = None

        if self._running:
            ok, frame = cap.read()
            if ok:
                self._frame = frame
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def _reader(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame

    def isOpened(self):
        return self._cap.isOpened()

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame

    def release(self):
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.3)
        self._cap.release()


def _open_raw(index: int):
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    else:
        cap = cv2.VideoCapture(index)
    return cap


def open_camera(index: int = CAMERA_INDEX):
    cap = _open_raw(index)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return LatestFrameCapture(cap)
