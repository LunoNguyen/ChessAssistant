"""Mở camera (webcam, DroidCam/Iriun, hoặc URL luồng video) và tự kết nối lại khi mất."""

from __future__ import annotations

import logging
import sys
import time
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)


def open_camera(source: int | str = 0, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """`source` là chỉ số camera hoặc URL luồng (ví dụ http://<ip>:4747/video của DroidCam)."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    if isinstance(source, int):
        # Trên Windows, DirectShow mở camera nhanh hơn nhiều so với backend mặc định (MSMF).
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(source, backend)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if isinstance(source, str):
            raise RuntimeError(
                f"Không mở được luồng {source}. Kiểm tra: điện thoại và laptop cùng WiFi, app DroidCam đang mở, "
                "và không có thiết bị nào khác (DroidCam Client, tab trình duyệt) đang xem — DroidCam chỉ nhận 1 kết nối."
            )
        raise RuntimeError(f"Không mở được camera {source}. Thử chỉ số khác với --camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def actual_size(cap: cv2.VideoCapture) -> tuple[int, int]:
    """Độ phân giải camera thật sự trả về (có thể khác giá trị đã yêu cầu)."""
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


class ReconnectingCamera:
    """Đọc frame; khi camera rớt (WiFi/USB của điện thoại) thì tự mở lại sau mỗi `retry_every` giây."""

    def __init__(self, factory: Callable[[], cv2.VideoCapture], retry_every: float = 1.0, clock=time.monotonic):
        self._factory = factory
        self._retry_every = retry_every
        self._clock = clock
        self._cap = None
        self._last_attempt = clock()
        self._open()

    @property
    def connected(self) -> bool:
        return self._cap is not None

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            if self._clock() - self._last_attempt < self._retry_every:
                return None
            self._open()
            if self._cap is None:
                return None
        ok, frame = self._cap.read()
        if ok and frame is not None:
            return frame
        log.warning("Mất kết nối camera, sẽ thử lại")
        self._drop()
        return None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _open(self) -> None:
        self._last_attempt = self._clock()
        try:
            self._cap = self._factory()
        except Exception as exc:
            log.warning("Không mở được camera: %s", exc)
            self._cap = None

    def _drop(self) -> None:
        self.release()
        self._last_attempt = self._clock()
