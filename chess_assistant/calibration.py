"""Hiệu chỉnh: người dùng bấm chọn 4 góc bàn cờ theo thứ tự a1, h1, h8, a8."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "calibration.json"

CORNER_NAMES = ("a1", "h1", "h8", "a8")


@dataclass
class Calibration:
    camera_index: int
    width: int
    height: int
    corners: list[list[float]]  # 4 điểm [x, y] theo thứ tự a1, h1, h8, a8

    def save(self, path: Path = DEFAULT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "Calibration":
        if not path.exists():
            raise FileNotFoundError(f"Chưa có file hiệu chỉnh {path}. Chạy: python -m scripts.calibrate")
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def _draw(frame: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    img = frame.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(img, CORNER_NAMES[i], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if len(points) > 1:
        closed = len(points) == 4
        cv2.polylines(img, [np.int32(points)], closed, (0, 255, 0), 2)
    if len(points) < 4:
        msg = f"Bam goc ngoai cua o {CORNER_NAMES[len(points)]}  |  r: lam lai  Esc: huy"
    else:
        msg = "Enter: luu  |  r: lam lai  Esc: huy"
    cv2.rectangle(img, (0, 0), (img.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(img, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def pick_corners(frame: np.ndarray, window: str = "Hieu chinh ban co") -> list[list[float]] | None:
    """Mở cửa sổ cho người dùng bấm 4 góc. Trả về None nếu người dùng huỷ.

    Mẹo nhận góc: a1 là ô tối ở góc dưới-trái nhìn từ phía Trắng; h1 cùng hàng
    với a1, ở phía Vua trắng.
    """
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            cv2.imshow(window, _draw(frame, points))
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # Esc
                return None
            if key == ord("r"):
                points.clear()
            if key in (13, 10) and len(points) == 4:  # Enter
                return [[float(x), float(y)] for x, y in points]
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None
    finally:
        cv2.destroyWindow(window)
