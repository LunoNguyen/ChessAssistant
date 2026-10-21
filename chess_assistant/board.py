"""Hình học bàn cờ: nắn phẳng ảnh camera và cắt 64 ô.

Quy ước "canvas" (ảnh đã nắn phẳng):
- Luôn vẽ theo hướng sơ đồ chuẩn: bên Trắng ở dưới, a1 ở góc dưới-trái.
- Bàn cờ BOARD_PX x BOARD_PX nằm giữa, có viền MARGIN_PX mỗi phía để cắt được
  phần đầu quân cờ cao bị "đổ" ra ngoài bàn.

Góc bàn cờ trong ảnh camera luôn được truyền theo thứ tự a1, h1, h8, a8
(góc ngoài của 4 ô đó). Nhờ vậy hướng bàn cờ được xác định ngay từ lúc
hiệu chỉnh, dù camera đặt ở phía Trắng, phía Đen hay bên cạnh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import chess
import cv2
import numpy as np

SQUARE_PX = 100
BOARD_PX = 8 * SQUARE_PX
MARGIN_PX = SQUARE_PX
CANVAS_PX = BOARD_PX + 2 * MARGIN_PX

Lean = Literal["up", "down", "left", "right"]

# Số lần xoay 90° (np.rot90) để đưa quân cờ về tư thế "đứng thẳng" (đầu quân hướng lên).
_ROTATIONS: dict[str, int] = {"up": 0, "down": 2, "left": -1, "right": 1}


def _canvas_corners() -> np.ndarray:
    m, s = MARGIN_PX, BOARD_PX
    # a1, h1, h8, a8
    return np.array([[m, m + s], [m + s, m + s], [m + s, m], [m, m]], dtype=np.float32)


def homography(corners: np.ndarray | list) -> np.ndarray:
    """Ma trận biến đổi ảnh camera -> canvas, từ 4 góc theo thứ tự a1, h1, h8, a8."""
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(src, _canvas_corners())


def warp(frame: np.ndarray, H: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(frame, H, (CANVAS_PX, CANVAS_PX))


def _apply(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def lean_direction(H: np.ndarray, corners: np.ndarray | list) -> Lean:
    """Hướng mà quân cờ "đổ" về trong canvas.

    Vật đứng thẳng trên bàn trông như vươn lên phía trên ảnh camera. Lấy một
    đoạn hướng lên tại tâm bàn cờ trong ảnh camera, chiếu sang canvas, rồi
    làm tròn về hướng trục gần nhất.
    """
    center = np.asarray(corners, dtype=np.float32).reshape(4, 2).mean(axis=0)
    p, q = _apply(H, np.array([center, center - [0, 10]]))
    dx, dy = q - p
    if abs(dy) >= abs(dx):
        return "up" if dy < 0 else "down"
    return "left" if dx < 0 else "right"


def square_rect(square: chess.Square) -> tuple[int, int, int, int]:
    """Hình chữ nhật (x0, y0, x1, y1) của một ô trong canvas."""
    file, rank = chess.square_file(square), chess.square_rank(square)
    x0 = MARGIN_PX + file * SQUARE_PX
    y0 = MARGIN_PX + (7 - rank) * SQUARE_PX
    return x0, y0, x0 + SQUARE_PX, y0 + SQUARE_PX


def square_at_canvas(x: float, y: float) -> chess.Square | None:
    """Ô chứa điểm (x, y) của canvas, None nếu nằm ngoài bàn cờ."""
    file = int((x - MARGIN_PX) // SQUARE_PX)
    row = int((y - MARGIN_PX) // SQUARE_PX)
    if not (0 <= file < 8 and 0 <= row < 8):
        return None
    return chess.square(file, 7 - row)


def user_color_from_corners(corners: np.ndarray) -> chess.Color:
    """Bên gần camera (đáy ảnh) là bên người dùng: so độ cao trung bình của cạnh hàng 1 và hàng 8."""
    c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return chess.WHITE if c[:2, 1].mean() > c[2:, 1].mean() else chess.BLACK


def crop_square(canvas: np.ndarray, square: chess.Square, lean: Lean = "up", pad: float = 0.75) -> np.ndarray:
    """Cắt một ô, mở rộng thêm `pad` lần cạnh ô theo hướng quân cờ đổ,
    rồi xoay sao cho đầu quân hướng lên. Kết quả luôn có kích thước
    (SQUARE_PX * (1 + pad)) x SQUARE_PX (cao x rộng)."""
    if not 0 <= pad * SQUARE_PX <= MARGIN_PX:
        raise ValueError(f"pad phải nằm trong [0, {MARGIN_PX / SQUARE_PX}]")
    x0, y0, x1, y1 = square_rect(square)
    extra = round(pad * SQUARE_PX)
    if lean == "up":
        y0 -= extra
    elif lean == "down":
        y1 += extra
    elif lean == "left":
        x0 -= extra
    else:
        x1 += extra
    return np.ascontiguousarray(np.rot90(canvas[y0:y1, x0:x1], _ROTATIONS[lean]))


def crop_squares(canvas: np.ndarray, lean: Lean = "up", pad: float = 0.75) -> list[np.ndarray]:
    """64 ảnh ô theo thứ tự chess.SQUARES (a1, b1, ..., h8)."""
    return [crop_square(canvas, sq, lean, pad) for sq in chess.SQUARES]


def square_center_image(H: np.ndarray, square: chess.Square) -> tuple[int, int]:
    """Tâm của một ô, chiếu ngược về toạ độ ảnh camera (dùng để vẽ mũi tên)."""
    x0, y0, x1, y1 = square_rect(square)
    (x, y), = _apply(np.linalg.inv(H), np.array([[(x0 + x1) / 2, (y0 + y1) / 2]]))
    return round(x), round(y)


def grid_lines_image(H: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """18 đoạn thẳng của lưới 8x8, chiếu ngược về toạ độ ảnh camera."""
    H_inv = np.linalg.inv(H)
    lo, hi = MARGIN_PX, MARGIN_PX + BOARD_PX
    segments = []
    for i in range(9):
        t = MARGIN_PX + i * SQUARE_PX
        segments.append(((t, lo), (t, hi)))
        segments.append(((lo, t), (hi, t)))
    pts = _apply(H_inv, np.array([p for seg in segments for p in seg], dtype=np.float32))
    pts = np.rint(pts).astype(int)
    return [(tuple(pts[2 * i]), tuple(pts[2 * i + 1])) for i in range(len(segments))]


@dataclass
class BoardView:
    """Gói H + hướng đổ của quân, tính một lần sau khi hiệu chỉnh."""

    corners: np.ndarray
    H: np.ndarray
    lean: Lean

    @classmethod
    def from_corners(cls, corners: np.ndarray | list) -> "BoardView":
        corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        H = homography(corners)
        return cls(corners=corners, H=H, lean=lean_direction(H, corners))

    def warp(self, frame: np.ndarray) -> np.ndarray:
        return warp(frame, self.H)

    def crops(self, frame: np.ndarray, pad: float = 0.75) -> list[np.ndarray]:
        return crop_squares(self.warp(frame), self.lean, pad)
