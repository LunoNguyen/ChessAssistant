"""Tự gán nhãn: dùng khung quân của model hiện có, nhưng lấy tên quân từ thế cờ đã biết.

Model v0 chỉ cần tìm đúng *vị trí* quân; tên quân (thường là chỗ model mới hay sai)
lấy từ FEN mà người dùng đã bày, nên nhãn luôn đúng.
"""

from __future__ import annotations

import chess
import cv2
import numpy as np

from chess_assistant.board import square_at_canvas
from chess_assistant.reading import foot_point
from chess_assistant.schema import PieceBox, class_of

Label = tuple[int, tuple[float, float, float, float]]


def autolabel(boxes: list[PieceBox], H: np.ndarray, board: chess.Board, foot_lift: float = 0.10) -> list[Label] | None:
    """Mỗi ô có quân nhận đúng một khung (conf cao nhất); thiếu khung ở ô nào -> None (bỏ frame)."""
    best: dict[chess.Square, PieceBox] = {}
    if boxes:
        feet = np.float32([foot_point(b, foot_lift) for b in boxes]).reshape(-1, 1, 2)
        canvas = cv2.perspectiveTransform(feet, H).reshape(-1, 2)
        for box, (x, y) in zip(boxes, canvas):
            sq = square_at_canvas(float(x), float(y))
            if sq is not None and board.piece_at(sq) and (sq not in best or box.conf > best[sq].conf):
                best[sq] = box
    occupied = board.piece_map()
    if set(best) != set(occupied):
        return None
    # Thế khởi đầu xoay 180° vẫn khớp vị trí quân: dùng lớp model đoán để chắc tracker không
    # đoán ngược hướng (nếu ngược, mọi nhãn sẽ bị đảo màu).
    rotated = board.transform(chess.flip_vertical).transform(chess.flip_horizontal)
    agree = sum(best[sq].cls == class_of(board.piece_at(sq)) for sq in best)
    agree_rotated = sum(best[sq].cls == class_of(rotated.piece_at(sq)) for sq in best)
    if agree_rotated > agree:
        return None
    return [(class_of(occupied[sq]), best[sq].xyxy) for sq in sorted(occupied)]


def to_yolo_lines(labels: list[Label], width: int, height: int) -> list[str]:
    lines = []
    for cls, (x1, y1, x2, y2) in labels:
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines
