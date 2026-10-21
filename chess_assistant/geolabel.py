"""Gán nhãn quân cờ theo hình học — không cần model.

Biết 4 góc bàn cờ (từ bộ dò) và thế cờ (FEN/PGN) thì biết mỗi quân đứng ở ô nào. Khung quân được
dựng từ đế quân (ô cờ thu nhỏ, chiếu lên ảnh) cộng chiều cao quân theo hướng lên của ảnh. Chiều cao
ước bằng hệ số theo loại quân × kích thước ô ở chỗ đó (vật đứng gần camera trông to hơn).

Dùng để thu dữ liệu cho bộ cờ / game mà model hiện tại chưa nhận ra.
"""

from __future__ import annotations

import io
from pathlib import Path

import chess
import chess.pgn
import cv2
import numpy as np

from chess_assistant.board import homography, square_rect
from chess_assistant.schema import class_of

# Chiều cao quân / cạnh ô, theo kiểu quân Staunton thông thường. Chỉnh chung bằng height_scale.
PIECE_HEIGHT = {
    chess.PAWN: 0.9,
    chess.ROOK: 1.0,
    chess.KNIGHT: 1.15,
    chess.BISHOP: 1.25,
    chess.QUEEN: 1.45,
    chess.KING: 1.6,
}
FOOTPRINT = 0.75  # đế quân chiếm khoảng 3/4 cạnh ô

Label = tuple[int, tuple[float, float, float, float]]


def _square_quad(square: chess.Square, shrink: float = 1.0) -> np.ndarray:
    x0, y0, x1, y1 = square_rect(square)
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 * shrink
    return np.float32([[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half]])


def geometric_boxes(
    corners: np.ndarray,
    board: chess.Board,
    image_size: tuple[int, int],
    height_scale: float = 1.0,
) -> list[Label]:
    """Khung (lớp, xyxy) cho mọi quân của `board`, theo thứ tự ô tăng dần, cắt trong ảnh (rộng, cao)."""
    H_inv = np.linalg.inv(homography(corners))
    width, img_h = image_size
    labels: list[Label] = []
    for sq, piece in sorted(board.piece_map().items()):
        full = cv2.perspectiveTransform(_square_quad(sq).reshape(-1, 1, 2), H_inv).reshape(4, 2)
        foot = cv2.perspectiveTransform(_square_quad(sq, FOOTPRINT).reshape(-1, 1, 2), H_inv).reshape(4, 2)
        scale = np.linalg.norm(np.roll(full, -1, axis=0) - full, axis=1).max()
        lift = height_scale * PIECE_HEIGHT[piece.piece_type] * scale
        pts = np.vstack([foot, foot - [0, lift]])
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        x1, x2 = float(np.clip(x1, 0, width)), float(np.clip(x2, 0, width))
        y1, y2 = float(np.clip(y1, 0, img_h)), float(np.clip(y2, 0, img_h))
        if x2 - x1 > 1 and y2 - y1 > 1:
            labels.append((class_of(piece), (x1, y1, x2, y2)))
    return labels


def positions_from_pgn(path: Path) -> list[chess.Board]:
    """Mọi thế cờ của ván trong file PGN, từ thế đầu đến thế cuối."""
    game = chess.pgn.read_game(io.StringIO(Path(path).read_text(encoding="utf-8")))
    if game is None:
        raise ValueError(f"Không đọc được PGN {path}")
    if game.errors:  # python-chess bỏ qua nước sai và dừng ở đó mà không báo
        raise ValueError("; ".join(str(e) for e in game.errors))
    board = game.board()
    boards = [board.copy()]
    for move in game.mainline_moves():
        board.push(move)
        boards.append(board.copy())
    return boards
