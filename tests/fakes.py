"""Đối tượng giả dùng chung cho test (không dùng trong code chạy thật)."""

from __future__ import annotations

import chess
import numpy as np

from chess_assistant.board import square_center_image
from chess_assistant.schema import PieceBox, class_of

FOOT_LIFT = 0.10


def boxes_for_board(board: chess.Board, H: np.ndarray, conf: float = 0.9, height: float = 60.0) -> list[PieceBox]:
    """Khung quân giả cao `height`, rộng 40, đặt sao cho điểm chân trùng tâm ô."""
    boxes = []
    for sq, piece in board.piece_map().items():
        cx, cy = square_center_image(H, sq)
        y2 = cy + FOOT_LIFT * height
        boxes.append(PieceBox(class_of(piece), conf, (cx - 20.0, y2 - height, cx + 20.0, y2)))
    return boxes


class FakeDetector:
    """Thay YOLO: trả khung quân đúng vị trí của `board_fn()` trên ảnh có 4 góc thật `corners`."""

    def __init__(self, board_fn, corners):
        from chess_assistant.board import BoardView

        self.board_fn = board_fn
        self.H = BoardView.from_corners(corners).H

    def __call__(self, frame) -> list[PieceBox]:
        return boxes_for_board(self.board_fn(), self.H)
