"""Kiểu dữ liệu dùng chung giữa nhận diện quân, đọc bàn cờ và theo dõi ván.

Một "reading" là mảng (64, 13): mỗi hàng là một ô theo thứ tự chess.SQUARES
(a1=0 … h8=63), mỗi cột là xác suất một lớp: 12 quân theo PIECE_CLASSES rồi ô trống.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

PIECE_CLASSES = "PNBRQKpnbrqk"
EMPTY = 12
NUM_CLASSES = 13
EPS = 0.01


@dataclass(frozen=True)
class PieceBox:
    cls: int                                  # chỉ số trong PIECE_CLASSES
    conf: float
    xyxy: tuple[float, float, float, float]   # toạ độ ảnh camera


def class_of(piece: chess.Piece | None) -> int:
    return EMPTY if piece is None else PIECE_CLASSES.index(piece.symbol())


def reading_from_board(board: chess.Board, confidence: float = 0.95) -> np.ndarray:
    """Reading "lý tưởng" của một thế cờ: `confidence` ở lớp đúng, phần còn lại chia đều."""
    reading = np.full((64, NUM_CLASSES), (1 - confidence) / (NUM_CLASSES - 1))
    for sq in chess.SQUARES:
        reading[sq, class_of(board.piece_at(sq))] = confidence
    return reading


def argmax_labels(reading: np.ndarray) -> list[int]:
    return [int(i) for i in np.argmax(reading, axis=1)]
