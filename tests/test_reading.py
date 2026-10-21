import chess
import numpy as np
import pytest

from chess_assistant.board import (
    MARGIN_PX,
    SQUARE_PX,
    BoardView,
    square_at_canvas,
    square_center_image,
    user_color_from_corners,
)
from chess_assistant.reading import read_board
from chess_assistant.schema import PIECE_CLASSES, PieceBox, argmax_labels, reading_from_board
from tests.fakes import boxes_for_board
from tests.synthetic import POSES


def H_of(pose: str) -> np.ndarray:
    return BoardView.from_corners(POSES[pose]).H


@pytest.mark.parametrize("pose", POSES)
def test_board_round_trip(pose):
    H = H_of(pose)
    board = chess.Board()
    assert argmax_labels(read_board(boxes_for_board(board, H), H)) == argmax_labels(reading_from_board(board))


def box_at(H, sq, cls, conf, height=60.0):
    cx, cy = square_center_image(H, sq)
    y2 = cy + 0.10 * height
    return PieceBox(PIECE_CLASSES.index(cls), conf, (cx - 20.0, y2 - height, cx + 20.0, y2))


def test_duplicate_boxes_keep_highest_conf():
    H = H_of("white_side")
    reading = read_board([box_at(H, chess.E4, "P", 0.6), box_at(H, chess.E4, "B", 0.9)], H)
    assert argmax_labels(reading)[chess.E4] == PIECE_CLASSES.index("B")


def test_off_board_box_ignored():
    H = H_of("white_side")
    off = PieceBox(0, 0.9, (0.0, -50.0, 10.0, 5.0))  # chân ở khoảng (5, 0) ảnh
    empty = read_board([], H)
    assert np.allclose(read_board([off], H), empty)


def test_rows_valid():
    H = H_of("rotated")
    reading = read_board(boxes_for_board(chess.Board(), H), H)
    assert np.allclose(reading.sum(1), 1)
    assert reading.min() > 0


def test_square_at_canvas():
    c = MARGIN_PX + SQUARE_PX // 2
    e4_x = MARGIN_PX + 4 * SQUARE_PX + SQUARE_PX // 2
    e4_y = MARGIN_PX + (7 - 3) * SQUARE_PX + SQUARE_PX // 2
    assert square_at_canvas(e4_x, e4_y) == chess.E4
    assert square_at_canvas(c, MARGIN_PX + 7 * SQUARE_PX + SQUARE_PX // 2) == chess.A1
    assert square_at_canvas(1, 1) is None


def test_user_color_from_corners():
    assert user_color_from_corners(np.float32(POSES["white_side"])) == chess.WHITE
    assert user_color_from_corners(np.float32(POSES["black_side"])) == chess.BLACK
