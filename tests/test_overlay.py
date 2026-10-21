import chess
import numpy as np

from chess_assistant.advisor import Advice
from chess_assistant.board import BoardView, square_center_image
from chess_assistant.overlay import OverlayState, draw_arrow, mini_board_layout, render
from tests.synthetic import POSES


def test_arrow_pixels():
    img = np.zeros((720, 1280, 3), np.uint8)
    view = BoardView.from_corners(POSES["white_side"])
    draw_arrow(img, view.H, chess.Move.from_uci("e2e4"))
    (x1, y1), (x2, y2) = square_center_image(view.H, chess.E2), square_center_image(view.H, chess.E4)
    mid = img[(y1 + y2) // 2, (x1 + x2) // 2]
    assert tuple(int(v) for v in mid) == (0, 220, 0)


def test_render_without_view():
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = render(frame, None, OverlayState("Dang tim ban co...", (0, 200, 255)))
    assert out.shape == frame.shape
    assert not np.shares_memory(out, frame)


def test_render_full_state():
    frame = np.full((720, 1280, 3), 90, np.uint8)
    view = BoardView.from_corners(POSES["white_side"])
    board = chess.Board()
    advice = Advice(board.fen(), [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")], "+0.30", 30, blunder_cp=250)
    state = OverlayState("Dang bam", (0, 255, 0), board=board, advice=advice,
                         last_move=None, message="test")
    out = render(frame, view, state)
    assert out.shape == frame.shape
    assert (out != frame).any()


def test_mini_board_layout():
    assert mini_board_layout(chess.WHITE)[7][0] == chess.A1
    assert mini_board_layout(chess.BLACK)[7][0] == chess.H8
    assert mini_board_layout(chess.BLACK)[0][0] == chess.H1


def test_alt_moves_drawn_in_blue():
    from chess_assistant.overlay import ARROW_BLACK

    frame = np.zeros((720, 1280, 3), np.uint8)
    view = BoardView.from_corners(POSES["white_side"])
    advice = Advice("x", [], "", 0, alt_moves=[chess.Move.from_uci("e7e5")])
    out = render(frame, view, OverlayState("", (0, 255, 0), advice=advice))
    (x1, y1), (x2, y2) = square_center_image(view.H, chess.E7), square_center_image(view.H, chess.E5)
    assert tuple(int(v) for v in out[(y1 + y2) // 2, (x1 + x2) // 2]) == ARROW_BLACK
