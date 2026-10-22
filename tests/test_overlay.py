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


# ---------------------------------------------------------------------------
# Giao diện tô ô (thay mũi tên) và chỉ vẽ viền bàn cờ
# ---------------------------------------------------------------------------

def _center(view, sq):
    return square_center_image(view.H, sq)


def test_suggestion_highlights_from_and_to_squares():
    from chess_assistant.overlay import FROM_COLOR, TO_COLOR

    frame = np.full((720, 1280, 3), 90, np.uint8)
    view = BoardView.from_corners(POSES["white_side"])
    advice = Advice("x", [chess.Move.from_uci("e2e4")], "", 0)
    out = render(frame, view, OverlayState("", (0, 255, 0), advice=advice))
    for sq, color in ((chess.E2, FROM_COLOR), (chess.E4, TO_COLOR)):
        x, y = _center(view, sq)
        px = out[y, x].astype(int)
        # Ô được tô: điểm ảnh lệch hẳn về màu tô so với nền xám.
        assert np.abs(px - np.array(color)).sum() < np.abs(np.array([90, 90, 90]) - np.array(color)).sum() * 0.7


def test_black_suggestion_uses_blue_tones():
    from chess_assistant.overlay import ALT_TO_COLOR

    frame = np.full((720, 1280, 3), 90, np.uint8)
    view = BoardView.from_corners(POSES["white_side"])
    advice = Advice("x", [], "", 0, alt_moves=[chess.Move.from_uci("e7e5")])
    out = render(frame, view, OverlayState("", (0, 255, 0), advice=advice))
    x, y = _center(view, chess.E5)
    b, g, r = (int(v) for v in out[y, x])
    assert b > r + 40  # tông xanh dương


def test_no_grid_lines_only_board_outline():
    frame = np.full((720, 1280, 3), 90, np.uint8)
    view = BoardView.from_corners(POSES["white_side"])
    out = render(frame, view, OverlayState("", (0, 255, 0)))
    # Điểm trên đường kẻ giữa c4 và d4 (bên trong bàn): không còn vẽ lưới.
    (x1, y1), (x2, y2) = _center(view, chess.C4), _center(view, chess.D4)
    assert tuple(out[(y1 + y2) // 2, (x1 + x2) // 2]) == (90, 90, 90)
    # Viền ngoài bàn cờ vẫn được vẽ: điểm giữa cạnh a1-h1.
    a1, h1 = np.float32(POSES["white_side"][0]), np.float32(POSES["white_side"][1])
    mx, my = ((a1 + h1) / 2).astype(int)
    assert tuple(out[my, mx]) != (90, 90, 90)
