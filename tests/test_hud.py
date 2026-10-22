import chess
import numpy as np

from chess_assistant.advisor import Advice
from chess_assistant.hud import HUD_SIZE, hud_square_center, render_hud
from chess_assistant.overlay import ALT_TO_COLOR, FROM_COLOR, TO_COLOR, OverlayState


def state_with(move: str | None = None, user_color=chess.WHITE, alt: str | None = None, board=None):
    board = board or chess.Board()
    advice = None
    if move or alt:
        advice = Advice(board.fen(), [chess.Move.from_uci(move)] if move else [], "+0.30", 30,
                        alt_moves=[chess.Move.from_uci(alt)] if alt else [])
    return OverlayState("Dang bam van co", (0, 255, 0), board=board, user_color=user_color, advice=advice)


def closer_to(px, color, than):
    px, color, than = (np.array(v, int) for v in (px, color, than))
    return np.abs(px - color).sum() < np.abs(than - color).sum()


def test_size():
    img = render_hud(state_with("e2e4"))
    assert img.shape == (HUD_SIZE, HUD_SIZE, 3) and img.dtype == np.uint8


def test_from_and_to_squares_highlighted():
    img = render_hud(state_with("e2e4"))
    plain = render_hud(state_with())
    for sq, color in ((chess.E2, FROM_COLOR), (chess.E4, TO_COLOR)):
        # Lấy điểm gần góc ô (tránh chữ quân ở giữa ô).
        x, y = hud_square_center(sq, chess.WHITE)
        px = (x - 8, y - 8)
        assert closer_to(img[px[1], px[0]], color, plain[px[1], px[0]])


def test_black_suggestion_blue_when_turn_unknown():
    img = render_hud(state_with(alt="e7e5"))
    x, y = hud_square_center(chess.E5, chess.WHITE)
    b, g, r = (int(v) for v in img[y - 8, x - 8])
    assert b > r + 40
    assert closer_to(img[y - 8, x - 8], ALT_TO_COLOR, (0, 0, 0)) or b > 150


def test_user_side_at_bottom():
    # Người dùng cầm Trắng: e2 ở nửa dưới; cầm Đen: e2 ở nửa trên.
    assert hud_square_center(chess.E2, chess.WHITE)[1] > HUD_SIZE / 2
    assert hud_square_center(chess.E2, chess.BLACK)[1] < HUD_SIZE / 2
    assert hud_square_center(chess.A1, chess.WHITE)[0] < hud_square_center(chess.H1, chess.WHITE)[0]
    assert hud_square_center(chess.A1, chess.BLACK)[0] > hud_square_center(chess.H1, chess.BLACK)[0]


def test_mirror_for_reflective_hud():
    s = state_with("e2e4")
    assert np.array_equal(render_hud(s, mirror=True), render_hud(s)[:, ::-1])


def test_no_board_shows_status_only():
    s = OverlayState("Dang tim ban co...", (0, 200, 255))
    img = render_hud(s)
    assert img.shape == (HUD_SIZE, HUD_SIZE, 3)
    assert img.std() > 1  # có chữ, không phải ảnh trơn
