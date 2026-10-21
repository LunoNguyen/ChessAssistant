import chess

from chess_assistant.schema import reading_from_board
from chess_assistant.stability import StabilityFilter

START = reading_from_board(chess.Board())
AFTER_E4 = reading_from_board(chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"))


def board_without(n: int) -> chess.Board:
    """Thế khởi đầu bớt `n` Tốt đen."""
    board = chess.Board()
    for sq in list(chess.SquareSet(chess.BB_RANK_7))[:n]:
        board.remove_piece_at(sq)
    return board


def push_many(f, reading, n, locked=True):
    return [f.push(reading, locked) for _ in range(n)]


def test_emits_once_after_stable_frames():
    f = StabilityFilter(stable_frames=8)
    assert all(r is None for r in push_many(f, START, 7))
    assert f.push(START, True) is not None
    assert all(r is None for r in push_many(f, START, 12))


def test_change_restarts_window():
    f = StabilityFilter(stable_frames=8)
    push_many(f, START, 5)
    f.push(AFTER_E4, True)
    assert all(r is None for r in push_many(f, START, 7))
    assert f.push(START, True) is not None


def test_unlocked_resets():
    f = StabilityFilter(stable_frames=8)
    push_many(f, START, 7)
    assert f.push(START, False) is None
    assert all(r is None for r in push_many(f, START, 7))


def test_none_reading_resets():
    f = StabilityFilter(stable_frames=8)
    push_many(f, START, 7)
    assert f.push(None, True) is None
    assert all(r is None for r in push_many(f, START, 7))


def test_capture_drop_of_one_passes():
    f = StabilityFilter(stable_frames=8, max_drop=4)
    push_many(f, START, 8)
    assert push_many(f, reading_from_board(board_without(1)), 8)[-1] is not None


def test_large_drop_blocked():
    f = StabilityFilter(stable_frames=8, max_drop=4)
    push_many(f, START, 8)
    assert all(r is None for r in push_many(f, reading_from_board(board_without(5)), 20))


def test_emits_mean_of_window():
    f = StabilityFilter(stable_frames=2)
    a = START.copy()
    b = START.copy() * 0.5 + reading_from_board(chess.Board()) * 0.5
    f.push(a, True)
    out = f.push(b, True)
    assert out is not None and abs(out - (a + b) / 2).max() < 1e-12


def test_large_drop_accepted_if_persistent():
    """Giảm mạnh nhưng giữ yên lâu (4x stable_frames) thì là thay đổi thật, không phải tay che."""
    f = StabilityFilter(stable_frames=8, max_drop=4)
    push_many(f, START, 8)
    out = push_many(f, reading_from_board(board_without(5)), 32)
    assert out[-1] is not None and all(r is None for r in out[:-1])


def test_window_memory_bounded_on_still_board():
    from chess_assistant.stability import PERSIST_FACTOR

    f = StabilityFilter(stable_frames=8)
    push_many(f, START, 1000)
    assert len(f._window) <= 8 * PERSIST_FACTOR
