import time

import chess
import numpy as np

from chess_assistant.game import GameTracker, SyncState, match_count, position_score
from chess_assistant.schema import PIECE_CLASSES, EMPTY, NUM_CLASSES, reading_from_board


def noisy(board: chess.Board, wrong: dict[chess.Square, str] | None = None, conf: float = 0.9) -> np.ndarray:
    """Reading của `board`; các ô trong `wrong` bị đọc thành ký hiệu chỉ định ('.' = trống)."""
    r = reading_from_board(board, conf)
    for sq, sym in (wrong or {}).items():
        r[sq] = (1 - conf) / (NUM_CLASSES - 1)
        r[sq, EMPTY if sym == "." else PIECE_CLASSES.index(sym)] = conf
    return r


def after(fen_or_board, *ucis: str) -> chess.Board:
    b = chess.Board(fen_or_board) if isinstance(fen_or_board, str) else fen_or_board.copy()
    for u in ucis:
        b.push_uci(u)
    return b


def synced(board: chess.Board | None = None) -> GameTracker:
    t = GameTracker()
    t.start_from(board or chess.Board())
    return t


def mv(u: str) -> chess.Move:
    return chess.Move.from_uci(u)


def test_scores():
    b = chess.Board()
    assert match_count(noisy(b), b) == 64
    assert position_score(noisy(b), b) > position_score(noisy(b), after(b, "e2e4"))


def test_no_change_returns_none():
    t = synced()
    assert t.update(noisy(chess.Board())) is None
    assert t.state == SyncState.SYNCED


def test_simple_move():
    t = synced()
    ev = t.update(noisy(after(chess.Board(), "e2e4")))
    assert ev.kind == "moves" and ev.moves == [mv("e2e4")]
    assert t.board.fen() == after(chess.Board(), "e2e4").fen()


def test_castling():
    fen = "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    t = synced(chess.Board(fen))
    ev = t.update(noisy(after(fen, "e1g1")))
    assert ev.moves == [mv("e1g1")]


def test_en_passant():
    fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
    t = synced(chess.Board(fen))
    ev = t.update(noisy(after(fen, "e5f6")))
    assert ev.moves == [mv("e5f6")]


def test_promotion_ambiguous_picks_queen():
    fen = "8/P6k/8/8/8/8/8/K7 w - - 0 1"
    t = synced(chess.Board(fen))
    r = noisy(after(fen, "a7a8q"))
    r[chess.A8] = 0.01
    for sym in "QRBN":
        r[chess.A8, PIECE_CLASSES.index(sym)] = 0.22
    r[chess.A8] /= r[chess.A8].sum()
    ev = t.update(r)
    assert ev.kind == "moves" and ev.moves == [mv("a7a8q")]


def test_two_missed_plies():
    t = synced()
    ev = t.update(noisy(after(chess.Board(), "e2e4", "e7e5")))
    assert ev.moves == [mv("e2e4"), mv("e7e5")]


def test_one_misread_square_tolerated():
    t = synced()
    ev = t.update(noisy(after(chess.Board(), "g1f3"), wrong={chess.A7: "."}))
    assert ev.moves == [mv("g1f3")]


def test_ambiguous_is_unsure():
    t = synced()
    r = (noisy(after(chess.Board(), "e2e4")) + noisy(after(chess.Board(), "e2e3"))) / 2
    ev = t.update(r)
    assert ev.kind == "unsure"
    assert t.board.fen() == chess.Board().fen()


def test_lost_after_three_unsure():
    t = synced()
    other = noisy(chess.Board("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R2QKB1R w KQ - 0 8"))
    kinds = [t.update(other).kind for _ in range(3)]
    assert kinds == ["unsure", "unsure", "lost"]
    assert t.state == SyncState.LOST


def test_undo_and_pgn():
    t = synced()
    t.update(noisy(after(chess.Board(), "e2e4")))
    t.update(noisy(after(chess.Board(), "e2e4", "e7e5")))
    assert "1. e4" in t.pgn()
    assert t.undo() == mv("e7e5")
    assert t.board.fen() == after(chess.Board(), "e2e4").fen()


def test_two_ply_speed():
    t = synced()
    r = noisy(after(chess.Board(), "e2e4", "e7e5"))
    start = time.perf_counter()
    t.update(r)
    assert time.perf_counter() - start < 0.2


# ---------------------------------------------------------------------------
# Task 5: đồng bộ lại
# ---------------------------------------------------------------------------

MIDGAME = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R2QKB1R"


def rotate180(reading: np.ndarray) -> np.ndarray:
    return reading[::-1].copy()


def test_start_position_autosyncs():
    t = GameTracker()
    ev = t.update(noisy(chess.Board()))
    assert ev.kind == "synced" and t.state == SyncState.SYNCED


def test_resync_midgame_needs_turn():
    t = GameTracker()
    ev = t.update(noisy(chess.Board(MIDGAME + " w - - 0 1")))
    assert ev.kind == "need_turn" and ev.flipped is False
    assert t.update(noisy(chess.Board(MIDGAME + " w - - 0 1"))) is None  # chờ chọn lượt
    assert t.set_turn(chess.WHITE) is True
    assert t.state == SyncState.SYNCED
    assert t.board.board_fen() == MIDGAME and t.board.turn == chess.WHITE


def test_resync_detects_flipped_view():
    t = GameTracker()
    ev = t.update(rotate180(noisy(chess.Board(MIDGAME + " w - - 0 1"))))
    assert ev.kind == "need_turn" and ev.flipped is True
    assert t.pending.board_fen() == MIDGAME


def test_castling_rights_inferred():
    fen = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/1R1QKB1R"
    t = GameTracker()
    t.update(noisy(chess.Board(fen + " w - - 0 1")))
    assert t.set_turn(chess.WHITE)
    assert t.board.has_kingside_castling_rights(chess.WHITE)
    assert not t.board.has_queenside_castling_rights(chess.WHITE)
    assert not t.board.has_castling_rights(chess.BLACK)


def test_set_turn_invalid_side():
    t = GameTracker()
    assert t.update(noisy(chess.Board(MIDGAME + " w - - 0 1"))).kind == "need_turn"
    t.pending.set_piece_at(chess.D3, chess.Piece.from_symbol("n"))  # Mã đen chiếu Vua e1: Đen không thể đến lượt
    assert t.set_turn(chess.BLACK) is False
    assert t.state == SyncState.NEED_TURN
    assert t.set_turn(chess.WHITE) is True


def test_resync_auto_turn_when_only_one_side_can_move():
    """Bên không đi đang bị chiếu thì chỉ còn một khả năng: tự chọn, không hỏi."""
    t = GameTracker()
    ev = t.update(noisy(chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")))
    assert ev.kind == "synced" and t.state == SyncState.SYNCED
    assert t.board.turn == chess.WHITE


def test_need_turn_resolved_by_next_white_move():
    t = GameTracker()
    t.update(noisy(chess.Board(MIDGAME + " w - - 0 1")))
    ev = t.update(noisy(after(MIDGAME + " w - - 0 1", "h2h3")))
    assert ev.kind == "moves" and ev.moves == [mv("h2h3")]
    assert t.state == SyncState.SYNCED and t.board.turn == chess.BLACK


def test_need_turn_resolved_by_next_black_move():
    t = GameTracker()
    t.update(noisy(chess.Board(MIDGAME + " w - - 0 1")))
    ev = t.update(noisy(after(MIDGAME + " b - - 0 1", "h7h6")))
    assert ev.kind == "moves" and ev.moves == [mv("h7h6")]
    assert t.state == SyncState.SYNCED and t.board.turn == chess.WHITE


def test_lost_then_resync():
    t = synced()
    far = noisy(after(chess.Board(), "e2e4", "e7e5", "g1f3", "b8c6"))
    events = [t.update(far) for _ in range(4)]
    assert [e.kind for e in events] == ["unsure", "unsure", "lost", "need_turn"]
    assert not any(e.kind == "moves" for e in events)


def test_invalid_reading_unsure():
    t = GameTracker()
    b = chess.Board(MIDGAME + " w - - 0 1")
    b.remove_piece_at(chess.G8)
    ev = t.update(noisy(b))
    assert ev.kind == "unsure" and t.state == SyncState.UNSYNCED


# ---------------------------------------------------------------------------
# Sửa sau review cuối
# ---------------------------------------------------------------------------

def test_history_kept_across_resync():
    t = synced()
    t.update(noisy(after(chess.Board(), "e2e4")))
    far = noisy(after(chess.Board(), "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"))
    for _ in range(4):
        t.update(far)
    assert t.set_turn(chess.WHITE)
    t.update(noisy(after(chess.Board(), "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3")))
    pgn = t.pgn()
    assert "1. e4" in pgn and "d3" in pgn


def test_need_turn_refreshes_to_start_position():
    t = GameTracker()
    partial = chess.Board("4k3/8/8/8/8/8/PPPP4/4K3 w - - 0 1")
    assert t.update(noisy(partial)).kind == "need_turn"
    ev = t.update(noisy(chess.Board()))
    assert ev.kind == "synced" and t.state == SyncState.SYNCED


def test_need_turn_new_reading_replaces_pending():
    t = GameTracker()
    t.update(noisy(chess.Board(MIDGAME + " w - - 0 1")))
    other = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R2Q1RK1"
    ev = t.update(noisy(chess.Board(other + " w - - 0 1")))
    assert ev.kind == "need_turn" and t.pending.board_fen() == other
    assert t.update(noisy(chess.Board(other + " w - - 0 1"))) is None  # không đổi -> không phát lại


def test_flip_pending():
    t = GameTracker()
    t.update(noisy(chess.Board(MIDGAME + " w - - 0 1")))
    t.flip_pending()
    rotated = chess.Board(MIDGAME + " w - - 0 1").transform(chess.flip_vertical).transform(chess.flip_horizontal)
    assert t.pending.board_fen() == rotated.board_fen()
    assert t.pending_flipped is True
