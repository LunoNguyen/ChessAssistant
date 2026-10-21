import chess
import pytest

from chess_assistant.engine import Engine, find_stockfish

try:
    find_stockfish()
    HAVE_STOCKFISH = True
except FileNotFoundError:
    HAVE_STOCKFISH = False

pytestmark = pytest.mark.skipif(not HAVE_STOCKFISH, reason="Chưa cài Stockfish (xem engines/README.md)")


def test_suggests_legal_move():
    board = chess.Board()
    with Engine() as engine:
        s = engine.suggest(board, think_time=0.1)
    assert s is not None and s.move in board.legal_moves


def test_finds_mate_in_one():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
    with Engine() as engine:
        s = engine.suggest(board, think_time=0.2)
    assert s.move == chess.Move.from_uci("d1d8")
    assert s.score_text() == "#1"


def test_no_suggestion_when_game_over():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
    board.push_uci("d1d8")
    with Engine() as engine:
        assert engine.suggest(board) is None


def test_analyse_multipv():
    board = chess.Board()
    with Engine() as engine:
        lines = engine.analyse(board, think_time=0.3, multipv=3)
    moves = [s.move for s in lines]
    assert len(moves) == 3 and len(set(moves)) == 3
    assert all(m in board.legal_moves for m in moves)


def test_play_limited_legal():
    board = chess.Board()
    with Engine() as engine:
        assert engine.play_limited(board, elo=1350, think_time=0.1) in board.legal_moves


def test_white_cp_mate():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
    with Engine() as engine:
        assert engine.analyse(board, think_time=0.2)[0].white_cp() == 10000
