import time

import chess
import chess.engine
import pytest

from chess_assistant.advisor import Advisor
from chess_assistant.engine import Engine, find_stockfish

try:
    find_stockfish()
    HAVE_STOCKFISH = True
except FileNotFoundError:
    HAVE_STOCKFISH = False

needs_stockfish = pytest.mark.skipif(not HAVE_STOCKFISH, reason="Chưa cài Stockfish")


def wait_latest(advisor: Advisor, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        advice = advisor.latest()
        if advice is not None:
            return advice
        time.sleep(0.02)
    return None


@pytest.fixture
def engine():
    if not HAVE_STOCKFISH:
        pytest.skip("Chưa cài Stockfish")
    with Engine() as e:
        yield e


@needs_stockfish
def test_suggests_on_user_turn(engine):
    board = chess.Board()
    advisor = Advisor(engine, think_time=0.2)
    try:
        advisor.request(board, chess.WHITE)
        advice = wait_latest(advisor)
    finally:
        advisor.close()
    assert advice is not None and advice.fen == board.fen()
    assert advice.moves[0] in board.legal_moves


@needs_stockfish
def test_no_advice_on_opponent_turn_best_mode(engine):
    board = chess.Board()
    board.push_uci("e2e4")
    advisor = Advisor(engine, think_time=0.2)
    try:
        advisor.request(board, chess.WHITE)
        time.sleep(1.5)
        assert advisor.latest() is None
    finally:
        advisor.close()


@needs_stockfish
def test_stale_request_dropped(engine):
    a = chess.Board()
    b = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    advisor = Advisor(engine, think_time=0.3)
    try:
        advisor.request(a, chess.WHITE)
        advisor.request(b, chess.BLACK)
        advice = wait_latest(advisor)
        time.sleep(0.5)
        advice = advisor.latest()
    finally:
        advisor.close()
    assert advice is not None and advice.fen == b.fen()


@needs_stockfish
def test_training_blunder_detected(engine):
    before = chess.Board("4k3/8/8/3r4/8/8/8/3QK3 w - - 0 1")
    after = before.copy()
    after.push_uci("d1d4")  # Hậu đứng vào ô bị Xe ăn
    advisor = Advisor(engine, think_time=0.3, mode="training", elo=1500)
    try:
        advisor.request(before, chess.WHITE)
        assert wait_latest(advisor) is not None
        advisor.request(after, chess.WHITE)
        advice = wait_latest(advisor)
    finally:
        advisor.close()
    assert advice is not None and advice.blunder_cp is not None and advice.blunder_cp >= 200


class BrokenEngine:
    def analyse(self, *args, **kwargs):
        raise chess.engine.EngineTerminatedError("engine died")

    def play_limited(self, *args, **kwargs):
        raise chess.engine.EngineTerminatedError("engine died")


def test_engine_error_reported():
    advisor = Advisor(BrokenEngine(), think_time=0.1)
    try:
        advisor.request(chess.Board(), chess.WHITE)
        deadline = time.time() + 2
        while advisor.error is None and time.time() < deadline:
            time.sleep(0.02)
        assert advisor.latest() is None
        assert advisor.error is not None
    finally:
        advisor.close()


@needs_stockfish
def test_user_color_change_invalidates_advice(engine):
    board = chess.Board()
    advisor = Advisor(engine, think_time=0.2)
    try:
        advisor.request(board, chess.WHITE)
        assert wait_latest(advisor) is not None
        advisor.request(board, chess.BLACK)  # người dùng đổi màu: giờ là lượt đối thủ
        time.sleep(0.5)
        assert advisor.latest() is None
    finally:
        advisor.close()


@needs_stockfish
def test_both_sides_mode_suggests_on_opponent_turn(engine):
    board = chess.Board()
    board.push_uci("e2e4")
    advisor = Advisor(engine, think_time=0.2, side="both")
    try:
        advisor.request(board, chess.WHITE)
        advice = wait_latest(advisor)
    finally:
        advisor.close()
    assert advice is not None and advice.moves[0] in board.legal_moves


@needs_stockfish
def test_both_turns_for_unknown_turn(engine):
    board = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R2QKB1R w KQ - 0 1")
    advisor = Advisor(engine, think_time=0.2)
    try:
        advisor.request(board, chess.WHITE, both_turns=True)
        advice = wait_latest(advisor)
    finally:
        advisor.close()
    white, black = board.copy(), board.copy()
    black.turn = chess.BLACK
    assert advice is not None
    assert advice.moves[0] in white.legal_moves
    assert advice.alt_moves[0] in black.legal_moves
