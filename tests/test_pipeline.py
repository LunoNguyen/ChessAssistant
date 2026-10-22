import chess
import numpy as np

from chess_assistant.game import GameTracker
from chess_assistant.pipeline import Pipeline
from chess_assistant.tracker import BoardTracker, TrackerConfig
from tests.fakes import FakeDetector
from tests.synthetic import POSES, scene

MIDGAME = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R2QKB1R w - - 0 1"


class Scenario:
    def __init__(self, pose: str, board: chess.Board):
        self.corners = POSES[pose]
        self.board = board
        self.pipeline = Pipeline(
            FakeDetector(lambda: self.board, self.corners),
            BoardTracker(TrackerConfig(redetect_every=5), background=False),
            GameTracker(),
        )
        self.events = []

    def run(self, frames: int, **scene_kw):
        result = None
        for i in range(frames):
            result = self.pipeline.process(scene(self.corners, fen=self.board.board_fen(), seed=i, **scene_kw))
            self.events += result.events
        return result


def kinds(events):
    return [e.kind for e in events]


def test_syncs_and_records_move():
    s = Scenario("white_side", chess.Board())
    s.run(10)
    assert kinds(s.events) == ["synced"]
    s.board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    s.run(10)
    assert kinds(s.events) == ["synced", "moves"]
    assert s.events[-1].moves == [chess.Move.from_uci("e2e4")]


def test_piece_in_hand_frames_ignored():
    s = Scenario("white_side", chess.Board())
    s.run(10)
    lifted = chess.Board()
    lifted.remove_piece_at(chess.E2)
    s.board = lifted
    s.run(5, hand=(640, 500, 120, 80))
    s.board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    s.run(10)
    assert kinds(s.events) == ["synced", "moves"]
    assert s.events[-1].moves == [chess.Move.from_uci("e2e4")]


def test_flip_applied_on_resync():
    s = Scenario("black_side", chess.Board(MIDGAME))
    s.run(1)
    s.pipeline.tracker.flip()  # giả lập tracker đoán nhầm phía Trắng
    s.run(12)
    assert "need_turn" in kinds(s.events)
    err = np.linalg.norm(s.pipeline.tracker.corners - np.float32(POSES["black_side"]), axis=1).max()
    assert err < 6


def test_user_color_from_view():
    s = Scenario("black_side", chess.Board())
    assert s.run(1).overlay.user_color == chess.BLACK


E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def test_undo_does_not_rerecord_same_move():
    s = Scenario("white_side", chess.Board())
    s.run(10)
    s.board = chess.Board(E4)
    s.run(10)
    assert s.pipeline.undo() == chess.Move.from_uci("e2e4")
    s.run(10)
    assert kinds(s.events) == ["synced", "moves"]
    assert s.pipeline.game.board.fen() == chess.STARTING_FEN


def test_flip_key_during_need_turn_rotates_pending():
    s = Scenario("white_side", chess.Board(MIDGAME))
    s.run(10)
    assert s.pipeline.game.state.value == "need_turn"
    before = s.pipeline.game.pending.board_fen()
    s.pipeline.flip()
    assert s.pipeline.game.pending.board_fen() != before
    s.pipeline.flip()
    assert s.pipeline.game.pending.board_fen() == before


class StaleAdvisor:
    error = None

    def __init__(self):
        from chess_assistant.advisor import Advice

        self.advice = Advice(chess.STARTING_FEN, [chess.Move.from_uci("e2e4")], "+0.2", 20)

    def request(self, board, user_color):
        pass

    def latest(self):
        return self.advice


def test_advice_hidden_when_unsynced():
    no_black_king = chess.Board(MIDGAME)
    no_black_king.remove_piece_at(chess.G8)
    s = Scenario("white_side", no_black_king)
    s.pipeline.advisor = StaleAdvisor()
    result = s.run(10)
    assert s.pipeline.game.state.value == "unsynced"
    assert result.overlay.advice is None


class RecordingAdvisor(StaleAdvisor):
    def __init__(self):
        super().__init__()
        self.requests = []

    def request(self, board, user_color, both_turns=False):
        self.requests.append((board.board_fen(), both_turns))


def test_pending_position_gets_advice_for_both_sides():
    s = Scenario("white_side", chess.Board(MIDGAME))
    advisor = RecordingAdvisor()
    s.pipeline.advisor = advisor
    result = s.run(10)
    assert s.pipeline.game.state.value == "need_turn"
    assert (chess.Board(MIDGAME).board_fen(), True) in advisor.requests
    assert result.overlay.advice is advisor.advice


def test_board_rotated_90_degrees_is_realigned():
    """Cùng tứ giác trên ảnh nhưng bàn cờ xoay 90°: tracker giữ góc cũ, pipeline phải tự sửa hướng."""
    s = Scenario("white_side", chess.Board())
    s.run(10)
    assert kinds(s.events) == ["synced"]
    s.corners = POSES["a_file_side"]  # cùng 4 điểm, a1 ở góc khác
    s.pipeline.detector = FakeDetector(lambda: s.board, s.corners)
    s.run(10)
    s.board = chess.Board(E4)
    s.run(12)
    assert kinds(s.events) == ["synced", "moves"]
    assert s.events[-1].moves == [chess.Move.from_uci("e2e4")]


def test_rotation_and_move_at_same_time():
    """Ảnh mới vừa xoay hướng vừa có một nước đi (như ChessReD): vẫn phải sửa hướng và nhận nước."""
    s = Scenario("white_side", chess.Board())
    s.run(10)
    s.corners = POSES["a_file_side"]

    def missing_a7():  # ảnh thật hay có 1 ô đọc sót
        b = s.board.copy()
        b.remove_piece_at(chess.A7)
        return b

    s.pipeline.detector = FakeDetector(missing_a7, s.corners)
    s.board = chess.Board(E4)
    s.run(12)
    assert kinds(s.events) == ["synced", "moves"]
    assert s.events[-1].moves == [chess.Move.from_uci("e2e4")]


def test_turn_key_before_position_is_read_explains_why():
    """Bấm w/b khi chưa đọc được thế cờ: phải nói rõ là chưa đọc được, không phải 'đang bị chiếu'."""
    no_kings = chess.Board(None)
    no_kings.set_piece_at(chess.E4, chess.Piece.from_symbol("P"))
    s = Scenario("white_side", no_kings)
    s.run(10)
    assert s.pipeline.set_turn(chess.BLACK) is False
    msg = s.run(1).overlay.message.lower()
    assert "chua doc duoc" in msg and "chieu" not in msg


ENDGAME = "8/1p3pp1/p7/P4k2/8/4PP2/5K1P/8 w - - 0 1"  # ChessReD ván 33, ảnh 78: chỉ còn 10 quân


def missing(board_fn, square):
    def fn():
        b = board_fn().copy()
        b.remove_piece_at(square)
        return b
    return fn


def test_endgame_rotation_realigned():
    """Ít quân: hướng sai vẫn khớp ~46 ô trống, nên hướng đúng chỉ hơn ~15 ô — vẫn phải sửa được."""
    s = Scenario("white_side", chess.Board(ENDGAME))
    s.pipeline.game.start_from(chess.Board(ENDGAME))
    s.run(10)
    s.corners = POSES["a_file_side"]
    s.pipeline.detector = FakeDetector(missing(lambda: s.board, chess.B7), s.corners)
    s.board = chess.Board("8/1p3pp1/p7/P4k2/8/4PP2/5K1P/8 w - - 0 1")
    s.board.push_uci("h2h4")
    s.run(12)
    assert [e.moves for e in s.events if e.kind == "moves"] == [[chess.Move.from_uci("h2h4")]]


def test_rotation_realigned_while_waiting_for_turn():
    """Đang chờ biết lượt (sau khi đọc lại) mà bàn xoay 90° và có nước đi: vẫn phải nhận ra."""
    s = Scenario("white_side", chess.Board(MIDGAME))
    s.run(10)
    assert s.pipeline.game.state.value == "need_turn"
    s.corners = POSES["a_file_side"]
    s.pipeline.detector = FakeDetector(lambda: s.board, s.corners)
    s.board = chess.Board(MIDGAME)
    s.board.push_uci("h2h3")
    s.run(12)
    assert [e.moves for e in s.events if e.kind == "moves"] == [[chess.Move.from_uci("h2h3")]]


def test_orientation_found_from_piece_colors_before_sync():
    """Chưa bám ván mà tracker chọn sai góc a1 (lệch 90°): nhờ màu quân, tự xoay đúng — không cần bấm f."""
    s = Scenario("white_side", chess.Board())
    s.run(1)
    s.pipeline.tracker.rotate(1)
    s.run(12)
    assert "synced" in kinds(s.events)
    err = np.linalg.norm(s.pipeline.tracker.corners - np.float32(POSES["white_side"]), axis=1).max()
    assert err < 6


def test_orientation_found_for_midgame_before_sync():
    s = Scenario("white_side", chess.Board(MIDGAME))
    s.run(1)
    s.pipeline.tracker.rotate(3)
    s.run(12)
    assert s.pipeline.game.state.value == "need_turn"
    assert s.pipeline.game.pending.board_fen() == chess.Board(MIDGAME).board_fen()
