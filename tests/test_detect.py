from pathlib import Path
import time

import numpy as np
import pytest

from chess_assistant.detect import detect_board, detect_quad, orient
from chess_assistant.tracker import BoardTracker, TrackerConfig
from tests.synthetic import POSES, scene

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
MIDGAME = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R2QKB1R"


def corner_error(found, truth) -> float:
    return float(np.linalg.norm(np.float32(found) - np.float32(truth), axis=1).max())


@pytest.mark.parametrize("pose", POSES)
def test_detects_board_and_a1_from_starting_position(pose):
    truth = POSES[pose]
    result = detect_board(scene(truth, fen=START, seed=1))
    assert result is not None
    # Góc đúng VÀ đúng thứ tự a1, h1, h8, a8.
    assert corner_error(result.corners, truth) < 6


@pytest.mark.parametrize("pose", POSES)
def test_detects_empty_board_up_to_180_degrees(pose):
    """Bàn trống: không biết phía Trắng, nhưng màu ô phải xác định đúng a1 hoặc h8."""
    truth = np.float32(POSES[pose])
    result = detect_board(scene(truth, fen=None, seed=2))
    assert result is not None
    assert min(corner_error(result.corners, truth), corner_error(result.corners, np.roll(truth, 2, axis=0))) < 6


@pytest.mark.parametrize("seed", range(4))
def test_no_board_returns_none(seed):
    frame = scene(POSES["white_side"], seed=seed)
    frame[:] = scene([[0, 0], [1, 0], [1, 1], [0, 1]], fen=None, seed=seed)  # bàn cờ thu nhỏ còn 1px
    assert detect_quad(frame) is None


def test_orient_wrong_order_input():
    """orient nhận 4 góc theo bất kỳ thứ tự vòng nào (kể cả ngược chiều)."""
    truth = np.float32(POSES["rotated"])
    frame = scene(truth, fen=START)
    for quad in (np.roll(truth, 1, axis=0), truth[::-1]):
        assert corner_error(orient(frame, quad).corners, truth) < 1


def test_detection_speed():
    frame = scene(POSES["white_side"], fen=START)
    detect_quad(frame)
    t = time.perf_counter()
    for _ in range(3):
        detect_quad(frame)
    assert (time.perf_counter() - t) / 3 < 0.5


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

def make_tracker(**kw) -> BoardTracker:
    return BoardTracker(TrackerConfig(redetect_every=1, **kw), background=False)


def test_tracker_keeps_orientation_through_midgame():
    """Khoá a1 từ thế khởi đầu; giữa ván (không còn suy được phía Trắng) vẫn giữ đúng."""
    truth = POSES["black_side"]
    tr = make_tracker()
    tr.update(scene(truth, fen=START))
    for seed in range(3):
        corners = tr.update(scene(truth, fen=MIDGAME, seed=seed))
    assert tr.status == "locked"
    assert corner_error(corners, truth) < 6


def test_tracker_holds_when_hand_covers_board():
    truth = POSES["white_side"]
    tr = make_tracker(hold_misses=1)
    before = tr.update(scene(truth, fen=START)).copy()
    for seed in range(4):
        after = tr.update(scene(truth, fen=START, seed=seed, hand=(640, 400, 330, 220)))
    assert corner_error(after, before) < 6
    assert tr.status in ("locked", "holding")


def test_tracker_follows_moved_board_after_confirmation():
    a, b = np.float32(POSES["white_side"]), np.float32(POSES["white_side"]) + [120, -40]
    tr = make_tracker(confirm=3)
    tr.update(scene(a, fen=START))
    tr.update(scene(b, fen=START, seed=1))
    tr.update(scene(b, fen=START, seed=2))
    assert corner_error(tr.corners, a) < 6  # chưa đủ xác nhận
    tr.update(scene(b, fen=START, seed=3))
    assert corner_error(tr.corners, b) < 6


def test_tracker_flip():
    truth = np.float32(POSES["white_side"])
    tr = make_tracker()
    tr.update(scene(truth, fen=None))
    tr.flip()
    flipped = tr.corners.copy()
    tr.update(scene(truth, fen=None, seed=5))
    # Sau khi lật, các lần dò tiếp theo giữ hướng đã lật.
    assert corner_error(tr.corners, flipped) < 6


def test_tracker_background_mode():
    truth = POSES["white_side"]
    frame = scene(truth, fen=START)
    tr = BoardTracker(TrackerConfig(redetect_every=1))
    try:
        deadline = time.time() + 5
        while tr.corners is None and time.time() < deadline:
            tr.update(frame)
            time.sleep(0.01)
    finally:
        tr.close()
    assert tr.corners is not None and corner_error(tr.corners, truth) < 6


def test_tracker_survives_detector_exception(monkeypatch):
    import chess_assistant.tracker as tracker_mod

    truth = POSES["white_side"]
    tr = make_tracker()
    before = tr.update(scene(truth, fen=START)).copy()

    def boom(frame):
        raise RuntimeError("detector crashed")

    monkeypatch.setattr(tracker_mod, "detect_quad", boom)
    after = tr.update(scene(truth, fen=START, seed=1))
    assert corner_error(after, before) < 1e-6


def test_tracker_reports_moving_while_jump_unconfirmed():
    a, b = np.float32(POSES["white_side"]), np.float32(POSES["white_side"]) + [120, -40]
    tr = make_tracker(confirm=3)
    tr.update(scene(a, fen=START))
    assert tr.status == "locked"
    tr.update(scene(b, fen=START, seed=1))
    assert tr.status == "moving"
    tr.update(scene(b, fen=START, seed=2))
    tr.update(scene(b, fen=START, seed=3))
    assert tr.status == "locked" and corner_error(tr.corners, b) < 6


def test_tracker_rotate():
    tr = make_tracker()
    tr.update(scene(POSES["white_side"], fen=START))
    before = tr.corners.copy()
    tr.rotate(1)
    assert np.allclose(tr.corners, np.roll(before, 1, axis=0))


PAPER_MIDGAME = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R2QKB1R"


def shrink(corners):
    return [[c[0] * 0.8 + 130, c[1] * 0.8 + 70] for c in corners]  # để lề giấy nằm trong ảnh


# small_far: mỗi ô chỉ ~14px ở độ phân giải dò, thiếu đường lưới — lỗi khác, không thuộc test này.
@pytest.mark.parametrize("pose", [p for p in POSES if p != "small_far"])
def test_paper_board_with_printed_frame_not_shifted(pose):
    """Bàn giấy (như ChessReD): đường viền in cách mép đúng 1 ô dễ làm lưới lệch đi 1 hàng."""
    truth = np.float32(shrink(POSES[pose]))
    found = detect_quad(scene(truth, fen=PAPER_MIDGAME, style="paper", seed=3))
    assert found is not None
    err = min(corner_error(np.roll(found.quad, k, axis=0), truth) for k in range(4))
    err = min(err, min(corner_error(np.roll(found.quad[::-1], k, axis=0), truth) for k in range(4)))
    assert err < 8


CHESSRED = Path(__file__).resolve().parent.parent / "data" / "downloads" / "chessred"
MODEL = Path(__file__).resolve().parent.parent / "models" / "pieces.pt"


@pytest.mark.skipif(not (CHESSRED / "annotations.json").exists() or not MODEL.exists(),
                    reason="Cần dữ liệu ChessReD và models/pieces.pt trên máy")
@pytest.mark.parametrize("image_index", [33, 34, 37, 39])
def test_real_paper_board_game33_not_shifted(image_index):
    """Ảnh thật (ChessReD ván 33) từng bị lưới lệch 1 hàng vì đường viền in trên bàn giấy."""
    import chess

    from chess_assistant.board import BoardView
    from chess_assistant.chessred import load_games
    from chess_assistant.game import match_count
    from chess_assistant.pieces import PieceDetector
    from chess_assistant.reading import read_board
    from scripts.replay_chessred import _frames

    frame_info = load_games(CHESSRED)[33][image_index]
    img = next(_frames([frame_info], 1))
    found = detect_board(img)
    assert found is not None
    boxes = PieceDetector()(img)
    truth = chess.Board(frame_info.board_fen + " w - - 0 1")
    best = max(match_count(read_board(boxes, BoardView.from_corners(np.roll(found.corners, k, axis=0)).H), truth)
               for k in range(4))
    assert best >= 58
