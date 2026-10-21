import chess
import cv2

from chess_assistant.evaluation import move_accuracy, square_accuracy
from chess_assistant.schema import reading_from_board
from scripts.replay import run_replay
from tests.fakes import FakeDetector
from tests.synthetic import POSES, scene


def mv(*ucis):
    return [chess.Move.from_uci(u) for u in ucis]


def test_square_accuracy_perfect():
    b = chess.Board()
    assert square_accuracy(reading_from_board(b), b) == 1.0
    assert square_accuracy(reading_from_board(b), chess.Board(None)) == 32 / 64


def test_move_accuracy():
    truth = mv("e2e4", "e7e5", "g1f3")
    assert move_accuracy(mv("e2e4", "e7e5"), truth) == {"correct": 2, "wrong": 0, "missed": 1}
    assert move_accuracy(mv("e2e4", "d7d5"), truth) == {"correct": 1, "wrong": 1, "missed": 1}
    assert move_accuracy(mv("e2e4", "e7e5", "g1f3"), truth) == {"correct": 3, "wrong": 0, "missed": 0}


def test_replay_synthetic(tmp_path):
    corners = POSES["white_side"]
    boards = [chess.Board()]
    for u in ("e2e4", "e7e5"):
        boards.append(boards[-1].copy())
        boards[-1].push_uci(u)
    per_frame = [b for b in boards for _ in range(15)]

    video = tmp_path / "game.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 15, (1280, 720))
    for i, b in enumerate(per_frame):
        writer.write(scene(corners, fen=b.board_fen(), seed=i))
    writer.release()
    pgn = tmp_path / "game.pgn"
    pgn.write_text("1. e4 e5 *\n")

    calls = []
    fake = FakeDetector(lambda: per_frame[min(len(calls), len(per_frame) - 1)], corners)

    def detector(frame):
        out = fake(frame)
        calls.append(1)
        return out

    report = run_replay(video, pgn, detector)
    assert report["correct"] == 2 and report["wrong"] == 0 and report["missed"] == 0
    assert report["frames"] == len(per_frame)


def test_replay_chessred_game(tmp_path):
    from chess_assistant.chessred import GameFrame
    from scripts.replay_chessred import replay_game

    corners = POSES["rotated"]
    boards = [chess.Board()]
    for u in ("d2d4", "g8f6", "c2c4"):
        boards.append(boards[-1].copy())
        boards[-1].push_uci(u)
    frames = []
    for i, b in enumerate(boards):
        path = tmp_path / f"G001_IMG{i:03d}.jpg"
        cv2.imwrite(str(path), scene(corners, fen=b.board_fen(), seed=i))
        frames.append(GameFrame(path, b.board_fen(), i))

    repeat = 12
    calls = []
    fake = FakeDetector(lambda: boards[min(len(calls) // repeat, len(boards) - 1)], corners)

    def detector(frame):
        out = fake(frame)
        calls.append(1)
        return out

    report = replay_game(frames, detector, repeat=repeat)
    assert report["correct"] == 3 and report["wrong"] == 0 and report["missed"] == 0


def keyed(board: chess.Board, ucis):
    """(khoá thế cờ trước nước, nước) theo đúng thứ tự đi."""
    from chess_assistant.evaluation import position_key

    b, out = board.copy(), []
    for u in ucis:
        m = chess.Move.from_uci(u)
        out.append((position_key(b), m))
        b.push(m)
    return out


def test_move_accuracy_by_position_after_resync():
    from chess_assistant.evaluation import move_accuracy_by_position

    truth = mv("e2e4", "e7e5", "g1f3", "b8c6", "f1c4")
    after4 = chess.Board()
    for m in truth[:4]:
        after4.push(m)
    # Ghi nhận e4 e5, lạc 2 nước, đọc lại rồi ghi nhận đúng Bc4 từ đúng thế cờ.
    recognized = keyed(chess.Board(), ["e2e4", "e7e5"]) + keyed(after4, ["f1c4"])
    assert move_accuracy_by_position(recognized, chess.Board(), truth) == {"correct": 3, "wrong": 0, "missed": 2}


def test_move_accuracy_by_position_counts_wrong_move():
    from chess_assistant.evaluation import move_accuracy_by_position

    truth = mv("e2e4", "e7e5")
    recognized = keyed(chess.Board(), ["d2d4"])
    assert move_accuracy_by_position(recognized, chess.Board(), truth) == {"correct": 0, "wrong": 1, "missed": 2}
