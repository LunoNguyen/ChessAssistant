import chess
import cv2
import numpy as np

from chess_assistant.schema import PIECE_CLASSES
from chess_assistant.screen2d import board_from_filename, flat_boxes, simulate_screen_photo


def test_board_from_filename():
    b = board_from_filename("1B1B1K2-3p1N2-6k1-R7-5P2-4q3-7R-1B6.jpeg")
    assert b.board_fen() == "1B1B1K2/3p1N2/6k1/R7/5P2/4q3/7R/1B6"


def test_flat_boxes_one_per_piece_on_its_square():
    board = chess.Board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")
    labels = dict((PIECE_CLASSES[c], box) for c, box in flat_boxes(board, 400))
    assert set(labels) == {"k", "P", "K"}
    x1, y1, x2, y2 = labels["P"]  # e4: cột 4, hàng thứ 4 từ trên xuống (Trắng ở dưới)
    assert 200 <= x1 < x2 <= 250 and 200 <= y1 < y2 <= 250
    x1, y1, x2, y2 = labels["K"]  # e1: hàng dưới cùng
    assert 350 <= y1 < y2 <= 400


def red_marker_board(square: chess.Square) -> np.ndarray:
    img = np.full((400, 400, 3), 230, np.uint8)
    f, r = chess.square_file(square), chess.square_rank(square)
    cv2.rectangle(img, (f * 50 + 5, (7 - r) * 50 + 5), (f * 50 + 44, (7 - r) * 50 + 44), (0, 0, 255), -1)
    return img


def test_simulated_photo_keeps_boxes_on_pieces():
    board = chess.Board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")
    img = red_marker_board(chess.E4)
    labels = flat_boxes(board, 400)
    rng = np.random.default_rng(0)
    for _ in range(5):
        photo, out = simulate_screen_photo(img, labels, rng, effects=False)
        assert photo.shape == (720, 1280, 3)
        assert len(out) == 3
        x1, y1, x2, y2 = dict((PIECE_CLASSES[c], b) for c, b in out)["P"]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        b, g, r = (int(v) for v in photo[cy, cx])
        assert r > 150 and g < 100 and b < 100  # tâm khung e4 vẫn nằm trên ô đỏ
        for _, (x1, y1, x2, y2) in out:
            assert 0 <= x1 < x2 <= 1280 and 0 <= y1 < y2 <= 720


def test_simulated_photo_with_effects_is_deterministic():
    board = chess.Board()
    img = red_marker_board(chess.E4)
    labels = flat_boxes(board, 400)
    a = simulate_screen_photo(img, labels, np.random.default_rng(7))
    b = simulate_screen_photo(img, labels, np.random.default_rng(7))
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
    assert len(a[1]) == 32


def test_convert_writes_yolo_dataset(tmp_path):
    import yaml

    from scripts.import_chess_positions import convert

    src = tmp_path / "src"; src.mkdir()
    for name in ("4k3-8-8-8-4P3-8-8-4K3.jpeg", "rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR.jpeg", "broken-name.jpeg"):
        cv2.imwrite(str(src / name), red_marker_board(chess.E4))
    counts = convert(src, tmp_path / "out", variants=2)
    assert counts["skipped"] == 1 and counts["train"] + counts["valid"] + counts["test"] == 4
    labels = sorted((tmp_path / "out").rglob("labels/*.txt"))
    assert sorted(len(p.read_text().split("\n")) - 1 for p in labels) == [3, 3, 32, 32]
    assert yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())["names"] == list(PIECE_CLASSES)


def test_simulated_board_not_squashed():
    """Điện thoại cầm dọc thì màn hình xoay 90° nhưng không bị kéo méo: bàn cờ vẫn gần vuông."""
    board = chess.Board()
    img = red_marker_board(chess.E4)
    labels = flat_boxes(board, 400)
    for seed in range(40):
        _, out = simulate_screen_photo(img, labels, np.random.default_rng(seed), effects=False)
        centers = {chess.SQUARES[sq]: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                   for sq, (_, b) in zip(sorted(board.piece_map()), out)}
        a1, h1, a8 = (np.array(centers[s]) for s in (chess.A1, chess.H1, chess.A8))
        ratio = np.linalg.norm(h1 - a1) / np.linalg.norm(a8 - a1)
        assert 0.4 < ratio < 2.5, (seed, ratio)
