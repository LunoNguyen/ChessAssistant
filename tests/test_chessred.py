import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from chess_assistant.chessred import convert_chessred
from chess_assistant.schema import PIECE_CLASSES


def write_image(path: Path, w: int = 200, h: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((h, w, 3), np.uint8))


def make_chessred(root: Path, category_names: list[str], with_2k_split: bool = True) -> None:
    """ChessReD thu nhỏ: 4 ảnh; ảnh 3 thiếu bbox một quân, ảnh 4 không có bbox nào (không thuộc 2K)."""
    cats = [{"id": i, "name": n} for i, n in enumerate(category_names)]
    images = []
    for i in range(1, 5):
        rel = f"images/{i}/G000_IMG{i:03d}.jpg"
        write_image(root / rel)
        images.append({"id": i, "path": rel, "width": 200, "height": 100, "game_id": 0})
    pieces = [
        {"image_id": 1, "category_id": 0, "chessboard_position": "e2", "bbox": [10, 20, 30, 40]},
        {"image_id": 1, "category_id": 1, "chessboard_position": "e8", "bbox": [100, 0, 50, 50]},
        {"image_id": 2, "category_id": 0, "chessboard_position": "d2", "bbox": [0, 0, 20, 20]},
        {"image_id": 3, "category_id": 0, "chessboard_position": "a2", "bbox": [0, 0, 20, 20]},
        {"image_id": 3, "category_id": 1, "chessboard_position": "a7"},
        {"image_id": 4, "category_id": 0, "chessboard_position": "a2"},
    ]
    splits = {"train": {"image_ids": [1, 3, 4], "n_samples": 3}, "val": {"image_ids": [2], "n_samples": 1},
              "test": {"image_ids": [], "n_samples": 0}}
    if with_2k_split:
        splits["chessred2k"] = {"train": {"image_ids": [1, 3], "n_samples": 2},
                                "val": {"image_ids": [2], "n_samples": 1},
                                "test": {"image_ids": [], "n_samples": 0}}
    data = {"images": images, "categories": cats, "annotations": {"pieces": pieces, "corners": []}, "splits": splits}
    (root / "annotations.json").write_text(json.dumps(data))


@pytest.mark.parametrize("names", [["white-pawn", "black-king", "empty"], ["P", "k", "empty"]])
def test_convert_basic(tmp_path, names):
    src, out = tmp_path / "chessred", tmp_path / "out"
    make_chessred(src, names)
    counts = convert_chessred(src, out)
    # Ảnh 3 thiếu bbox một quân -> bỏ (nhãn thiếu sẽ dạy model rằng quân đó là nền).
    assert counts == {"train": 1, "valid": 1, "test": 0, "skipped_incomplete": 1}
    lines = (out / "train" / "labels" / "G000_IMG001.txt").read_text().splitlines()
    assert lines == [
        f"{PIECE_CLASSES.index('P')} 0.125000 0.400000 0.150000 0.400000",
        f"{PIECE_CLASSES.index('k')} 0.625000 0.250000 0.250000 0.500000",
    ]
    assert (out / "train" / "images" / "G000_IMG001.jpg").exists()
    assert yaml.safe_load((out / "data.yaml").read_text())["names"] == list(PIECE_CLASSES)


def test_without_2k_split_uses_bbox_images(tmp_path):
    src, out = tmp_path / "chessred", tmp_path / "out"
    make_chessred(src, ["white-pawn", "black-king", "empty"], with_2k_split=False)
    counts = convert_chessred(src, out)
    assert counts["train"] == 1 and counts["valid"] == 1


def test_finds_images_moved_by_unzip(tmp_path):
    """Giải nén có thể thêm một tầng thư mục: tìm ảnh theo tên file."""
    src, out = tmp_path / "chessred", tmp_path / "out"
    make_chessred(src, ["white-pawn", "black-king", "empty"])
    (src / "images").rename(src / "nested")
    (src / "chessred2k").mkdir()
    (src / "nested").rename(src / "chessred2k" / "images")
    assert convert_chessred(src, out)["train"] == 1


def test_missing_annotations_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="annotations.json"):
        convert_chessred(tmp_path, tmp_path / "out")


# ---------------------------------------------------------------------------
# Phát lại ván ChessReD
# ---------------------------------------------------------------------------

from chess_assistant.chessred import load_games, moves_between  # noqa: E402

import chess  # noqa: E402

NAMES = ["white-pawn", "white-rook", "white-knight", "white-bishop", "white-queen", "white-king",
         "black-pawn", "black-rook", "black-knight", "black-bishop", "black-queen", "black-king", "empty"]
SYM = {n: i for i, n in enumerate(NAMES)}


def board_pieces(board: chess.Board, image_id: int) -> list[dict]:
    word = {chess.PAWN: "pawn", chess.ROOK: "rook", chess.KNIGHT: "knight",
            chess.BISHOP: "bishop", chess.QUEEN: "queen", chess.KING: "king"}
    return [{"image_id": image_id, "chessboard_position": chess.square_name(sq),
             "category_id": SYM[("white-" if p.color else "black-") + word[p.piece_type]]}
            for sq, p in board.piece_map().items()]


def make_game(root: Path, game_id: int, ucis: list[str], start_id: int) -> list[int]:
    board, ids, images, pieces = chess.Board(), [], [], []
    for move_id in range(len(ucis) + 1):
        if move_id:
            board.push_uci(ucis[move_id - 1])
        iid = start_id + move_id
        rel = f"images/{game_id}/G{game_id:03d}_IMG{move_id:03d}.jpg"
        write_image(root / rel)
        images.append({"id": iid, "path": rel, "game_id": game_id, "move_id": move_id})
        pieces += board_pieces(board, iid)
        ids.append(iid)
    return ids, images, pieces


def write_games(root: Path) -> None:
    ids_a, img_a, pc_a = make_game(root, 7, ["e2e4", "e7e5", "g1f3"], 0)
    ids_b, img_b, pc_b = make_game(root, 9, ["d2d4"], 100)
    data = {"images": list(reversed(img_a)) + img_b,  # thứ tự trong file không theo nước đi
            "categories": [{"id": i, "name": n} for i, n in enumerate(NAMES)],
            "annotations": {"pieces": pc_a + pc_b, "corners": []},
            "splits": {"chessred2k": {"test": {"image_ids": ids_a + ids_b, "n_samples": 6}}}}
    (root / "annotations.json").write_text(json.dumps(data))


def test_load_games_orders_frames_and_builds_positions(tmp_path):
    write_games(tmp_path)
    games = load_games(tmp_path, split="test")
    assert sorted(games) == [7, 9]
    frames = games[7]
    assert [f.move_id for f in frames] == [0, 1, 2, 3]
    assert frames[0].board_fen == chess.STARTING_BOARD_FEN
    assert frames[1].board_fen == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    assert frames[3].path.name == "G007_IMG003.jpg" and frames[3].path.exists()


def test_moves_between_recovers_moves():
    b, fens = chess.Board(), [chess.STARTING_BOARD_FEN]
    for u in ["e2e4", "d7d5", "e4d5", "d8d5", "e1e2"]:
        b.push_uci(u)
        fens.append(b.board_fen())
    fens.insert(3, fens[2])  # ảnh chụp lặp cùng thế cờ
    assert [m.uci() for m in moves_between(fens)] == ["e2e4", "d7d5", "e4d5", "d8d5", "e1e2"]


def test_moves_between_handles_one_missing_photo():
    b, fens = chess.Board(), [chess.STARTING_BOARD_FEN]
    for u in ["e2e4", "e7e5", "g1f3"]:
        b.push_uci(u)
        fens.append(b.board_fen())
    del fens[2]  # thiếu ảnh sau e7e5
    assert [m.uci() for m in moves_between(fens)] == ["e2e4", "e7e5", "g1f3"]
