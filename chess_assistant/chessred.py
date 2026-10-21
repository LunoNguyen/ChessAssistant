"""Chuyển ChessReD2K (ảnh chụp bằng điện thoại, có khung quân) sang thư mục YOLO.

Nguồn: https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f (CC BY-NC-SA 4.0).
Chỉ phần ChessReD2K có khung quân (bbox, dạng COCO [x, y, w, h] pixel). Ảnh mà có quân
thiếu khung bị bỏ: nhãn thiếu sẽ dạy model rằng quân đó là nền.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import chess
import cv2
import yaml

from chess_assistant.dataset import normalize_class_name
from chess_assistant.schema import PIECE_CLASSES

OUT_SPLITS = {"train": "train", "val": "valid", "test": "test"}


def _place(src: Path, dst: Path) -> None:
    """Hard link nếu được (không tốn thêm dung lượng), không thì chép."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class _ImageFinder:
    """Tìm ảnh theo `path` trong annotations; nếu giải nén tạo cấu trúc khác thì tìm theo tên file."""

    def __init__(self, root: Path):
        self.root = root
        self._by_name: dict[str, Path] | None = None

    def __call__(self, rel: str) -> Path | None:
        direct = self.root / rel
        if direct.exists():
            return direct
        if self._by_name is None:
            self._by_name = {p.name: p for p in self.root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        return self._by_name.get(Path(rel).name)


def convert_chessred(root: Path, out_dir: Path) -> dict[str, int]:
    """root chứa annotations.json và ảnh đã giải nén. Ghi out_dir/{train,valid,test}/{images,labels} + data.yaml."""
    root, out_dir = Path(root), Path(out_dir)
    ann_path = root / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Không thấy {ann_path}. Tải annotations.json từ trang 4TU và đặt vào {root}")
    data = json.loads(ann_path.read_text(encoding="utf-8"))

    to_class: dict[int, int | None] = {}
    for cat in data["categories"]:
        fen = normalize_class_name(str(cat["name"]))
        to_class[cat["id"]] = PIECE_CLASSES.index(fen) if fen and len(fen) == 1 else None  # "empty" -> None

    pieces = data["annotations"]["pieces"] if isinstance(data["annotations"], dict) else data["annotations"]
    by_image: dict[int, list[dict]] = defaultdict(list)
    for p in pieces:
        if to_class.get(p["category_id"]) is not None:
            by_image[p["image_id"]].append(p)
    has_bbox = {i for i, ps in by_image.items() if any(p.get("bbox") for p in ps)}

    splits = data.get("splits", {})
    source_splits = splits.get("chessred2k", splits)
    images = {img["id"]: img for img in data["images"]}
    find = _ImageFinder(root)

    counts = {"train": 0, "valid": 0, "test": 0, "skipped_incomplete": 0}
    for split, out_split in OUT_SPLITS.items():
        (out_dir / out_split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / out_split / "labels").mkdir(parents=True, exist_ok=True)
        for image_id in source_splits.get(split, {}).get("image_ids", []):
            if image_id not in has_bbox:
                continue
            ps = by_image[image_id]
            if not all(p.get("bbox") for p in ps):
                counts["skipped_incomplete"] += 1
                continue
            info = images[image_id]
            src = find(info["path"])
            if src is None:
                raise FileNotFoundError(f"Không thấy ảnh {info['path']} trong {root}. Đã giải nén chessred2k.zip vào đây chưa?")
            w, h = info.get("width"), info.get("height")
            if not (w and h):
                h, w = cv2.imread(str(src)).shape[:2]
            lines = []
            for p in ps:
                x, y, bw, bh = p["bbox"]
                lines.append(f"{to_class[p['category_id']]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
            _place(src, out_dir / out_split / "images" / src.name)
            (out_dir / out_split / "labels" / f"{src.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts[out_split] += 1

    (out_dir / "data.yaml").write_text(yaml.safe_dump({"names": list(PIECE_CLASSES)}), encoding="utf-8")
    return counts


# ---------------------------------------------------------------------------
# Phát lại các ván ChessReD: mỗi ảnh là thế cờ sau một nước đi
# ---------------------------------------------------------------------------

@dataclass
class GameFrame:
    path: Path
    board_fen: str   # phần xếp quân của FEN
    move_id: int


def load_games(root: Path, split: str = "test", subset: str | None = "chessred2k") -> dict[int, list[GameFrame]]:
    """{game_id: các ảnh theo thứ tự nước đi} cho một split của ChessReD (mặc định: test của ChessReD2K)."""
    root = Path(root)
    data = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    splits = data["splits"]
    if subset and subset in splits:
        splits = splits[subset]
    wanted = set(splits[split]["image_ids"])

    names = {c["id"]: normalize_class_name(str(c["name"])) for c in data["categories"]}
    boards: dict[int, chess.Board] = defaultdict(lambda: chess.Board(None))
    pieces = data["annotations"]["pieces"] if isinstance(data["annotations"], dict) else data["annotations"]
    for p in pieces:
        fen = names.get(p["category_id"])
        if p["image_id"] in wanted and fen and len(fen) == 1:
            boards[p["image_id"]].set_piece_at(chess.parse_square(p["chessboard_position"]), chess.Piece.from_symbol(fen))

    find = _ImageFinder(root)
    games: dict[int, list[GameFrame]] = defaultdict(list)
    for img in data["images"]:
        if img["id"] in wanted:
            path = find(img["path"])
            if path is None:
                raise FileNotFoundError(f"Không thấy ảnh {img['path']} trong {root}")
            games[img["game_id"]].append(GameFrame(path, boards[img["id"]].board_fen(), img["move_id"]))
    return {gid: sorted(frames, key=lambda f: f.move_id) for gid, frames in sorted(games.items())}


def moves_between(board_fens: list[str]) -> list[chess.Move]:
    """Chuỗi nước đi thật từ các thế cờ liên tiếp (bỏ ảnh lặp; chịu được thiếu 1 ảnh giữa hai nước)."""
    board = chess.Board(board_fens[0] + " w KQkq - 0 1")
    board.castling_rights = board.clean_castling_rights()
    moves: list[chess.Move] = []
    for target in board_fens[1:]:
        if target == board.board_fen():
            continue
        found = _find_moves(board, target, depth=1) or _find_moves(board, target, depth=2)
        if found is None:
            raise ValueError(f"Không nối được thế {board.board_fen()} -> {target} bằng 1-2 nước hợp lệ")
        for m in found:
            board.push(m)
        moves += found
    return moves


def _find_moves(board: chess.Board, target: str, depth: int) -> list[chess.Move] | None:
    for m in board.legal_moves:
        board.push(m)
        try:
            if depth == 1 and board.board_fen() == target:
                return [m]
            if depth == 2:
                rest = _find_moves(board, target, 1)
                if rest:
                    return [m, *rest]
        finally:
            board.pop()
    return None
