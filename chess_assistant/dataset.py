"""Gộp các dataset YOLO (Roboflow/Kaggle/tự thu) về 12 lớp FEN.

Mỗi nguồn là một thư mục YOLO: data.yaml (có `names`) và các thư mục
train|valid|val|test/{images,labels}.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Literal

import yaml

from chess_assistant.schema import PIECE_CLASSES

PIECE_WORDS = {"pawn": "p", "knight": "n", "bishop": "b", "rook": "r", "queen": "q", "king": "k"}
IGNORED = {"board", "chessboard"}
SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "test"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _with_color(letter: str, color: str) -> str:
    return letter.upper() if color == "w" else letter


def normalize_class_name(name: str) -> str | None | Literal["ignore"]:
    """Tên lớp bất kỳ -> ký hiệu FEN; "ignore" cho lớp bàn cờ; None nếu không xác định được."""
    if name in PIECE_CLASSES:
        return name
    tokens = [t for t in re.split(r"[-_\s]+", name.strip().lower()) if t]
    joined = "".join(tokens)
    if joined in IGNORED:
        return "ignore"

    color = None
    if "white" in tokens:
        color = "w"
    elif "black" in tokens:
        color = "b"
    piece = next((PIECE_WORDS[t] for t in tokens if t in PIECE_WORDS), None)
    if piece is not None:
        return _with_color(piece, color) if color else None

    # Dạng viết tắt: "wp", "bk", "w_p"...
    if len(joined) == 2 and joined[0] in "wb" and joined[1] in "pnbrqk":
        return _with_color(joined[1], joined[0])
    return None


def remap_label_file(lines: list[str], src_names: list[str]) -> list[str] | None:
    """Đổi chỉ số lớp sang PIECE_CLASSES; bỏ dòng lớp bàn cờ; lớp không rõ -> None (bỏ cả ảnh)."""
    out = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        target = normalize_class_name(src_names[int(parts[0])])
        if target == "ignore":
            continue
        if target is None:
            return None
        out.append(" ".join([str(PIECE_CLASSES.index(target)), *parts[1:]]))
    return out


def _names(src: Path) -> list[str]:
    names = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def build_dataset(sources: list[Path], out_dir: Path, holdout: set[str]) -> dict[str, int]:
    """Gộp các nguồn vào out_dir/{train,val,test}. Nguồn có tên trong `holdout` vào hết `test`."""
    counts = {"train": 0, "val": 0, "test": 0}
    for split in counts:
        for sub in ("images", "labels"):
            (out_dir / split / sub).mkdir(parents=True, exist_ok=True)

    for src in map(Path, sources):
        names = _names(src)
        for src_split, split in SPLITS.items():
            img_dir = src / src_split / "images"
            if not img_dir.is_dir():
                continue
            dest = "test" if src.name in holdout else split
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMAGE_EXTS:
                    continue
                label = src / src_split / "labels" / f"{img.stem}.txt"
                lines = label.read_text(encoding="utf-8").splitlines() if label.exists() else []
                remapped = remap_label_file(lines, names)
                if remapped is None:
                    continue
                stem = f"{src.name}_{src_split}_{img.stem}"
                shutil.copy2(img, out_dir / dest / "images" / f"{stem}{img.suffix}")
                (out_dir / dest / "labels" / f"{stem}.txt").write_text(
                    "\n".join(remapped) + ("\n" if remapped else ""), encoding="utf-8"
                )
                counts[dest] += 1

    data = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": dict(enumerate(PIECE_CLASSES)),
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return counts
