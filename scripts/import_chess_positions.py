"""Chuyển dataset Chess Positions (bàn cờ 2D, CC0) sang YOLO, kèm biến đổi giống camera chụp màn hình.

Nguồn: koryakinp/chess-positions (Kaggle, CC0) — bản mirror smallchess/kaggle-chess-positions trên Hugging Face.

    python -m scripts.import_chess_positions
    python -m scripts.import_chess_positions --src data/downloads/chess_positions/train --variants 2 --limit 4000

Mỗi ảnh gốc sinh `--variants` ảnh "điện thoại chụp màn hình" (nghiêng, moiré, chói, mờ, nhiễu, JPEG).
Kết quả: data/public/chess2d/{train,valid,test}/{images,labels} + data.yaml — nguồn cho prepare_dataset.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import yaml

from chess_assistant.autolabel import to_yolo_lines
from chess_assistant.schema import PIECE_CLASSES
from chess_assistant.screen2d import board_from_filename, flat_boxes, simulate_screen_photo


def _split(name: str, val: float, test: float) -> str:
    """Chia cố định theo tên file (chạy lại vẫn ra cùng kết quả)."""
    x = int(hashlib.md5(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if x < test else "valid" if x < test + val else "train"


def convert(src: Path, out: Path, variants: int = 1, limit: int | None = None,
            val: float = 0.1, test: float = 0.1, seed: int = 0) -> dict[str, int]:
    files = sorted(p for p in Path(src).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:limit]
    if not files:
        raise FileNotFoundError(f"Không có ảnh trong {src}")
    for split in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            (out / split / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    counts = {"train": 0, "valid": 0, "test": 0, "skipped": 0}
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        try:
            board = board_from_filename(f.name)
        except ValueError:
            board = None
        if img is None or board is None:
            counts["skipped"] += 1
            continue
        split = _split(f.name, val, test)
        labels = flat_boxes(board, img.shape[1])
        for v in range(variants):
            photo, out_labels = simulate_screen_photo(img, labels, rng)
            stem = f"c2d_{i:05d}_{v}"
            cv2.imwrite(str(out / split / "images" / f"{stem}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 90])
            h, w = photo.shape[:2]
            (out / split / "labels" / f"{stem}.txt").write_text("\n".join(to_yolo_lines(out_labels, w, h)) + "\n")
            counts[split] += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)} ảnh gốc", flush=True)
    (out / "data.yaml").write_text(yaml.safe_dump({"names": list(PIECE_CLASSES)}), encoding="utf-8")
    (out / "SOURCE.md").write_text(
        "Chess Positions — koryakinp (Kaggle), CC0-1.0. Mirror: huggingface.co/datasets/smallchess/kaggle-chess-positions\n"
        "Đã biến đổi thành ảnh mô phỏng điện thoại chụp màn hình bằng scripts/import_chess_positions.py\n",
        encoding="utf-8")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/downloads/chess_positions/train"))
    ap.add_argument("--out", type=Path, default=Path("data/public/chess2d"))
    ap.add_argument("--variants", type=int, default=1, help="Số ảnh mô phỏng cho mỗi ảnh gốc")
    ap.add_argument("--limit", type=int, help="Chỉ dùng N ảnh gốc đầu tiên")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    try:
        counts = convert(args.src, args.out, args.variants, args.limit, seed=args.seed)
    except FileNotFoundError as exc:
        raise SystemExit(f"Lỗi: {exc}")
    print(f"Đã ghi {args.out}: train={counts['train']} valid={counts['valid']} test={counts['test']}"
          f" (bỏ {counts['skipped']} ảnh lỗi)")


if __name__ == "__main__":
    main()
