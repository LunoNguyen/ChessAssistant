"""Gộp các dataset YOLO về 12 lớp FEN để train nhận diện quân.

    python -m scripts.prepare_dataset --sources data/public/* data/captured/* --out data/pieces --holdout setX

Mỗi nguồn là một thư mục YOLOv8 (data.yaml + train/valid/test). Nguồn có tên trong --holdout
được đưa hết vào tập test (bộ cờ "chưa từng thấy" để đánh giá).
"""

import argparse
import glob
from pathlib import Path

from chess_assistant.dataset import build_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", required=True, help="Thư mục nguồn (hỗ trợ glob)")
    ap.add_argument("--out", type=Path, default=Path("data/pieces"))
    ap.add_argument("--holdout", nargs="*", default=[], help="Tên nguồn dành riêng cho test")
    args = ap.parse_args()

    sources = sorted({Path(p) for pattern in args.sources for p in glob.glob(pattern) if Path(p, "data.yaml").exists()})
    if not sources:
        raise SystemExit("Không thấy nguồn nào có data.yaml")
    print("Nguồn:", ", ".join(s.name for s in sources))
    counts = build_dataset(sources, args.out, set(args.holdout))
    print(f"Đã ghi {args.out}: train={counts['train']} val={counts['val']} test={counts['test']}")


if __name__ == "__main__":
    main()
