"""Nhập ChessReD2K (ảnh chụp bằng điện thoại, CC BY-NC-SA 4.0) thành nguồn YOLO cho prepare_dataset.

Tải từ https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f :
  - annotations.json (~22 MB)
  - chessred2k.zip   (~4.7 GB)
Đặt annotations.json và giải nén chessred2k.zip vào cùng thư mục data/downloads/chessred/, rồi:

    python -m scripts.import_chessred
    python -m scripts.import_chessred --src D:/Downloads/chessred --out data/public/chessred2k
"""

import argparse
from pathlib import Path

from chess_assistant.chessred import convert_chessred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/downloads/chessred"))
    ap.add_argument("--out", type=Path, default=Path("data/public/chessred2k"))
    args = ap.parse_args()

    try:
        counts = convert_chessred(args.src, args.out)
    except FileNotFoundError as exc:
        raise SystemExit(f"Lỗi: {exc}")
    print(f"Đã ghi {args.out}: train={counts['train']} valid={counts['valid']} test={counts['test']}"
          f" (bỏ {counts['skipped_incomplete']} ảnh thiếu khung quân)")
    if counts["train"] == 0:
        raise SystemExit("Không có ảnh nào có khung quân. Kiểm tra lại đã giải nén chessred2k.zip (không phải chessred.zip) chưa.")
    print('Tiếp theo: python -m scripts.prepare_dataset --sources "data/public/*" --out data/pieces')


if __name__ == "__main__":
    main()
