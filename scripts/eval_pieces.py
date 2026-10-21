"""Đánh giá model nhận diện quân.

    python -m scripts.eval_pieces --data data/pieces/data.yaml --captured data/captured/<bộ holdout>

1. mAP50 theo lớp trên tập test của dataset (Ultralytics val).
2. Tỉ lệ ô đúng sau khi quy về ô, trên ảnh tự thu của bộ cờ để riêng (thế cờ lấy từ sessions/*.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import cv2
import numpy as np

from chess_assistant.board import BoardView
from chess_assistant.detect import detect_board
from chess_assistant.evaluation import square_accuracy
from chess_assistant.pieces import MODELS_DIR, PieceDetector
from chess_assistant.reading import read_board


def captured_square_accuracy(captured: Path, detector: PieceDetector) -> tuple[float, int, int]:
    """(độ đúng ô trung bình, số ảnh đánh giá được, số ảnh không tìm thấy bàn cờ)."""
    fens, per_image = {}, {}
    for p in (captured / "sessions").glob("*.json"):
        meta = json.loads(p.read_text(encoding="utf-8"))
        fens[p.stem] = meta["fen"]
        per_image.update(meta.get("images", {}))  # capture_geometric --pgn: mỗi ảnh một thế
    scores, no_board = [], 0
    for img_path in sorted((captured / "train" / "images").glob("*.jpg")):
        session = img_path.stem.rsplit("_", 1)[0]
        fen = per_image.get(img_path.stem, fens.get(session))
        if fen is None:
            continue
        frame = cv2.imread(str(img_path))
        found = detect_board(frame)
        if found is None:
            no_board += 1
            continue
        board = chess.Board(fen + " w - - 0 1")
        reading = read_board(detector(frame), BoardView.from_corners(found.corners).H)
        # Ảnh tĩnh không biết phía Trắng: lấy hướng cho kết quả tốt hơn.
        scores.append(max(square_accuracy(reading, board), square_accuracy(reading[::-1], board)))
    return (float(np.mean(scores)) if scores else 0.0), len(scores), no_board


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/pieces/data.yaml"))
    ap.add_argument("--captured", type=Path, help="Thư mục một bộ cờ tự thu (data/captured/<set>)")
    ap.add_argument("--weights", type=Path, default=MODELS_DIR / "pieces.pt")
    args = ap.parse_args()

    detector = PieceDetector(args.weights)
    if args.data.exists():
        metrics = detector.model.val(data=str(args.data), split="test", verbose=False)
        names = detector.model.names
        print(f"mAP50 (test): {metrics.box.map50:.3f}")
        for i, ap50 in zip(metrics.box.ap_class_index, metrics.box.ap50):
            print(f"  {names[int(i)]}: {ap50:.3f}")
    if args.captured:
        acc, n, missing = captured_square_accuracy(args.captured, detector)
        print(f"Tỉ lệ ô đúng trên {args.captured.name}: {acc:.1%} ({n} ảnh, {missing} ảnh không thấy bàn cờ)")


if __name__ == "__main__":
    main()
