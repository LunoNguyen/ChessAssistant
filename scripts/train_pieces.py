"""Train YOLO nhận diện 12 loại quân cờ rồi chép trọng số tốt nhất sang models/pieces.pt.

    python -m scripts.train_pieces --data data/pieces/data.yaml
    python -m scripts.train_pieces --data data/pieces/data.yaml --model models/pieces.pt --epochs 50   # fine-tune

Kiểm tra trong log dòng "albumentations:" có Blur và ImageCompression (mô phỏng camera điện thoại).
"""

import argparse
import shutil
import sys
from pathlib import Path

from chess_assistant.pieces import MODELS_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/pieces/data.yaml"))
    ap.add_argument("--model", default="yolo11s.pt", help="Trọng số khởi đầu")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    if not args.data.exists():
        sys.exit(
            f"Chưa có dataset {args.data}.\n"
            "  1. Tải dataset quân cờ định dạng YOLOv8 (Roboflow/Kaggle), giải nén vào data/public/<tên>/\n"
            '  2. python -m scripts.prepare_dataset --sources "data/public/*" --out data/pieces\n'
            "  3. Chạy lại lệnh train này."
        )

    from ultralytics import YOLO

    results = YOLO(args.model).train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        close_mosaic=10,
        project="runs/pieces",
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy2(best, MODELS_DIR / "pieces.pt")
    print(f"Đã chép {best} -> {MODELS_DIR / 'pieces.pt'}")


if __name__ == "__main__":
    main()
