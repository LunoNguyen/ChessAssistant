"""Đóng gói dataset (+ model hiện tại) để train trên Google Colab thay vì máy local.

    python -m scripts.export_colab --dataset data/pieces_v2

Tạo thư mục colab_upload/ gồm:
    pieces_dataset.zip   dataset YOLO (data.yaml dùng đường dẫn tương đối)
    pieces.pt            model hiện tại (để fine-tune tiếp, không train lại từ đầu)
Tải cả thư mục lên Google Drive tại MyDrive/ChessAssistant/, rồi mở notebooks/train_colab.ipynb trên Colab.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import cv2
import yaml

from chess_assistant.pieces import MODELS_DIR

SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _image_bytes(path: Path, max_side: int | None) -> bytes | None:
    """Ảnh đã thu nhỏ để cạnh dài nhất = max_side (JPEG); None nếu giữ nguyên file gốc."""
    if not max_side or path.suffix.lower() not in IMAGE_EXTS:
        return None
    img = cv2.imread(str(path))
    if img is None or max(img.shape[:2]) <= max_side:
        return None
    s = max_side / max(img.shape[:2])
    small = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else None


def export_dataset(dataset: Path, out_zip: Path, max_side: int | None = 1280) -> Path:
    """Nén dataset YOLO thành zip; data.yaml được viết lại với path tương đối để dùng trên Colab.

    Ảnh lớn hơn `max_side` được thu nhỏ (YOLO train ở 640px nên ảnh 3072px chỉ làm gói nặng thêm).
    Nhãn YOLO dùng toạ độ tỉ lệ nên giữ nguyên."""
    dataset, out_zip = Path(dataset), Path(out_zip)
    data_yaml = dataset / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Không thấy {data_yaml}. Gộp dữ liệu trước: python -m scripts.prepare_dataset --sources ... --out {dataset}"
        )
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    data["path"] = "."
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as z:  # ảnh jpg đã nén sẵn
        z.writestr("data.yaml", yaml.safe_dump(data, allow_unicode=True))
        for split in SPLITS:
            for sub in ("images", "labels"):
                folder = dataset / split / sub
                if folder.is_dir():
                    for f in sorted(folder.iterdir()):
                        small = _image_bytes(f, max_side) if sub == "images" else None
                        if small is None:
                            z.write(f, f"{split}/{sub}/{f.name}")
                        else:
                            z.writestr(f"{split}/{sub}/{f.name}", small)
    return out_zip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("data/pieces"))
    ap.add_argument("--out", type=Path, default=Path("colab_upload"))
    ap.add_argument("--max-side", type=int, default=1280, help="Thu nhỏ ảnh về cạnh dài này (0 = giữ nguyên)")
    args = ap.parse_args()

    try:
        zip_path = export_dataset(args.dataset, args.out / "pieces_dataset.zip", args.max_side or None)
    except FileNotFoundError as exc:
        raise SystemExit(f"Lỗi: {exc}")
    print(f"Đã tạo {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)")
    model = MODELS_DIR / "pieces.pt"
    if model.exists():
        shutil.copy2(model, args.out / "pieces.pt")
        print(f"Đã chép model hiện tại -> {args.out / 'pieces.pt'}")
    notebook = Path(__file__).resolve().parent.parent / "notebooks" / "train_colab.ipynb"
    if notebook.exists():
        shutil.copy2(notebook, args.out / notebook.name)
        print(f"Đã chép notebook -> {args.out / notebook.name}")
    print("Tiếp theo: tải thư mục này lên Google Drive tại MyDrive/ChessAssistant/, "
          "mở notebooks/train_colab.ipynb trên colab.research.google.com.")


if __name__ == "__main__":
    main()
