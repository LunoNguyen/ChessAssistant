"""Đóng gói dữ liệu train FOMO (tâm quân + 4 góc bàn cờ) để chạy notebooks/train_fomo_colab.ipynb.

    python -m scripts.prepare_fomo
    python -m scripts.prepare_fomo --max-side 384 --out colab_upload

Đọc thẳng các nguồn đã có (không cần data/pieces_v4):
- ChessReD2K (data/public/chessred2k + data/downloads/chessred/annotations.json): góc có sẵn trong
  annotations, tâm quân = tâm ô chiếu theo góc (đúng chỗ đế quân).
- Cờ 2D (data/public/chess2d + tên file FEN trong data/downloads/chess_positions/train): góc suy ra
  bằng cách khớp phép chiếu từ khung quân về ô cờ.
- Ảnh tự chụp (data/captured/game3d): tâm = điểm chân quân của khung YOLO, góc dò bằng OpenCV.

Kết quả: colab_upload/fomo_dataset.zip gồm images/{train,val,test}/*.jpg (cạnh dài <= --max-side),
labels.json và fomo.py (notebook dùng chung code với chess_assistant/fomo.py).
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import chess
import cv2
import numpy as np

from chess_assistant.dataset import normalize_class_name
from chess_assistant.fomo import CORNER, SCHEMES, order_corners, solve_homography, square_centers
from chess_assistant.schema import PIECE_CLASSES, class_of
from chess_assistant.screen2d import board_from_filename

ROOT = Path(__file__).resolve().parent.parent
SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "test"}
CORNER_KEYS = ("bottom_left", "bottom_right", "top_right", "top_left")  # = a1, h1, h8, a8 (đã kiểm tra trên ChessReD)

Record = dict  # {"file", "split", "source", "points": [[cls, x, y], ...]}


def _corner_points(corners_px: np.ndarray, w: int, h: int) -> list[list]:
    return [[CORNER, float(x) / w, float(y) / h] for x, y in corners_px]


def _square_points(board_squares: dict[int, int], corners_px: np.ndarray, w: int, h: int) -> list[list]:
    centers = square_centers(corners_px)
    return [[cls, float(centers[sq][0]) / w, float(centers[sq][1]) / h] for sq, cls in sorted(board_squares.items())]


def _yolo_boxes(label_file: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append((int(parts[0]), *map(float, parts[1:])))
    return rows


# ---------------------------------------------------------------------------
# Từng nguồn: trả về (đường dẫn ảnh, split, điểm chuẩn hoá theo ảnh gốc)
# ---------------------------------------------------------------------------

def chessred_items(public_dir: Path, annotations: Path):
    data = json.loads(annotations.read_text(encoding="utf-8"))
    names = {}
    for cat in data["categories"]:
        fen = normalize_class_name(str(cat["name"]))
        names[cat["id"]] = PIECE_CLASSES.index(fen) if fen and len(fen) == 1 else None
    images = {img["file_name"]: img for img in data["images"]}
    corners = {c["image_id"]: c["corners"] for c in data["annotations"]["corners"]}
    pieces: dict[int, dict[int, int]] = {}
    for p in data["annotations"]["pieces"]:
        cls = names.get(p["category_id"])
        if cls is not None:
            pieces.setdefault(p["image_id"], {})[chess.parse_square(p["chessboard_position"])] = cls
    for split_dir in sorted(public_dir.iterdir()):
        if split_dir.name not in SPLITS:
            continue
        for img_path in sorted((split_dir / "images").glob("*.jpg")):
            info = images.get(img_path.name)
            if info is None or info["id"] not in corners:
                continue
            c = corners[info["id"]]
            px = np.array([c[k] for k in CORNER_KEYS], np.float64)
            w, h = info["width"], info["height"]
            pts = _corner_points(px, w, h) + _square_points(pieces.get(info["id"], {}), px, w, h)
            yield img_path, SPLITS[split_dir.name], pts


FLAT_CORNERS = np.array([[0, 8], [8, 8], [8, 0], [0, 0]], np.float64)  # a1, h1, h8, a8 trên ảnh 2D (đơn vị: ô)


def flat_fit(board: chess.Board, boxes: list[tuple]) -> np.ndarray | None:
    """4 góc (chuẩn hoá theo ảnh) của bàn 2D đã biến đổi, khớp từ tâm khung quân về tâm ô.

    import_chess_positions ghi khung theo thứ tự ô tăng dần, nên khung thứ k là quân thứ k."""
    pieces = sorted(board.piece_map().items())
    if len(pieces) != len(boxes) or len(boxes) < 5:
        return None
    if any(class_of(p) != b[0] for (_, p), b in zip(pieces, boxes)):
        return None
    src = np.array([(chess.square_file(sq) + 0.5, 7 - chess.square_rank(sq) + 0.5) for sq, _ in pieces], np.float64)
    dst = np.array([(b[1], b[2]) for b in boxes], np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        return None
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    if np.abs(proj - dst).max() > 0.01:  # sai > 1% ảnh: khung bị cắt mép hoặc không khớp
        return None
    corners = cv2.perspectiveTransform(FLAT_CORNERS.reshape(-1, 1, 2), H).reshape(4, 2)
    if cv2.contourArea(corners.astype(np.float32)) < 0.01 or not cv2.isContourConvex(corners.astype(np.float32)):
        return None  # quân thẳng hàng: phép chiếu suy biến (bàn cờ co về một điểm)
    return corners


def chess2d_items(public_dir: Path, src_dir: Path):
    sources = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for split_dir in sorted(public_dir.iterdir()):
        if split_dir.name not in SPLITS:
            continue
        for img_path in sorted((split_dir / "images").glob("c2d_*.jpg")):
            idx = int(img_path.stem.split("_")[1])
            label = split_dir / "labels" / f"{img_path.stem}.txt"
            if idx >= len(sources) or not label.exists():
                continue
            board = board_from_filename(sources[idx].name)
            corners = flat_fit(board, _yolo_boxes(label))
            if corners is None:
                continue
            squares = {sq: class_of(p) for sq, p in board.piece_map().items()}
            pts = _corner_points(corners, 1, 1) + _square_points(squares, corners, 1, 1)
            yield img_path, SPLITS[split_dir.name], pts


def captured_items(captured_dir: Path, foot_lift: float = 0.10):
    from chess_assistant.detect import detect_board

    for split_dir in sorted(captured_dir.iterdir()):
        if split_dir.name not in SPLITS:
            continue
        for img_path in sorted((split_dir / "images").glob("*.jpg")):
            label = split_dir / "labels" / f"{img_path.stem}.txt"
            frame = cv2.imread(str(img_path))
            found = detect_board(frame) if frame is not None and label.exists() else None
            if found is None:
                continue
            h, w = frame.shape[:2]
            corners = order_corners(np.asarray(found.corners, np.float64).reshape(4, 2))
            pts = _corner_points(corners, w, h)
            for cls, cx, cy, bw, bh in _yolo_boxes(label):
                pts.append([cls, cx, cy + bh / 2 - foot_lift * bh])
            yield img_path, SPLITS[split_dir.name], pts


# ---------------------------------------------------------------------------

def _small_jpeg(path: Path, max_side: int) -> bytes | None:
    probe = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)
    if probe is None:
        return None
    img = probe
    if max(probe.shape[:2]) >= 4 * max_side:  # ảnh rất lớn (ChessReD 3072px): đọc lại ở 1/4 cho nhanh
        img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else None


def build(sources: dict[str, object], out_zip: Path, max_side: int = 384, limit: int | None = None) -> Counter:
    """sources: {tên nguồn: iterator (ảnh, split, điểm)}. Ghi zip, trả về số ảnh theo (nguồn, split)."""
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    records: list[Record] = []
    counts: Counter = Counter()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as z:
        for name, items in sources.items():
            n = 0
            for img_path, split, pts in items:
                if limit is not None and n >= limit:
                    break
                data = _small_jpeg(img_path, max_side)
                if data is None:
                    counts[(name, "unreadable")] += 1
                    continue
                arc = f"images/{split}/{name}_{img_path.stem}.jpg"
                z.writestr(arc, data)
                records.append({"file": arc, "split": split, "source": name,
                                "points": [[int(c), round(x, 5), round(y, 5)] for c, x, y in pts]})
                counts[(name, split)] += 1
                n += 1
                if n % 1000 == 0:
                    print(f"  {name}: {n} ảnh", flush=True)
        meta = {"version": 1, "point_classes": list(PIECE_CLASSES) + ["corner"], "corner_class": CORNER,
                "schemes": {k: list(v) for k, v in SCHEMES.items()}, "records": records}
        z.writestr("labels.json", json.dumps(meta))
        z.write(ROOT / "chess_assistant" / "fomo.py", "fomo.py")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "colab_upload")
    ap.add_argument("--max-side", type=int, default=384, help="Cạnh dài ảnh trong gói (model chỉ dùng 160-192px)")
    ap.add_argument("--limit", type=int, help="Mỗi nguồn tối đa N ảnh (thử nhanh)")
    ap.add_argument("--chessred", type=Path, default=ROOT / "data/public/chessred2k")
    ap.add_argument("--chessred-ann", type=Path, default=ROOT / "data/downloads/chessred/annotations.json")
    ap.add_argument("--chess2d", type=Path, default=ROOT / "data/public/chess2d")
    ap.add_argument("--chess2d-src", type=Path, default=ROOT / "data/downloads/chess_positions/train")
    ap.add_argument("--captured", type=Path, nargs="*", default=[ROOT / "data/captured/game3d"])
    args = ap.parse_args()

    sources: dict[str, object] = {}
    if args.chessred.is_dir() and args.chessred_ann.exists():
        sources["chessred2k"] = chessred_items(args.chessred, args.chessred_ann)
    else:
        print(f"Bỏ qua ChessReD: cần {args.chessred} và {args.chessred_ann}")
    if args.chess2d.is_dir() and args.chess2d_src.is_dir():
        sources["chess2d"] = chess2d_items(args.chess2d, args.chess2d_src)
    else:
        print(f"Bỏ qua cờ 2D: cần {args.chess2d} và {args.chess2d_src} (tên file FEN gốc)")
    for d in args.captured:
        if d.is_dir():
            sources[d.name] = captured_items(d)
    if not sources:
        raise SystemExit("Lỗi: không có nguồn dữ liệu nào.")

    out_zip = args.out / "fomo_dataset.zip"
    counts = build(sources, out_zip, args.max_side, args.limit)
    for name in sources:
        row = {s: counts[(name, s)] for s in ("train", "val", "test")}
        print(f"{name:12s} train={row['train']:5d} val={row['val']:4d} test={row['test']:4d}")
    print(f"Đã tạo {out_zip} ({out_zip.stat().st_size / 1e6:.0f} MB)")
    notebook = ROOT / "notebooks" / "train_fomo_colab.ipynb"
    if notebook.exists():
        shutil.copy2(notebook, args.out / notebook.name)
        print(f"Đã chép notebook -> {args.out / notebook.name}")
    print("Tiếp theo: tải fomo_dataset.zip lên Google Drive vào MyDrive/ChessAssistant/, "
          "mở notebooks/train_fomo_colab.ipynb trên Colab.")


if __name__ == "__main__":
    main()
