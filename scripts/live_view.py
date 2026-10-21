"""Xem trực tiếp: tự động tìm và bám theo bàn cờ, vẽ lưới 8x8 và hiện 64 ô đã cắt.

    python -m scripts.live_view                 # tự động dò bàn cờ (mặc định)
    python -m scripts.live_view --camera 1
    python -m scripts.live_view --manual        # dùng 4 góc từ scripts.calibrate
    python -m scripts.live_view --image anh.jpg

Phím: f = lật 180° (khi chương trình đoán nhầm phía Trắng)
      r = dò lại từ đầu
      s = lưu 64 ô vào data/raw/<thời gian>/ (dùng làm dataset tuần 2)
      q / Esc = thoát

Mẹo: bắt đầu với thế cờ khởi đầu để chương trình tự nhận ra phía Trắng.
"""

import argparse
import time
from pathlib import Path

import chess
import cv2
import numpy as np

from chess_assistant.board import BoardView
from chess_assistant.overlay import draw_grid
from chess_assistant.calibration import Calibration
from chess_assistant.camera import open_camera
from chess_assistant.tracker import BoardTracker

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

STATUS_TEXT = {
    "searching": ("Dang tim ban co...", (0, 200, 255)),
    "locked": ("Da khoa ban co", (0, 255, 0)),
    "holding": ("Mat dau - giu vi tri cu", (0, 165, 255)),
    "manual": ("Hieu chinh tay", (255, 255, 0)),
}


def draw_status(img: np.ndarray, status: str) -> None:
    text, color = STATUS_TEXT[status]
    cv2.rectangle(img, (0, 0), (img.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(img, f"{text}   |  f: lat 180  r: do lai  s: luu  q: thoat", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def montage(crops: list[np.ndarray], scale: float = 0.5) -> np.ndarray:
    """Ghép 64 ô thành sơ đồ (Trắng ở dưới), mỗi ô ghi tên."""
    h, w = crops[0].shape[:2]
    h, w = int(h * scale), int(w * scale)
    rows = []
    for rank in range(7, -1, -1):
        row = []
        for file in range(8):
            sq = chess.square(file, rank)
            tile = cv2.resize(crops[sq], (w, h))
            cv2.rectangle(tile, (0, 0), (w - 1, h - 1), (80, 80, 80), 1)
            cv2.putText(tile, chess.square_name(sq), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            row.append(tile)
        rows.append(np.hstack(row))
    return np.vstack(rows)


def save_crops(frame: np.ndarray, crops: list[np.ndarray]) -> Path:
    out = DATA_DIR / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "frame.jpg"), frame)
    for sq, crop in zip(chess.SQUARES, crops):
        cv2.imwrite(str(out / f"{chess.square_name(sq)}.png"), crop)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--manual", action="store_true", help="Dùng 4 góc đã hiệu chỉnh tay thay vì tự dò")
    ap.add_argument("--image", help="Dùng ảnh tĩnh thay cho camera")
    ap.add_argument("--pad", type=float, default=0.75, help="Phần đệm phía đầu quân, tính theo cạnh ô (0..1)")
    args = ap.parse_args()

    tracker = None
    manual_corners = None
    camera, width, height = args.camera, args.width, args.height
    if args.manual:
        cal = Calibration.load()
        manual_corners = np.float32(cal.corners)
        camera, width, height = cal.camera_index, cal.width, cal.height
    else:
        tracker = BoardTracker(background=not args.image)

    cap = None
    still = None
    if args.image:
        still = cv2.imread(args.image)
        if still is None:
            raise SystemExit(f"Không đọc được ảnh {args.image}")
    else:
        cap = open_camera(camera, width, height)

    view = None
    view_corners = None
    try:
        while True:
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Không đọc được khung hình từ camera")
            else:
                frame = still

            if tracker is not None:
                corners, status = tracker.update(frame), tracker.status
            else:
                corners, status = manual_corners, "manual"

            if corners is not None and (view_corners is None or not np.array_equal(corners, view_corners)):
                view, view_corners = BoardView.from_corners(corners), corners.copy()
            elif corners is None:
                view = view_corners = None

            shown = frame.copy()
            if view is not None:
                draw_grid(shown, view)
            draw_status(shown, status)
            cv2.imshow("Camera + luoi", shown)
            crops = None
            if view is not None:
                crops = view.crops(frame, args.pad)
                cv2.imshow("64 o", montage(crops))

            key = cv2.waitKey(1 if cap is not None else 30) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("f") and tracker is not None:
                tracker.flip()
            if key == ord("r") and tracker is not None:
                tracker.reset()
            if key == ord("s") and crops is not None:
                print(f"Đã lưu 64 ô vào {save_crops(frame, crops)}")
    finally:
        if tracker is not None:
            tracker.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
