"""Thu dữ liệu quân cờ KHÔNG cần model: khung quân tính từ vị trí bàn cờ + thế cờ đã biết.

Dùng khi model hiện tại chưa nhận ra kiểu quân của bạn (bộ cờ lạ, game cờ 3D trên màn hình...).

    python -m scripts.capture_geometric --set game3d --camera http://192.168.0.100:4747/video
    python -m scripts.capture_geometric --set game3d --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    python -m scripts.capture_geometric --set game3d --pgn van1.pgn     # n/p để bước qua từng thế của ván

Cách dùng:
  1. Bày đúng thế cờ (mặc định thế khởi đầu). Với --pgn: mở ván đó trong game ở chế độ xem lại,
     đặt game và chương trình cùng một nước, rồi bấm n/p để bước cùng nhau.
  2. Nhìn khung vàng (quân Trắng) / tím (quân Đen) và chữ trên khung: phải khớp quân thật.
     Sai hướng → f (lật 180°) hoặc o (xoay 90°). Khung thấp/cao quá → - / +.
  3. Space để bắt đầu/dừng ghi. Trong lúc ghi, đổi góc nhìn và khoảng cách camera cho đa dạng.
  4. q để thoát. Xem lại ảnh trong preview/ và xoá ảnh hỏng (cả ảnh, nhãn, preview cùng tên).

Kết quả (dùng được ngay làm nguồn cho prepare_dataset):
    data/captured/<set>/train/images|labels, preview/, sessions/<phiên>.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess
import cv2
import numpy as np

from chess_assistant.autolabel import to_yolo_lines
from chess_assistant.camera import ReconnectingCamera, open_camera
from chess_assistant.geolabel import geometric_boxes, positions_from_pgn
from chess_assistant.schema import PIECE_CLASSES
from chess_assistant.tracker import BoardTracker
from scripts.capture_labeled import CAPTURED_DIR, prepare_set_dir

WINDOW = "Thu du lieu (hinh hoc)"


def draw_labels(img: np.ndarray, labels) -> None:
    for cls, (x1, y1, x2, y2) in labels:
        sym = PIECE_CLASSES[cls]
        color = (0, 255, 255) if sym.isupper() else (255, 0, 255)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
        cv2.putText(img, sym, (int(x1) + 2, int(y1) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="Tên bộ cờ / game (mỗi tên một thư mục)")
    ap.add_argument("--camera", default="0")
    ap.add_argument("--fen", default=chess.STARTING_BOARD_FEN, help="Phần xếp quân của FEN")
    ap.add_argument("--pgn", type=Path, help="Bước qua các thế của một ván (n/p)")
    ap.add_argument("--every", type=int, default=10, help="Lưu 1 ảnh mỗi N frame khi đang ghi")
    ap.add_argument("--height-scale", type=float, default=1.0, help="Hệ số chiều cao quân (+/- để chỉnh)")
    args = ap.parse_args()

    if args.pgn and not args.pgn.exists():
        hint = ""
        if Path(str(args.pgn) + ".txt").exists():
            hint = f" Thấy file {args.pgn}.txt: Notepad đã tự thêm đuôi .txt — đổi tên bỏ .txt đi."
        raise SystemExit(
            f"Không thấy file {args.pgn}.{hint}\n"
            "Tạo file bằng Notepad, gõ danh sách nước đi (vd: 1. d4 d5 2. c4 e6), "
            'Save as type: "All Files", tên van1.pgn, lưu trong thư mục data của dự án.'
        )
    try:
        positions = positions_from_pgn(args.pgn) if args.pgn else [chess.Board(args.fen.split()[0] + " w - - 0 1")]
    except ValueError as exc:
        raise SystemExit(f"File PGN có lỗi: {exc}")
    pos_idx, scale = 0, args.height_scale

    root = CAPTURED_DIR / args.set
    prepare_set_dir(root)
    session = time.strftime("%Y%m%d-%H%M%S")
    meta = {"set": args.set, "fen": positions[0].board_fen(), "method": "geometric", "images": {}}

    camera = ReconnectingCamera(lambda: open_camera(args.camera))
    tracker = BoardTracker()
    recording, frame_idx, saved = False, 0, 0
    try:
        while True:
            frame = camera.read()
            if frame is None:
                if cv2.waitKey(30) & 0xFF in (27, ord("q")):
                    break
                continue
            frame_idx += 1
            board = positions[pos_idx]
            corners = tracker.update(frame)
            shown = frame.copy()
            labels = []
            if corners is not None:
                h, w = frame.shape[:2]
                labels = geometric_boxes(corners, board, (w, h), scale)
                draw_labels(shown, labels)
                if recording and tracker.status == "locked" and labels and frame_idx % args.every == 0:
                    name = f"{session}_{saved:05d}"
                    cv2.imwrite(str(root / "train" / "images" / f"{name}.jpg"), frame)
                    (root / "train" / "labels" / f"{name}.txt").write_text(
                        "\n".join(to_yolo_lines(labels, w, h)) + "\n", encoding="utf-8"
                    )
                    preview = frame.copy()
                    draw_labels(preview, labels)
                    cv2.imwrite(str(root / "preview" / f"{name}.jpg"), preview)
                    meta["images"][name] = board.board_fen()
                    saved += 1

            step = f" | the {pos_idx + 1}/{len(positions)} (n/p)" if len(positions) > 1 else ""
            status = (f"{'DANG GHI' if recording else 'Tam dung'} | da luu {saved} | cao x{scale:.2f}{step}"
                      f" | {tracker.status} | Space ghi  f lat  o xoay  +/- cao  q thoat")
            cv2.rectangle(shown, (0, 0), (shown.shape[1], 32), (0, 0, 0), -1)
            cv2.putText(shown, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 255) if recording else (255, 255, 255), 2)
            cv2.imshow(WINDOW, shown)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == 32:
                recording = not recording
            elif key == ord("f"):
                tracker.flip()
            elif key == ord("o"):
                tracker.rotate(1)
            elif key in (ord("+"), ord("=")):
                scale = round(scale + 0.05, 2)
            elif key == ord("-"):
                scale = max(0.3, round(scale - 0.05, 2))
            elif key == ord("n"):
                pos_idx = min(pos_idx + 1, len(positions) - 1)
            elif key == ord("p"):
                pos_idx = max(pos_idx - 1, 0)
    finally:
        meta["height_scale"] = scale
        (root / "sessions" / f"{session}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()
    print(f"Phiên {session}: lưu {saved} ảnh -> {root}")


if __name__ == "__main__":
    main()
