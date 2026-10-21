"""Quay và tự gán nhãn ảnh quân cờ cho một thế cờ đã biết.

    python -m scripts.capture_labeled --set go_vang --camera 1
    python -m scripts.capture_labeled --set go_vang --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"

Bày đúng thế cờ (mặc định thế khởi đầu), nhấn Space để bắt đầu/dừng ghi, f để lật hướng bàn cờ
(nếu a1 hiện sai góc — frame có lớp quân ngược màu sẽ bị bỏ qua), q để thoát.
Đổi góc nhìn, khoảng cách, ánh sáng trong lúc ghi. Cần model v0 (models/pieces.pt) để tìm vị trí quân;
tên quân lấy từ FEN nên nhãn luôn đúng.

Kết quả (thư mục YOLO, dùng được ngay làm nguồn cho prepare_dataset):
    data/captured/<set>/data.yaml
    data/captured/<set>/train/images/<phiên>_<n>.jpg   + train/labels/<phiên>_<n>.txt
    data/captured/<set>/preview/<phiên>_<n>.jpg        (ảnh có vẽ khung để kiểm tra, xoá ảnh hỏng ở cả 3 chỗ)
    data/captured/<set>/sessions/<phiên>.json          ({"set", "fen"})
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess
import cv2
import yaml

from chess_assistant.autolabel import autolabel, to_yolo_lines
from chess_assistant.board import BoardView
from chess_assistant.camera import ReconnectingCamera, open_camera
from chess_assistant.overlay import draw_grid
from chess_assistant.pieces import MODELS_DIR, PieceDetector
from chess_assistant.schema import PIECE_CLASSES
from chess_assistant.tracker import BoardTracker

CAPTURED_DIR = Path(__file__).resolve().parent.parent / "data" / "captured"


def prepare_set_dir(root: Path) -> None:
    for sub in ("train/images", "train/labels", "preview", "sessions"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        data_yaml.write_text(yaml.safe_dump({"names": list(PIECE_CLASSES)}), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="Tên bộ cờ (mỗi bộ một thư mục)")
    ap.add_argument("--fen", default=chess.STARTING_BOARD_FEN, help="Phần xếp quân của FEN")
    ap.add_argument("--camera", default="0")
    ap.add_argument("--every", type=int, default=10, help="Lưu 1 ảnh mỗi N frame")
    ap.add_argument("--weights", type=Path, default=MODELS_DIR / "pieces.pt")
    args = ap.parse_args()

    board = chess.Board(args.fen.split()[0] + " w - - 0 1")
    root = CAPTURED_DIR / args.set
    prepare_set_dir(root)
    session = time.strftime("%Y%m%d-%H%M%S")
    (root / "sessions" / f"{session}.json").write_text(
        json.dumps({"set": args.set, "fen": board.board_fen()}), encoding="utf-8"
    )

    detector = PieceDetector(args.weights, conf=0.15)
    camera = ReconnectingCamera(lambda: open_camera(args.camera))
    tracker = BoardTracker()
    recording, frame_idx, saved, skipped = False, 0, 0, 0
    try:
        while True:
            frame = camera.read()
            if frame is None:
                if cv2.waitKey(30) & 0xFF in (27, ord("q")):
                    break
                continue
            frame_idx += 1
            corners = tracker.update(frame)
            shown = frame.copy()
            if corners is not None:
                view = BoardView.from_corners(corners)
                draw_grid(shown, view)
                if recording and tracker.status == "locked" and frame_idx % args.every == 0:
                    labels = autolabel(detector(frame), view.H, board)
                    if labels is None:
                        skipped += 1
                    else:
                        name = f"{session}_{saved:05d}"
                        h, w = frame.shape[:2]
                        cv2.imwrite(str(root / "train" / "images" / f"{name}.jpg"), frame)
                        (root / "train" / "labels" / f"{name}.txt").write_text(
                            "\n".join(to_yolo_lines(labels, w, h)) + "\n", encoding="utf-8"
                        )
                        preview = frame.copy()
                        for cls, (x1, y1, x2, y2) in labels:
                            cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(preview, PIECE_CLASSES[cls], (int(x1), int(y1) - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.imwrite(str(root / "preview" / f"{name}.jpg"), preview)
                        saved += 1
            status = f"{'DANG GHI' if recording else 'Tam dung'} | da luu {saved} | bo qua {skipped} | Space: ghi/dung  f: lat  q: thoat"
            cv2.rectangle(shown, (0, 0), (shown.shape[1], 32), (0, 0, 0), -1)
            cv2.putText(shown, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255) if recording else (255, 255, 255), 2)
            cv2.imshow("Thu du lieu", shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == 32:
                recording = not recording
            if key == ord("f"):
                tracker.flip()
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()
    print(f"Phiên {session}: lưu {saved} ảnh, bỏ {skipped} frame không đủ khung quân -> {root}")


if __name__ == "__main__":
    main()
