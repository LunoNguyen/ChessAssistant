"""Trợ lý cờ vua qua camera: theo dõi ván và gợi ý nước đi.

    python -m scripts.play --camera 1                       # DroidCam/Iriun thường là camera 1 hoặc 2
    python -m scripts.play --camera http://192.168.1.5:4747/video
    python -m scripts.play --mode training --elo 1500

Gợi ý cho cả hai bên ở mọi lượt (--side me: chỉ lượt của bạn). Khi đọc lại thế cờ giữa ván mà chưa rõ
bên nào đến lượt, hiện 2 mũi tên (xanh lá: Trắng, xanh dương: Đen) và tự nhận lượt ở nước đi kế tiếp.

Phím: w/b  chọn bên đến lượt ngay (không bắt buộc)
      c    đổi màu quân của bạn (mặc định: bên gần camera)
      u    lùi nước vừa ghi nhận
      f    lật bàn cờ 180°
      r    dò lại bàn cờ
      p    tạm dừng / tiếp tục gợi ý
      q/Esc thoát (ván được lưu vào data/games/*.pgn)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import chess
import cv2
import numpy as np

from chess_assistant.advisor import Advisor
from chess_assistant.camera import ReconnectingCamera, open_camera
from chess_assistant.engine import Engine, find_stockfish
from chess_assistant.game import GameTracker
from chess_assistant.hud import WindowDisplay, render_hud
from chess_assistant.overlay import render
from chess_assistant.pieces import MODELS_DIR, PieceDetector
from chess_assistant.pipeline import Pipeline
from chess_assistant.tracker import BoardTracker

GAMES_DIR = Path(__file__).resolve().parent.parent / "data" / "games"
WINDOW = "Chess Assistant"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="0", help="Chỉ số camera hoặc URL luồng video")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--weights", type=Path, default=MODELS_DIR / "pieces.pt")
    ap.add_argument("--think-time", type=float, default=0.5)
    ap.add_argument("--multipv", type=int, default=1)
    ap.add_argument("--mode", choices=["best", "training"], default="best")
    ap.add_argument("--elo", type=int, default=1500)
    ap.add_argument("--hud", action="store_true", help="Mở thêm cửa sổ 240x240 mô phỏng màn hình kính")
    ap.add_argument("--hud-scale", type=int, default=1, help="Phóng to cửa sổ HUD (vd 2 = 480x480)")
    ap.add_argument("--hud-mirror", action="store_true", help="Lật gương HUD (kính dùng tấm phản xạ)")
    ap.add_argument("--side", choices=["both", "me"], default="both",
                    help="both: gợi ý mọi lượt (mặc định); me: chỉ lượt của bạn (bên gần camera)")
    return ap.parse_args()


def save_game(game: GameTracker) -> Path | None:
    if not game.board.move_stack:
        return None
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    path = GAMES_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.pgn"
    path.write_text(game.pgn(), encoding="utf-8")
    return path


def handle_key(key: int, pipeline: Pipeline) -> bool:
    """Xử lý phím; trả về False khi cần thoát."""
    if key in (27, ord("q")):
        return False
    if key in (ord("w"), ord("b")):
        pipeline.set_turn(chess.WHITE if key == ord("w") else chess.BLACK)
    elif key == ord("c"):
        pipeline.user_color = not pipeline.current_user_color()
    elif key == ord("u"):
        pipeline.undo()
    elif key == ord("f"):
        pipeline.flip()
    elif key == ord("r"):
        pipeline.tracker.reset()
        pipeline.stability.reset()
    elif key == ord("p"):
        pipeline.paused = not pipeline.paused
    return True


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    try:
        stockfish = find_stockfish()
        detector = PieceDetector(args.weights)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1

    camera = ReconnectingCamera(lambda: open_camera(args.camera, args.width, args.height))
    engine = Engine(stockfish)
    tracker = BoardTracker()
    advisor = Advisor(engine, args.think_time, args.multipv, args.mode, args.elo, side=args.side)
    pipeline = Pipeline(detector, tracker, GameTracker(), advisor)
    blank = np.zeros((args.height, args.width, 3), np.uint8)
    hud = WindowDisplay(scale=args.hud_scale) if args.hud else None
    try:
        while True:
            frame = camera.read()
            if frame is None:
                shown = blank.copy()
                cv2.putText(shown, "Mat ket noi camera, dang thu lai...", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            else:
                result = pipeline.process(frame)
                shown = render(frame, result.view, result.overlay)
                if hud is not None:
                    hud.show(render_hud(result.overlay, mirror=args.hud_mirror))
            cv2.imshow(WINDOW, shown)
            if not handle_key(cv2.waitKey(1) & 0xFF, pipeline):
                break
    finally:
        path = save_game(pipeline.game)
        if path:
            print(f"Đã lưu ván cờ: {path}")
        advisor.close()
        tracker.close()
        engine.close()
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
