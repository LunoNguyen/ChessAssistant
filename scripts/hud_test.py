"""Hiện HUD mẫu để kiểm tra màn hình kính (dây nối, màu, hướng) mà không cần camera hay model.

    python -m scripts.hud_test                       # cửa sổ 240x240 trên laptop
    python -m scripts.hud_test --scale 2             # phóng to 480x480 cho dễ nhìn
    python -m scripts.hud_test --backend st7789      # màn hình SPI trên Raspberry Pi
    python -m scripts.hud_test --mirror              # lật gương (kính dùng tấm phản xạ)

Lần lượt hiện 4 màn hình mẫu (mỗi màn 3 giây, hoặc bấm phím bất kỳ trên laptop): đang tìm bàn cờ,
gợi ý cho Trắng (e2-e4), gợi ý khi chưa rõ lượt (Trắng + Đen), góc nhìn phía Đen. q/Esc để thoát.
"""

from __future__ import annotations

import argparse
import time

import chess
import cv2

from chess_assistant.advisor import Advice
from chess_assistant.hud import ST7789Display, WindowDisplay, render_hud
from chess_assistant.overlay import OverlayState

MIDGAME = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R2QKB1R w KQ - 0 8"


def samples() -> list[OverlayState]:
    start = chess.Board()
    mid = chess.Board(MIDGAME)
    return [
        OverlayState("Dang tim ban co...", (0, 165, 255)),
        OverlayState("Dang bam van co", (0, 255, 0), board=start,
                     advice=Advice(start.fen(), [chess.Move.from_uci("e2e4")], "+0.30", 30)),
        OverlayState("Chua ro luot", (0, 255, 255), board=mid,
                     advice=Advice(mid.fen(), [chess.Move.from_uci("a2a3")], "Trang +0.6 | Den -0.2", 60,
                                   alt_moves=[chess.Move.from_uci("c6b4")])),
        OverlayState("Dang bam van co", (0, 255, 0), board=mid, user_color=chess.BLACK,
                     advice=Advice(mid.fen(), [chess.Move.from_uci("c4d5")], "+0.60", 60)),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["window", "st7789"], default="window")
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()

    display = WindowDisplay(scale=args.scale) if args.backend == "window" else ST7789Display()
    try:
        for state in samples():
            display.show(render_hud(state, mirror=args.mirror))
            if args.backend == "window":
                if cv2.waitKey(int(args.seconds * 1000)) & 0xFF in (27, ord("q")):
                    break
            else:
                time.sleep(args.seconds)
    finally:
        display.close()


if __name__ == "__main__":
    main()
