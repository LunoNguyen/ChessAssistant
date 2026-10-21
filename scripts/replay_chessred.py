"""Phát lại các ván thật của ChessReD như một camera ảo — test toàn bộ hệ thống không cần bàn cờ.

Mỗi ảnh ChessReD là thế cờ sau một nước đi, chụp bằng điện thoại. Mỗi ảnh được đưa vào pipeline
nhiều lần (giống camera nhìn bàn cờ đứng yên), rồi so các nước hệ thống ghi nhận với nước đi thật.

    python -m scripts.replay_chessred                    # 3 ván test của ChessReD2K
    python -m scripts.replay_chessred --show             # xem trực tiếp, có mũi tên gợi ý của Stockfish
    python -m scripts.replay_chessred --show --game 33 --delay 300

Phím khi --show: q/Esc thoát, n sang ván tiếp theo, Space tạm dừng/tiếp tục.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterator

import chess
import cv2
import numpy as np

from chess_assistant.chessred import GameFrame, load_games, moves_between
from chess_assistant.schema import PieceBox
from scripts.replay import replay_frames

WIDTH = 1280  # ảnh ChessReD 3072px; camera thật thường ~1280x720


def _frames(frames: list[GameFrame], repeat: int) -> Iterator[np.ndarray]:
    for f in frames:
        img = cv2.imread(str(f.path))
        if img is None:
            raise FileNotFoundError(f"Không đọc được ảnh {f.path}")
        if img.shape[1] > WIDTH:
            s = WIDTH / img.shape[1]
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        for _ in range(repeat):
            yield img


def replay_game(
    frames: list[GameFrame],
    detector: Callable[[np.ndarray], list[PieceBox]],
    repeat: int = 12,
    pipeline=None,
    on_frame=None,
) -> dict:
    truth = moves_between([f.board_fen for f in frames])
    start = chess.Board(frames[0].board_fen + " w KQkq - 0 1")
    start.castling_rights = start.clean_castling_rights()
    return replay_frames(_frames(frames, repeat), truth, detector, start, pipeline, on_frame)


def _show_setup(detector, engine, game_id: int, delay: int):
    from chess_assistant.advisor import Advisor
    from chess_assistant.game import GameTracker
    from chess_assistant.overlay import render
    from chess_assistant.pipeline import Pipeline
    from chess_assistant.tracker import BoardTracker

    advisor = Advisor(engine, think_time=0.3)
    pipeline = Pipeline(detector, BoardTracker(background=False), GameTracker(), advisor)
    title = f"ChessReD replay - van {game_id}"
    paused = [False]

    def on_frame(frame, result) -> bool:
        while True:
            cv2.imshow("ChessReD replay", render(frame, result.view, result.overlay))
            cv2.setWindowTitle("ChessReD replay", title)
            key = cv2.waitKey(delay) & 0xFF
            if key in (27, ord("q")):
                raise SystemExit(0)
            if key == ord("n"):
                return False
            if key == 32:
                paused[0] = not paused[0]
            if not paused[0]:
                return True

    return pipeline, on_frame, advisor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/downloads/chessred"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--game", type=int, action="append", help="Chỉ phát ván này (lặp lại để chọn nhiều ván)")
    ap.add_argument("--repeat", type=int, default=30, help="Số frame cho mỗi ảnh (~1 giây camera)")
    ap.add_argument("--show", action="store_true", help="Hiện cửa sổ như play.py, có gợi ý Stockfish")
    ap.add_argument("--delay", type=int, default=30, help="ms giữa các frame khi --show")
    args = ap.parse_args()

    from chess_assistant.pieces import PieceDetector

    games = load_games(args.root, split=args.split)
    if args.game:
        games = {g: f for g, f in games.items() if g in args.game}
    if not games:
        raise SystemExit("Không có ván nào để phát.")
    detector = PieceDetector()

    engine = None
    if args.show:
        from chess_assistant.engine import Engine
        engine = Engine()

    totals = {"correct": 0, "wrong": 0, "missed": 0}
    try:
        for game_id, frames in games.items():
            pipeline = on_frame = advisor = None
            if args.show:
                pipeline, on_frame, advisor = _show_setup(detector, engine, game_id, args.delay)
            report = replay_game(frames, detector, args.repeat, pipeline, on_frame)
            if advisor is not None:
                advisor.close()
            for k in totals:
                totals[k] += report[k]
            n = report["correct"] + report["wrong"] + report["missed"]
            print(f"Ván {game_id}: {len(frames)} ảnh, {n} nước | đúng {report['correct']} / sai {report['wrong']}"
                  f" / bỏ lỡ {report['missed']} | {report['ms_mean']:.0f} ms/frame")
    finally:
        if engine is not None:
            engine.close()
        if args.show:
            cv2.destroyAllWindows()
    n = sum(totals.values())
    print(f"Tổng: đúng {totals['correct']}/{n} ({totals['correct'] / max(n, 1):.1%}), sai {totals['wrong']}")


if __name__ == "__main__":
    main()
