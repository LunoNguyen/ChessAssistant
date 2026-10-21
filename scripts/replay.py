"""Chạy lại cả pipeline trên video một ván thật và so với PGN đúng.

    python -m scripts.replay data/videos/van1.mp4 data/videos/van1.pgn

In số nước ghi nhận đúng / sai / bỏ lỡ và thời gian xử lý mỗi frame.
Thoát mã 1 nếu có nước ghi nhận sai (tiêu chí: không được ghi nhận sai mà không báo).
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator

import chess
import chess.pgn
import cv2
import numpy as np

from chess_assistant.evaluation import move_accuracy_by_position, position_key
from chess_assistant.game import GameTracker
from chess_assistant.pieces import MODELS_DIR, PieceDetector
from chess_assistant.pipeline import Pipeline
from chess_assistant.schema import PieceBox
from chess_assistant.tracker import BoardTracker


def load_truth(pgn_path: Path) -> tuple[chess.Board, list[chess.Move]]:
    game = chess.pgn.read_game(io.StringIO(Path(pgn_path).read_text(encoding="utf-8")))
    if game is None:
        raise ValueError(f"Không đọc được PGN {pgn_path}")
    return game.board(), list(game.mainline_moves())


def replay_frames(
    frames: Iterable[np.ndarray],
    truth: list[chess.Move],
    detector: Callable[[np.ndarray], list[PieceBox]],
    start: chess.Board | None = None,
    pipeline: Pipeline | None = None,
    on_frame: Callable[[np.ndarray, object], bool] | None = None,
) -> dict:
    """Chạy pipeline trên chuỗi frame và so các nước ghi nhận với `truth`.

    `on_frame(frame, result)` (tuỳ chọn) được gọi sau mỗi frame; trả False để dừng sớm.
    """
    if pipeline is None:
        game = GameTracker()
        if start is not None and start.fen() != chess.STARTING_FEN:
            game.start_from(start)
        pipeline = Pipeline(detector, BoardTracker(background=False), game)

    recognized: list[tuple[str, chess.Move]] = []  # (thế cờ trước nước, nước)
    times: list[float] = []
    for frame in frames:
        t = time.perf_counter()
        result = pipeline.process(frame)
        times.append((time.perf_counter() - t) * 1000)
        for event in result.events:
            if event.kind == "moves":
                before = pipeline.game.board.copy()
                for _ in event.moves:
                    before.pop()
                for move in event.moves:
                    recognized.append((position_key(before), move))
                    before.push(move)
        if on_frame is not None and on_frame(frame, result) is False:
            break

    truth_start = start if start is not None else chess.Board()
    report = move_accuracy_by_position(recognized, truth_start, truth)
    report.update(
        frames=len(times),
        accuracy=report["correct"] / len(truth) if truth else 1.0,
        ms_mean=float(np.mean(times)) if times else 0.0,
        ms_p95=float(np.percentile(times, 95)) if times else 0.0,
        recognized=[m.uci() for _, m in recognized],
    )
    return report


def _video_frames(video: Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video {video}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def run_replay(video: Path, pgn: Path, detector: Callable[[np.ndarray], list[PieceBox]]) -> dict:
    start, truth = load_truth(pgn)
    return replay_frames(_video_frames(video), truth, detector, start)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("pgn", type=Path)
    ap.add_argument("--weights", type=Path, default=MODELS_DIR / "pieces.pt")
    args = ap.parse_args()

    report = run_replay(args.video, args.pgn, PieceDetector(args.weights))
    print(f"Đúng {report['correct']} / sai {report['wrong']} / bỏ lỡ {report['missed']}"
          f"  (tỉ lệ đúng {report['accuracy']:.1%})")
    print(f"{report['frames']} frame, xử lý trung bình {report['ms_mean']:.1f} ms, p95 {report['ms_p95']:.1f} ms")
    print("Nước đã ghi nhận:", " ".join(report["recognized"]))
    return 1 if report["wrong"] else 0


if __name__ == "__main__":
    sys.exit(main())
