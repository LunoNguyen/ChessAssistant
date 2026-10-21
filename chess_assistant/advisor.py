"""Gợi ý nước đi bằng Stockfish ở luồng nền, không làm đứng video."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

import chess

log = logging.getLogger(__name__)


@dataclass
class Advice:
    fen: str
    moves: list[chess.Move]
    score_text: str
    white_cp: int
    blunder_cp: int | None = None
    alt_moves: list[chess.Move] = field(default_factory=list)  # gợi ý cho Đen khi chưa rõ lượt


class Advisor:
    """- best: lượt người dùng -> nước tốt nhất (multipv biến); lượt đối thủ -> không làm gì.
    - training: lượt người dùng -> một nước ở sức mạnh `elo`; lượt đối thủ -> đánh giá lại
      và báo nếu nước người dùng vừa đi làm mất >= `blunder_cp` centipawn.
    """

    def __init__(
        self,
        engine,
        think_time: float = 0.5,
        multipv: int = 1,
        mode: Literal["best", "training"] = "best",
        elo: int = 1500,
        blunder_cp: int = 200,
        side: Literal["me", "both"] = "me",
    ):
        self.engine = engine
        self.think_time = think_time
        self.multipv = multipv
        self.mode = mode
        self.elo = elo
        self.blunder_cp = blunder_cp
        self.side = side  # "both": gợi ý ở mọi lượt, không chỉ lượt người dùng
        self.error: str | None = None
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._wanted: tuple | None = None  # (FEN, màu người dùng, cả hai lượt?)
        self._latest: Advice | None = None
        self._latest_key: tuple | None = None
        self._user_cp: int | None = None  # điểm (phía người dùng) ở lượt người dùng gần nhất

    def request(self, board: chess.Board, user_color: chess.Color, both_turns: bool = False) -> None:
        """both_turns=True: chưa biết bên nào đến lượt — gợi ý cho Trắng (moves) và Đen (alt_moves)."""
        key = (board.fen(), user_color, both_turns)
        with self._lock:
            if key == self._wanted:
                return
            self._wanted = key
        self._pool.submit(self._work, board.copy(), user_color, key)

    def latest(self) -> Advice | None:
        with self._lock:
            if self._latest is not None and self._latest_key == self._wanted:
                return self._latest
            return None

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------

    def _work(self, board: chess.Board, user_color: chess.Color, key: tuple) -> None:
        with self._lock:
            if key != self._wanted:
                return  # đã có yêu cầu mới hơn
        try:
            advice = self._analyse_both(board, key[0]) if key[2] else self._analyse(board, user_color, key[0])
        except Exception as exc:  # engine chết, lỗi giao tiếp...
            log.exception("Advisor: lỗi khi phân tích")
            with self._lock:
                self.error = f"{type(exc).__name__}: {exc}"
            return
        with self._lock:
            self.error = None
            if advice is not None and key == self._wanted:
                self._latest, self._latest_key = advice, key

    def _analyse_both(self, board: chess.Board, fen: str) -> Advice | None:
        best: dict[chess.Color, object] = {}
        for turn in (chess.WHITE, chess.BLACK):
            b = board.copy()
            b.turn = turn
            if b.is_valid():
                lines = self.engine.analyse(b, self.think_time, 1)
                if lines:
                    best[turn] = lines[0]
        if not best:
            return None
        white, black = best.get(chess.WHITE), best.get(chess.BLACK)
        parts = [f"Trang di {white.score_text()}"] if white else []
        parts += [f"Den di {black.score_text()}"] if black else []
        return Advice(
            fen,
            [white.move] if white else [],
            " | ".join(parts),
            (white or black).white_cp(),
            alt_moves=[black.move] if black else [],
        )

    def _analyse(self, board: chess.Board, user_color: chess.Color, fen: str) -> Advice | None:
        users_turn = board.turn == user_color or self.side == "both"
        if not users_turn and self.mode == "best":
            return None
        lines = self.engine.analyse(board, self.think_time, self.multipv if users_turn else 1)
        if not lines:
            return None
        top = lines[0]
        white_cp = top.white_cp()
        user_cp = white_cp if user_color == chess.WHITE else -white_cp

        if users_turn:
            self._user_cp = user_cp
            if self.mode == "best":
                moves = [s.move for s in lines]
            else:
                moves = [self.engine.play_limited(board, self.elo, self.think_time)]
            return Advice(fen, moves, top.score_text(), white_cp)

        drop = None if self._user_cp is None else self._user_cp - user_cp
        blunder = drop if drop is not None and drop >= self.blunder_cp else None
        return Advice(fen, [], top.score_text(), white_cp, blunder_cp=blunder)
