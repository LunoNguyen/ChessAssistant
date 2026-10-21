"""Kết nối Stockfish qua python-chess."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

ENGINES_DIR = Path(__file__).resolve().parent.parent / "engines"


def find_stockfish() -> Path:
    """Tìm Stockfish: STOCKFISH_PATH -> thư mục engines/ -> PATH."""
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        if Path(env).is_file():
            return Path(env)
        raise FileNotFoundError(f"STOCKFISH_PATH={env} không tồn tại")

    candidates = sorted(
        p for p in ENGINES_DIR.rglob("stockfish*")
        if p.is_file() and (p.suffix.lower() == ".exe" or (os.name != "nt" and os.access(p, os.X_OK)))
    )
    if candidates:
        return candidates[0]

    found = shutil.which("stockfish")
    if found:
        return Path(found)

    raise FileNotFoundError(
        "Không tìm thấy Stockfish. Tải tại https://stockfishchess.org/download/ "
        f"và đặt file .exe vào {ENGINES_DIR} (xem engines/README.md)."
    )


@dataclass
class Suggestion:
    move: chess.Move
    score: chess.engine.PovScore
    depth: int

    def white_cp(self) -> int:
        """Điểm theo phía Trắng (centipawn); chiếu hết quy đổi thành ±10000."""
        white = self.score.white()
        if white.is_mate():
            return 10000 if white.mate() > 0 else -10000
        return white.score()

    def score_text(self) -> str:
        """Điểm theo góc nhìn bên Trắng, ví dụ '+0.35' hoặc '#3'."""
        white = self.score.white()
        if white.is_mate():
            return f"#{white.mate()}"
        return f"{white.score() / 100:+.2f}"


class Engine:
    """Giữ một tiến trình Stockfish suốt phiên chơi. Dùng với `with Engine() as e:`."""

    def __init__(self, path: Path | None = None, skill_level: int = 20, threads: int = 2, hash_mb: int = 128):
        self._engine = chess.engine.SimpleEngine.popen_uci(str(path or find_stockfish()))
        self._engine.configure({"Threads": threads, "Hash": hash_mb})
        self.set_skill(skill_level)

    def set_skill(self, level: int) -> None:
        """Độ khó 0 (yếu nhất) .. 20 (mạnh nhất)."""
        self._engine.configure({"Skill Level": max(0, min(20, level))})

    @property
    def name(self) -> str:
        return self._engine.id.get("name", "unknown")

    def suggest(self, board: chess.Board, think_time: float = 0.5) -> Suggestion | None:
        lines = self.analyse(board, think_time)
        return lines[0] if lines else None

    def analyse(self, board: chess.Board, think_time: float = 0.5, multipv: int = 1) -> list[Suggestion]:
        """Các biến tốt nhất theo thứ tự; rỗng nếu ván đã kết thúc."""
        if board.is_game_over():
            return []
        infos = self._engine.analyse(board, chess.engine.Limit(time=think_time), multipv=multipv)
        return [
            Suggestion(move=info["pv"][0], score=info["score"], depth=info.get("depth", 0))
            for info in infos
            if info.get("pv")
        ]

    def play_limited(self, board: chess.Board, elo: int, think_time: float = 0.5) -> chess.Move:
        """Nước đi ở sức mạnh giới hạn (chế độ luyện tập); chỉ áp dụng cho lệnh này."""
        result = self._engine.play(
            board,
            chess.engine.Limit(time=think_time),
            options={"UCI_LimitStrength": True, "UCI_Elo": elo},
        )
        return result.move

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
