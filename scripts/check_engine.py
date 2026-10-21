"""Kiểm tra kết nối Stockfish.

    python -m scripts.check_engine
"""

import chess

from chess_assistant.engine import Engine, find_stockfish


def main() -> None:
    path = find_stockfish()
    print(f"Stockfish: {path}")
    board = chess.Board()
    with Engine(path) as engine:
        print(f"Engine:    {engine.name}")
        s = engine.suggest(board, think_time=1.0)
    print(f"Thế khởi đầu -> gợi ý {board.san(s.move)} ({s.move.uci()}), điểm {s.score_text()}, depth {s.depth}")


if __name__ == "__main__":
    main()
