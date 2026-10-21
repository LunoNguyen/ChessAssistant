"""Chỉ số đánh giá: độ đúng từng ô và độ đúng chuỗi nước đi."""

from __future__ import annotations

import chess
import numpy as np

from chess_assistant.game import match_count


def square_accuracy(reading: np.ndarray, board: chess.Board) -> float:
    return match_count(reading, board) / 64


def move_accuracy(recognized: list[chess.Move], truth: list[chess.Move]) -> dict[str, int]:
    """So từng ply. Gặp nước sai thì dừng (các nước sau đều lệch); phần thật chưa ghi nhận là `missed`."""
    correct = wrong = 0
    for got, want in zip(recognized, truth):
        if got != want:
            wrong = 1
            break
        correct += 1
    return {"correct": correct, "wrong": wrong, "missed": len(truth) - correct - wrong}


def position_key(board: chess.Board) -> str:
    """Thế cờ + bên đi (bỏ quyền nhập thành: khi đọc lại giữa ván chỉ suy đoán được)."""
    return f"{board.board_fen()} {'w' if board.turn else 'b'}"


def move_accuracy_by_position(
    recognized: list[tuple[str, chess.Move]], start: chess.Board, truth: list[chess.Move]
) -> dict[str, int]:
    """Một nước đúng khi từ đúng thế cờ đó ván thật cũng đi nước ấy. Khác `move_accuracy`, cách này
    không phạt các nước đúng ghi nhận được sau khi hệ thống lạc rồi đọc lại."""
    board, truth_keys = start.copy(), set()
    for move in truth:
        truth_keys.add((position_key(board), move))
        board.push(move)
    correct = sum(1 for pair in set(recognized) if pair in truth_keys)
    wrong = sum(1 for pair in recognized if pair not in truth_keys)
    return {"correct": correct, "wrong": wrong, "missed": len(truth) - correct}
