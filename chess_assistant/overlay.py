"""Vẽ kết quả lên video: lưới, mũi tên gợi ý, thanh điểm, sơ đồ mini, trạng thái.

Chữ không dấu vì cv2.putText không vẽ được tiếng Việt có dấu.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import cv2
import numpy as np

from chess_assistant.advisor import Advice
from chess_assistant.board import BoardView, grid_lines_image, square_center_image

FONT = cv2.FONT_HERSHEY_SIMPLEX
ARROW_MAIN = (0, 220, 0)
ARROW_ALT = (0, 170, 255)
ARROW_BLACK = (255, 140, 0)  # xanh dương (BGR): gợi ý cho Đen khi chưa rõ lượt


@dataclass
class OverlayState:
    status: str
    status_color: tuple[int, int, int]
    board: chess.Board | None = None
    user_color: chess.Color = chess.WHITE
    advice: Advice | None = None
    last_move: chess.Move | None = None
    message: str = ""


def draw_grid(img: np.ndarray, view: BoardView, color=(0, 255, 0)) -> None:
    for p, q in grid_lines_image(view.H):
        cv2.line(img, p, q, color, 1, cv2.LINE_AA)
    for sq in (chess.A1, chess.H1, chess.H8, chess.A8):
        x, y = square_center_image(view.H, sq)
        cv2.putText(img, chess.square_name(sq), (x - 12, y + 6), FONT, 0.6, (0, 0, 255), 2)


def draw_arrow(img: np.ndarray, H: np.ndarray, move: chess.Move, color=ARROW_MAIN, thickness: int = 6) -> None:
    start = square_center_image(H, move.from_square)
    end = square_center_image(H, move.to_square)
    cv2.arrowedLine(img, start, end, color, thickness, cv2.LINE_AA, tipLength=0.25)


def mini_board_layout(user_color: chess.Color) -> list[list[chess.Square]]:
    """8 hàng từ trên xuống, mỗi hàng 8 ô từ trái sang; phía người dùng ở dưới."""
    if user_color == chess.WHITE:
        return [[chess.square(f, r) for f in range(8)] for r in range(7, -1, -1)]
    return [[chess.square(f, r) for f in range(7, -1, -1)] for r in range(8)]


def draw_mini_board(img: np.ndarray, board: chess.Board, user_color: chess.Color, size: int = 200) -> None:
    cell = size // 8
    x0, y0 = img.shape[1] - 8 * cell - 10, 42
    for row, squares in enumerate(mini_board_layout(user_color)):
        for col, sq in enumerate(squares):
            x, y = x0 + col * cell, y0 + row * cell
            light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
            cv2.rectangle(img, (x, y), (x + cell - 1, y + cell - 1), (181, 217, 240) if light else (99, 136, 181), -1)
            piece = board.piece_at(sq)
            if piece:
                fg, bg = ((255, 255, 255), (0, 0, 0)) if piece.color else ((0, 0, 0), (255, 255, 255))
                org = (x + cell // 4, y + cell * 3 // 4)
                cv2.putText(img, piece.symbol().upper(), org, FONT, cell / 40, bg, 3, cv2.LINE_AA)
                cv2.putText(img, piece.symbol().upper(), org, FONT, cell / 40, fg, 1, cv2.LINE_AA)


def draw_eval_bar(img: np.ndarray, white_cp: int) -> None:
    x, y, w, h = 10, 50, 18, 300
    frac = 0.5 + max(-1000, min(1000, white_cp)) / 2000
    split = y + int(h * (1 - frac))
    cv2.rectangle(img, (x, y), (x + w, split), (40, 40, 40), -1)
    cv2.rectangle(img, (x, split), (x + w, y + h), (235, 235, 235), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (128, 128, 128), 1)


def _last_move_san(board: chess.Board) -> str | None:
    if not board.move_stack:
        return None
    b = board.copy()
    move = b.pop()
    return b.san(move)


def _bar(img: np.ndarray, y: int, text: str, color) -> None:
    cv2.rectangle(img, (0, y - 24), (img.shape[1], y + 8), (0, 0, 0), -1)
    cv2.putText(img, text, (10, y), FONT, 0.6, color, 2, cv2.LINE_AA)


def render(frame: np.ndarray, view: BoardView | None, state: OverlayState) -> np.ndarray:
    img = frame.copy()
    if view is not None:
        grid = img.copy()
        draw_grid(grid, view)
        img = cv2.addWeighted(grid, 0.4, img, 0.6, 0)
        if state.advice is not None:
            for move in reversed(state.advice.moves[1:]):
                draw_arrow(img, view.H, move, ARROW_ALT, 3)
            if state.advice.moves:
                draw_arrow(img, view.H, state.advice.moves[0])
            for move in state.advice.alt_moves:
                draw_arrow(img, view.H, move, ARROW_BLACK)

    if state.advice is not None:
        draw_eval_bar(img, state.advice.white_cp)
    if state.board is not None:
        draw_mini_board(img, state.board, state.user_color)

    top = state.status
    if state.board is not None:
        top += "  |  Luot: " + ("Trang" if state.board.turn == chess.WHITE else "Den")
        san = _last_move_san(state.board)
        if san:
            top += f"  |  Nuoc cuoi: {san}"
    if state.advice is not None and state.advice.score_text:
        top += f"  |  Diem: {state.advice.score_text}"
    _bar(img, 24, top, state.status_color)

    h = img.shape[0]
    if state.advice is not None and state.advice.blunder_cp is not None:
        _bar(img, h - 44, f"Nuoc vua roi mat {state.advice.blunder_cp / 100:.1f} diem", (0, 0, 255))
    if state.message:
        _bar(img, h - 12, state.message, (255, 255, 255))
    return img
