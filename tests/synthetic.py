"""Sinh ảnh camera mô phỏng một bàn cờ thật (có viền gỗ, quân cờ, nền nhiễu)
để kiểm thử bộ dò bàn cờ mà không cần camera."""

from __future__ import annotations

import chess
import cv2
import numpy as np

SQ = 100
BORDER = 60
FLAT = 8 * SQ + 2 * BORDER

LIGHT = (170, 205, 225)  # BGR
DARK = (70, 105, 150)
WOOD = (40, 75, 115)

# Góc vùng chơi trên ảnh phẳng, thứ tự a1, h1, h8, a8.
FLAT_PLAY_CORNERS = np.float32([
    [BORDER, BORDER + 8 * SQ],
    [BORDER + 8 * SQ, BORDER + 8 * SQ],
    [BORDER + 8 * SQ, BORDER],
    [BORDER, BORDER],
])


# Bàn cờ in trên giấy (như ChessReD): lề trắng rộng, đường viền đen in cách mép bàn đúng 1 ô.
PAPER_BORDER = 140
PAPER = (225, 228, 230)
PAPER_DARK = (40, 45, 55)


def _play_corners(border: int) -> np.ndarray:
    return np.float32([[border, border + 8 * SQ], [border + 8 * SQ, border + 8 * SQ],
                       [border + 8 * SQ, border], [border, border]])


def flat_board(fen: str | None = chess.STARTING_BOARD_FEN, style: str = "wood") -> np.ndarray:
    paper = style == "paper"
    border = PAPER_BORDER if paper else BORDER
    size = 8 * SQ + 2 * border
    img = np.full((size, size, 3), PAPER if paper else WOOD, np.uint8)
    if paper:
        a, b = border - SQ, border + 9 * SQ
        cv2.rectangle(img, (a, a), (b, b), (20, 20, 20), 5)
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        x0, y0 = border + f * SQ, border + (7 - r) * SQ
        color = (PAPER_DARK if paper else DARK) if (f + r) % 2 == 0 else (PAPER if paper else LIGHT)
        cv2.rectangle(img, (x0, y0), (x0 + SQ - 1, y0 + SQ - 1), color, -1)
    if fen:
        board = chess.Board(fen + " w - - 0 1") if " " not in fen else chess.Board(fen)
        for sq, piece in board.piece_map().items():
            f, r = chess.square_file(sq), chess.square_rank(sq)
            c = (border + f * SQ + SQ // 2, border + (7 - r) * SQ + SQ // 2)
            fill, edge = ((235, 235, 235), (90, 90, 90)) if piece.color else ((35, 35, 35), (150, 150, 150))
            radius = 22 if piece.piece_type == chess.PAWN else 32
            cv2.circle(img, c, radius, fill, -1, cv2.LINE_AA)
            cv2.circle(img, c, radius, edge, 2, cv2.LINE_AA)
    return img


def scene(
    corners,
    fen: str | None = chess.STARTING_BOARD_FEN,
    size: tuple[int, int] = (1280, 720),
    seed: int = 0,
    noise: float = 6.0,
    hand: tuple[int, int, int, int] | None = None,
    style: str = "wood",
) -> np.ndarray:
    """Ảnh camera có bàn cờ mà 4 góc vùng chơi (a1, h1, h8, a8) nằm ở `corners`.

    `hand` = (cx, cy, rx, ry): vẽ một mảng màu da che bàn cờ.
    """
    rng = np.random.default_rng(seed)
    w, h = size
    # Nền: mặt bàn có vân và gradient ánh sáng, cùng vài đường thẳng gây nhiễu.
    bg = rng.normal(120, 18, (h // 8, w // 8, 3)).clip(0, 255).astype(np.uint8)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(3):
        p = tuple(int(v) for v in rng.integers(0, [w, h]))
        q = tuple(int(v) for v in rng.integers(0, [w, h]))
        cv2.line(bg, p, q, tuple(int(v) for v in rng.integers(0, 255, 3)), 3)

    flat = flat_board(fen, style)
    border = PAPER_BORDER if style == "paper" else BORDER
    M = cv2.getPerspectiveTransform(_play_corners(border), np.float32(corners))
    board = cv2.warpPerspective(flat, M, (w, h))
    mask = cv2.warpPerspective(np.full(flat.shape[:2], 255, np.uint8), M, (w, h)) > 0
    img = bg.copy()
    img[mask] = board[mask]

    if hand:
        cx, cy, rx, ry = hand
        cv2.ellipse(img, (cx, cy), (rx, ry), 30, 0, 360, (120, 160, 210), -1)

    gain = np.linspace(0.85, 1.1, w)[None, :, None]
    img = img.astype(np.float32) * gain + rng.normal(0, noise, img.shape)
    img = cv2.GaussianBlur(img.clip(0, 255).astype(np.uint8), (3, 3), 0)
    return img


# Một số tư thế camera điển hình: 4 góc vùng chơi a1, h1, h8, a8 trong ảnh 1280x720.
POSES = {
    "white_side": [[330, 640], [950, 640], [840, 170], [440, 170]],
    "black_side": [[840, 170], [440, 170], [330, 640], [950, 640]],
    "a_file_side": [[950, 640], [840, 170], [440, 170], [330, 640]],
    "rotated": [[380, 660], [930, 590], [850, 130], [360, 190]],
    "small_far": [[520, 520], [800, 520], [760, 300], [560, 300]],
    "top_down": [[400, 680], [880, 680], [880, 200], [400, 200]],
}
