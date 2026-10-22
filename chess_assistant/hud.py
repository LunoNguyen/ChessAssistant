"""HUD cho kính: ảnh vuông nhỏ (mặc định 240x240) hiện sơ đồ bàn cờ + ô gợi ý.

Màn hình trên kính quá nhỏ để vẽ đè lên video, nên hiện sơ đồ bàn cờ (phía người dùng ở dưới):
ô quân cần đi tô vàng, ô đích tô xanh lá; khi chưa rõ lượt, gợi ý cho Đen dùng tông xanh dương.

Xuất hình qua:
- WindowDisplay: cửa sổ OpenCV (thử trên laptop);
- ST7789Display: màn hình SPI 1.3" 240x240 trên Raspberry Pi (thư viện luma.lcd, chỉ cần khi chạy trên Pi).

Chữ không dấu vì cv2.putText không vẽ được tiếng Việt có dấu.
"""

from __future__ import annotations

import chess
import cv2
import numpy as np

from chess_assistant.overlay import (
    ALT_FROM_COLOR,
    ALT_TO_COLOR,
    ARROW_ALT,
    FROM_COLOR,
    TO_COLOR,
    OverlayState,
    mini_board_layout,
)

HUD_SIZE = 240
BAR = 20                 # cao dòng trạng thái (trên) và dòng gợi ý (dưới)
BOARD = HUD_SIZE - 2 * BAR
CELL = BOARD // 8
BOARD_X = (HUD_SIZE - 8 * CELL) // 2
BOARD_Y = BAR

BACKGROUND = (20, 20, 20)
LIGHT = (200, 210, 215)
DARK = (110, 125, 140)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FILL_ALPHA = 0.7


def _cell_origin(square: chess.Square, user_color: chess.Color) -> tuple[int, int]:
    for row, squares in enumerate(mini_board_layout(user_color)):
        if square in squares:
            return BOARD_X + squares.index(square) * CELL, BOARD_Y + row * CELL
    raise ValueError(square)


def hud_square_center(square: chess.Square, user_color: chess.Color) -> tuple[int, int]:
    x, y = _cell_origin(square, user_color)
    return x + CELL // 2, y + CELL // 2


def _fill_cell(img, square, user_color, color, alpha=FILL_ALPHA) -> None:
    x, y = _cell_origin(square, user_color)
    roi = img[y:y + CELL, x:x + CELL]
    roi[:] = (np.array(color, np.float32) * alpha + roi.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    cv2.rectangle(img, (x, y), (x + CELL - 1, y + CELL - 1), color, 2)


def _text(img, text, org, color, scale=0.42, thickness=1) -> None:
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def _san(board: chess.Board, move: chess.Move, turn: chess.Color) -> str:
    b = board.copy()
    b.turn = turn
    return b.san(move) if move in b.legal_moves else move.uci()


def render_hud(state: OverlayState, mirror: bool = False) -> np.ndarray:
    img = np.full((HUD_SIZE, HUD_SIZE, 3), BACKGROUND, np.uint8)
    board = state.board

    # Dòng trên: trạng thái (rút gọn) + ô màu cho bên đến lượt.
    _text(img, state.status[:26], (4, 14), state.status_color)
    if board is not None:
        turn_color = (240, 240, 240) if board.turn == chess.WHITE else (40, 40, 40)
        cv2.rectangle(img, (HUD_SIZE - 16, 3), (HUD_SIZE - 4, 15), turn_color, -1)
        cv2.rectangle(img, (HUD_SIZE - 16, 3), (HUD_SIZE - 4, 15), (150, 150, 150), 1)

    if board is None:
        # Chưa có thế cờ: hiện trạng thái to ở giữa màn hình.
        words, lines, line = state.status.split(), [], ""
        for w in words:
            if len(line) + len(w) > 16:
                lines.append(line); line = ""
            line = f"{line} {w}".strip()
        lines.append(line)
        for i, ln in enumerate(lines[:4]):
            _text(img, ln, (12, 100 + 22 * i), state.status_color, 0.6, 1)
        return img[:, ::-1].copy() if mirror else img

    user = state.user_color
    for sq in chess.SQUARES:
        x, y = _cell_origin(sq, user)
        light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
        cv2.rectangle(img, (x, y), (x + CELL - 1, y + CELL - 1), LIGHT if light else DARK, -1)

    advice = state.advice
    bottom = ""
    if advice is not None:
        for move in advice.moves[1:]:  # các phương án phụ: chỉ kẻ viền ô đích
            x, y = _cell_origin(move.to_square, user)
            cv2.rectangle(img, (x, y), (x + CELL - 1, y + CELL - 1), ARROW_ALT, 2)
        if advice.moves:
            best = advice.moves[0]
            _fill_cell(img, best.from_square, user, FROM_COLOR)
            _fill_cell(img, best.to_square, user, TO_COLOR)
        for move in advice.alt_moves:
            _fill_cell(img, move.from_square, user, ALT_FROM_COLOR)
            _fill_cell(img, move.to_square, user, ALT_TO_COLOR)
        if advice.alt_moves:  # chưa rõ lượt: gợi ý cho cả hai bên
            parts = [f"T {_san(board, advice.moves[0], chess.WHITE)}"] if advice.moves else []
            parts += [f"D {_san(board, advice.alt_moves[0], chess.BLACK)}"]
            bottom = "  ".join(parts)
        elif advice.moves:
            bottom = f"{_san(board, advice.moves[0], board.turn)}  {advice.score_text}"

    # Quân = chấm tròn (Trắng: nền trắng chữ đen; Đen: nền đen chữ trắng) — rõ trên cả ô sáng lẫn ô tối.
    radius = CELL // 2 - 3
    for sq, piece in board.piece_map().items():
        cx, cy = hud_square_center(sq, user)
        disc, letter = ((245, 245, 245), (20, 20, 20)) if piece.color else ((25, 25, 25), (245, 245, 245))
        cv2.circle(img, (cx, cy), radius, disc, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), radius, (90, 90, 90), 1, cv2.LINE_AA)
        sym = piece.symbol().upper()
        (tw, th), _ = cv2.getTextSize(sym, FONT, 0.45, 1)
        cv2.putText(img, sym, (cx - tw // 2, cy + th // 2), FONT, 0.45, letter, 1, cv2.LINE_AA)

    if bottom:
        _text(img, bottom, (4, HUD_SIZE - 5), (255, 255, 255), 0.5, 1)
    if advice is not None and advice.blunder_cp is not None:
        _text(img, f"HO -{advice.blunder_cp / 100:.1f}", (HUD_SIZE - 70, HUD_SIZE - 5), (0, 0, 255), 0.45)
    return img[:, ::-1].copy() if mirror else img


class WindowDisplay:
    """Cửa sổ OpenCV mô phỏng màn hình kính (thử trên laptop). `scale` để phóng to cho dễ nhìn."""

    def __init__(self, name: str = "HUD kinh", scale: int = 1):
        self.name, self.scale = name, scale

    def show(self, img: np.ndarray) -> None:
        if self.scale != 1:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(self.name, img)

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.name)
        except cv2.error:
            pass


class ST7789Display:
    """Màn hình SPI ST7789 240x240 trên Raspberry Pi (chân theo docs/hardware/glasses_plan.md)."""

    def __init__(self, port: int = 0, device: int = 0, gpio_dc: int = 24, gpio_rst: int = 25,
                 gpio_light: int = 18, rotate: int = 0, spi_hz: int = 32_000_000):
        from luma.core.interface.serial import spi  # chỉ có trên Pi (pip install luma.lcd)
        from luma.lcd.device import st7789

        serial = spi(port=port, device=device, gpio_DC=gpio_dc, gpio_RST=gpio_rst, bus_speed_hz=spi_hz)
        self.device = st7789(serial, width=HUD_SIZE, height=HUD_SIZE, rotate=rotate, gpio_LIGHT=gpio_light)

    def show(self, img: np.ndarray) -> None:
        from PIL import Image

        self.device.display(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    def close(self) -> None:
        self.device.cleanup()
