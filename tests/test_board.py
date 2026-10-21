import chess
import cv2
import numpy as np
import pytest

from chess_assistant.board import (
    CANVAS_PX,
    SQUARE_PX,
    BoardView,
    crop_squares,
    grid_lines_image,
    square_center_image,
)

FLAT = 800
# Góc của ảnh sơ đồ phẳng theo thứ tự a1, h1, h8, a8 (Trắng ở dưới).
FLAT_CORNERS = np.float32([[0, FLAT], [FLAT, FLAT], [FLAT, 0], [0, 0]])

# Vị trí 4 góc a1, h1, h8, a8 trong ảnh camera 1280x720 cho từng góc đặt camera.
# Cạnh gần camera (đáy ảnh) rộng hơn cạnh xa (đỉnh ảnh).
CAMERA_SETUPS = {
    "white_side": ([[300, 650], [980, 650], [860, 180], [420, 180]], "up"),
    "black_side": ([[860, 180], [420, 180], [300, 650], [980, 650]], "down"),
    # Camera đặt bên cột a: cột a ở đáy ảnh, cột h ở đỉnh ảnh.
    "a_file_side": ([[980, 650], [860, 180], [420, 180], [300, 650]], "right"),
}


def flat_board_with_marker(square: chess.Square) -> np.ndarray:
    img = np.zeros((FLAT, FLAT, 3), np.uint8)
    s = FLAT // 8
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        color = (60, 60, 60) if (f + r) % 2 == 0 else (200, 200, 200)
        cv2.rectangle(img, (f * s, (7 - r) * s), ((f + 1) * s - 1, (8 - r) * s - 1), color, -1)
    f, r = chess.square_file(square), chess.square_rank(square)
    cv2.circle(img, (f * s + s // 2, (7 - r) * s + s // 2), s // 3, (0, 0, 255), -1)
    return img


def simulate_camera(flat: np.ndarray, corners) -> np.ndarray:
    M = cv2.getPerspectiveTransform(FLAT_CORNERS, np.float32(corners))
    return cv2.warpPerspective(flat, M, (1280, 720))


def red_fraction(img: np.ndarray) -> float:
    b, g, r = cv2.split(img.astype(int))
    return float(np.mean((r > 150) & (g < 80) & (b < 80)))


@pytest.mark.parametrize("setup", CAMERA_SETUPS)
def test_marker_lands_in_correct_square(setup):
    corners, _ = CAMERA_SETUPS[setup]
    frame = simulate_camera(flat_board_with_marker(chess.E4), corners)
    view = BoardView.from_corners(corners)

    crops = view.crops(frame, pad=0.75)
    # Sau khi xoay, ô gốc luôn là phần dưới cùng của ảnh cắt.
    base = {sq: crop[-SQUARE_PX:] for sq, crop in zip(chess.SQUARES, crops)}

    assert red_fraction(base[chess.E4]) > 0.25
    for sq in (chess.D4, chess.F4, chess.E3, chess.E5, chess.A1, chess.H8):
        assert red_fraction(base[sq]) < 0.01, chess.square_name(sq)


@pytest.mark.parametrize("setup", CAMERA_SETUPS)
def test_lean_direction(setup):
    corners, expected = CAMERA_SETUPS[setup]
    assert BoardView.from_corners(corners).lean == expected


@pytest.mark.parametrize("lean", ["up", "down", "left", "right"])
@pytest.mark.parametrize("pad", [0.0, 0.5, 1.0])
def test_crop_shapes(lean, pad):
    canvas = np.zeros((CANVAS_PX, CANVAS_PX, 3), np.uint8)
    crops = crop_squares(canvas, lean, pad)
    assert len(crops) == 64
    assert {c.shape for c in crops} == {(round(SQUARE_PX * (1 + pad)), SQUARE_PX, 3)}


def test_pad_too_large():
    canvas = np.zeros((CANVAS_PX, CANVAS_PX, 3), np.uint8)
    with pytest.raises(ValueError):
        crop_squares(canvas, "up", pad=1.5)


def test_projection_back_to_image():
    corners, _ = CAMERA_SETUPS["white_side"]
    view = BoardView.from_corners(corners)
    # Tâm ô chiếu ngược phải trùng với tâm ô thật trong ảnh camera mô phỏng.
    M = cv2.getPerspectiveTransform(FLAT_CORNERS, np.float32(corners))
    s = FLAT // 8
    for sq in (chess.A1, chess.H8, chess.E4):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        flat_center = np.float32([[[f * s + s / 2, (7 - r) * s + s / 2]]])
        expected = cv2.perspectiveTransform(flat_center, M).reshape(2)
        assert np.linalg.norm(np.subtract(square_center_image(view.H, sq), expected)) <= 1, chess.square_name(sq)
    lines = grid_lines_image(view.H)
    assert len(lines) == 18
    # Đường ngoài cùng đi qua đúng các góc đã chọn.
    flat_pts = {p for seg in lines for p in seg}
    for c in corners:
        assert min(np.linalg.norm(np.subtract(p, c)) for p in flat_pts) <= 1
