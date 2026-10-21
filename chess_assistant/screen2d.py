"""Dữ liệu bàn cờ 2D: nhãn từ tên file FEN + biến đổi giống "camera điện thoại chụp màn hình".

Nguồn chính: Chess Positions (koryakinp, Kaggle, CC0) — ảnh bàn cờ 2D 400x400, Trắng ở dưới,
phần xếp quân của FEN nằm trong tên file (các hàng nối bằng '-'). Quân 2D nằm gọn trong ô nên
khung quân chính là ô cờ (thu vào một chút).
"""

from __future__ import annotations

from pathlib import Path

import chess
import cv2
import numpy as np

from chess_assistant.schema import class_of

Label = tuple[int, tuple[float, float, float, float]]
INSET = 0.04  # thu khung vào 4% mỗi phía: hình quân không chạm mép ô


def board_from_filename(name: str) -> chess.Board:
    fen = Path(name).stem.replace("-", "/")
    return chess.Board(fen + " w - - 0 1")


def flat_boxes(board: chess.Board, size: int) -> list[Label]:
    """Khung = ô cờ của mỗi quân trên ảnh bàn cờ vuông `size` px, Trắng ở dưới."""
    s = size / 8
    labels = []
    for sq, piece in sorted(board.piece_map().items()):
        x0, y0 = chess.square_file(sq) * s, (7 - chess.square_rank(sq)) * s
        d = s * INSET
        labels.append((class_of(piece), (x0 + d, y0 + d, x0 + s - d, y0 + s - d)))
    return labels


def _transform_labels(labels: list[Label], M: np.ndarray, size: tuple[int, int]) -> list[Label]:
    w, h = size
    out = []
    for cls, (x1, y1, x2, y2) in labels:
        pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
        q = cv2.perspectiveTransform(pts, M).reshape(4, 2)
        bx1, by1 = q.min(axis=0)
        bx2, by2 = q.max(axis=0)
        cx1, cy1, cx2, cy2 = max(bx1, 0), max(by1, 0), min(bx2, w), min(by2, h)
        if cx2 - cx1 > 2 and cy2 - cy1 > 2:
            out.append((cls, (float(cx1), float(cy1), float(cx2), float(cy2))))
    return out


def _screen_canvas(board_img: np.ndarray, rng: np.random.Generator, size=(1600, 900)):
    """Đặt bàn cờ lên một 'trang web' giả: nền giao diện, vài khung chữ nhật, bàn cờ ở vị trí ngẫu nhiên."""
    W, H = size
    canvas = np.empty((H, W, 3), np.uint8)
    canvas[:] = rng.integers(20, 90, 3)
    for _ in range(int(rng.integers(2, 7))):
        x1, y1 = int(rng.integers(0, W - 100)), int(rng.integers(0, H - 60))
        x2, y2 = x1 + int(rng.integers(60, 400)), y1 + int(rng.integers(30, 300))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), tuple(int(v) for v in rng.integers(30, 200, 3)), -1)
    bs = int(rng.integers(420, 820))
    bx, by = int(rng.integers(20, W - bs - 20)), int(rng.integers(20, H - bs - 20))
    canvas[by:by + bs, bx:bx + bs] = cv2.resize(board_img, (bs, bs), interpolation=cv2.INTER_AREA)
    k = bs / board_img.shape[1]
    S = np.float64([[k, 0, bx], [0, k, by], [0, 0, 1]])
    return canvas, S, (bx, by, bs)


def _effects(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    # Nhiễu sọc moiré của màn hình chụp qua camera.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fx, fy = rng.uniform(0.02, 0.25, 2) * rng.choice([-1, 1], 2)
    f *= (1 + rng.uniform(0, 0.08) * np.sin(2 * np.pi * (xx * fx + yy * fy)))[..., None]
    # Độ sáng, tương phản, ám màu.
    f = f * rng.uniform(0.6, 1.3) + rng.uniform(-30, 30)
    f *= rng.uniform(0.85, 1.15, 3)
    # Chói sáng: một đốm sáng mờ.
    if rng.random() < 0.5:
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(80, 400)
        glare = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2)) * rng.uniform(40, 140)
        f += glare[..., None]
    img = f.clip(0, 255).astype(np.uint8)
    # Mờ (lấy nét kém / rung tay) và nhiễu cảm biến.
    k = int(rng.choice([0, 3, 5, 7]))
    if k and rng.random() < 0.5:
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1 / k
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1)
        img = cv2.filter2D(img, -1, cv2.warpAffine(kernel, M, (k, k)))
    elif k:
        img = cv2.GaussianBlur(img, (k, k), 0)
    img = (img.astype(np.float32) + rng.normal(0, rng.uniform(0, 8), img.shape)).clip(0, 255).astype(np.uint8)
    # Nén JPEG (luồng video qua WiFi).
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 92))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def simulate_screen_photo(
    board_img: np.ndarray,
    labels: list[Label],
    rng: np.random.Generator,
    out_size: tuple[int, int] = (1280, 720),
    effects: bool = True,
) -> tuple[np.ndarray, list[Label]]:
    """Ảnh 'điện thoại chụp màn hình đang hiện bàn cờ' + khung quân đã biến đổi theo."""
    canvas, S, (bx, by, bs) = _screen_canvas(board_img, rng)
    H_s, W_s = canvas.shape[:2]
    w, h = out_size
    board_quad = np.float32([[bx, by], [bx + bs, by], [bx + bs, by + bs], [bx, by + bs]]).reshape(-1, 1, 2)
    for _ in range(50):
        # Điện thoại đôi khi cầm dọc: cả màn hình xoay 90° nguyên khối (giữ tỉ lệ, không kéo méo).
        portrait = rng.random() < 0.2
        # Màn hình nhìn nghiêng: cạnh xa hẹp hơn, xoay nhẹ, lệch khỏi tâm.
        sw = rng.uniform(0.75, 1.35) * (h * 1.2 if portrait else w)
        sh = sw * H_s / W_s * rng.uniform(0.75, 1.0)
        top = rng.uniform(0.0, 0.18) * sw
        cx, cy = w / 2 + rng.uniform(-0.15, 0.15) * w, h / 2 + rng.uniform(-0.12, 0.12) * h
        quad = np.float32([[cx - sw / 2 + top, cy - sh / 2], [cx + sw / 2 - top, cy - sh / 2],
                           [cx + sw / 2, cy + sh / 2], [cx - sw / 2, cy + sh / 2]])
        a = np.deg2rad(rng.uniform(-8, 8) + (rng.choice([-90, 90]) if portrait else 0))
        R = np.float32([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        quad = (quad - [cx, cy]) @ R.T + [cx, cy]
        P = cv2.getPerspectiveTransform(np.float32([[0, 0], [W_s, 0], [W_s, H_s], [0, H_s]]), quad.astype(np.float32))
        b = cv2.perspectiveTransform(board_quad, P).reshape(4, 2)
        if (b >= 0).all() and (b[:, 0] < w).all() and (b[:, 1] < h).all():
            break  # bàn cờ phải nằm trọn trong ảnh

    background = rng.normal(rng.uniform(20, 120), 15, (h // 8, w // 8, 3)).clip(0, 255).astype(np.uint8)
    photo = cv2.resize(background, (w, h), interpolation=cv2.INTER_CUBIC)
    screen = cv2.warpPerspective(canvas, P, (w, h))
    mask = cv2.warpPerspective(np.full((H_s, W_s), 255, np.uint8), P, (w, h)) > 0
    photo[mask] = screen[mask]
    if effects:
        photo = _effects(photo, rng)
    return photo, _transform_labels(labels, P @ S, (w, h))
