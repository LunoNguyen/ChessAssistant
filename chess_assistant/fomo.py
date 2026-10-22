"""FOMO (Faster Objects, More Objects): nhận diện quân bằng *tâm* thay vì khung, đủ nhẹ cho ESP32-S3.

Model ra một lưới xác suất (G, G, C+1) với G = cạnh ảnh / STRIDE; kênh 0 là nền. Mỗi quân là một
điểm ở tâm đế quân (đúng điểm cần để biết quân đứng ô nào), 4 góc bàn cờ cũng là điểm (lớp "corner")
để ESP32 tự tính phép chiếu phối cảnh mà không cần dò bàn cờ bằng OpenCV.

File này chỉ dùng numpy + cv2 (không import python-chess) vì được chép nguyên vào gói dữ liệu để
notebook Colab dùng chung — firmware ESP32 (firmware/esp32s3_fomo/main/fomo_decode.cc) làm lại
đúng các bước decode / góc / chiếu ô ở đây.

Quy ước:
- Điểm: (lớp, x, y), x/y chuẩn hoá về [0, 1) theo ảnh. Lớp 0..11 theo PIECE_CLASSES, CORNER = 12.
- Góc bàn cờ theo thứ tự a1, h1, h8, a8 (như chess_assistant.board).
- Ô: 0..63 = a1, b1, ..., h8 (như python-chess).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PIECE_CLASSES = "PNBRQKpnbrqk"  # trùng chess_assistant.schema (không import để khỏi cần python-chess)
CORNER = 12
STRIDE = 4  # lưới ra = ảnh vào / 4 (FOMO gốc là 8: quá thô cho hàng cờ xa camera, xem notebook)

# Kênh ra của model (kênh 0 = nền, không liệt kê).
SCHEMES: dict[str, tuple[str, ...]] = {
    "color": ("white", "black", "corner"),
    "full": tuple(PIECE_CLASSES) + ("corner",),
}

Point = tuple[int, float, float]


def channel_of(point_cls: int, scheme: str) -> int:
    """Kênh model (1-based) của một lớp điểm."""
    names = SCHEMES[scheme]
    if point_cls == CORNER:
        return len(names)
    if scheme == "color":
        return 1 if point_cls < 6 else 2
    return point_cls + 1


def make_target(points: list[Point], grid: int, scheme: str) -> np.ndarray:
    """Nhãn (grid, grid) uint8: kênh của điểm rơi vào ô lưới, 0 = nền.

    Góc ghi trước, quân ghi sau: nếu góc và quân ở góc bàn chung một ô lưới thì giữ quân."""
    target = np.zeros((grid, grid), np.uint8)
    for cls, x, y in sorted(points, key=lambda p: p[0] != CORNER):
        i, j = int(y * grid), int(x * grid)
        if 0 <= i < grid and 0 <= j < grid:
            target[i, j] = channel_of(cls, scheme)
    return target


# ---------------------------------------------------------------------------
# Decode lưới xác suất -> điểm
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    channel: int     # kênh model (1-based)
    x: float         # tâm, chuẩn hoá [0, 1)
    y: float
    score: float     # xác suất lớn nhất trong cụm


def decode(probs: np.ndarray, threshold: float = 0.5) -> list[Detection]:
    """Mỗi ô lưới có kênh thắng (khác nền) >= threshold và không thua ô nào xung quanh (3x3) ở kênh đó
    là một điểm, đặt ở tâm ô lưới. Sắp theo score giảm dần.

    Không gộp cụm liền nhau như FOMO gốc: ở hàng xa camera mỗi ô cờ chỉ rộng ~1 ô lưới, gộp cụm sẽ
    nhập cả hàng tốt thành một điểm. Một quân sinh 2 điểm thì vẫn rơi cùng ô cờ nên không hại gì."""
    grid_h, grid_w, _ = probs.shape
    winner = probs.argmax(axis=-1)
    best = probs.max(axis=-1)
    out = []
    for i in range(grid_h):
        for j in range(grid_w):
            c = int(winner[i, j])
            if c == 0 or best[i, j] < threshold:
                continue
            around = probs[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2, c]
            if best[i, j] >= around.max():
                out.append(Detection(c, (j + 0.5) / grid_w, (i + 0.5) / grid_h, float(best[i, j])))
    return sorted(out, key=lambda d: -d.score)


# ---------------------------------------------------------------------------
# Góc bàn cờ và chiếu điểm về ô
# ---------------------------------------------------------------------------

def complete_corners(points: list[tuple[float, float]]) -> np.ndarray | None:
    """4 góc từ các góc tìm được (đã sắp theo score): >4 lấy 4 điểm đầu; 3 điểm thì suy ra góc thứ 4
    theo hình bình hành (đỉnh có góc lớn nhất là đỉnh kề hai đỉnh kia); <3 -> None."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    if len(pts) >= 4:
        return pts[:4]
    if len(pts) < 3:
        return None

    def angle(k: int) -> float:
        p, a, b = pts[k], pts[(k + 1) % 3], pts[(k + 2) % 3]
        u, v = a - p, b - p
        return float(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12), -1, 1)))

    k = max(range(3), key=angle)
    a, b = pts[(k + 1) % 3], pts[(k + 2) % 3]
    return np.vstack([pts, a + b - pts[k]])


def order_corners(points: np.ndarray) -> np.ndarray:
    """Sắp 4 góc theo vòng: dưới-trái, dưới-phải, trên-phải, trên-trái của ảnh.

    Đây chỉ là thứ tự *tạm* a1, h1, h8, a8 (giả sử Trắng ở gần camera); hướng thật được xác định
    sau bằng màu quân hoặc thế khởi đầu (thử 4 phép xoay, xem rotate_squares)."""
    pts = np.asarray(points, np.float64).reshape(4, 2)
    center = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ring = pts[np.argsort(-ang)]          # y hướng xuống: dưới-trái -> dưới-phải -> trên-phải -> trên-trái
    start = int(np.argmax(ring[:, 1] - ring[:, 0]))
    return np.roll(ring, -start, axis=0)


def solve_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """H 3x3 (h33 = 1) đưa 4 điểm src -> dst, giải hệ 8x8 (firmware giải đúng hệ này)."""
    A, b = [], []
    for (x, y), (u, v) in zip(np.asarray(src, np.float64), np.asarray(dst, np.float64)):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    h = np.linalg.solve(np.array(A), np.array(b))
    return np.append(h, 1.0).reshape(3, 3)


# Toạ độ "bàn cờ đơn vị": u theo cột a->h, v theo hàng 1->8, cả hai trong [0, 1].
UNIT_CORNERS = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64)  # a1, h1, h8, a8


def board_homography(corners: np.ndarray) -> np.ndarray:
    """Ảnh -> bàn cờ đơn vị, từ 4 góc a1, h1, h8, a8."""
    return solve_homography(corners, UNIT_CORNERS)


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    return q[:, :2] / q[:, 2:3]


def square_of(H: np.ndarray, x: float, y: float) -> int | None:
    (u, v), = apply_homography(H, [[x, y]])
    f, r = int(np.floor(u * 8)), int(np.floor(v * 8))
    return r * 8 + f if 0 <= f < 8 and 0 <= r < 8 else None


def square_centers(corners: np.ndarray) -> np.ndarray:
    """(64, 2) tâm 64 ô trên ảnh, theo thứ tự a1..h8."""
    Hinv = np.linalg.inv(board_homography(corners))
    uv = [((f + 0.5) / 8, (r + 0.5) / 8) for r in range(8) for f in range(8)]
    return apply_homography(Hinv, uv)


def occupancy(detections: list[Detection], corners: np.ndarray, scheme: str) -> list[int]:
    """64 kênh (0 = trống) theo thứ tự a1..h8: mỗi ô giữ quân có score cao nhất rơi vào ô đó."""
    corner_ch = len(SCHEMES[scheme])
    occ, score = [0] * 64, [0.0] * 64
    try:
        H = board_homography(corners)
    except np.linalg.LinAlgError:  # góc trùng / thẳng hàng: coi như không thấy bàn cờ
        return occ
    for d in detections:
        if d.channel == corner_ch:
            continue
        sq = square_of(H, d.x, d.y)
        if sq is not None and d.score > score[sq]:
            occ[sq], score[sq] = d.channel, d.score
    return occ


def rotate_squares(occ: list[int], k: int) -> list[int]:
    """Xoay bàn cờ k lần 90° (dùng khi chưa biết góc nào là a1)."""
    grid = np.array(occ).reshape(8, 8)  # [hàng, cột]
    return [int(v) for v in np.rot90(grid, k).reshape(-1)]


def target_occupancy(points: list[Point], scheme: str) -> tuple[list[int], np.ndarray] | None:
    """Đáp án (64 kênh, 4 góc) từ nhãn điểm; None nếu nhãn không có đủ 4 góc."""
    corners = np.array([(x, y) for c, x, y in points if c == CORNER])
    if len(corners) != 4:
        return None
    try:
        H = board_homography(corners)
    except np.linalg.LinAlgError:
        return None
    occ = [0] * 64
    for c, x, y in points:
        if c != CORNER and (sq := square_of(H, x, y)) is not None:
            occ[sq] = channel_of(c, scheme)
    return occ, corners


def square_accuracy(pred: list[int], truth: list[int], any_rotation: bool = False) -> float:
    """Tỉ lệ ô đúng; any_rotation: lấy phép xoay tốt nhất (góc dự đoán chỉ có thứ tự tạm)."""
    ks = range(4) if any_rotation else [0]
    return max(float(np.mean(np.array(rotate_squares(pred, k)) == np.array(truth))) for k in ks)


# ---------------------------------------------------------------------------
# Tăng cường dữ liệu: cắt khung vuông quanh bàn cờ (giống camera kính nhìn xuống bàn)
# ---------------------------------------------------------------------------

def board_view(
    img: np.ndarray,
    points: list[Point],
    size: int,
    fill: float = 0.8,
    angle: float = 0.0,
    flip: bool = False,
    offset: tuple[float, float] = (0.0, 0.0),
    border: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, list[Point]]:
    """Ảnh vuông size x size: bàn cờ xoay `angle` độ, chiếm `fill` cạnh ảnh, tâm lệch `offset`
    (tỉ lệ cạnh ảnh). Một lần warpAffine nên ảnh và điểm luôn khớp nhau."""
    h, w = img.shape[:2]
    px = np.array([(x * w, y * h) for _, x, y in points], np.float64).reshape(-1, 2)
    ref = np.array([(x * w, y * h) for c, x, y in points if c == CORNER], np.float64).reshape(-1, 2)
    if len(ref) < 2:
        ref = px if len(px) >= 2 else np.array([[0, 0], [w, h]], np.float64)
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    if flip:
        R = R @ np.diag([-1.0, 1.0])
    rot = (ref - ref.mean(axis=0)) @ R.T
    extent = max(float((rot.max(axis=0) - rot.min(axis=0)).max()), 1.0)
    s = fill * size / extent
    center = (rot.max(axis=0) + rot.min(axis=0)) / 2 + ref.mean(axis=0) @ R.T  # tâm khung của bàn, sau xoay
    t = np.array([size / 2 + offset[0] * size, size / 2 + offset[1] * size]) - s * center
    M = np.c_[s * R, t]  # toạ độ liên tục (tâm pixel i ở i + 0.5)
    src, A = img, s * R
    if s < 0.8:  # thu nhỏ nhiều: warpAffine không có INTER_AREA nên thu nhỏ trước cho đỡ răng cưa
        src = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        A = s * R @ np.diag([w / src.shape[1], h / src.shape[0]])
    # OpenCV đặt tâm pixel ở số nguyên: đổi toạ độ liên tục -> toạ độ OpenCV.
    M_cv = np.c_[A, A @ [0.5, 0.5] + t - 0.5]
    out = cv2.warpAffine(src, M_cv, (size, size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    moved = px @ M[:, :2].T + M[:, 2] if len(px) else px
    new_points = [(c, float(x / size), float(y / size))
                  for (c, _, _), (x, y) in zip(points, moved) if 0 <= x < size and 0 <= y < size]
    return out, new_points


def random_view(img: np.ndarray, points: list[Point], size: int, rng: np.random.Generator,
                fill: tuple[float, float] = (0.5, 0.97)) -> tuple[np.ndarray, list[Point]]:
    f = float(rng.uniform(*fill))
    slack = (1 - f) / 2
    angle = float(rng.integers(0, 4) * 90 + rng.uniform(-12, 12))
    out, pts = board_view(img, points, size, fill=f, angle=angle, flip=bool(rng.random() < 0.5),
                          offset=tuple(rng.uniform(-slack, slack, 2)),
                          border=tuple(int(v) for v in rng.integers(0, 256, 3)))
    return photometric(out, rng), pts


def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sáng/tối, tương phản, ám màu, mờ, nhiễu, nén JPEG — camera OV2640 rẻ tiền."""
    f = img.astype(np.float32) * rng.uniform(0.6, 1.4) + rng.uniform(-35, 35)
    f *= rng.uniform(0.85, 1.15, 3)
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        f = cv2.GaussianBlur(f, (k, k), 0)
    f += rng.normal(0, rng.uniform(0, 8), f.shape)
    img = f.clip(0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 90))])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> RGB float32 [-1, 1] (chuẩn MobileNetV2). Firmware: q = (rgb/127.5 - 1)/scale + zp."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
