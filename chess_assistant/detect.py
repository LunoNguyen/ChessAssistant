"""Tự động tìm bàn cờ trong ảnh camera, không cần train model.

Các bước:
1. Tìm các đường thẳng dài (Canny + Hough), chia làm 2 họ theo hướng
   (họ "cột" và họ "hàng" của lưới).
2. Mỗi cặp đường liền kề của họ này + cặp liền kề của họ kia tạo một "ô hạt giống".
   Từ ô đó suy ra phép chiếu lưới -> ảnh, rồi lần lượt khớp thêm các đường khác vào
   toạ độ nguyên của lưới và fit lại phép chiếu cho chính xác.
3. Trong lưới tìm được, chọn cửa sổ 8x8 có nhiều đường khớp nhất và màu ô xen kẽ
   sáng/tối rõ nhất (viền gỗ bên ngoài không xen kẽ nên bị loại).
4. Xác định hướng (góc nào là a1): a1 là ô tối, và phía có quân trắng là phía Trắng.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from chess_assistant.board import MARGIN_PX, SQUARE_PX, homography, warp

WORK_WIDTH = 640          # dò trên ảnh thu nhỏ cho nhanh
MAX_LINES_PER_FAMILY = 25
MATCH_TOL = 0.2           # sai số cho phép khi khớp đường vào lưới (đơn vị: cạnh ô)
MIN_LINES = 10            # số đường lưới tối thiểu (tối đa 18) để chấp nhận
MIN_ALTERNATION = 0.35    # độ xen kẽ sáng/tối tối thiểu (0..1)
OUTSIDE_WEIGHT = 1.0      # phạt khi dải ô ngay ngoài cửa sổ vẫn xen kẽ (cửa sổ lệch vào trong bàn)
WINDOW_SLACK = 1          # xét thêm cửa sổ lệch 1 ô ra ngoài vùng đường lưới đã khớp: bàn giấy có
                          # đường viền in cách mép 1 ô, đôi khi được khớp thay cho mép xa bị quân che


@dataclass
class Detection:
    quad: np.ndarray      # 4x2, toạ độ ảnh gốc, thứ tự vòng quanh bàn cờ (chưa biết a1)
    lines: int            # số đường lưới khớp (tối đa 18)
    alternation: float    # độ xen kẽ sáng/tối của 64 ô (0..1)

    @property
    def score(self) -> float:
        return self.lines / 18 + self.alternation


# ---------------------------------------------------------------------------
# 1. Đường thẳng
# ---------------------------------------------------------------------------

_ANGLE_BINS = 12
_ANGLE_SLACK = np.deg2rad(6)


def _find_lines(gray: np.ndarray) -> np.ndarray:
    """Các đường thẳng (rho, theta), đã loại trùng, mạnh nhất trước.

    Mỗi điểm cạnh chỉ bỏ phiếu cho đường có pháp tuyến cùng hướng gradient tại đó.
    Nhờ vậy loại được "đường chéo" đi qua các góc ô — với bàn cờ nhỏ trong ảnh,
    Hough thường lại ưu tiên những đường chéo này hơn đường lưới thật.
    """
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, 0.5 * otsu, otsu) > 0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    h, w = gray.shape
    threshold = int(0.12 * min(h, w))

    found = []
    step = np.pi / _ANGLE_BINS
    for b in range(_ANGLE_BINS):
        lo, hi = b * step, (b + 1) * step
        dist = np.abs(np.mod(angle - (lo + hi) / 2 + np.pi / 2, np.pi) - np.pi / 2)
        mask = (edges & (dist < step / 2 + _ANGLE_SLACK)).astype(np.uint8) * 255
        res = cv2.HoughLinesWithAccumulator(mask, 1, np.pi / 360, threshold, min_theta=lo, max_theta=hi)
        if res is not None:
            found.append(res.reshape(-1, 3))
    if not found:
        return np.empty((0, 2))
    raw = np.concatenate(found)
    raw = raw[np.argsort(-raw[:, 2])][:300, :2]

    center = np.array([w / 2, h / 2])
    kept: list[tuple[float, float]] = []
    for rho, theta in raw:
        if not any(_same_line((rho, theta), k, center) for k in kept):
            kept.append((rho, theta))
        if len(kept) >= 2 * MAX_LINES_PER_FAMILY + 10:
            break
    return np.array(kept)


def _same_line(l1, l2, center, angle_tol=np.deg2rad(2.0), dist_tol=4.0) -> bool:
    d = abs(l1[1] - l2[1])
    if min(d, np.pi - d) > angle_tol:
        return False
    # Chân đường vuông góc từ tâm ảnh xuống l1, đo khoảng cách tới l2.
    n1 = np.array([np.cos(l1[1]), np.sin(l1[1])])
    foot = center + (l1[0] - n1 @ center) * n1
    n2 = np.array([np.cos(l2[1]), np.sin(l2[1])])
    return abs(n2 @ foot - l2[0]) < dist_tol


def _split_families(lines: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Chia đường thành 2 họ theo hướng. Trả về 2 mảng đường dạng thuần nhất
    (a, b, c) với ax + by + c = 0, mỗi họ đã sắp xếp theo vị trí cắt ngang ảnh."""
    if len(lines) < 6:
        return None
    rho, theta = lines[:, 0], lines[:, 1]
    doubled = np.column_stack([np.cos(2 * theta), np.sin(2 * theta)]).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    _, labels, centers = cv2.kmeans(doubled, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()

    families = []
    for k in range(2):
        phi = np.arctan2(centers[k, 1], centers[k, 0]) / 2
        mean_normal = np.array([np.cos(phi), np.sin(phi)])
        idx = np.flatnonzero(labels == k)
        n = np.column_stack([np.cos(theta[idx]), np.sin(theta[idx])])
        r = rho[idx].copy()
        # Lật pháp tuyến cho cùng chiều với hướng trung bình để so vị trí được.
        flip = n @ mean_normal < 0
        n[flip] *= -1
        r[flip] *= -1
        # Bỏ đường lệch quá xa hướng chung của họ.
        ok = n @ mean_normal > np.cos(np.deg2rad(35))
        n, r = n[ok][:MAX_LINES_PER_FAMILY], r[ok][:MAX_LINES_PER_FAMILY]
        if len(r) < 3:
            return None
        order = np.argsort(r - n @ center)
        families.append(np.column_stack([n, -r])[order])

    a_dir, b_dir = (f[:, :2].mean(0) for f in families)
    cos_between = abs(a_dir @ b_dir) / (np.linalg.norm(a_dir) * np.linalg.norm(b_dir))
    if cos_between > np.cos(np.deg2rad(25)):
        return None
    return families[0], families[1]


# ---------------------------------------------------------------------------
# 2. Khớp lưới
# ---------------------------------------------------------------------------

def _match(L: np.ndarray, H: np.ndarray, axis: int, span: tuple[int, int], allowed: tuple[int, int]) -> dict[int, int]:
    """Khớp các đường ảnh `L` vào đường lưới toạ độ nguyên.

    H: lưới -> ảnh. Đường ảnh l thành đường lưới l^T H = (p, q, r): p*u + q*v + r = 0.
    axis=0: họ u = hằng số (đo u tại hai đầu `span` của v); axis=1: ngược lại.
    Trả về {chỉ số đường lưới: chỉ số đường ảnh}.
    """
    Lp = L @ H
    if axis == 0:
        p, q, r = Lp[:, 0], Lp[:, 1], Lp[:, 2]
    else:
        p, q, r = Lp[:, 1], Lp[:, 0], Lp[:, 2]
    s = np.array(span, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = -(q[:, None] * s[None, :] + r[:, None]) / p[:, None]
    n = np.round(vals.mean(axis=1))
    err = np.abs(vals - n[:, None]).max(axis=1)
    ok = np.isfinite(err) & (err < MATCH_TOL) & (n >= allowed[0]) & (n <= allowed[1])
    best: dict[int, int] = {}
    for i in np.flatnonzero(ok):
        k = int(n[i])
        if k not in best or err[i] < err[best[k]]:
            best[k] = int(i)
    return best


def _all_intersections(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Giao điểm của mọi cặp (A[i], B[k]): mảng (len(A), len(B), 2), NaN nếu song song."""
    P = np.cross(A[:, None, :], B[None, :, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        X = P[..., :2] / P[..., 2:]
    X[np.abs(P[..., 2]) < 1e-9] = np.nan
    return X


def _fit(X: np.ndarray, Ma: dict[int, int], Mb: dict[int, int]) -> np.ndarray | None:
    na, ia = np.array(list(Ma.keys())), np.array(list(Ma.values()))
    nb, ib = np.array(list(Mb.keys())), np.array(list(Mb.values()))
    image = X[np.ix_(ia, ib)].reshape(-1, 2)
    lattice = np.stack(np.meshgrid(na, nb, indexing="ij"), -1).reshape(-1, 2)
    ok = np.isfinite(image).all(axis=1)
    if ok.sum() < 4:
        return None
    H, _ = cv2.findHomography(lattice[ok].astype(np.float32), image[ok].astype(np.float32), 0)
    return H


def _grow(A: np.ndarray, B: np.ndarray, X: np.ndarray, i: int, k: int, shape: tuple[int, int]):
    """Từ ô hạt giống (A[i], A[i+1], B[k], B[k+1]) mở rộng thành lưới."""
    quad = np.float32([X[i, k], X[i + 1, k], X[i + 1, k + 1], X[i, k + 1]])
    if not np.isfinite(quad).all():
        return None
    h, w = shape
    side = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    if side.min() < 6 or side.max() > min(h, w) / 5 or not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
        return None
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), quad)

    Ma, Mb = {0: i, 1: i + 1}, {0: k, 1: k + 1}
    for _ in range(12):
        ua, ub, va, vb = min(Ma), max(Ma), min(Mb), max(Mb)
        newA = _match(A, H, 0, (va, vb), (ua - 2, ub + 2))
        newB = _match(B, H, 1, (ua, ub), (va - 2, vb + 2))
        if len(newA) < 2 or len(newB) < 2:
            return None
        H2 = _fit(X, newA, newB)
        if H2 is None:
            return None
        done = newA == Ma and newB == Mb
        Ma, Mb, H = newA, newB, H2
        if done:
            break
    return H, Ma, Mb


# ---------------------------------------------------------------------------
# 3. Chọn cửa sổ 8x8
# ---------------------------------------------------------------------------

_SIGN = np.where(np.add.outer(np.arange(8), np.arange(8)) % 2 == 0, 1.0, -1.0)
_OFFSETS = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]  # góc ô: ít bị chân quân che


def _window_range(M: dict[int, int]) -> np.ndarray:
    lo, hi = min(M), max(M)
    k = WINDOW_SLACK
    return np.arange(hi - 8 - k, lo + 1 + k) if hi - lo <= 8 else np.arange(lo - k, hi - 7 + k)


def _strip_alternation(strips: np.ndarray, sign: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Độ xen kẽ của các dải 8 ô ngay ngoài từng cửa sổ (0 nếu dải ra ngoài ảnh). strips: (nu, nv, 8)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        alt = np.abs((sign * (strips - mean[..., None])).mean(axis=-1)) / sd
    return np.where(np.isnan(strips).any(axis=-1), 0.0, alt)


def _best_window(gray: np.ndarray, H: np.ndarray, Ma: dict[int, int], Mb: dict[int, int]):
    us, vs = _window_range(Ma), _window_range(Mb)
    nu, nv = len(us), len(vs)
    # Lấy thêm 1 ô mỗi phía để xét các dải ngay ngoài cửa sổ.
    u_lo, v_lo = us[0] - 1, vs[0] - 1
    I, J = np.meshgrid(np.arange(u_lo, us[-1] + 9), np.arange(v_lo, vs[-1] + 9), indexing="ij")
    pts = np.stack([np.stack([I + dx, J + dy], -1) for dx, dy in _OFFSETS], axis=2)
    img = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(pts.shape)
    h, w = gray.shape
    x, y = np.rint(img[..., 0]).astype(int), np.rint(img[..., 1]).astype(int)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    samples = np.where(inside, gray[y.clip(0, h - 1), x.clip(0, w - 1)], np.nan)
    C = samples.mean(axis=-1)

    # Mọi cửa sổ 8x8 cùng lúc: cửa sổ (u0, v0) bắt đầu ở hàng u0 - u_lo, cột v0 - v_lo của C.
    win = np.lib.stride_tricks.sliding_window_view(C, (8, 8))[1:1 + nu, 1:1 + nv]  # (nu, nv, 8, 8)
    valid = ~np.isnan(win).any(axis=(-1, -2))  # bàn cờ phải nằm trọn trong khung hình
    with np.errstate(invalid="ignore", divide="ignore"):
        m = win.mean(axis=(-1, -2))
        sd = win.std(axis=(-1, -2))
        alt = np.abs((_SIGN * (win - m[..., None, None])).mean(axis=(-1, -2))) / sd
    valid &= sd >= 4

    # Cửa sổ đúng: dải ngay ngoài là lề/viền, không xen kẽ. Cửa sổ lệch 1 hàng: một dải ngoài lại là
    # hàng thật của bàn cờ, vẫn xen kẽ -> trừ điểm. (Lấy trị tuyệt đối nên chỉ cần mẫu xen kẽ.)
    edge_sign = np.where(np.arange(8) % 2 == 0, -1.0, 1.0)
    rows = np.lib.stride_tricks.sliding_window_view(C, 8, axis=1)  # (hàng, cột bắt đầu, 8)
    cols = np.moveaxis(np.lib.stride_tricks.sliding_window_view(C, 8, axis=0), -1, -1)  # (hàng bắt đầu, cột, 8)
    outside = np.maximum.reduce([
        _strip_alternation(rows[0:nu, 1:1 + nv], edge_sign, m, sd),          # hàng ngay trên
        _strip_alternation(rows[9:9 + nu, 1:1 + nv], edge_sign, m, sd),      # hàng ngay dưới
        _strip_alternation(cols[1:1 + nu, 0:nv], edge_sign, m, sd),          # cột ngay trái
        _strip_alternation(cols[1:1 + nu, 9:9 + nv], edge_sign, m, sd),      # cột ngay phải
    ])

    na = np.array(list(Ma)); nb = np.array(list(Mb))
    lines_u = ((na[None, :] >= us[:, None]) & (na[None, :] <= us[:, None] + 8)).sum(axis=1)
    lines_v = ((nb[None, :] >= vs[:, None]) & (nb[None, :] <= vs[:, None] + 8)).sum(axis=1)
    lines = lines_u[:, None] + lines_v[None, :]
    score = np.where(valid, lines / 18 + alt - OUTSIDE_WEIGHT * outside, -np.inf)
    if not np.isfinite(score).any():
        return None
    i, j = np.unravel_index(int(np.argmax(score)), score.shape)
    return float(score[i, j]), int(us[i]), int(vs[j]), int(lines[i, j]), float(alt[i, j])


def detect_quad(frame: np.ndarray) -> Detection | None:
    """Tìm vùng chơi 8x8. Trả về None nếu không thấy bàn cờ đủ tin cậy."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    scale = min(1.0, WORK_WIDTH / gray.shape[1])
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape

    fam = _split_families(_find_lines(gray), np.array([w / 2, h / 2]))
    if fam is None:
        return None
    A, B = fam
    X = _all_intersections(A, B)

    best = None
    seen: set[tuple] = set()
    covered: list[tuple[set[int], set[int]]] = []  # các lưới đã thử: (đường A, đường B)
    for i in range(len(A) - 1):
        for k in range(len(B) - 1):
            # Ô hạt giống nằm trong một lưới đã thử thì sẽ hội tụ về đúng lưới đó.
            if any({i, i + 1} <= ca and {k, k + 1} <= cb for ca, cb in covered):
                continue
            grown = _grow(A, B, X, i, k, (h, w))
            if grown is None:
                continue
            H, Ma, Mb = grown
            covered.append((set(Ma.values()), set(Mb.values())))
            key = (tuple(sorted(Ma.items())), tuple(sorted(Mb.items())))
            if key in seen:
                continue
            seen.add(key)
            win = _best_window(gray, H, Ma, Mb)
            if win is None:
                continue
            if best is None or win[0] > best[0][0]:
                best = (win, H)

    if best is None:
        return None
    (_, u0, v0, lines, alt), H = best
    if lines < MIN_LINES or alt < MIN_ALTERNATION:
        return None
    lattice = np.float32([[u0, v0], [u0 + 8, v0], [u0 + 8, v0 + 8], [u0, v0 + 8]])
    quad = cv2.perspectiveTransform(lattice.reshape(-1, 1, 2), H).reshape(4, 2) / scale
    return Detection(quad=quad.astype(np.float32), lines=lines, alternation=alt)


# ---------------------------------------------------------------------------
# 4. Hướng bàn cờ
# ---------------------------------------------------------------------------

def _signed_area(q: np.ndarray) -> float:
    x, y = q[:, 0], q[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


_DARK = np.add.outer(np.arange(8), np.arange(8)) % 2 == 0  # [rank, file]: a1 (0,0) là ô tối


def _square_stats(canvas_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Độ sáng góc ô (nền ô) và tâm ô (quân cờ), mảng 8x8 theo [rank, file]."""
    s, inset, patch = SQUARE_PX, 8, 18
    corners = np.zeros((8, 8))
    centers = np.zeros((8, 8))
    for rank in range(8):
        for file in range(8):
            x0 = MARGIN_PX + file * s
            y0 = MARGIN_PX + (7 - rank) * s
            sq = canvas_gray[y0:y0 + s, x0:x0 + s]
            corners[rank, file] = np.mean([
                sq[inset:inset + patch, inset:inset + patch].mean(),
                sq[inset:inset + patch, -inset - patch:-inset].mean(),
                sq[-inset - patch:-inset, inset:inset + patch].mean(),
                sq[-inset - patch:-inset, -inset - patch:-inset].mean(),
            ])
            c = s // 2
            centers[rank, file] = sq[c - 20:c + 20, c - 20:c + 20].mean()
    return corners, centers


@dataclass
class Orientation:
    corners: np.ndarray   # 4x2 theo thứ tự a1, h1, h8, a8
    parity: float         # nền ô sáng - nền ô tối (>0 là đúng màu ô)
    white_side: float     # độ sáng quân ở hàng 1-2 trừ hàng 7-8 (>0: quân trắng ở phía hàng 1)


def orient(frame: np.ndarray, quad: np.ndarray) -> Orientation:
    """Chọn góc a1 trong 4 góc của `quad`.

    - Camera không bao giờ nhìn bàn cờ qua gương, nên chiều a1->h1->h8->a8 cố định.
    - a1 là ô tối: loại 2 trong 4 cách xoay.
    - Phía có quân sáng màu (thế cờ khởi đầu) là phía Trắng: chọn 1 trong 2 cách còn lại.
      Nếu không phân biệt được (bàn trống), ưu tiên phía Trắng gần camera (đáy ảnh).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if _signed_area(q) > 0:
        q = q[::-1].copy()

    candidates = []
    for k in range(4):
        cand = np.roll(q, -k, axis=0)
        bg, fg = _square_stats(warp(gray, homography(cand)))
        parity = (bg[~_DARK].mean() - bg[_DARK].mean()) / 255
        white_side = (fg[:2].mean() - fg[6:].mean()) / 255
        candidates.append(Orientation(cand, float(parity), float(white_side)))

    right_colors = [c for c in candidates if c.parity > 0] or candidates
    decided = [c for c in right_colors if c.white_side > 0.1]
    if decided:
        return max(decided, key=lambda c: c.white_side)
    # Không rõ phía Trắng: chọn cách để a1 gần đáy ảnh nhất.
    return max(right_colors, key=lambda c: c.corners[0, 1] + c.corners[1, 1])


def detect_board(frame: np.ndarray) -> Orientation | None:
    """Tìm bàn cờ và trả về 4 góc theo thứ tự a1, h1, h8, a8."""
    det = detect_quad(frame)
    return None if det is None else orient(frame, det.quad)
