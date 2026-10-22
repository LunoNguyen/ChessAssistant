import json
import zipfile

import chess
import cv2
import numpy as np
import pytest

from chess_assistant import fomo
from chess_assistant.schema import PIECE_CLASSES

# Bàn cờ nhìn nghiêng: a1 dưới-trái, h1 dưới-phải, h8 trên-phải, a8 trên-trái (toạ độ chuẩn hoá).
CORNERS = np.array([[0.15, 0.85], [0.9, 0.8], [0.75, 0.2], [0.25, 0.25]])


def start_points() -> list[fomo.Point]:
    centers = fomo.square_centers(CORNERS)
    board = chess.Board()
    pts = [(fomo.CORNER, float(x), float(y)) for x, y in CORNERS]
    pts += [(PIECE_CLASSES.index(p.symbol()), *map(float, centers[sq])) for sq, p in board.piece_map().items()]
    return pts


def one_hot(target: np.ndarray, channels: int) -> np.ndarray:
    probs = np.full(target.shape + (channels,), 0.01)
    np.put_along_axis(probs, target[..., None].astype(int), 0.95, axis=-1)
    return probs


def test_piece_classes_match_schema():
    assert fomo.PIECE_CLASSES == PIECE_CLASSES


def test_channels():
    assert fomo.channel_of(PIECE_CLASSES.index("Q"), "color") == 1
    assert fomo.channel_of(PIECE_CLASSES.index("n"), "color") == 2
    assert fomo.channel_of(fomo.CORNER, "color") == 3
    assert fomo.channel_of(PIECE_CLASSES.index("n"), "full") == 8
    assert fomo.channel_of(fomo.CORNER, "full") == 13


def test_target_piece_wins_shared_cell():
    t = fomo.make_target([(0, 0.51, 0.51), (fomo.CORNER, 0.52, 0.52)], 10, "color")
    assert t[5, 5] == 1 and (t > 0).sum() == 1


def test_decode_keeps_adjacent_pieces_apart():
    """Hàng tốt ở xa camera: các quân nằm ở ô lưới kề nhau vẫn phải ra từng điểm riêng."""
    target = np.zeros((8, 8), np.uint8)
    target[2, 1:7] = 2
    dets = fomo.decode(one_hot(target, 4), 0.5)
    assert len(dets) == 6 and all(d.channel == 2 for d in dets)


def test_decode_suppresses_weaker_neighbour():
    probs = np.zeros((6, 6, 4))
    probs[..., 0] = 1
    probs[3, 3] = [0.1, 0.9, 0, 0]
    probs[3, 4] = [0.3, 0.7, 0, 0]
    dets = fomo.decode(probs, 0.5)
    assert len(dets) == 1 and (dets[0].x, dets[0].y) == (3.5 / 6, 3.5 / 6)


def test_complete_corners_from_three():
    for missing in range(4):
        three = np.array([c for k, c in enumerate(CORNERS) if k != missing])
        full = fomo.complete_corners([tuple(c) for c in three])
        # Chọn đúng đỉnh đối diện: trong 3 hình bình hành có thể, góc suy ra gần góc thật nhất.
        candidates = [three[(k + 1) % 3] + three[(k + 2) % 3] - three[k] for k in range(3)]
        dist = [np.linalg.norm(c - CORNERS[missing]) for c in candidates]
        assert np.isclose(np.linalg.norm(full[3] - CORNERS[missing]), min(dist))
    assert fomo.complete_corners([(0, 0), (1, 1)]) is None


def test_order_corners_any_input_order():
    rng = np.random.default_rng(0)
    for _ in range(5):
        shuffled = CORNERS[rng.permutation(4)]
        assert np.allclose(fomo.order_corners(shuffled), CORNERS)


def test_homography_and_square_of():
    H = fomo.board_homography(CORNERS)
    centers = fomo.square_centers(CORNERS)
    assert all(fomo.square_of(H, *centers[sq]) == sq for sq in range(64))
    assert fomo.square_of(H, 0.01, 0.01) is None
    assert np.allclose(fomo.board_homography(CORNERS), fomo.solve_homography(CORNERS, fomo.UNIT_CORNERS))


def test_occupancy_and_rotation():
    pts = start_points()
    occ, corners = fomo.target_occupancy(pts, "color")
    assert occ[:16] == [1] * 16 and occ[16:48] == [0] * 32 and occ[48:] == [2] * 16
    dets = [fomo.Detection(fomo.channel_of(c, "color"), x, y, 0.9) for c, x, y in pts]
    assert fomo.occupancy(dets, corners, "color") == occ
    # Góc dự đoán chỉ có thứ tự tạm: bắt đầu từ góc khác vẫn đúng sau khi thử 4 hướng.
    rolled = fomo.occupancy(dets, np.roll(corners, 1, axis=0), "color")
    assert fomo.square_accuracy(rolled, occ) < 1 and fomo.square_accuracy(rolled, occ, any_rotation=True) == 1
    assert fomo.rotate_squares(fomo.rotate_squares(occ, 1), 3) == occ
    assert fomo.occupancy(dets, np.array([[0.1, 0.1]] * 4), "color") == [0] * 64  # góc suy biến


@pytest.mark.parametrize("grid", [40])
def test_round_trip_perfect_model(grid):
    """Nhãn hoàn hảo -> decode -> 64 ô phải khớp (lưới 1/4 của ảnh 160px)."""
    img = np.full((300, 400, 3), 128, np.uint8)
    pts = [(c, x, y) for c, x, y in start_points()]
    view, vpts = fomo.board_view(img, pts, 160)
    dets = fomo.decode(one_hot(fomo.make_target(vpts, grid, "color"), 4), 0.5)
    occ_true, _ = fomo.target_occupancy(vpts, "color")
    corners = fomo.complete_corners([(d.x, d.y) for d in dets if d.channel == 3])
    occ = fomo.occupancy(dets, fomo.order_corners(corners), "color")
    assert fomo.square_accuracy(occ, occ_true, any_rotation=True) > 0.95


def test_board_view_moves_image_and_points_together():
    """Vẽ chấm màu ở các điểm, biến đổi, rồi kiểm tra chấm vẫn nằm đúng điểm mới."""
    img = np.zeros((300, 500, 3), np.uint8)
    pts = [(fomo.CORNER, 0.2, 0.3), (fomo.CORNER, 0.8, 0.3), (fomo.CORNER, 0.8, 0.9),
           (fomo.CORNER, 0.2, 0.9), (0, 0.5, 0.6)]
    for _, x, y in pts:
        cv2.circle(img, (int(x * 500), int(y * 300)), 12, (255, 255, 255), -1)
    rng = np.random.default_rng(3)
    for _ in range(10):
        view, new = fomo.board_view(img, pts, 160, fill=float(rng.uniform(0.5, 0.95)),
                                    angle=float(rng.uniform(0, 360)), flip=bool(rng.random() < 0.5))
        assert len(new) == 5
        for _, x, y in new:
            assert view[int(y * 160), int(x * 160)].mean() > 100


def test_random_view_shapes():
    img = np.random.default_rng(0).integers(0, 255, (384, 384, 3), dtype=np.uint8)
    out, pts = fomo.random_view(img, start_points(), 160, np.random.default_rng(1))
    assert out.shape == (160, 160, 3) and out.dtype == np.uint8
    assert all(0 <= x < 1 and 0 <= y < 1 for _, x, y in pts)
    x = fomo.preprocess(out)
    assert x.dtype == np.float32 and x.min() >= -1 and x.max() <= 1


# ---------------------------------------------------------------------------
# scripts.prepare_fomo
# ---------------------------------------------------------------------------

def test_flat_fit_recovers_corners():
    from chess_assistant.screen2d import flat_boxes, simulate_screen_photo
    from scripts.prepare_fomo import flat_fit

    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
    img = np.full((400, 400, 3), 200, np.uint8)
    labels = flat_boxes(board, 400)
    photo, out = simulate_screen_photo(img, labels, np.random.default_rng(5), effects=False)
    h, w = photo.shape[:2]
    boxes = [(c, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h) for c, (x1, y1, x2, y2) in out]
    corners = flat_fit(board, boxes)
    assert corners is not None
    H = fomo.board_homography(corners)
    for (sq, _), (_, cx, cy, _, _) in zip(sorted(board.piece_map().items()), boxes):
        assert fomo.square_of(H, cx, cy) == sq
    assert flat_fit(board, boxes[:-1]) is None  # thiếu khung -> không đoán bừa


def test_build_writes_zip(tmp_path):
    from scripts.prepare_fomo import build

    img = tmp_path / "a.jpg"
    cv2.imwrite(str(img), np.zeros((1000, 800, 3), np.uint8))
    pts = [[0, 0.5, 0.5], [fomo.CORNER, 0.1, 0.9]]
    counts = build({"demo": iter([(img, "train", pts)])}, tmp_path / "out" / "fomo_dataset.zip", max_side=200)
    assert counts[("demo", "train")] == 1
    with zipfile.ZipFile(tmp_path / "out" / "fomo_dataset.zip") as z:
        meta = json.loads(z.read("labels.json"))
        small = cv2.imdecode(np.frombuffer(z.read("images/train/demo_a.jpg"), np.uint8), cv2.IMREAD_COLOR)
        assert "fomo.py" in z.namelist()
    assert max(small.shape[:2]) == 200
    assert meta["records"][0]["points"] == pts and meta["point_classes"][fomo.CORNER] == "corner"


def test_notebook_is_valid_and_uses_fomo_module():
    from pathlib import Path

    nb = json.loads((Path(__file__).resolve().parent.parent / "notebooks" / "train_fomo_colab.ipynb")
                    .read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for code in [compile(("".join(c["source"])).replace("!nvidia-smi", "#"), "cell", "exec")
                 for c in nb["cells"] if c["cell_type"] == "code"]:
        assert code is not None
    for needed in ("import fomo", "fomo_dataset.zip", "TFLITE_BUILTINS_INT8", "fomo_model_data.cc", "fomo_settings.h"):
        assert needed in src
