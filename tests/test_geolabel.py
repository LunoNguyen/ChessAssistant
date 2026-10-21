import chess
import numpy as np

from chess_assistant.board import square_center_image, BoardView
from chess_assistant.geolabel import geometric_boxes, positions_from_pgn
from chess_assistant.schema import PIECE_CLASSES, class_of
from tests.synthetic import POSES

CORNERS = np.float32(POSES["white_side"])
H = BoardView.from_corners(CORNERS).H


def by_square(labels, board):
    """Ghép mỗi nhãn với ô của nó (nhãn trả về theo thứ tự ô tăng dần)."""
    return dict(zip(sorted(board.piece_map()), labels))


def height(box):
    return box[3] - box[1]


def test_one_box_per_piece_with_known_class():
    board = chess.Board()
    labels = geometric_boxes(CORNERS, board, (1280, 720))
    assert len(labels) == 32
    for sq, (cls, _) in by_square(labels, board).items():
        assert cls == class_of(board.piece_at(sq))


def test_box_stands_on_its_square():
    board = chess.Board()
    for sq, (_, (x1, y1, x2, y2)) in by_square(geometric_boxes(CORNERS, board, (1280, 720)), board).items():
        cx, cy = square_center_image(H, sq)
        assert x1 < cx < x2, chess.square_name(sq)
        assert y1 < cy <= y2 + 1, chess.square_name(sq)       # đế quân nằm trong khung
        assert y2 - cy < 0.6 * (y2 - y1), chess.square_name(sq)  # phần lớn khung ở phía trên đế


def test_taller_pieces_get_taller_boxes():
    boxes = by_square(geometric_boxes(CORNERS, chess.Board(), (1280, 720)), chess.Board())
    assert height(boxes[chess.E1][1]) > height(boxes[chess.D2][1])  # Vua cao hơn Tốt (cùng khoảng cách)


def test_near_pieces_larger_than_far_pieces():
    boxes = by_square(geometric_boxes(CORNERS, chess.Board(), (1280, 720)), chess.Board())
    assert height(boxes[chess.A2][1]) > height(boxes[chess.A7][1])  # hàng 2 gần camera hơn hàng 7


def test_height_scale():
    board = chess.Board()
    base = by_square(geometric_boxes(CORNERS, board, (1280, 720)), board)
    tall = by_square(geometric_boxes(CORNERS, board, (1280, 720), height_scale=1.5), board)
    # Khung = đế (không đổi) + chiều cao quân; chỉ phần chiều cao tăng 50%.
    base_box, tall_box = base[chess.E2][1], tall[chess.E2][1]
    assert tall_box[3] == base_box[3]                       # đáy (đế) giữ nguyên
    assert height(tall_box) > height(base_box) * 1.15


def test_boxes_clipped_to_image():
    labels = geometric_boxes(CORNERS, chess.Board(), (1280, 300))  # ảnh thấp: nhiều khung bị cắt
    for _, (x1, y1, x2, y2) in labels:
        assert 0 <= x1 < x2 <= 1280 and 0 <= y1 < y2 <= 300


def test_positions_from_pgn(tmp_path):
    pgn = tmp_path / "g.pgn"
    pgn.write_text('[Event "x"]\n\n1. e4 e5 2. Nf3 *\n')
    boards = positions_from_pgn(pgn)
    assert len(boards) == 4
    assert boards[0].board_fen() == chess.STARTING_BOARD_FEN
    assert boards[-1].piece_at(chess.F3) == chess.Piece.from_symbol("N")
    assert PIECE_CLASSES  # dùng import


def test_positions_from_pgn_rejects_illegal_move(tmp_path):
    import pytest

    pgn = tmp_path / "bad.pgn"
    pgn.write_text("1. e4 e5 2. Nf6 *\n")  # Nf6 không hợp lệ cho Trắng ở nước 2
    with pytest.raises(ValueError, match="Nf6"):
        positions_from_pgn(pgn)
