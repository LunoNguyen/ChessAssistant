import chess

from chess_assistant.autolabel import autolabel, to_yolo_lines
from chess_assistant.board import BoardView, square_center_image
from chess_assistant.schema import PIECE_CLASSES, PieceBox
from tests.fakes import boxes_for_board
from tests.synthetic import POSES

H = BoardView.from_corners(POSES["white_side"]).H


def test_autolabel_uses_known_classes():
    board = chess.Board()
    wrong = [PieceBox(0, b.conf, b.xyxy) for b in boxes_for_board(board, H)]
    labels = autolabel(wrong, H, board)
    assert labels is not None and len(labels) == 32
    assert sorted(c for c, _ in labels) == sorted(PIECE_CLASSES.index(p.symbol()) for p in board.piece_map().values())


def test_autolabel_missing_piece_returns_none():
    board = chess.Board()
    boxes = boxes_for_board(board, H)
    e1 = square_center_image(H, chess.E1)
    boxes = [b for b in boxes if abs((b.xyxy[0] + b.xyxy[2]) / 2 - e1[0]) > 1]
    assert autolabel(boxes, H, board) is None


def test_autolabel_drops_boxes_on_empty_squares():
    board = chess.Board()
    extra = boxes_for_board(chess.Board("8/8/8/8/4P3/8/8/8 w - - 0 1"), H)
    labels = autolabel(boxes_for_board(board, H) + extra, H, board)
    assert labels is not None and len(labels) == 32


def test_to_yolo_lines():
    assert to_yolo_lines([(5, (100, 200, 140, 260))], 1280, 720) == ["5 0.093750 0.319444 0.031250 0.083333"]


def test_autolabel_rejects_rotated_orientation():
    """Thế khởi đầu xoay 180° vẫn khớp vị trí; dùng lớp model dự đoán để phát hiện hướng sai."""
    board = chess.Board()
    rotated = board.transform(chess.flip_vertical).transform(chess.flip_horizontal)
    # Model thấy đúng quân thật; nhưng H (do tracker đoán hướng) bị xoay 180°.
    boxes = boxes_for_board(rotated, H)
    assert autolabel(boxes, H, board) is None
