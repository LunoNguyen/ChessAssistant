import chess
import numpy as np

from chess_assistant.schema import EMPTY, PIECE_CLASSES, argmax_labels, class_of, reading_from_board


def test_class_of():
    assert class_of(None) == EMPTY
    assert class_of(chess.Piece.from_symbol("q")) == 10


def test_reading_from_board_shape_and_rows():
    r = reading_from_board(chess.Board())
    assert r.shape == (64, 13)
    assert np.allclose(r.sum(1), 1)
    labels = argmax_labels(r)
    assert labels[chess.E1] == PIECE_CLASSES.index("K")
    assert labels[chess.E4] == EMPTY
