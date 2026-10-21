from pathlib import Path

import numpy as np
import pytest

from chess_assistant.pieces import PieceDetector
from chess_assistant.schema import PIECE_CLASSES


class _Arr:
    def __init__(self, a):
        self._a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Boxes:
    def __init__(self, cls, conf, xyxy):
        self.cls, self.conf, self.xyxy = _Arr(cls), _Arr(conf), _Arr(xyxy)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def __init__(self, names, boxes):
        self.names = names
        self._boxes = boxes
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [_Result(self._boxes)]


SHUFFLED = "kqrbnpKQRBNP"


def test_maps_class_names():
    names = dict(enumerate(SHUFFLED))
    boxes = _Boxes([0, 11], [0.9, 0.8], [[1, 2, 3, 4], [5, 6, 7, 8]])
    det = PieceDetector(model=FakeModel(names, boxes))
    out = det(np.zeros((10, 10, 3), np.uint8))
    assert [b.cls for b in out] == [PIECE_CLASSES.index("k"), PIECE_CLASSES.index("P")]
    assert out[0].conf == pytest.approx(0.9)
    assert out[1].xyxy == (5.0, 6.0, 7.0, 8.0)


def test_passes_inference_options():
    model = FakeModel(dict(enumerate(SHUFFLED)), _Boxes([], [], np.zeros((0, 4))))
    det = PieceDetector(model=model, conf=0.4, imgsz=512, device=0)
    assert det(np.zeros((10, 10, 3), np.uint8)) == []
    assert model.calls[0] == {"conf": 0.4, "imgsz": 512, "device": 0, "verbose": False}


def test_missing_class_raises():
    names = dict(enumerate(SHUFFLED.replace("q", "")))
    with pytest.raises(ValueError, match="q"):
        PieceDetector(model=FakeModel(names, None))


def test_missing_weights_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_pieces"):
        PieceDetector(weights=Path(tmp_path) / "none.pt")
