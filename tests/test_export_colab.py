import zipfile
from pathlib import Path

import yaml

from chess_assistant.schema import PIECE_CLASSES
from scripts.export_colab import export_dataset


def make_dataset(root: Path) -> Path:
    for split, n in {"train": 3, "val": 2, "test": 1}.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(n):
            (root / split / "images" / f"img{i}.jpg").write_bytes(b"jpg")
            (root / split / "labels" / f"img{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    data = {"path": "D:\\some\\windows\\path", "train": "train/images", "val": "val/images",
            "test": "test/images", "names": dict(enumerate(PIECE_CLASSES))}
    (root / "data.yaml").write_text(yaml.safe_dump(data))
    return root


def test_export_dataset_zip(tmp_path):
    ds = make_dataset(tmp_path / "pieces")
    out = export_dataset(ds, tmp_path / "upload" / "pieces_dataset.zip")
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        data = yaml.safe_load(z.read("data.yaml"))
    assert sum(n.startswith("train/images/") for n in names) == 3
    assert sum(n.startswith("val/labels/") for n in names) == 2
    assert "test/images/img0.jpg" in names
    # Đường dẫn Windows không dùng được trên Colab: phải là đường dẫn tương đối.
    assert data["path"] == "." and data["train"] == "train/images"
    assert [data["names"][i] for i in range(12)] == list(PIECE_CLASSES)


def test_export_rejects_missing_dataset(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="prepare_dataset"):
        export_dataset(tmp_path / "nope", tmp_path / "x.zip")


def test_export_shrinks_large_images(tmp_path):
    import cv2
    import numpy as np

    ds = make_dataset(tmp_path / "pieces")
    big = ds / "train" / "images" / "big.jpg"
    cv2.imwrite(str(big), np.zeros((3072, 2048, 3), np.uint8))
    (ds / "train" / "labels" / "big.txt").write_text("3 0.5 0.25 0.1 0.2\n")
    out = export_dataset(ds, tmp_path / "x.zip", max_side=1280)
    with zipfile.ZipFile(out) as z:
        img = cv2.imdecode(np.frombuffer(z.read("train/images/big.jpg"), np.uint8), cv2.IMREAD_COLOR)
        label = z.read("train/labels/big.txt").decode()
    assert max(img.shape[:2]) == 1280 and img.shape[:2] == (1280, 853)
    assert label.split() == "3 0.5 0.25 0.1 0.2".split()  # toạ độ tỉ lệ: không đổi khi thu nhỏ
