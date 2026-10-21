from pathlib import Path

import pytest
import yaml

from chess_assistant.dataset import build_dataset, normalize_class_name, remap_label_file
from chess_assistant.schema import PIECE_CLASSES


@pytest.mark.parametrize("name, expected", [
    ("White-Pawn", "P"),
    ("white_pawn", "P"),
    ("pawn-white", "P"),
    ("wp", "P"),
    ("w_p", "P"),
    ("bk", "k"),
    ("black_queen", "q"),
    ("Black Knight", "n"),
    ("wb", "B"),
    ("P", "P"),
    ("q", "q"),
    ("board", "ignore"),
    ("chess-board", "ignore"),
    ("bishop", None),
    ("pawn", None),
    ("dragon", None),
])
def test_normalize_names(name, expected):
    assert normalize_class_name(name) == expected


def test_remap_label_file():
    names = ["board", "white-king", "black-pawn"]
    assert remap_label_file(["1 0.5 0.5 0.1 0.2", "0 0.5 0.5 1 1", "2 0.1 0.1 0.05 0.05"], names) == [
        f"{PIECE_CLASSES.index('K')} 0.5 0.5 0.1 0.2",
        f"{PIECE_CLASSES.index('p')} 0.1 0.1 0.05 0.05",
    ]
    assert remap_label_file(["0 0.5 0.5 0.1 0.1"], ["bishop"]) is None


def make_source(root: Path, name: str, names: list[str], splits: dict[str, int]) -> Path:
    src = root / name
    for split, n in splits.items():
        for i in range(n):
            (src / split / "images").mkdir(parents=True, exist_ok=True)
            (src / split / "labels").mkdir(parents=True, exist_ok=True)
            (src / split / "images" / f"img{i}.jpg").write_bytes(b"jpg")
            (src / split / "labels" / f"img{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (src / "data.yaml").write_text(yaml.safe_dump({"names": names}))
    return src


def test_build_dataset_holdout(tmp_path):
    a = make_source(tmp_path, "setA", ["white-king"], {"train": 2, "valid": 1})
    b = make_source(tmp_path, "setB", ["black-pawn"], {"train": 2, "valid": 1})
    bad = make_source(tmp_path, "setC", ["bishop"], {"train": 1})
    out = tmp_path / "out"
    counts = build_dataset([a, b, bad], out, holdout={"setB"})
    assert counts == {"train": 2, "val": 1, "test": 3}
    test_imgs = sorted(p.name for p in (out / "test" / "images").iterdir())
    assert test_imgs == ["setB_train_img0.jpg", "setB_train_img1.jpg", "setB_valid_img0.jpg"]
    label = (out / "test" / "labels" / (Path(test_imgs[0]).stem + ".txt")).read_text().split()
    assert label[0] == str(PIECE_CLASSES.index("p"))
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert [data["names"][i] for i in range(12)] == list(PIECE_CLASSES)
