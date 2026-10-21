"""Nhận diện quân cờ trên toàn khung hình bằng YOLO (Ultralytics)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chess_assistant.schema import PIECE_CLASSES, PieceBox

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class PieceDetector:
    """Gọi như hàm: frame -> list[PieceBox]. Tên lớp của model phải là 12 ký hiệu FEN."""

    def __init__(
        self,
        weights: Path = MODELS_DIR / "pieces.pt",
        conf: float = 0.25,
        imgsz: int = 640,
        device: str | int | None = None,
        model=None,
    ):
        if model is None:
            weights = Path(weights)
            if not weights.is_file():
                raise FileNotFoundError(
                    f"Không thấy model {weights}. Train trước bằng: python -m scripts.train_pieces"
                )
            from ultralytics import YOLO  # nạp muộn: import ultralytics/torch mất vài giây

            model = YOLO(str(weights))
        self.model = model
        self.conf, self.imgsz, self.device = conf, imgsz, device

        names = {int(i): str(n) for i, n in dict(model.names).items()}
        missing = [c for c in PIECE_CLASSES if c not in names.values()]
        if missing:
            raise ValueError(f"Model thiếu lớp quân: {', '.join(missing)} (tên lớp phải là ký hiệu FEN)")
        self._to_piece = {i: PIECE_CLASSES.index(n) for i, n in names.items() if n in PIECE_CLASSES}

    def __call__(self, frame: np.ndarray) -> list[PieceBox]:
        result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        boxes = result.boxes
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        return [
            PieceBox(self._to_piece[c], float(s), tuple(float(v) for v in box))
            for c, s, box in zip(cls, conf, xyxy)
            if c in self._to_piece
        ]
