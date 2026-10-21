"""Quy khung quân (từ YOLO) về 64 ô: ra reading (64, 13) xác suất."""

from __future__ import annotations

import cv2
import numpy as np

from chess_assistant.board import square_at_canvas
from chess_assistant.schema import EMPTY, EPS, NUM_CLASSES, PieceBox


def foot_point(box: PieceBox, foot_lift: float = 0.10) -> tuple[float, float]:
    """Điểm chân quân: giữa cạnh dưới khung, nâng lên `foot_lift` lần chiều cao khung.

    Nhìn nghiêng, cạnh dưới khung là mép trước đế quân; nâng lên một chút cho
    gần tâm đế hơn.
    """
    x1, y1, x2, y2 = box.xyxy
    return (x1 + x2) / 2, y2 - foot_lift * (y2 - y1)


def read_board(boxes: list[PieceBox], H: np.ndarray, foot_lift: float = 0.10) -> np.ndarray:
    """Reading (64, 13): mỗi ô giữ khung có độ tin cậy cao nhất có chân rơi vào ô đó."""
    best: dict[int, PieceBox] = {}
    if boxes:
        feet = np.float32([foot_point(b, foot_lift) for b in boxes]).reshape(-1, 1, 2)
        canvas = cv2.perspectiveTransform(feet, H).reshape(-1, 2)
        for box, (x, y) in zip(boxes, canvas):
            sq = square_at_canvas(float(x), float(y))
            if sq is not None and (sq not in best or box.conf > best[sq].conf):
                best[sq] = box

    reading = np.full((64, NUM_CLASSES), EPS)
    reading[:, EMPTY] = 1 - EPS
    for sq, box in best.items():
        reading[sq, EMPTY] = 1 - box.conf
        reading[sq, box.cls] = box.conf
    reading = reading.clip(EPS, 1 - EPS)
    return reading / reading.sum(axis=1, keepdims=True)
