"""Chỉ cho kết quả đọc bàn cờ đi tiếp khi nó đứng yên đủ lâu.

Lúc tay đang cầm quân hoặc camera đang lia, kết quả đọc thay đổi liên tục hoặc
thiếu nhiều quân; những frame đó không được đưa vào GameTracker.
"""

from __future__ import annotations

import numpy as np

from chess_assistant.schema import EMPTY, argmax_labels

PERSIST_FACTOR = 4  # giảm quân mạnh nhưng giữ yên chừng này lần stable_frames thì coi là thật


class StabilityFilter:
    def __init__(self, stable_frames: int = 8, max_drop: int = 4):
        self.stable_frames = stable_frames
        self.max_drop = max_drop
        self._last_occupied: int | None = None
        self.reset()

    def reset(self) -> None:
        self._labels: list[int] | None = None
        self._window: list[np.ndarray] = []  # chỉ giữ tối đa stable_frames * PERSIST_FACTOR frame
        self._still = 0                      # số frame liên tiếp có cùng nhãn
        self._done = False

    def push(self, reading: np.ndarray | None, board_locked: bool) -> np.ndarray | None:
        if reading is None or not board_locked:
            self.reset()
            return None
        labels = argmax_labels(reading)
        if labels != self._labels:
            self.reset()
            self._labels = labels
        self._window.append(reading)
        self._window = self._window[-self.stable_frames * PERSIST_FACTOR:]
        self._still += 1
        if self._done or self._still < self.stable_frames:
            return None

        occupied = sum(label != EMPTY for label in labels)
        big_drop = self._last_occupied is not None and self._last_occupied - occupied > self.max_drop
        if big_drop and self._still < self.stable_frames * PERSIST_FACTOR:
            return None
        self._done = True
        self._last_occupied = occupied
        return np.mean(self._window[-self.stable_frames:], axis=0)
