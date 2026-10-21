"""Theo dõi bàn cờ qua nhiều frame.

- Dò lại định kỳ (mặc định trong luồng nền để không làm giật video).
- Giữ hướng a1 nhất quán: chỉ quyết định a1 ở đâu lúc bắt đầu khoá; các lần dò sau
  chỉ cần xoay 4 góc mới cho khớp nhất với 4 góc cũ (giữa ván cờ không còn dựa vào
  thế khởi đầu được nữa).
- Làm mượt rung nhẹ; vị trí nhảy xa (camera/bàn bị xê dịch) phải lặp lại vài lần
  mới chấp nhận, tránh bị kéo lệch bởi một lần dò sai lúc tay che bàn cờ.
- Không dò được (tay che) thì giữ nguyên vị trí cũ.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np

from chess_assistant.detect import detect_quad, orient

Status = Literal["searching", "locked", "holding", "moving"]  # moving: đang chờ xác nhận vị trí mới

log = logging.getLogger(__name__)


def _safe_detect(frame: np.ndarray):
    """Lỗi bất ngờ trong bộ dò được ghi log và tính như một lần dò hụt."""
    try:
        return detect_quad(frame)
    except Exception:
        log.exception("BoardTracker: lỗi khi dò bàn cờ")
        return None


@dataclass
class TrackerConfig:
    redetect_every: int = 5     # dò lại sau mỗi N frame khi đã khoá
    jitter: float = 0.02        # dịch chuyển < 2% cạnh bàn: coi là rung, làm mượt
    jump: float = 0.15          # dịch chuyển > 15% cạnh bàn: phải xác nhận
    confirm: int = 3            # số lần dò liên tiếp cần để chấp nhận một cú nhảy
    smooth: float = 0.3         # hệ số làm mượt cho rung nhẹ
    hold_misses: int = 3        # số lần dò hụt liên tiếp trước khi báo "holding"


class BoardTracker:
    def __init__(self, config: TrackerConfig | None = None, background: bool = True):
        self.cfg = config or TrackerConfig()
        self._pool = ThreadPoolExecutor(max_workers=1) if background else None
        self._pending: Future | None = None
        self.reset()

    def reset(self) -> None:
        self.corners: np.ndarray | None = None  # 4x2, thứ tự a1, h1, h8, a8
        self.status: Status = "searching"
        self._frame_idx = 0
        self._misses = 0
        self._jump: list[np.ndarray] = []

    def flip(self) -> None:
        """Xoay 180°: dùng khi chương trình đoán nhầm phía Trắng."""
        self.rotate(2)

    def rotate(self, k: int) -> None:
        """Đổi góc a1 đi k góc (k=2: lật 180°). Dùng khi biết hướng bàn cờ đang sai."""
        if self.corners is not None:
            self.corners = np.roll(self.corners, k, axis=0)
            self._jump = [np.roll(c, k, axis=0) for c in self._jump]

    def update(self, frame: np.ndarray) -> np.ndarray | None:
        """Đưa vào một frame, trả về 4 góc hiện tại (a1, h1, h8, a8) hoặc None."""
        self._frame_idx += 1
        if self._pool is None:
            if self._due():
                self._integrate(frame, _safe_detect(frame))
            return self.corners

        if self._pending is not None and self._pending.done():
            frame_used, det = self._pending.result()
            self._pending = None
            self._integrate(frame_used, det)
        if self._pending is None and self._due():
            snapshot = frame.copy()
            self._pending = self._pool.submit(lambda: (snapshot, _safe_detect(snapshot)))
        return self.corners

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------

    def _due(self) -> bool:
        return self.corners is None or self._frame_idx % self.cfg.redetect_every == 0

    def _integrate(self, frame: np.ndarray, det) -> None:
        if det is None:
            self._misses += 1
            if self.corners is not None and self._misses >= self.cfg.hold_misses:
                self.status = "holding"
            return
        self._misses = 0

        if self.corners is None:
            self.corners = orient(frame, det.quad).corners
            self.status = "locked"
            return

        cand = self._align(det.quad)
        size = np.linalg.norm(self.corners[0] - self.corners[2])
        shift = np.linalg.norm(cand - self.corners, axis=1).mean() / size

        if shift < self.cfg.jitter:
            self.corners = (1 - self.cfg.smooth) * self.corners + self.cfg.smooth * cand
            self._jump.clear()
        elif shift < self.cfg.jump:
            self.corners = cand
            self._jump.clear()
        else:
            # Chỉ chấp nhận khi vài lần dò liên tiếp cùng chỉ về một vị trí mới.
            if self._jump and np.linalg.norm(cand - self._jump[-1], axis=1).mean() / size > self.cfg.jitter * 2:
                self._jump.clear()
            self._jump.append(cand)
            if len(self._jump) < self.cfg.confirm:
                # Vị trí cũ có thể đã sai: báo "moving" để không ai đọc bàn cờ theo nó.
                self.status = "moving"
                return
            self.corners = cand
            self._jump.clear()
        self.corners = self.corners.astype(np.float32)
        self.status = "locked"

    def _align(self, quad: np.ndarray) -> np.ndarray:
        """Xoay vòng 4 góc mới sao cho khớp nhất với 4 góc hiện tại (giữ hướng a1)."""
        q = np.asarray(quad, dtype=np.float32)
        options = [np.roll(q, -k, axis=0) for k in range(4)]
        options += [np.roll(q[::-1], -k, axis=0) for k in range(4)]
        return min(options, key=lambda c: np.linalg.norm(c - self.corners, axis=1).sum())
