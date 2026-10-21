"""Ghép các khối thành một bước xử lý cho mỗi frame.

frame -> BoardTracker -> PieceDetector -> read_board -> StabilityFilter -> GameTracker
      -> Advisor -> OverlayState

Dùng chung cho scripts/play.py (camera) và scripts/replay.py (video offline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import chess
import numpy as np

from chess_assistant.advisor import Advisor
from chess_assistant.board import BoardView, user_color_from_corners
from chess_assistant.game import GameEvent, GameTracker, SyncState, match_count
from chess_assistant.overlay import OverlayState
from chess_assistant.reading import read_board
from chess_assistant.schema import PieceBox
from chess_assistant.stability import StabilityFilter
from chess_assistant.tracker import BoardTracker

REALIGN_MOVE_SLACK = 4   # một nước đi đổi tối đa 4 ô (nhập thành)
REALIGN_MIN_GAIN = 6     # hướng mới phải khớp hơn hướng cũ ít nhất chừng này ô (tàn cuộc ít quân:
                         # hướng sai vẫn khớp nhiều ô trống, nên chênh lệch chỉ ~15 ô)

GREEN, YELLOW, ORANGE, RED, CYAN = (0, 255, 0), (0, 255, 255), (0, 165, 255), (0, 0, 255), (255, 255, 0)


@dataclass
class FrameResult:
    corners: np.ndarray | None
    view: BoardView | None
    reading: np.ndarray | None
    events: list[GameEvent] = field(default_factory=list)
    overlay: OverlayState | None = None


class Pipeline:
    def __init__(
        self,
        detector: Callable[[np.ndarray], list[PieceBox]],
        tracker: BoardTracker,
        game: GameTracker | None = None,
        advisor: Advisor | None = None,
        stability: StabilityFilter | None = None,
        foot_lift: float = 0.10,
    ):
        self.detector = detector
        self.tracker = tracker
        self.game = game or GameTracker()
        self.advisor = advisor
        self.stability = stability or StabilityFilter()
        self.foot_lift = foot_lift
        self.user_color: chess.Color | None = None  # None = bên gần camera
        self.paused = False
        self._view: BoardView | None = None
        self._view_corners: np.ndarray | None = None
        self._message = ""

    def process(self, frame: np.ndarray) -> FrameResult:
        corners = self.tracker.update(frame)
        view = self._view_for(corners)

        reading = None
        if view is not None:
            boxes = self.detector(frame)
            reading = read_board(boxes, view.H, self.foot_lift)
            realigned = self._realign(boxes, reading)
            if realigned is not None:
                reading = realigned
                view = self._view_for(self.tracker.corners)

        events: list[GameEvent] = []
        stable = self.stability.push(reading, self.tracker.status == "locked")
        if stable is not None:
            event = self.game.update(stable)
            if event is not None:
                events.append(event)
                self._message = event.message
                if event.kind == "moves":
                    self._message = ""
                if event.flipped:
                    # Đọc lại cho thấy hướng bàn cờ đang ngược: lật để các frame sau đọc đúng.
                    self.tracker.flip()
                    self.stability.reset()
                    corners = self.tracker.corners
                    view = self._view_for(corners)

        user_color = self.current_user_color()
        advice = None
        if self.advisor is not None and not self.paused:
            # Chỉ gợi ý cho thế cờ đang tin được: đang bám ván, hoặc vừa đọc lại và chỉ chưa rõ lượt
            # (khi đó gợi ý cho cả hai bên). Khi lạc/chưa đọc được thì không.
            if self.game.state == SyncState.SYNCED:
                self.advisor.request(self.game.board, user_color)
                advice = self.advisor.latest()
            elif self.game.state == SyncState.NEED_TURN and self.game.pending is not None:
                self.advisor.request(self.game.pending, user_color, both_turns=True)
                advice = self.advisor.latest()
            if self.advisor.error:
                self._message = f"Loi engine: {self.advisor.error}"

        return FrameResult(corners, view, reading, events, self._overlay(user_color, advice))

    def current_user_color(self) -> chess.Color:
        if self.user_color is not None:
            return self.user_color
        if self.tracker.corners is not None:
            return user_color_from_corners(self.tracker.corners)
        return chess.WHITE

    def set_turn(self, color: chess.Color) -> bool:
        if self.game.state == SyncState.SYNCED:
            self._message = "Da biet luot roi - khong can bam w/b"
            return False
        if self.game.state != SyncState.NEED_TURN:
            self._message = "Chua doc duoc the co (nhan dien quan chua du) - w/b chua dung duoc"
            return False
        ok = self.game.set_turn(color)
        self._message = "" if ok else "Ben nay khong the den luot (dang bi chieu?) - chon lai"
        return ok

    def undo(self) -> chess.Move | None:
        # Không reset bộ lọc ổn định: nếu reset, chính bàn cờ đang đứng yên sẽ được đọc lại
        # và nước vừa lùi bị ghi nhận lại ngay.
        return self.game.undo()

    def flip(self) -> None:
        """Lật hướng bàn cờ 180° (phím f); thế cờ đang chờ chọn lượt cũng được lật theo."""
        self.tracker.flip()
        self.stability.reset()
        self.game.flip_pending()

    # ------------------------------------------------------------------

    def _realign(self, boxes: list[PieceBox], reading: np.ndarray) -> np.ndarray | None:
        """Khi đang bám ván mà kết quả đọc lệch hẳn thế cờ đã biết, thử 3 hướng xoay còn lại của
        bàn cờ (tracker có thể chọn sai góc a1 khi camera/bàn xoay mạnh). Thế cờ đã biết cho biết
        hướng nào đúng. Trả về reading theo hướng mới, hoặc None nếu không đổi."""
        if self.tracker.status != "locked":
            return None
        if self.game.state == SyncState.SYNCED:
            board = self.game.board
        elif self.game.state == SyncState.NEED_TURN and self.game.pending is not None:
            board = self.game.pending  # đã đọc lại được thế cờ, chỉ chưa rõ lượt
        else:
            return None
        current = match_count(reading, board)
        if current >= self.game.min_match:
            return None
        # Ảnh mới có thể đã sau một nước đi (lệch tới 4 ô, như nhập thành) và có ô đọc sai,
        # nên hướng đúng có thể chưa đạt min_match; nhưng phải hơn hẳn hướng hiện tại.
        need = self.game.min_match - REALIGN_MOVE_SLACK
        best = None
        for k in (1, 2, 3):
            corners = np.roll(self.tracker.corners, k, axis=0)
            candidate = read_board(boxes, BoardView.from_corners(corners).H, self.foot_lift)
            score = match_count(candidate, board)
            if score >= need and score - current >= REALIGN_MIN_GAIN and (best is None or score > best[0]):
                best = (score, k, candidate)
        if best is None:
            return None
        self.tracker.rotate(best[1])
        self.stability.reset()
        return best[2]

    def _view_for(self, corners: np.ndarray | None) -> BoardView | None:
        if corners is None:
            self._view = self._view_corners = None
        elif self._view_corners is None or not np.array_equal(corners, self._view_corners):
            self._view, self._view_corners = BoardView.from_corners(corners), corners.copy()
        return self._view

    def _overlay(self, user_color: chess.Color, advice) -> OverlayState:
        state = self.game.state
        if self.tracker.status == "searching":
            status, color = "Dang tim ban co...", ORANGE
        elif self.tracker.status == "moving":
            status, color = "Camera dang di chuyen...", ORANGE
        elif self.tracker.status == "holding":
            status, color = "Bi che / mat dau - giu vi tri cu", ORANGE
        elif state == SyncState.SYNCED:
            status, color = ("Tam dung goi y" if self.paused else "Dang bam van co"), GREEN
        elif state == SyncState.NEED_TURN:
            status, color = "Da doc lai the co - chua ro luot (xanh la: Trang, xanh duong: Den)", YELLOW
        elif state == SyncState.LOST:
            status, color = "LAC - dang doc lai the co", RED
        else:
            status, color = "Cho the co (bay the khoi dau hoac giu yen)", CYAN

        board = self.game.board if state == SyncState.SYNCED else self.game.pending
        last = board.peek() if board is not None and board.move_stack else None
        return OverlayState(status, color, board, user_color, advice, last, self._message)
