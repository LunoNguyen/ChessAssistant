"""Theo dõi ván cờ từ kết quả đọc bàn cờ.

Không tin trực tiếp vào từng ô đọc được: luôn giữ một chess.Board và chỉ chấp nhận
thế cờ mới nếu nó là kết quả của 1 (hoặc 2) nước đi hợp lệ khớp rõ ràng với
reading. Nhờ vậy vài ô đọc sai không làm hỏng ván, và nhập thành / bắt tốt qua
đường / phong cấp được xử lý tự động.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import chess
import chess.pgn
import numpy as np

from chess_assistant.schema import EMPTY, PIECE_CLASSES, argmax_labels, class_of

_ROWS = np.arange(64)


class SyncState(Enum):
    UNSYNCED = "unsynced"
    SYNCED = "synced"
    NEED_TURN = "need_turn"
    LOST = "lost"


@dataclass
class GameEvent:
    kind: Literal["moves", "synced", "need_turn", "lost", "unsure"]
    moves: list[chess.Move] = field(default_factory=list)
    flipped: bool = False
    message: str = ""


def _labels(board: chess.Board) -> np.ndarray:
    labels = np.full(64, EMPTY)
    for sq, piece in board.piece_map().items():
        labels[sq] = class_of(piece)
    return labels


def position_score(reading: np.ndarray, board: chess.Board) -> float:
    """Log-likelihood của thế cờ theo reading: Σ log p(ô = quân của thế đó)."""
    return float(np.log(reading)[_ROWS, _labels(board)].sum())


def match_count(reading: np.ndarray, board: chess.Board) -> int:
    return int((np.array(argmax_labels(reading)) == _labels(board)).sum())


def board_from_labels(labels: list[int], flip: bool) -> chess.Board | None:
    """Dựng thế cờ trực tiếp từ nhãn 64 ô (flip=True: bàn cờ xoay 180°).

    Quyền nhập thành suy từ Vua/Xe còn ở ô gốc; không có bắt tốt qua đường.
    Trả về None nếu thế cờ không thể có thật với bất kỳ bên nào đến lượt.
    """
    board = chess.Board(None)
    for sq in chess.SQUARES:
        label = labels[63 - sq if flip else sq]
        if label != EMPTY:
            board.set_piece_at(sq, chess.Piece.from_symbol(PIECE_CLASSES[label]))
    if len(board.pieces(chess.KING, chess.WHITE)) != 1 or len(board.pieces(chess.KING, chess.BLACK)) != 1:
        return None
    if board.pieces(chess.PAWN, chess.WHITE) & chess.BB_BACKRANKS or board.pieces(chess.PAWN, chess.BLACK) & chess.BB_BACKRANKS:
        return None
    board.set_castling_fen("KQkq")
    board.castling_rights = board.clean_castling_rights()
    for turn in (chess.WHITE, chess.BLACK):
        board.turn = turn
        if board.is_valid():
            board.turn = chess.WHITE
            return board
    return None


def _valid_turns(board: chess.Board) -> list[chess.Color]:
    turns = []
    for turn in (chess.WHITE, chess.BLACK):
        b = board.copy()
        b.turn = turn
        if b.is_valid():
            turns.append(turn)
    return turns


def _plausibility(board: chess.Board) -> int:
    """Tốt trắng ở nửa dưới, Tốt đen ở nửa trên: dùng để chọn hướng bàn cờ."""
    white = sum(1 for sq in board.pieces(chess.PAWN, chess.WHITE) if 1 <= chess.square_rank(sq) <= 3)
    black = sum(1 for sq in board.pieces(chess.PAWN, chess.BLACK) if 4 <= chess.square_rank(sq) <= 6)
    return white + black


@dataclass
class _Candidate:
    score: float
    moves: list[chess.Move]
    labels: np.ndarray


def _better(new: _Candidate, old: _Candidate | None) -> bool:
    """Trong cùng một nhóm (cùng ô đi/đến), giữ điểm cao hơn; hoà thì ưu tiên phong Hậu."""
    if old is None or new.score > old.score + 1e-6:
        return True
    tie = abs(new.score - old.score) <= 1e-6
    return tie and new.moves[-1].promotion == chess.QUEEN


def _key(move: chess.Move) -> tuple[int, int]:
    return move.from_square, move.to_square


class GameTracker:
    def __init__(self, margin: float = 3.0, min_match: int = 62, lost_after: int = 3):
        self.margin = margin
        self.min_match = min_match
        self.lost_after = lost_after
        self.board = chess.Board()
        self.state = SyncState.UNSYNCED
        self.pending: chess.Board | None = None
        self.pending_flipped = False
        self._misses = 0
        self._history: list[str] = []  # PGN các đoạn ván trước mỗi lần đồng bộ lại

    def start_from(self, board: chess.Board) -> None:
        if self.board.move_stack:
            self._history.append(str(chess.pgn.Game.from_board(self.board)))
        self.board = board.copy()
        self.state = SyncState.SYNCED
        self.pending = None
        self._misses = 0

    def update(self, reading: np.ndarray) -> GameEvent | None:
        if self.state == SyncState.SYNCED:
            return self._update_synced(reading)
        return self._resync(reading)

    def set_turn(self, color: chess.Color) -> bool:
        """Hoàn tất đồng bộ lại sau khi người dùng cho biết bên nào đến lượt."""
        if self.state != SyncState.NEED_TURN or self.pending is None:
            return False
        candidate = self.pending.copy()
        candidate.turn = color
        if not candidate.is_valid():
            return False
        self.start_from(candidate)
        return True

    def flip_pending(self) -> None:
        """Xoay 180° thế cờ đang chờ chọn lượt (người dùng lật bàn trong lúc chờ)."""
        if self.state != SyncState.NEED_TURN or self.pending is None:
            return
        rotated = self.pending.transform(chess.flip_vertical).transform(chess.flip_horizontal)
        rotated.set_castling_fen("KQkq")
        rotated.castling_rights = rotated.clean_castling_rights()
        self.pending, self.pending_flipped = rotated, not self.pending_flipped

    def undo(self) -> chess.Move | None:
        return self.board.pop() if self.board.move_stack else None

    def pgn(self) -> str:
        """PGN cả buổi: các đoạn trước mỗi lần đồng bộ lại, rồi đoạn hiện tại."""
        segments = list(self._history)
        if self.board.move_stack or not segments:
            segments.append(str(chess.pgn.Game.from_board(self.board)))
        return "\n\n".join(segments)

    # ------------------------------------------------------------------

    def _update_synced(self, reading: np.ndarray) -> GameEvent | None:
        log_r = np.log(reading)

        def score(labels: np.ndarray) -> float:
            return float(log_r[_ROWS, labels].sum())

        current = score(_labels(self.board))
        one = self._candidates(self.board, score, depth=1)
        if not one or current >= one[0].score:
            self._misses = 0
            return None

        best = self._accept(one, [current], reading)
        observed = np.array(argmax_labels(reading))
        if best is None or (observed != best.labels).any():
            # 1 nước chưa giải thích hết: có thể đã bỏ lỡ một nước.
            two = self._candidates(self.board, score, depth=2)
            best_two = self._accept(two, [current, one[0].score], reading)
            if best_two is not None:
                best = best_two
        if best is not None:
            for move in best.moves:
                self.board.push(move)
            self._misses = 0
            return GameEvent("moves", moves=best.moves)

        self._misses += 1
        if self._misses >= self.lost_after:
            self.state = SyncState.LOST
            return GameEvent("lost", message="mat dau van co, can doc lai toan bo")
        return GameEvent("unsure", message="khong khop nuoc di nao")

    def _resync(self, reading: np.ndarray) -> GameEvent | None:
        """Đọc lại toàn bộ thế cờ. Khi đang chờ chọn lượt vẫn tiếp tục đọc: người dùng có
        thể còn đang bày quân, nên thế khởi đầu hoặc một thế hợp lệ mới thay cho thế đang chờ."""
        waiting = self.state == SyncState.NEED_TURN
        if match_count(reading, chess.Board()) >= self.min_match:
            self.start_from(chess.Board())
            return GameEvent("synced", message="the co khoi dau")
        if waiting and self.pending is not None:
            resolved = self._resolve_turn_by_move(reading)
            if resolved is not None:
                return resolved
        labels = argmax_labels(reading)
        options = [(flip, b) for flip in (False, True) if (b := board_from_labels(labels, flip)) is not None]
        if not options:
            return None if waiting else GameEvent("unsure", message="khong doc duoc the co hop le")
        flip, board = max(options, key=lambda o: (_plausibility(o[1]), not o[0]))
        if waiting and self.pending is not None and board.board_fen() == self.pending.board_fen() and not flip:
            return None
        turns = _valid_turns(board)
        if len(turns) == 1:
            # Bên kia đang bị chiếu: chỉ có một bên có thể đến lượt.
            board.turn = turns[0]
            self.start_from(board)
            return GameEvent("synced", flipped=flip, message="doc lai xong")
        self.pending, self.pending_flipped = board, flip
        self.state = SyncState.NEED_TURN
        return GameEvent("need_turn", flipped=flip,
                         message="chua ro luot: goi y ca 2 ben, tu nhan luot khi co nuoc di (hoac bam w/b)")

    def _resolve_turn_by_move(self, reading: np.ndarray) -> GameEvent | None:
        """Đang chờ lượt: nếu reading là thế cờ sau một nước hợp lệ của bên nào đó từ thế đang chờ,
        thì bên đó vừa đi. Đồng bộ với lượt ấy và ghi nhận nước đi luôn."""
        log_r = np.log(reading)

        def score(labels: np.ndarray) -> float:
            return float(log_r[_ROWS, labels].sum())

        current = score(_labels(self.pending))
        cands: list[tuple[_Candidate, chess.Color]] = []
        for turn in _valid_turns(self.pending):
            b = self.pending.copy()
            b.turn = turn
            cands += [(c, turn) for c in self._candidates(b, score, depth=1)]
        if not cands:
            return None
        cands.sort(key=lambda ct: ct[0].score, reverse=True)
        best = self._accept([c for c, _ in cands], [current], reading)
        if best is None:
            return None
        start = self.pending.copy()
        start.turn = next(t for c, t in cands if c is best)
        self.start_from(start)
        self.board.push(best.moves[0])
        return GameEvent("moves", moves=best.moves, message="")

    def _accept(self,cands: list[_Candidate], rivals: list[float], reading: np.ndarray) -> _Candidate | None:
        """Ứng viên tốt nhất, nếu nó hơn rõ ràng mọi đối thủ và khớp gần như toàn bộ."""
        if not cands:
            return None
        best = cands[0]
        second = max([c.score for c in cands[1:2]] + rivals)
        if best.score - second < self.margin:
            return None
        if int((np.array(argmax_labels(reading)) == best.labels).sum()) < self.min_match:
            return None
        return best

    @staticmethod
    def _candidates(board: chess.Board, score, depth: int) -> list[_Candidate]:
        """Mọi thế sau `depth` nước, gộp theo (ô đi, ô đến) để các cách phong cấp không tranh nhau."""
        groups: dict[tuple, _Candidate] = {}
        b = board.copy(stack=False)
        for m1 in list(b.legal_moves):
            b.push(m1)
            if depth == 1:
                labels = _labels(b)
                cand = _Candidate(score(labels), [m1], labels)
                if _better(cand, groups.get(_key(m1))):
                    groups[_key(m1)] = cand
            else:
                for m2 in list(b.legal_moves):
                    b.push(m2)
                    labels = _labels(b)
                    cand = _Candidate(score(labels), [m1, m2], labels)
                    key = (_key(m1), _key(m2))
                    if _better(cand, groups.get(key)):
                        groups[key] = cand
                    b.pop()
            b.pop()
        return sorted(groups.values(), key=lambda c: c.score, reverse=True)
