# Trợ lý cờ vua qua camera (bản 1) — Kế hoạch triển khai

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Từ camera điện thoại (làm webcam cho laptop), nhận diện quân bằng YOLO, theo dõi ván cờ bằng nước đi hợp lệ, gợi ý nước tốt nhất từ Stockfish và vẽ mũi tên lên video.

**Architecture:** Giữ các module tuần 1 (`board`, `detect`, `tracker`, `engine`, `camera`). Thêm chuỗi module độc lập, nối với nhau bằng mảng numpy / `chess.Board` / dataclass: `pieces` (YOLO → khung quân) → `reading` (khung → xác suất 64×13) → `stability` → `game` (GameTracker) → `advisor` (Stockfish nền) → `overlay`. `pipeline.Pipeline` ghép chúng cho cả `scripts/play.py` (camera) và `scripts/replay.py` (video offline).

**Tech Stack:** Python 3.12, OpenCV, numpy, python-chess, Stockfish 19, Ultralytics YOLO11s, PyTorch (CUDA, RTX 3050 6GB), pytest.

**Spec:** `docs/superpowers/specs/2026-10-20-camera-chess-assistant-design.md`

## Global Constraints

- Chạy trên Windows 11, Python ≥ 3.10; lệnh chạy bằng `.venv\Scripts\python.exe`.
- Lớp quân dùng ký hiệu FEN theo thứ tự cố định `PIECE_CLASSES = "PNBRQKpnbrqk"` (chỉ số 0–11), ô trống là chỉ số `EMPTY = 12`, ký tự `"."`; `NUM_CLASSES = 13`.
- Một "reading" là `np.ndarray` float64 hình `(64, 13)`, hàng theo thứ tự `chess.SQUARES` (a1=0 … h8=63), mỗi hàng tổng = 1, mọi phần tử ∈ [ε, 1−ε] trước chuẩn hoá, `ε = 0.01`.
- Giá trị mặc định từ spec: `foot_lift = 0.10`, `stable_frames = 8`, `max_drop = 4`, `margin = 3.0`, `min_match = 62`, `lost_after = 3`, `think_time = 0.5`, `multipv = 1`, ngưỡng blunder `200` centipawn.
- Mục tiêu hiệu năng: video ≥ 15 FPS; gợi ý ≤ ~1 s sau khi nước của đối thủ được ghi nhận.
- Không bao giờ ghi nhận nước đi khi chưa đạt ngưỡng; báo "không chắc" thay vì đoán.
- Trọng số model ở `models/pieces.pt`; `models/` và `data/` không đưa vào git.
- Phím trong `play.py`: `w`/`b` bên đến lượt khi đồng bộ lại · `c` đổi màu người dùng · `u` lùi nước · `f` lật bàn · `r` dò lại bàn cờ · `p` tạm dừng gợi ý · `q`/Esc thoát.
- Chữ vẽ bằng `cv2.putText` dùng tiếng Việt không dấu (OpenCV không vẽ được dấu).
- Ngoài phạm vi: kính, chạy trên điện thoại, app PyQt/web, đồng hồ cờ, giọng nói, tính năng che giấu việc dùng trợ lý.

## Review Focus

1. **Phong cấp khi model không phân biệt được quân phong** → phải ghi nhận phong Hậu, không được báo "không chắc" (test ở Task 4).
2. **Đổi quân liên tiếp làm số quân giảm 1 mỗi nước** → `StabilityFilter` vẫn phải cho qua; chỉ chặn khi giảm > `max_drop` (test ở Task 3).
3. **Mất camera điện thoại giữa ván (WiFi/USB rớt)** → giữ nguyên thế cờ, tự mở lại camera, không thoát (test ở Task 8).
4. **Camera nhìn đi chỗ khác ≥ 3 nước** → chuyển LOST sau `lost_after` lần không khớp, đọc lại toàn bộ và hỏi lượt; không được ghi nhận chuỗi nước bịa (test ở Task 5).
5. **Người dùng chọn sai bên đến lượt khi đồng bộ lại** (bên không đi đang bị chiếu) → `set_turn` trả `False`, giữ trạng thái chờ, hiện lý do (test ở Task 5).

---

### Task 0: Git, kiểu dữ liệu dùng chung, phụ thuộc ML

**Files:**
- Create: `chess_assistant/schema.py`, `tests/test_schema.py`
- Modify: `.gitignore` (thêm `models/`), `requirements-ml.txt` (ghim `ultralytics>=8.3`, `torch` bản CUDA — ghi chú lệnh cài trong README)

**Interfaces:**
- Produces:
  - `PIECE_CLASSES: str = "PNBRQKpnbrqk"`, `EMPTY: int = 12`, `NUM_CLASSES: int = 13`, `EPS: float = 0.01`
  - `@dataclass(frozen=True) class PieceBox: cls: int; conf: float; xyxy: tuple[float, float, float, float]`
  - `def class_of(piece: chess.Piece | None) -> int` — `None` → `EMPTY`, ngược lại `PIECE_CLASSES.index(piece.symbol())`
  - `def reading_from_board(board: chess.Board, confidence: float = 0.95) -> np.ndarray` — reading (64,13) đặt `confidence` vào lớp đúng, phần còn lại chia đều cho 12 lớp khác
  - `def argmax_labels(reading: np.ndarray) -> list[int]` — 64 chỉ số lớp

- [ ] **Step 1:** `git init`, commit toàn bộ code tuần 1 hiện có (`git add -A && git commit -m "chore: tuần 1 - dò bàn cờ, tracker, engine"`). Kiểm tra `.venv/`, `data/`, `engines/*.exe`, `models/` không nằm trong `git status`.
- [ ] **Step 2: Viết test** `tests/test_schema.py`:
  - `test_class_of`: `class_of(None) == EMPTY`, `class_of(chess.Piece.from_symbol("q")) == 10`.
  - `test_reading_from_board_shape_and_rows`: với `chess.Board()`, shape `(64, 13)`, `np.allclose(r.sum(1), 1)`, `argmax_labels(r)[chess.E1] == PIECE_CLASSES.index("K")`, `argmax_labels(r)[chess.E4] == EMPTY`.
- [ ] **Step 3:** Chạy `.venv\Scripts\python.exe -m pytest tests/test_schema.py -v` → FAIL (module chưa có).
- [ ] **Step 4:** Viết `chess_assistant/schema.py` theo Interfaces.
- [ ] **Step 5:** Chạy lại → PASS. Cài ML: `.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124` rồi `pip install -r requirements-ml.txt`; kiểm tra `python -c "import torch; print(torch.cuda.is_available())"` in `True`.
- [ ] **Step 6:** Commit `feat: kiểu dữ liệu dùng chung cho quân cờ và reading`.

---

### Task 1: Quy khung quân về ô (`read_board`)

**Files:**
- Modify: `chess_assistant/board.py` (thêm `square_at_canvas`, `user_color_from_corners`)
- Create: `chess_assistant/reading.py`, `tests/test_reading.py`, `tests/fakes.py`

**Interfaces:**
- Consumes: `PieceBox`, `EMPTY`, `NUM_CLASSES`, `EPS` (Task 0); `square_center_image`, `MARGIN_PX`, `SQUARE_PX` (board.py)
- Produces:
  - `board.square_at_canvas(x: float, y: float) -> chess.Square | None` — ô chứa điểm canvas, `None` nếu ngoài bàn
  - `board.user_color_from_corners(corners: np.ndarray) -> chess.Color` — `WHITE` nếu trung bình y của a1,h1 lớn hơn của h8,a8 (hàng 1 gần đáy ảnh = gần camera)
  - `reading.foot_point(box: PieceBox, foot_lift: float = 0.10) -> tuple[float, float]` — `((x1+x2)/2, y2 - foot_lift*(y2-y1))`
  - `reading.read_board(boxes: list[PieceBox], H: np.ndarray, foot_lift: float = 0.10) -> np.ndarray` — reading (64,13)
  - `tests/fakes.py`: `boxes_for_board(board: chess.Board, H: np.ndarray, conf: float = 0.9, height: float = 60.0) -> list[PieceBox]` — khung giả cao `height`, rộng 40, đặt sao cho điểm chân trùng tâm ô (dùng `square_center_image`)

Quy tắc xác suất mỗi ô (spec §4): khởi tạo hàng = `EPS`; ô có khung lớp `c`, conf `s` → `p[c] = s`, `p[EMPTY] = 1 − s`; ô không khung → `p[EMPTY] = 1 − EPS`; kẹp `[EPS, 1−EPS]`; chuẩn hoá hàng. Nhiều khung cùng ô → giữ conf cao nhất.

- [ ] **Step 1: Viết test** `tests/test_reading.py` (dùng `tests/synthetic.POSES` và `BoardView.from_corners(...).H`):
  - `test_board_round_trip[pose]` (tham số hoá theo mọi pose): `argmax_labels(read_board(boxes_for_board(chess.Board(), H), H)) == argmax_labels(reading_from_board(chess.Board()))`.
  - `test_duplicate_boxes_keep_highest_conf`: hai khung cùng chân ở e4, lớp `P` conf 0.6 và lớp `B` conf 0.9 → argmax e4 là `B`.
  - `test_off_board_box_ignored`: khung có chân ở (5, 5) ảnh → reading bằng reading của bàn trống.
  - `test_rows_valid`: mọi hàng tổng 1, `reading.min() > 0`.
  - `test_square_at_canvas`: tâm ô e4 trong canvas → `chess.E4`; `(1, 1)` → `None`.
  - `test_user_color_from_corners`: `POSES["white_side"]` → `WHITE`, `POSES["black_side"]` → `BLACK`.
- [ ] **Step 2:** Chạy `pytest tests/test_reading.py -v` → FAIL.
- [ ] **Step 3:** Cài đặt `square_at_canvas`, `user_color_from_corners`, `foot_point`, `read_board` (chiếu điểm chân bằng `cv2.perspectiveTransform` với H), và `tests/fakes.boxes_for_board`.
- [ ] **Step 4:** Chạy lại → PASS; chạy toàn bộ `pytest` → không hồi quy.
- [ ] **Step 5:** Commit `feat: quy khung quân YOLO về 64 ô`.

---

### Task 2: Mở rộng Engine (MultiPV, giới hạn sức mạnh)

**Files:**
- Modify: `chess_assistant/engine.py`, `tests/test_engine.py`

**Interfaces:**
- Produces:
  - `Engine.analyse(board: chess.Board, think_time: float = 0.5, multipv: int = 1) -> list[Suggestion]` — sắp theo thứ tự PV; rỗng nếu ván đã kết thúc. `suggest()` giữ nguyên, gọi `analyse(...)[0]`.
  - `Engine.play_limited(board: chess.Board, elo: int, think_time: float = 0.5) -> chess.Move` — bật `UCI_LimitStrength=True`, `UCI_Elo=elo` chỉ cho lệnh này rồi tắt lại.
  - `Suggestion.white_cp() -> int` — điểm theo phía Trắng, chiếu hết quy đổi `±10000`.

- [ ] **Step 1: Viết test** (bỏ qua khi thiếu Stockfish như hiện có):
  - `test_analyse_multipv`: thế khởi đầu, `multipv=3` → 3 nước khác nhau, đều hợp lệ.
  - `test_play_limited_legal`: `play_limited(chess.Board(), elo=1350)` ∈ `legal_moves`.
  - `test_white_cp_mate`: thế chiếu hết 1 nước của test hiện có → `analyse(...)[0].white_cp() == 10000`.
- [ ] **Step 2:** Chạy → FAIL. **Step 3:** Cài đặt. **Step 4:** Chạy `pytest tests/test_engine.py -v` → PASS.
- [ ] **Step 5:** Commit `feat: engine hỗ trợ multipv và giới hạn Elo`.

---

### Task 3: `StabilityFilter`

**Files:**
- Create: `chess_assistant/stability.py`, `tests/test_stability.py`

**Interfaces:**
- Consumes: `argmax_labels`, `EMPTY` (Task 0)
- Produces: `class StabilityFilter(stable_frames: int = 8, max_drop: int = 4)` với
  - `push(reading: np.ndarray | None, board_locked: bool) -> np.ndarray | None` — trả reading trung bình **đúng một lần** khi chuỗi argmax giống nhau đạt `stable_frames` frame liên tiếp; `None` các lần khác.
  - `reset() -> None`

Quy tắc: `reading is None` hoặc `not board_locked` → xoá cửa sổ, trả `None`. argmax khác cửa sổ → bắt đầu cửa sổ mới từ frame này. Khi đạt đủ: đếm số ô có quân; nếu đã có lần phát trước và `số quân lần trước − số quân hiện tại > max_drop` → không phát (coi là bị che) nhưng vẫn đánh dấu đã xử lý cửa sổ này. Khi phát, lưu số quân làm mốc.

- [ ] **Step 1: Viết test** (reading dựng bằng `reading_from_board`):
  - `test_emits_once_after_stable_frames`: 7 lần push → `None`; lần 8 → mảng; lần 9–20 cùng reading → `None`.
  - `test_change_restarts_window`: 5 frame A, 1 frame B, 7 frame A → chưa phát; frame A thứ 8 liên tiếp → phát.
  - `test_unlocked_resets`: 7 frame A, 1 lần `board_locked=False`, 7 frame A → `None`.
  - `test_capture_drop_of_one_passes` *(Review Focus 2)*: phát thế khởi đầu, sau đó 8 frame thế đã bớt 1 quân → phát.
  - `test_large_drop_blocked`: phát thế khởi đầu, sau đó 8 frame thế bớt 5 quân → `None`.
- [ ] **Step 2:** Chạy → FAIL. **Step 3:** Cài đặt. **Step 4:** Chạy → PASS.
- [ ] **Step 5:** Commit `feat: bộ lọc ổn định cho kết quả đọc bàn cờ`.

---

### Task 4: `GameTracker` — trạng thái SYNCED (ghi nhận nước đi)

**Files:**
- Create: `chess_assistant/game.py`, `tests/test_game.py`

**Interfaces:**
- Consumes: `class_of`, `argmax_labels`, `reading_from_board`, `EMPTY` (Task 0)
- Produces:
  - `position_score(reading: np.ndarray, board: chess.Board) -> float` — `Σ log reading[sq, class_of(board.piece_at(sq))]`
  - `match_count(reading: np.ndarray, board: chess.Board) -> int` — số ô argmax trùng thế cờ
  - `class SyncState(Enum): UNSYNCED, SYNCED, NEED_TURN, LOST`
  - `@dataclass class GameEvent: kind: Literal["moves", "synced", "need_turn", "lost", "unsure"]; moves: list[chess.Move] = field(default_factory=list); flipped: bool = False; message: str = ""`
  - `class GameTracker(margin: float = 3.0, min_match: int = 62, lost_after: int = 3)` với thuộc tính `board: chess.Board`, `state: SyncState`; phương thức `update(reading) -> GameEvent | None`, `undo() -> chess.Move | None`, `pgn() -> str`, `start_from(board: chess.Board) -> None` (đặt thế, chuyển SYNCED — dùng cho test và đầu ván).

Thuật toán `update` khi SYNCED:
1. Ứng viên 1 nước: nhóm `legal_moves` theo `(from_square, to_square)`; trong mỗi nhóm có phong cấp chọn quân có `position_score` cao nhất, hoà điểm (chênh < 1e-6) → Hậu. Thêm ứng viên "không đổi" (thế hiện tại).
2. Ứng viên tốt nhất là "không đổi" → `None`, đặt lại bộ đếm hụt.
3. Ứng viên tốt nhất là nước đi, `match_count ≥ min_match` và hơn ứng viên thứ hai ≥ `margin` → `push`, trả `GameEvent("moves", [move])`.
4. Không đạt → thử mọi chuỗi 2 nước (`m1` hợp lệ, `m2` hợp lệ sau `m1`, cùng quy tắc nhóm phong cấp) với cùng điều kiện so với ứng viên thứ hai trong tập 2 nước và với ứng viên "không đổi" → `GameEvent("moves", [m1, m2])`.
5. Vẫn không → tăng bộ đếm hụt, trả `GameEvent("unsure", message=...)`; đạt `lost_after` → `state = LOST`, trả `GameEvent("lost")`.

- [ ] **Step 1: Viết test** `tests/test_game.py` với helper `noisy(board, wrong: dict[chess.Square, str] = {}, conf=0.9)` = `reading_from_board(board, conf)` rồi đổi các ô trong `wrong` sang ký hiệu chỉ định:
  - `test_no_change_returns_none`: `start_from(Board())`, `update(noisy(Board()))` → `None`.
  - `test_simple_move`: reading của thế sau `e2e4` → `kind == "moves"`, `moves == [e2e4]`, `board.fen()` khớp.
  - `test_castling`, `test_en_passant`: dựng FEN trước, reading thế sau `e1g1` / nước bắt tốt qua đường → nhận đúng nước.
  - `test_promotion_ambiguous_picks_queen` *(Review Focus 1)*: thế có Tốt trắng ở a7, reading sau phong cấp có ô a8 với xác suất bằng nhau giữa `Q`,`R`,`B`,`N` (0.22 mỗi lớp) → `moves == [a7a8q]`.
  - `test_two_missed_plies`: reading thế sau `e2e4 e7e5` → `moves == [e2e4, e7e5]`.
  - `test_one_misread_square_tolerated`: thế sau `g1f3` với 1 ô xa bị đọc nhầm (`wrong={chess.A7: "."}`) → vẫn nhận `g1f3`.
  - `test_ambiguous_is_unsure`: reading trộn đều hai thế sau `e2e4` và `e2e3` (trung bình hai reading) → `kind == "unsure"`, `board` không đổi.
  - `test_lost_after_three_unsure`: 3 lần reading vô nghĩa (thế đã xếp lại hoàn toàn khác) → sự kiện cuối `kind == "lost"`, `state == LOST`.
  - `test_undo_and_pgn`: sau 2 nước, `undo()` trả nước cuối; `"1. e4" in pgn()`.
- [ ] **Step 2:** Chạy `pytest tests/test_game.py -v` → FAIL.
- [ ] **Step 3:** Cài đặt phần SYNCED + `undo` + `pgn` (dùng `chess.pgn.Game.from_board`). Tính điểm vectơ hoá: dựng mảng chỉ số lớp 64 ô cho mỗi ứng viên rồi `np.log(reading)[np.arange(64), idx].sum()`.
- [ ] **Step 4:** Chạy → PASS. Đo: `update` với chuỗi 2 nước ở thế khởi đầu < 200 ms (thêm `test_two_ply_speed`).
- [ ] **Step 5:** Commit `feat: GameTracker ghi nhận nước đi bằng khớp nước hợp lệ`.

---

### Task 5: `GameTracker` — UNSYNCED / LOST / đồng bộ lại

**Files:**
- Modify: `chess_assistant/game.py`, `tests/test_game.py`

**Interfaces:**
- Produces:
  - `board_from_labels(labels: list[int], flip: bool) -> chess.Board | None` — dựng thế từ argmax (`flip=True` dùng ô `63 − sq`), đặt `"KQkq"` rồi `board.castling_rights = board.clean_castling_rights()`; trả `None` nếu không đúng 1 Vua mỗi bên hoặc có Tốt ở hàng 1/8.
  - `GameTracker.set_turn(color: chess.Color) -> bool` — chỉ hợp lệ ở `NEED_TURN`; đặt lượt, nếu `is_valid()` thì vào SYNCED và trả `True`, ngược lại trả `False` và giữ `NEED_TURN`.
  - `GameTracker.pending: chess.Board | None`, `GameTracker.pending_flipped: bool`

Hành vi `update` khi UNSYNCED/LOST:
1. `match_count(reading, chess.Board()) ≥ min_match` → `start_from(Board())`, trả `GameEvent("synced")`.
2. Ngược lại thử `flip=False` và `flip=True`; bỏ thế `None`; nếu cả hai hợp lệ chọn thế có điểm hợp lý cao hơn = (số Tốt trắng ở hàng 2–4) + (số Tốt đen ở hàng 5–7), hoà → `flip=False`.
3. Không thế nào hợp lệ → `GameEvent("unsure", message="khong doc duoc the co hop le: <lý do>")`.
4. Có thế → `pending`, `state = NEED_TURN`, trả `GameEvent("need_turn", flipped=<flip đã chọn>)`. `play.py` sẽ gọi `BoardTracker.flip()` khi `flipped` là `True` (Task 9).
5. Ở `NEED_TURN`, `update` trả `None` (chờ `set_turn`).

- [ ] **Step 1: Viết test**:
  - `test_start_position_autosyncs`: tracker mới, reading thế khởi đầu → `kind == "synced"`, `state == SYNCED`.
  - `test_resync_midgame_needs_turn`: reading thế giữa ván → `kind == "need_turn"`; `set_turn(chess.WHITE)` → `True`; `board.board_fen()` đúng.
  - `test_resync_detects_flipped_view`: reading thế giữa ván đã xoay 180° (đảo hàng reading theo `63 − sq`) → `flipped is True`, `pending.board_fen()` là thế gốc.
  - `test_castling_rights_inferred`: thế có Vua trắng e1, Xe h1, Xe a1 đã đi → sau `set_turn`, `board.has_kingside_castling_rights(WHITE)` và không có quyền phía Hậu.
  - `test_set_turn_invalid_side` *(Review Focus 5)*: thế Đen đang chiếu Vua trắng; `set_turn(chess.BLACK)` → `False`, `state == NEED_TURN`; `set_turn(chess.WHITE)` → `True`.
  - `test_lost_then_resync` *(Review Focus 4)*: đang SYNCED, reading của thế sau 3 nước → 3 lần `update` cho `unsure, unsure, lost`; lần tiếp → `need_turn`; không có sự kiện `"moves"` nào được phát.
  - `test_invalid_reading_unsure`: reading không có Vua đen → `kind == "unsure"`.
- [ ] **Step 2:** Chạy → FAIL. **Step 3:** Cài đặt. **Step 4:** Chạy toàn bộ `tests/test_game.py` → PASS.
- [ ] **Step 5:** Commit `feat: GameTracker đồng bộ lại khi lạc`.

---

### Task 6: `Advisor` (Stockfish ở luồng nền)

**Files:**
- Create: `chess_assistant/advisor.py`, `tests/test_advisor.py`

**Interfaces:**
- Consumes: `Engine.analyse`, `Engine.play_limited`, `Suggestion.white_cp` (Task 2)
- Produces:
  - `@dataclass class Advice: fen: str; moves: list[chess.Move]; score_text: str; white_cp: int; blunder_cp: int | None = None`
  - `class Advisor(engine: Engine, think_time: float = 0.5, multipv: int = 1, mode: Literal["best", "training"] = "best", elo: int = 1500, blunder_cp: int = 200)` với
    - `request(board: chess.Board, user_color: chess.Color) -> None` — không chặn; gửi bản sao board cho luồng nền (1 worker); yêu cầu mới thay yêu cầu cũ chưa chạy; kết quả của FEN cũ bị bỏ.
    - `latest() -> Advice | None` — chỉ trả kết quả có `fen` trùng yêu cầu gần nhất.
    - `close() -> None`

Lỗi trong luồng nền (engine chết, exception) → ghi `logging.exception`, `latest()` trả `None`, `Advisor.error: str | None` giữ thông báo để overlay hiển thị; lần `request` sau thử lại bình thường.

Hành vi: lượt người dùng → `best`: `analyse(multipv)` → `moves` là các PV đầu; `training`: `moves = [play_limited(board, elo)]`, điểm vẫn từ `analyse`. Đồng thời lưu điểm phía người dùng. Lượt đối thủ ở `training` → `analyse`; nếu điểm phía người dùng giảm ≥ `blunder_cp` so với lần lưu trước → `Advice(moves=[], blunder_cp=<mức giảm>)`. Lượt đối thủ ở `best` → không phân tích.

- [ ] **Step 1: Viết test** (bỏ qua khi thiếu Stockfish; helper `wait_latest(advisor, timeout=5)`):
  - `test_suggests_on_user_turn`: thế khởi đầu, `user_color=WHITE` → `latest().moves[0]` hợp lệ, `fen` khớp.
  - `test_no_advice_on_opponent_turn_best_mode`: sau `e2e4`, `user_color=WHITE` → `latest()` vẫn `None` sau 1.5 s.
  - `test_stale_request_dropped`: `request(A)` rồi ngay `request(B)` → kết quả cuối có `fen == B.fen()`.
  - `test_engine_error_reported`: engine giả có `analyse` ném `chess.engine.EngineTerminatedError` → `latest() is None`, `advisor.error` khác `None`.
  - `test_training_blunder_detected`: thế mà người dùng (Trắng) vừa thí Hậu không bù → `blunder_cp >= 200`.
- [ ] **Step 2:** Chạy → FAIL. **Step 3:** Cài đặt bằng `concurrent.futures.ThreadPoolExecutor(max_workers=1)`. **Step 4:** Chạy → PASS.
- [ ] **Step 5:** Commit `feat: Advisor phân tích Stockfish ở luồng nền`.

---

### Task 7: `PieceDetector` (bọc Ultralytics)

**Files:**
- Create: `chess_assistant/pieces.py`, `tests/test_pieces.py`

**Interfaces:**
- Consumes: `PieceBox`, `PIECE_CLASSES` (Task 0)
- Produces: `class PieceDetector(weights: Path = MODELS_DIR / "pieces.pt", conf: float = 0.25, imgsz: int = 640, device: str | int | None = None, model=None)`, `__call__(frame: np.ndarray) -> list[PieceBox]`. `MODELS_DIR = <gốc dự án>/models`.

Hành vi: `model` truyền vào thì dùng luôn (cho test); không thì nạp `ultralytics.YOLO(weights)`, file thiếu → `FileNotFoundError` hướng dẫn chạy `scripts/train_pieces.py`. Ánh xạ `model.names` (dict chỉ số → tên) sang chỉ số `PIECE_CLASSES` theo tên đúng bằng ký hiệu FEN; thiếu lớp nào → `ValueError` liệt kê lớp thiếu. Gọi `model.predict(frame, conf=..., imgsz=..., device=..., verbose=False)[0].boxes` và đọc `.cls`, `.conf`, `.xyxy` qua `.cpu().numpy()`.

- [ ] **Step 1: Viết test** với `FakeModel` (có `names` và `predict` trả đối tượng giả có `.boxes.cls/.conf/.xyxy` là mảng bọc `.cpu().numpy()`):
  - `test_maps_class_names`: `names={0: "k", 1: "P", ...}` (đủ 12, thứ tự xáo) → `PieceBox.cls` đúng chỉ số trong `PIECE_CLASSES`.
  - `test_missing_class_raises`: thiếu `"q"` → `ValueError` chứa `"q"`.
  - `test_missing_weights_message`: đường dẫn không tồn tại → `FileNotFoundError` chứa `"train_pieces"`.
- [ ] **Step 2:** Chạy → FAIL. **Step 3:** Cài đặt. **Step 4:** Chạy → PASS.
- [ ] **Step 5:** Commit `feat: PieceDetector bọc YOLO`.

---

### Task 8: Camera tự kết nối lại + Overlay

**Files:**
- Modify: `chess_assistant/camera.py`, `scripts/live_view.py` (dùng `overlay.draw_grid` thay hàm cục bộ)
- Create: `chess_assistant/overlay.py`, `tests/test_camera.py`, `tests/test_overlay.py`

**Interfaces:**
- Consumes: `Advice` (Task 6), `BoardView`, `grid_lines_image`, `square_center_image` (board.py)
- Produces:
  - `camera.open_camera(source: int | str = 0, width=1280, height=720)` — `str` là URL luồng (ví dụ `http://<ip>:4747/video` của DroidCam) thì mở bằng backend mặc định.
  - `class ReconnectingCamera(factory: Callable[[], cv2.VideoCapture], retry_every: float = 1.0, clock=time.monotonic)` với `read() -> np.ndarray | None` (lỗi đọc → nhả capture, trả `None`; thử mở lại không sớm hơn `retry_every` giây) và `connected: bool`, `release()`.
  - `@dataclass class OverlayState: status: str; status_color: tuple[int, int, int]; board: chess.Board | None = None; user_color: chess.Color = chess.WHITE; advice: Advice | None = None; last_move: chess.Move | None = None; message: str = ""`
  - `overlay.draw_grid(img, view: BoardView) -> None` (chuyển từ live_view, vẽ tại chỗ)
  - `overlay.draw_arrow(img, H, move: chess.Move, color=(0, 220, 0), thickness=6) -> None` — từ `square_center_image(H, from)` tới `square_center_image(H, to)`.
  - `overlay.mini_board_layout(user_color: chess.Color) -> list[list[chess.Square]]` — 8 hàng từ trên xuống, mỗi hàng 8 ô từ trái sang; phía người dùng ở dưới.
  - `overlay.draw_mini_board(img, board: chess.Board, user_color: chess.Color, size: int = 200) -> None` — góc trên phải, theo `mini_board_layout`.
  - `overlay.draw_eval_bar(img, white_cp: int) -> None` — cạnh trái, kẹp ±1000 cp.
  - `overlay.render(frame, view: BoardView | None, state: OverlayState) -> np.ndarray` — trả bản sao đã vẽ: lưới mờ (`draw_grid` trộn alpha 0.4) khi có `view`; mũi tên cho mọi nước trong `state.advice.moves` (nước đầu đậm nhất); thanh điểm từ `state.advice.white_cp`; sơ đồ mini khi có `state.board`; dòng trạng thái ở trên cùng kèm lượt đi (`"Luot: Trang/Den"`) và nước cuối (SAN); `state.message` ở dưới cùng; blunder hiện chữ đỏ `"Nuoc vua roi mat X diem"`.

- [ ] **Step 1: Viết test**:
  - `test_reconnect_after_failure` *(Review Focus 3)*: `factory` giả trả capture đọc được 2 frame rồi lỗi; đồng hồ giả. `read()` lần 3 → `None`, `connected is False`; tiến đồng hồ 0.5 s → `read()` không gọi `factory`; tiến 1.0 s → gọi `factory`, trả frame.
  - `test_arrow_pixels`: ảnh đen 1280×720, `view` từ `POSES["white_side"]`, `draw_arrow(e2e4)` → điểm giữa đoạn từ tâm e2 tới tâm e4 có màu `(0, 220, 0)`.
  - `test_render_without_view`: `render(frame, None, OverlayState("Dang tim ban co...", (0, 200, 255)))` trả ảnh cùng shape, không lỗi.
  - `test_mini_board_layout`: `mini_board_layout(WHITE)[7][0] == chess.A1`, `mini_board_layout(BLACK)[7][0] == chess.H8`, `mini_board_layout(BLACK)[0][0] == chess.A1`.
- [ ] **Step 2:** Chạy → FAIL. **Step 3:** Cài đặt; cập nhật `live_view.py` import `draw_grid` từ overlay. **Step 4:** Chạy toàn bộ `pytest` → PASS.
- [ ] **Step 5:** Commit `feat: camera tự kết nối lại và lớp vẽ overlay`.

---

### Task 9: `Pipeline` + `scripts/play.py` (chạy đầu-cuối)

**Files:**
- Create: `chess_assistant/pipeline.py`, `scripts/play.py`, `tests/test_pipeline.py`
- Modify: `tests/fakes.py` (thêm `FakeDetector`), `chess_assistant/tracker.py` (exception từ luồng dò → `logging.exception` và tính như một lần dò hụt, không ném ra vòng lặp chính), `tests/test_detect.py` (thêm `test_tracker_survives_detector_exception`: thay `detect_quad` bằng hàm ném lỗi qua `monkeypatch`, `update` không ném, `corners` giữ nguyên)

**Interfaces:**
- Consumes: `BoardTracker` (có), `PieceDetector.__call__`, `read_board`, `StabilityFilter.push`, `GameTracker.update/set_turn/undo`, `Advisor.request/latest`, `user_color_from_corners`, `overlay.OverlayState`
- Produces:
  - `@dataclass class FrameResult: corners: np.ndarray | None; view: BoardView | None; reading: np.ndarray | None; events: list[GameEvent]; overlay: OverlayState`
  - `class Pipeline(detector: Callable[[np.ndarray], list[PieceBox]], tracker: BoardTracker, game: GameTracker | None = None, advisor: Advisor | None = None, stability: StabilityFilter | None = None)` với `process(frame) -> FrameResult`, `user_color: chess.Color | None` (`None` = tự suy từ góc camera), `paused: bool`.
  - `tests/fakes.FakeDetector(board_fn: Callable[[], chess.Board], corners)` — trả `boxes_for_board(board_fn(), H)`.

Thứ tự trong `process`: tracker → (có góc) `BoardView` → detector → `read_board` → `stability.push(reading, tracker.status == "locked")` → nếu có reading ổn định: `game.update` → sự kiện `need_turn` có `flipped` thì `tracker.flip()` → khi `game.state == SYNCED`, chưa `paused` và thế cờ đổi: `advisor.request(game.board, user_color)` → dựng `OverlayState` (status: tìm bàn / bám / bị che / LẠC / CHỜ LƯỢT + message của sự kiện gần nhất).

`scripts/play.py`: tham số `--camera` (int hoặc URL), `--think-time`, `--multipv`, `--mode best|training`, `--elo`, `--weights`. Kiểm tra Stockfish + weights khi khởi động (báo lỗi rõ, thoát mã 1). Vòng lặp: `ReconnectingCamera.read()` (None → hiện "Mat ket noi camera, dang thu lai..." trên khung đen) → `pipeline.process` → `overlay.render` → phím theo Global Constraints. Thoát: ghi `game.pgn()` vào `data/games/<YYYYmmdd-HHMMSS>.pgn`, đóng engine/tracker/camera.

- [ ] **Step 1: Viết test** `tests/test_pipeline.py` (tracker `background=False`, advisor `None`, ảnh từ `tests/synthetic.scene`):
  - `test_syncs_and_records_move`: 10 frame thế khởi đầu → có sự kiện `synced`; đổi `board_fn` sang thế sau `e2e4`, 10 frame → có sự kiện `moves == [e2e4]`.
  - `test_occlusion_frames_ignored`: giữa hai thế, 5 frame có `hand=` che → không sự kiện `moves` sai; sau đó nhận đúng nước.
  - `test_flip_applied_on_resync`: bắt đầu từ thế giữa ván với camera `black_side` nhưng tracker đoán nhầm hướng (gọi `tracker.flip()` sau frame đầu) → sau sự kiện `need_turn`, `tracker.corners` đúng thứ tự a1,h1,h8,a8 thật (sai số < 6 px).
  - `test_user_color_from_view`: pose `black_side` → `pipeline.process(...).overlay.user_color == BLACK`.
- [ ] **Step 2:** Chạy → FAIL. **Step 3:** Cài đặt `pipeline.py`, `FakeDetector`, `scripts/play.py`. **Step 4:** Chạy toàn bộ `pytest` → PASS; `.venv\Scripts\python.exe -m scripts.play --help` chạy được.
- [ ] **Step 5:** Commit `feat: pipeline đầu-cuối và chương trình play`.

---

### Task 10: Dữ liệu công khai + train model v0

**Files:**
- Create: `chess_assistant/dataset.py`, `scripts/prepare_dataset.py`, `scripts/train_pieces.py`, `tests/test_dataset.py`
- Modify: `README.md` (mục "Huấn luyện model")

**Interfaces:**
- Produces:
  - `normalize_class_name(name: str) -> str | None | Literal["ignore"]` — chấp nhận ký hiệu FEN và các biến thể tên phổ biến không phân biệt hoa thường, `-`/`_`/khoảng trắng: `white-pawn`, `white_pawn`, `pawn-white`, `wp`, `w_p`, … → `"P"`; `black-king`, `bk` … → `"k"`; `board`, `chessboard`, `chess-board` → `"ignore"`; tên không có màu (`pawn`, `bishop`) và tên lạ → `None`.
  - `remap_label_file(lines: list[str], src_names: list[str]) -> list[str] | None` — đổi chỉ số lớp sang chỉ số `PIECE_CLASSES`; bỏ dòng `"ignore"`; có dòng `None` → trả `None` (bỏ ảnh).
  - `build_dataset(sources: list[Path], out_dir: Path, holdout: set[str]) -> dict[str, int]` — mỗi nguồn là thư mục YOLO (có `data.yaml` với `names`, các thư mục `train|valid|test/images|labels`). Nguồn có tên trong `holdout` → toàn bộ vào `test`; nguồn khác giữ split gốc (`valid` → `val`). Ghi `out_dir/{train,val,test}/{images,labels}` (tên file có tiền tố tên nguồn) và `out_dir/data.yaml` với `names` = 12 ký hiệu FEN. Trả số ảnh mỗi split.
  - `scripts/train_pieces.py --data data/pieces/data.yaml --epochs 100 --batch 16 --model yolo11s.pt` → `YOLO(model).train(data, epochs, imgsz=640, batch, device=0, close_mosaic=10)`, sao `runs/.../weights/best.pt` sang `models/pieces.pt`.

- [ ] **Step 1: Viết test**:
  - `test_normalize_names` (tham số hoá): `"White-Pawn" → "P"`, `"bk" → "k"`, `"black_queen" → "q"`, `"board" → "ignore"`, `"bishop" → None`.
  - `test_remap_label_file`: nguồn `["board", "white-king", "black-pawn"]`, dòng `"1 0.5 0.5 0.1 0.2"` → `"5 0.5 0.5 0.1 0.2"`; dòng lớp 0 bị bỏ; nguồn có `"bishop"` → `None`.
  - `test_build_dataset_holdout(tmp_path)`: hai nguồn giả nhỏ (mỗi nguồn 2 ảnh train, 1 valid), `holdout={"setB"}` → mọi ảnh của setB nằm ở `test`, `data.yaml` có `names` đúng 12 ký hiệu.
- [ ] **Step 2:** Chạy → FAIL. **Step 3:** Cài đặt. **Step 4:** Chạy → PASS.
- [ ] **Step 5 (người dùng làm tay):** Tải 2–4 dataset "chess pieces" định dạng YOLOv8 từ Roboflow Universe/Kaggle, **kiểm tra giấy phép** (CC BY 4.0 / public domain…), giải nén vào `data/public/<tên>/`. Ghi nguồn + giấy phép vào `data/public/SOURCES.md`.
- [ ] **Step 6:** `python -m scripts.prepare_dataset --sources data/public/* --out data/pieces --holdout <tên một bộ>` → in số ảnh mỗi split, `test` > 0.
- [ ] **Step 7:** `python -m scripts.train_pieces ...` (chạy nền, ~1–2 giờ trên RTX 3050). Kiểm tra dòng log `albumentations:` có `Blur` và `ImageCompression`; `models/pieces.pt` tồn tại; mAP50 trên `val` được ghi vào README.
- [ ] **Step 8:** Chạy thử `python -m scripts.play --camera <chỉ số DroidCam/Iriun>` với bàn cờ thật, ghi nhận vấn đề vào `docs/notes/v0-field-test.md`.
- [ ] **Step 9:** Commit `feat: chuẩn bị dataset công khai và script train` (không commit `data/`, `models/`).

---

### Task 11: Công cụ tự gán nhãn + fine-tune v1

**Files:**
- Create: `chess_assistant/autolabel.py`, `scripts/capture_labeled.py`, `tests/test_autolabel.py`

**Interfaces:**
- Consumes: `PieceBox`, `foot_point`, `square_at_canvas` (Task 1), `PieceDetector` (Task 7), `BoardTracker`
- Produces:
  - `autolabel(boxes: list[PieceBox], H: np.ndarray, board: chess.Board, foot_lift: float = 0.10) -> list[tuple[int, tuple[float, float, float, float]]] | None` — mỗi ô có quân trong `board` nhận đúng 1 khung (conf cao nhất trong các khung có chân rơi vào ô đó), lớp lấy từ `board`; khung ở ô trống bị bỏ; có ô có quân không có khung → `None`.
  - `to_yolo_lines(labels, width: int, height: int) -> list[str]` — `"<cls> <cx> <cy> <w> <h>"` chuẩn hoá về [0,1], 6 chữ số thập phân.
  - `scripts/capture_labeled.py --set <tên bộ cờ> --fen <FEN, mặc định thế khởi đầu> --camera <...> --every 10` → ghi `data/captured/<set>/<YYYYmmdd-HHMMSS>/{images,labels,preview}/` + `meta.json` (`{"set", "fen"}`); Space bật/tắt ghi, `q` thoát; chỉ lưu frame khi tracker `locked` và `autolabel` khác `None`.

- [ ] **Step 1: Viết test**:
  - `test_autolabel_uses_known_classes`: khung giả từ `boxes_for_board` nhưng **mọi khung mang lớp sai** (`cls=0`) → nhãn trả về có lớp đúng theo `board`.
  - `test_autolabel_missing_piece_returns_none`: bỏ khung của e1 → `None`.
  - `test_autolabel_drops_boxes_on_empty_squares`: thêm khung ở e4 (trống) → không có trong kết quả.
  - `test_to_yolo_lines`: khung `(100, 200, 140, 260)` ảnh 1280×720 lớp 5 → `"5 0.093750 0.319444 0.031250 0.083333"`.
- [ ] **Step 2:** Chạy → FAIL. **Step 3:** Cài đặt. **Step 4:** Chạy → PASS.
- [ ] **Step 5 (người dùng làm tay):** Với mỗi bộ cờ có được, quay ít nhất 3 thế (khởi đầu + 2 FEN giữa ván), mỗi thế ~1 phút, đổi góc/ánh sáng. Xem `preview/`, xoá ảnh hỏng.
- [ ] **Step 6:** `build_dataset` gồm `data/public/*` + `data/captured/*` (mỗi bộ cờ tự thu là một nguồn; YOLO dir tương thích), `--holdout <một bộ cờ tự thu>`; train lại từ `models/pieces.pt` → v1, ghi mAP vào README.
- [ ] **Step 7:** Commit `feat: công cụ tự gán nhãn theo thế cờ đã biết`.

---

### Task 12: Đánh giá — `eval_pieces.py` và `replay.py`

**Files:**
- Create: `chess_assistant/evaluation.py`, `scripts/eval_pieces.py`, `scripts/replay.py`, `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `Pipeline` (Task 9), `read_board`, `argmax_labels`, `class_of`
- Produces:
  - `square_accuracy(reading: np.ndarray, board: chess.Board) -> float` — `match_count / 64`
  - `move_accuracy(recognized: list[chess.Move], truth: list[chess.Move]) -> dict[str, int]` với khoá `correct`, `wrong`, `missed`: so theo từng ply; nước ghi nhận khác nước thật cùng ply → `wrong` (và dừng so); phần thật còn lại chưa ghi nhận → `missed`.
  - `scripts/eval_pieces.py --data data/pieces/data.yaml --captured data/captured/<holdout set>` → in mAP50 theo lớp (`YOLO.val(split="test")`) và square accuracy trung bình trên các phiên (dùng `meta.json["fen"]`).
  - `scripts/replay.py <video> <pgn> [--start-fen ...]` → chạy `Pipeline` không advisor, tracker `background=False`, in `correct/wrong/missed`, tỉ lệ đúng, thời gian xử lý trung bình và p95 mỗi frame (ms). Thoát mã 1 nếu `wrong > 0`.

- [ ] **Step 1: Viết test**:
  - `test_square_accuracy_perfect`: `reading_from_board(b)` với `b` → `1.0`.
  - `test_move_accuracy`: truth `[e4, e5, Nf3]`, recognized `[e4, e5]` → `{"correct": 2, "wrong": 0, "missed": 1}`; recognized `[e4, d5]` → `{"correct": 1, "wrong": 1, "missed": 1}`.
  - `test_replay_synthetic(tmp_path)`: ghi video ngắn từ `tests/synthetic.scene` cho chuỗi `e4 e5` (mỗi thế 15 frame) + PGN; chạy `replay` với `FakeDetector` (tham số `--fake-detector` chỉ dùng cho test) → `correct == 2`, `wrong == 0`.
- [ ] **Step 2:** Chạy → FAIL. **Step 3:** Cài đặt. **Step 4:** Chạy toàn bộ `pytest` → PASS.
- [ ] **Step 5 (người dùng làm tay):** Quay 2–3 ván thật (ít nhất 1 với bộ cờ holdout), ghi PGN đúng; chạy `replay` cho từng ván. Đạt tiêu chí spec: ≥ 95% nước đúng, `wrong == 0`. Không đạt → ghi lại ca sai, chỉnh `foot_lift`/`margin`/`stable_frames` hoặc thu thêm dữ liệu, lặp lại.
- [ ] **Step 6:** Ghi kết quả đánh giá vào README; commit `feat: công cụ đánh giá model và chạy lại video`.
