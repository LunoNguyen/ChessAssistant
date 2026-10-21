# Thiết kế: Trợ lý cờ vua qua camera (bản 1 — camera điện thoại)

Ngày: 2026-10-20 · Trạng thái: chờ duyệt

## 1. Mục tiêu & phạm vi

**Mục tiêu.** Một hệ thống nhìn bàn cờ thật qua camera, theo dõi ván đang chơi và gợi ý nước đi tốt nhất cho người dùng, hiển thị bằng mũi tên vẽ lên video.

**Bối cảnh sử dụng.**
- Giai đoạn này: camera điện thoại làm webcam cho laptop (DroidCam / Iriun → `cv2.VideoCapture`). Mọi xử lý và hiển thị trên laptop.
- Camera ở góc nhìn người chơi, có thể di chuyển (mô phỏng kính có camera — mục tiêu tương lai).
- Phải chạy được với **nhiều loại bộ cờ**, kể cả bộ chưa từng thấy khi train.
- Phần cứng: laptop RTX 3050 6GB, i5-12450HX, 24GB RAM, Windows 11.

**Tiêu chí hoàn thành bản 1.**
- Chơi trọn ván thật với ít nhất 2–3 bộ cờ khác nhau (ít nhất 1 bộ không có trong dữ liệu train): ≥ 95% nước đi được ghi nhận đúng, không ghi nhận sai nước nào mà không báo.
- Gợi ý xuất hiện ≤ ~1 giây sau khi nước của đối thủ được ghi nhận.
- Video hiển thị ≥ 15 FPS.

**Ngoài phạm vi.** Tích hợp kính; chạy trên điện thoại; app giao diện (PyQt/web); đồng hồ cờ; giọng nói; thiết kế nhằm che giấu việc dùng trợ lý khi đấu với người không biết.

**Giữ lại từ tuần 1.** `board.py` (homography, cắt ô), `detect.py` (tự dò bàn cờ), `tracker.py` (bám bàn cờ), `engine.py` (Stockfish), `camera.py`. Hướng dò tay (`calibration.py`) giữ làm dự phòng.

## 2. Kiến trúc

```text
Camera ─frame─▶ BoardTracker ─H, 4 góc─▶ PieceDetector ─khung quân─▶ BoardReading ─xác suất 64 ô─▶
StabilityFilter ─đọc ổn định─▶ GameTracker ─chess.Board, lượt─▶ Advisor ─gợi ý─▶ Overlay
```

| Module (file) | Trạng thái | Vào → Ra | Phụ thuộc |
|---|---|---|---|
| `tracker.BoardTracker` | có | frame → 4 góc (a1,h1,h8,a8) | `detect` |
| `pieces.PieceDetector` | mới | frame → `list[PieceBox]` (lớp, conf, xyxy) | ultralytics |
| `reading.read_board` | mới | `list[PieceBox]`, H → `BoardReading` (8×8×13 xác suất) | `board` |
| `stability.StabilityFilter` | mới | `BoardReading` mỗi frame → `BoardReading` ổn định hoặc None | — |
| `game.GameTracker` | mới | `BoardReading` ổn định → sự kiện (nước mới / lạc / đã đồng bộ) + `chess.Board` | python-chess |
| `advisor.Advisor` | mới | `chess.Board` → `Suggestion` (luồng nền) | `engine` |
| `overlay` | mới | frame, H, trạng thái → ảnh hiển thị | `board` |
| `scripts/play.py` | mới | vòng lặp chính, phím điều khiển | tất cả |

Quy ước chung: lớp quân dùng ký hiệu FEN (`P N B R Q K p n b r q k`), lớp ô trống là `.`; ô theo chỉ số `chess.SQUARES`. `GameTracker` không phụ thuộc ảnh — test được hoàn toàn bằng dữ liệu giả.

Đường lên kính sau này: thay nguồn frame và `overlay`; các module giữa giữ nguyên.

## 3. Nhận diện quân

**Model.** Ultralytics YOLO11s detect, 12 lớp, ảnh vào 640. Chạy trên GPU (~10–20ms/frame). Trọng số đặt ở `models/pieces.pt` (không đưa vào git).

**Dữ liệu — 3 vòng.**
1. *Công khai*: tải vài dataset "chess pieces detection" (Roboflow Universe/Kaggle) có giấy phép cho phép dùng; script `scripts/prepare_public_data.py` gộp và ánh xạ tên lớp về 12 lớp FEN, loại ảnh có lớp không ánh xạ được. Train **v0**.
2. *Tự gán nhãn có hỗ trợ*: `scripts/capture_labeled.py` — người dùng quay video một bộ cờ đang bày một thế đã biết (nhập FEN, mặc định thế khởi đầu). Mỗi frame được lấy mẫu: v0 cho khung quân, `BoardTracker` + `read_board` gán khung vào ô, **nhãn lớp lấy từ FEN đã biết**; khung không rơi vào ô có quân bị bỏ; ô có quân mà không có khung → frame bị loại. Lưu theo định dạng YOLO, mỗi phiên một thư mục, kèm ảnh xem trước để người dùng loại ảnh hỏng.
3. *Fine-tune* trên công khai + tự thu → **v1**.

**Chia dữ liệu.** Theo bộ cờ / phiên quay, không chia ngẫu nhiên. Giữ riêng ≥ 1 bộ cờ hoàn toàn cho test.

**Tăng cường.** Mặc định của Ultralytics (HSV, perspective, mosaic) + motion blur và giảm chất lượng JPEG (mô phỏng camera điện thoại qua WiFi).

**Đánh giá** (`scripts/eval_pieces.py`): mAP50 theo lớp và tỉ lệ ô đúng sau `read_board` trên bộ cờ giữ riêng. Mục tiêu sơ bộ: ≥ 97% ô đúng.

## 4. Quy khung quân về ô (`read_board`)

- Điểm chân = (giữa cạnh dưới khung) nâng lên `foot_lift × chiều cao khung`, mặc định `foot_lift = 0.10`, tinh chỉnh bằng dữ liệu thật.
- Chiếu điểm chân qua H sang canvas → ô. Điểm rơi ngoài bàn bị bỏ.
- Nhiều khung cùng ô: giữ khung có conf cao nhất.
- Xác suất 13 lớp mỗi ô: ô có khung lớp `c` với conf `s` → `p[c] = s`, `p[.] = 1 − s`, các lớp khác chia đều một lượng nhỏ ε; ô không có khung → `p[.] = 1 − ε`. Mọi xác suất được kẹp vào [ε, 1−ε] (ε = 0.01) để log không vô cực.

## 5. Ổn định & theo dõi ván

**StabilityFilter.** Chỉ trả về kết quả khi: `BoardTracker.status == "locked"`; argmax 64 ô không đổi trong `stable_frames` frame liên tiếp (mặc định 8); số ô có quân không giảm quá `max_drop` (mặc định 4) so với lần ổn định trước (dấu hiệu tay che). Trả về trung bình xác suất của các frame ổn định.

**GameTracker — trạng thái `SYNCED`** (giữ `chess.Board`):
- Khi nhận đọc ổn định khác thế hiện tại: chấm điểm mọi thế cờ thu được sau 1 nước hợp lệ: `score(P) = Σ_ô log p_ô(P[ô])`. So với `score(thế hiện tại)`.
- Không có nước 1-ply nào đạt ngưỡng → thử chuỗi 2 nước (bỏ lỡ một nước).
- Chấp nhận khi: nước tốt nhất hơn nước thứ hai ≥ `margin` (log-likelihood, mặc định 3.0) **và** argmax khớp ≥ 62/64 ô với thế kết quả.
- Phong cấp: chọn theo xác suất lớp ở ô phong (Hậu nếu không phân biệt được).
- `lost_after` (mặc định 3) lần đọc ổn định liên tiếp không khớp → `LOST`.

**Trạng thái `LOST` / bắt đầu (`UNSYNCED`):**
- Đọc khớp thế khởi đầu (≥ 62/64) → `SYNCED` từ thế khởi đầu, Trắng đi.
- Ngược lại dựng thế cờ từ argmax; thử cả hai hướng 180°, chọn hướng hợp lệ và hợp lý hơn (không có Tốt ở hàng 1/8, mỗi bên 1 Vua, `is_valid()` với một trong hai bên đi). Không hợp lệ → chờ, hiện lý do.
- Hỏi bên đến lượt (phím `w`/`b`); quyền nhập thành suy từ Vua/Xe ở ô gốc; không có bắt tốt qua đường.

**Màu quân người dùng.** Mặc định là bên gần camera (tính từ vị trí hàng 1/hàng 8 trong ảnh); phím `c` để đổi.

**Sửa sai & lưu.** Phím `u` lùi nước gần nhất. Ván được ghi ra `data/games/<thời gian>.pgn`.

## 6. Gợi ý & hiển thị

**Advisor.** Một tiến trình Stockfish, luồng nền. Khi đến lượt người dùng và thế cờ mới: `analyse` với `think_time` (mặc định 0.5s), `multipv` (mặc định 1, tuỳ chọn 3). Thế cờ đổi giữa chừng → bỏ kết quả cũ. Chế độ `best` (mặc định) và `training` (`UCI_LimitStrength` + `UCI_Elo`, chỉ báo khi người dùng vừa đi nước làm điểm đánh giá giảm ≥ 200 centipawn, tức ≥ 2 Tốt).

**Overlay (cửa sổ OpenCV).** Lưới mờ; mũi tên gợi ý từ tâm ô đi → ô đến, chiếu ngược về ảnh camera; thanh điểm; nước cuối; lượt; dòng trạng thái (đang tìm / đang bám / bị che / LẠC + lý do); sơ đồ 2D thu nhỏ của thế cờ hệ thống đang hiểu.

**Phím.** `w`/`b` bên đến lượt khi đồng bộ lại · `c` đổi màu người dùng · `u` lùi nước · `f` lật bàn · `r` dò lại bàn cờ · `p` tạm dừng gợi ý · `q` thoát.

## 7. Xử lý lỗi

- Mất camera: thử mở lại mỗi giây, hiện thông báo, không thoát.
- Thiếu Stockfish hoặc `models/pieces.pt`: báo cách khắc phục khi khởi động và thoát sạch.
- Kết quả mâu thuẫn: không bao giờ ghi nhận nước khi không đạt ngưỡng; hiện "không chắc" thay vì đoán.
- Lỗi trong luồng nền (dò bàn cờ, engine): ghi log, hiển thị trạng thái, vòng lặp chính tiếp tục.

## 8. Kiểm thử

- **Unit test** (pytest, không cần GPU/camera):
  - `read_board`: khung quân giả + homography biết trước (dùng `tests/synthetic.py`).
  - `StabilityFilter`: chuỗi đọc giả có nhiễu, che tay.
  - `GameTracker`: sinh xác suất 64 ô từ thế cờ thật + nhiễu; kiểm tra nước thường, nhập thành, bắt tốt qua đường, phong cấp, bỏ lỡ 1 nước, đọc sai 1–2 ô, lạc rồi đồng bộ lại, lùi nước.
  - `Advisor`: dùng Stockfish thật nếu có (bỏ qua nếu không).
- **Đánh giá model**: `scripts/eval_pieces.py` trên bộ cờ giữ riêng.
- **Chạy lại video**: `scripts/replay.py <video> <pgn>` chạy toàn pipeline offline và in tỉ lệ nước ghi nhận đúng, số nước ghi nhận sai, độ trễ. Đây là thước đo chính cho tiêu chí hoàn thành.

## 9. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Bộ dò bàn cờ thuật toán yếu với góc nghiêng / camera rung | Đo bằng video thật sớm; nếu tỉ lệ khoá thấp, bổ sung YOLO-pose 4 góc (hướng C) |
| Dataset công khai ít bộ cờ đa dạng, nhãn không thống nhất | Script ánh xạ lớp + vòng tự gán nhãn có hỗ trợ |
| Nhầm Vua/Hậu, Tượng/Tốt | GameTracker chỉ chấp nhận nước hợp lệ khớp toàn cục |
| Kết nối camera điện thoại qua WiFi trễ/nén mạnh | Ưu tiên USB (DroidCam/Iriun hỗ trợ), tăng cường dữ liệu bằng nén JPEG |

## 10. Thứ tự xây dựng (gợi ý cho kế hoạch)

1. `GameTracker` + `StabilityFilter` + test (không cần model).
2. Chuẩn bị dữ liệu công khai, train v0, `PieceDetector` + `read_board`.
3. `Advisor` + `overlay` + `scripts/play.py` → chạy thử đầu-cuối với camera điện thoại.
4. Công cụ tự gán nhãn, thu dữ liệu nhiều bộ cờ, fine-tune v1.
5. `replay.py` + đánh giá theo tiêu chí hoàn thành; tinh chỉnh ngưỡng.
