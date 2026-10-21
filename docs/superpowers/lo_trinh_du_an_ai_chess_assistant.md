# Lộ Trình Triển Khai Hệ Thống AI Nhận Diện Bàn Cờ Vua & Gợi Ý Nước Đi

Tài liệu này tổng hợp danh sách các công việc, công nghệ, thư viện, bộ dữ liệu (dataset) và lộ trình chi tiết từng bước để xây dựng hệ thống AI hỗ trợ chơi cờ vua qua camera real-time.

> **Nguyên tắc chung:** Làm cho pipeline chạy được từ đầu đến cuối càng sớm càng tốt (dù thô), rồi mới thay từng khối bằng phiên bản tốt hơn (AI). Tránh dành nhiều tuần train model trước khi có một hệ thống chạy được.

---

## 1. Công Nghệ & Công Cụ Sử Dụng (Tech Stack)

### 1.1 Ngôn ngữ lập trình
*   **Python 3.10+** (máy hiện có 3.12): Ngôn ngữ chính cho toàn bộ hệ thống (Computer Vision, Deep Learning, Logic Cờ Vua, UI).

### 1.2 Framework & Core Libraries
*   **Deep Learning / Machine Learning**:
    *   `PyTorch`: Framework huấn luyện mô hình (Ultralytics chạy trên PyTorch, nên không cần TensorFlow).
    *   `Ultralytics` (YOLOv8 / YOLO11): Phân loại quân cờ (`-cls`) và, nếu cần, phát hiện góc bàn cờ (`-pose`).
*   **Computer Vision & Xử lý ảnh**:
    *   `OpenCV` (`opencv-python`): Xử lý luồng video camera, biến đổi hình học (Perspective Transform), vẽ GUI/overlay.
    *   `Pillow`: Thao tác với hình ảnh cơ bản.
    *   `Albumentations`: Tăng cường dữ liệu ảnh (chỉnh sáng, xoay nhẹ, làm mờ, thêm nhiễu).
*   **Xử lý Logic Cờ Vua & Engine**:
    *   `python-chess`: Quản lý luật chơi, trạng thái ván cờ, sinh nước đi hợp lệ, xuất FEN/PGN.
    *   `Stockfish`: Engine cờ vua mạnh nhất hiện nay. **Lưu ý:** không cài qua `pip` — phải tải file thực thi (`stockfish-windows-x86-64-avx2.exe`) từ https://stockfishchess.org/download/ và trỏ đường dẫn tới nó qua `chess.engine.SimpleEngine.popen_uci(...)`.
*   **Giao diện người dùng (UI/App)** *(Tùy chọn)*:
    *   `PyQt6` hoặc `CustomTkinter`: Ứng dụng Desktop.
    *   `Streamlit` / `Gradio`: Demo Web nhanh (khó làm video real-time mượt; phù hợp demo ảnh tĩnh hơn).

---

## 2. Nguồn Dataset Dự Kiến (Tìm trên Mạng & Tự Thu Tập)

Nên kết hợp dataset có sẵn trên Kaggle/Roboflow và dataset tự chụp bằng camera thực tế.

### 2.1 Dataset tìm trên mạng
*   **Roboflow Universe (Khuyên dùng)**:
    *   Từ khóa: `chess board detection`, `chess pieces detection`, `chess square classification`.
    *   Đã được gán nhãn sẵn theo chuẩn YOLO.
    *   **Lưu ý:** phần lớn là nhãn *bounding box cho cả ảnh*, không phải ảnh từng ô. Muốn dùng cho mô hình phân loại ô, cần viết script cắt lại theo ô (xem Giai đoạn 3).
*   **Kaggle Datasets**:
    *   *Chess Positions Dataset*: Ảnh bàn cờ **2D kỹ thuật số** kèm FEN — chỉ hữu ích để test logic FEN, **không** dùng để train nhận diện bàn cờ thật.
    *   *Chess Pieces Detection*: Ảnh quân cờ thật từ nhiều góc độ.

### 2.2 Dataset tự chụp (Bắt buộc để tối ưu thực tế)
*   Dataset trên mạng hiếm khi khớp bộ cờ, góc camera và ánh sáng của bạn — đây là nguồn dữ liệu quan trọng nhất.
*   **Cách thu thập nhanh — "tự gán nhãn" bằng vị trí đã biết:**
    1.  Cố định camera, hiệu chỉnh 4 góc bàn cờ một lần.
    2.  Bày một thế cờ đã biết (thế khởi đầu, hoặc nhập FEN bạn tự xếp).
    3.  Chụp vài chục frame, đổi ánh sáng / dịch nhẹ camera.
    4.  Script tự cắt 64 ô và gán nhãn theo FEN đã biết → hàng nghìn ảnh ô có nhãn mà không phải gán tay.
*   Chỉ cần **Label Studio / Roboflow** nếu đi theo hướng train YOLO-pose phát hiện góc bàn cờ.

---

## 3. Danh Sách Các Việc Cần Làm (Checklist Chi Tiết)

### Giai đoạn 1: Chuẩn bị Môi trường & Dữ liệu
- [ ] **Khởi tạo môi trường lập trình**:
  - [ ] Tạo virtualenv (`python -m venv .venv`).
  - [ ] Cài thư viện: `opencv-python`, `ultralytics`, `torch`, `python-chess`, `albumentations`, `matplotlib`.
  - [ ] Tải Stockfish binary, đặt vào thư mục `engines/`, kiểm tra kết nối bằng một script nhỏ.
- [ ] **Chuẩn bị Dataset**:
  - [ ] Tải dataset quân cờ từ Roboflow/Kaggle (để pre-train / bổ sung).
  - [ ] Thu thập ảnh thực tế bằng cách "tự gán nhãn" (mục 2.2) — làm sau khi Giai đoạn 2 xong.
  - [ ] Chia dữ liệu Train (70%) – Validation (20%) – Test (10%). **Chia theo ván/phiên chụp**, không chia ngẫu nhiên theo ô — nếu không, các ô gần như giống hệt nhau sẽ rơi vào cả train và test, làm accuracy ảo cao.

---

### Giai đoạn 2: Phát Hiện & Nắn Phẳng Bàn Cờ (Computer Vision)
- [ ] **Kết nối Camera**:
  - [ ] Mở luồng Webcam (`cv2.VideoCapture`), đặt độ phân giải cố định (ví dụ 1280×720).
- [x] **Xác định 4 góc bàn cờ**:
  - [x] *Dự phòng*: người dùng bấm chọn 4 góc một lần (`scripts.calibrate`).
  - [x] *Tự động (Thuật toán, đang dùng)*: Hough Lines có lọc theo hướng gradient → khớp lưới 9×9 đường bằng homography → chọn cửa sổ 8×8 theo số đường khớp + độ xen kẽ màu ô. Hoạt động cả khi có quân trên bàn.
  - [x] *Theo dõi (tracking)*: dò lại định kỳ ở luồng nền, làm mượt rung, giữ vị trí khi tay che, chỉ chấp nhận cú nhảy lớn sau vài lần xác nhận.
  - [ ] *Nâng cấp (AI, nếu cần)*: Train YOLOv8-pose với 4 keypoint góc — chỉ khi thuật toán trên thất bại nhiều với camera/ánh sáng thật.
- [x] **Xác định hướng bàn cờ (Orientation)**:
  - [x] Tự suy ra: a1 là ô tối (loại 2/4 cách xoay), phía có quân sáng màu ở thế khởi đầu là phía Trắng. Sau khi khoá, giữ hướng nhất quán qua các frame. Phím `f` để lật nếu đoán nhầm.
- [ ] **Biến đổi hình học (Perspective Transform)**:
  - [ ] `cv2.getPerspectiveTransform()` + `cv2.warpPerspective()` → ảnh vuông 2D chuẩn (ví dụ 800×800px).
- [ ] **Chia lưới 64 ô cờ**:
  - [ ] Mỗi ô 100×100px (a1 → h8, theo hướng đã xác định).
  - [ ] **Cắt ô có phần đệm phía trên:** sau khi nắn phẳng, quân cao (Vua, Hậu) bị "đổ" sang ô phía trên do góc nhìn nghiêng. Nên cắt mỗi ô cao gấp ~1.5–2 lần (lấy thêm phần phía trên), rồi resize về kích thước cố định cho model. Cách khác: cắt trực tiếp từ ảnh gốc theo bounding box chiếu của khối hộp đứng trên mỗi ô.

---

### Giai đoạn 3: Nhận Diện Quân Cờ trên Từng Ô
- [ ] **Huấn luyện Mô hình Phân loại (Classification)**:
  - [ ] Train **YOLOv8-cls** hoặc **ResNet18 / MobileNetV3** với 13 lớp:
    - 6 quân Trắng: `P`, `N`, `B`, `R`, `Q`, `K`
    - 6 quân Đen: `p`, `n`, `b`, `r`, `q`, `k`
    - 1 ô trống: `empty`
  - [ ] **Mẹo:** tách làm 2 bước nếu thiếu dữ liệu: (1) ô trống / quân trắng / quân đen (dễ, rất chính xác), (2) loại quân. Bước (1) đã đủ để theo dõi nước đi (xem Giai đoạn 4).
  - [ ] Lưu ý mất cân bằng lớp: `empty` chiếm ~50% ô, `P/p` nhiều, `K/Q` rất ít → dùng class weights hoặc oversample.
- [ ] **Pipeline dự đoán**:
  - [ ] Đưa **64 ô theo một batch** vào model (không gọi 64 lần riêng lẻ) để giảm độ trễ.
  - [ ] Giữ lại **xác suất** của từng lớp (không chỉ nhãn top-1) — dùng ở Giai đoạn 4 để chọn nước đi hợp lệ khớp nhất.

---

### Giai đoạn 4: Theo Dõi Trạng Thái Ván Cờ (Game State Tracking)

> **Thay đổi quan trọng so với bản cũ:** Không dựng FEN trực tiếp từ một ảnh rồi gửi cho Stockfish. Một ảnh tĩnh **không chứa** thông tin: đến lượt bên nào, quyền nhập thành, ô bắt tốt qua đường, số nước đi. Ngoài ra `board.is_valid()` vẫn trả về `True` với nhiều thế cờ sai do nhận diện nhầm (ví dụ nhầm Tượng thành Tốt).

- [ ] **Duy trì một `chess.Board` xuyên suốt ván**:
  - [ ] Bắt đầu từ thế khởi đầu (hoặc FEN người dùng nhập).
  - [ ] Khi phát hiện bàn cờ thay đổi, **sinh tất cả nước đi hợp lệ** (`board.legal_moves`), với mỗi nước tính thế cờ kết quả, và chọn nước có độ khớp cao nhất với xác suất model ở 64 ô.
  - [ ] Nếu không nước nào đủ khớp → báo "không nhận ra nước đi", giữ nguyên trạng thái, yêu cầu người dùng kiểm tra lại.
  - [ ] Cách này tự xử lý đúng nhập thành, bắt tốt qua đường, phong cấp, và sửa được phần lớn lỗi phân loại.
- [ ] **Bộ lọc ổn định (Khử nhiễu)**:
  - [ ] Chỉ xử lý khi bàn cờ đứng yên trong $N$ frame liên tiếp (ví dụ 10–15 frame).
  - [ ] Phát hiện tay che bàn cờ bằng frame differencing hoặc số ô thay đổi bất thường (> 4 ô thay đổi cùng lúc = có vật che) → bỏ qua frame đó.
- [ ] **Xuất FEN**: `board.fen()` lúc này luôn chính xác và đầy đủ (lượt đi, nhập thành, en passant) để đưa vào Stockfish. Có thể xuất thêm PGN cả ván.

---

### Giai đoạn 5: Tích Hợp Engine Gợi Ý Nước Đi
- [ ] **Tích hợp Stockfish Engine**:
  - [ ] Mở engine **một lần** khi khởi động (`SimpleEngine.popen_uci`), đóng khi thoát — không mở lại mỗi nước.
  - [ ] Cấu hình độ khó: `Skill Level` (0–20) hoặc `UCI_LimitStrength` + `UCI_Elo`.
- [ ] **Tính toán gợi ý**:
  - [ ] `engine.analyse(board, chess.engine.Limit(time=0.5))` → nước đi tốt nhất (`pv[0]`) và điểm số (`score`).
  - [ ] Chạy engine trong **luồng riêng** (thread) để không làm đứng luồng camera.
  - [ ] Chỉ gợi ý khi đến lượt bên người dùng chọn (Trắng/Đen).

---

### Giai đoạn 6: Hiển Thị Kết Quả & Hoàn Thiện Giao Diện (UI)
- [ ] **Vẽ Overlay trực tiếp lên Video**:
  - [ ] Vẽ khung bàn cờ và lưới 8×8 (chiếu ngược từ ảnh phẳng về ảnh camera bằng ma trận nghịch đảo của Perspective Transform).
  - [ ] Vẽ mũi tên gợi ý từ ô xuất phát đến ô đích (ví dụ E2 → E4) bằng `cv2.arrowedLine`, toạ độ tâm ô được chiếu ngược về ảnh gốc.
  - [ ] Hiển thị thanh đánh giá (evaluation bar) và nước đi cuối cùng.
- [ ] **Đóng gói ứng dụng**:
  - [ ] Giao diện điều khiển: Hiệu chỉnh góc, Bắt đầu, Tạm dừng, Hoàn tác nước đi, Chọn bên Trắng/Đen, Độ khó AI.
  - [ ] Kiểm thử toàn bộ hệ thống từ lúc đặt bàn cờ đến khi chơi hết 1 ván.

---

## 4. Đánh Giá (Evaluation)

| Chỉ số | Cách đo | Mục tiêu tham khảo |
|---|---|---|
| Accuracy phân loại ô | Trên tập Test (chia theo ván) | ≥ 99% |
| Accuracy cả bàn | % ảnh mà cả 64 ô đều đúng | ≥ 90% |
| Accuracy nhận diện nước đi | % nước đi được ghi nhận đúng trong ván thật | ≥ 98% |
| Độ trễ | Từ lúc bàn đứng yên đến khi hiện mũi tên | < 1 giây |
| FPS hiển thị | Luồng camera + overlay | ≥ 15 FPS |

> Accuracy 99% mỗi ô nghe cao nhưng nghĩa là trung bình ~0.64 ô sai mỗi ảnh — đó là lý do cần Giai đoạn 4 (đối chiếu với nước đi hợp lệ).

---

## 5. Lộ Trình Tóm Tắt Theo Tuần (Ví dụ trong 6 Tuần)

```text
Tuần 1: Môi trường + Stockfish ➔ Camera ➔ Tự dò & bám bàn cờ + hướng a1 ➔ Perspective Transform ➔ Cắt 64 ô (có đệm).
Tuần 2: Thu thập dataset "tự gán nhãn" theo thế cờ đã biết ➔ Train model phân loại (bắt đầu với 3 lớp: trống/trắng/đen).
Tuần 3: Theo dõi ván cờ: chess.Board + so khớp legal_moves ➔ Bộ lọc ổn định / phát hiện tay che.
        ➔ Mốc: pipeline chạy được từ đầu đến cuối (chưa cần đẹp).
Tuần 4: Mở rộng model lên 13 lớp ➔ Kết nối Stockfish (thread riêng) ➔ Vẽ mũi tên gợi ý lên video.
Tuần 5: (Tuỳ chọn) YOLO-pose cho góc bàn cờ nếu bộ dò thuật toán chưa đủ ổn ➔ Cải thiện dữ liệu ở các ca sai.
Tuần 6: Hoàn thiện UI/App ➔ Đánh giá theo bảng mục 4 ➔ Chơi thử nhiều ván trọn vẹn ➔ Refactor code.
```
