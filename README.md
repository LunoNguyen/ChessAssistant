# AI Chess Assistant

Nhận diện bàn cờ vua thật qua camera và gợi ý nước đi bằng Stockfish.
Lộ trình chi tiết: [lo_trinh_du_an_ai_chess_assistant.md](lo_trinh_du_an_ai_chess_assistant.md).

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Stockfish phải tải riêng — xem [engines/README.md](engines/README.md).

Phần nhận diện quân (YOLO) cần thêm Ultralytics và PyTorch bản CUDA. Cài theo đúng thứ tự này,
vì Ultralytics tự kéo PyTorch bản chỉ-CPU từ PyPI:

```bash
pip install -r requirements-ml.txt
pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip uninstall -y opencv-python-headless
pip install --force-reinstall --no-deps opencv-python
python -c "import torch; print(torch.cuda.is_available())"
```

Hai dòng `opencv` ở trên là bắt buộc: `albumentations` kéo theo `opencv-python-headless`, bản này ghi đè
OpenCV bằng bản không có cửa sổ và `play`/`live_view` sẽ báo lỗi "The function is not implemented".

## Chơi với trợ lý (camera điện thoại)

1. Cài app **DroidCam** hoặc **Iriun** trên điện thoại và laptop (nên cắm USB để ít trễ).
   Điện thoại sẽ hiện như một webcam (thường là camera 1 hoặc 2), hoặc dùng URL luồng của DroidCam.
2. Cần model nhận diện quân `models/pieces.pt` (xem mục "Huấn luyện model").
3. Chạy:

```bash
python -m scripts.play --camera 1
```

Bày thế cờ khởi đầu để hệ thống tự bắt đầu theo dõi. Giữa ván mà bị lạc, hệ thống đọc lại toàn bộ thế cờ
và hỏi bên đến lượt (`w`/`b`). Phím khác: `c` đổi màu quân của bạn, `u` lùi nước, `f` lật bàn,
`r` dò lại bàn cờ, `p` tạm dừng gợi ý, `q` thoát (ván lưu vào `data/games/*.pgn`).
Chế độ luyện tập: `--mode training --elo 1500` (gợi ý ở sức mạnh giới hạn + cảnh báo nước hớ).

## Huấn luyện model nhận diện quân

1. **Dataset công khai**:
   - **ChessReD2K** (khuyên dùng — ~2.000 ảnh chụp bằng điện thoại, nhiều góc, CC BY-NC-SA 4.0, chỉ dùng phi thương mại):
     tải `annotations.json` và `chessred2k.zip` từ
     [4TU.ResearchData](https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f), đặt `annotations.json`
     và nội dung giải nén của `chessred2k.zip` vào `data/downloads/chessred/`, rồi chạy
     `python -m scripts.import_chessred` → `data/public/chessred2k/`.
   - **Chess Pieces (Roboflow)** (bổ sung — Public Domain, 292 ảnh một góc):
     [trang dataset](https://public.roboflow.com/object-detection/chess-full) → Download → định dạng **YOLOv8**,
     giải nén vào `data/public/chess-pieces/`.
   - **Bàn cờ 2D (Chess Positions, CC0)** — 8.750 ảnh bàn cờ 2D với 28 kiểu bàn × 32 kiểu quân, nhãn từ tên file FEN.
     `python -m scripts.import_chess_positions` biến mỗi ảnh thành ảnh mô phỏng "điện thoại chụp màn hình"
     (nghiêng, moiré, chói, mờ, nhiễu, JPEG) → `data/public/chess2d/`. Ảnh gốc tải từ bản mirror
     `huggingface.co/datasets/smallchess/kaggle-chess-positions` vào `data/downloads/chess_positions/train/`.
     Lưu ý: không dùng để nhận gợi ý khi đấu với người thật trên các trang cờ (vi phạm luật của trang).
   - Dataset khác định dạng YOLOv8 cũng được: giải nén vào `data/public/<tên>/`, kiểm tra có đủ 12 lớp **có màu**.
   Ghi nguồn + giấy phép vào `data/public/SOURCES.md`.
2. **Dữ liệu tự thu** (sau khi có model v0): `python -m scripts.capture_labeled --set <tên bộ cờ> --camera 1`
   — bày thế cờ đã biết (mặc định thế khởi đầu, hoặc `--fen ...`), nhấn Space để ghi. Nhãn lấy từ FEN,
   không cần vẽ khung. Xem `preview/` và xoá ảnh hỏng.
   - **Model chưa nhận ra quân của bạn** (bộ cờ lạ, game cờ 3D…): dùng gán nhãn theo hình học, không cần model:
     `python -m scripts.capture_geometric --set <tên> --camera <...>` (thêm `--fen ...` hoặc `--pgn van.pgn` rồi bấm
     n/p để bước qua từng thế). Kiểm tra khung vàng/tím khớp quân thật (f lật, o xoay 90°, +/- chiều cao), Space để ghi.
3. **Gộp**: `python -m scripts.prepare_dataset --sources "data/public/*" "data/captured/*" --out data/pieces --holdout <một bộ cờ>`
4. **Train** (~1–2 giờ trên RTX 3050): `python -m scripts.train_pieces --data data/pieces/data.yaml`
   → `models/pieces.pt`. Fine-tune lần sau: thêm `--model models/pieces.pt`.
   Fine-tune với dữ liệu mới thường chỉ cần `--epochs 30`.
5. **Đánh giá**: `python -m scripts.eval_pieces --captured data/captured/<bộ holdout>`;
   cả ván thật: `python -m scripts.replay <video> <pgn>`.

### Train trên Google Colab (không chạy nặng máy local)

1. Gộp dữ liệu: `python -m scripts.prepare_dataset --sources "data/public/*" "data/captured/*" --out data/pieces_v3`
2. Đóng gói: `python -m scripts.export_colab --dataset data/pieces_v3` → thư mục `colab_upload/`
   (`pieces_dataset.zip` + `pieces.pt` hiện tại).
3. Tải hai file đó lên Google Drive vào `MyDrive/ChessAssistant/`.
4. Mở [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) trên colab.research.google.com
   (File → Upload notebook), chọn `Runtime → Change runtime type → T4 GPU`, chạy lần lượt từng ô.
5. Tải `MyDrive/ChessAssistant/pieces_new.pt` về, đổi tên thành `models/pieces.pt` (giữ bản cũ để so sánh).

Checkpoint lưu trên Drive: Colab bị ngắt thì đặt `RESUME = True` ở ô cấu hình và chạy lại để train tiếp.

## Sử dụng (Tuần 1)

1. **Kiểm tra engine:** `python -m scripts.check_engine`
2. **Xem trực tiếp:** `python -m scripts.live_view` — chương trình **tự tìm và bám theo bàn cờ**,
   vẽ lưới 8×8 lên ảnh camera và hiện 64 ô đã cắt.
   - Bày thế cờ khởi đầu khi bắt đầu để chương trình tự nhận ra phía Trắng (a1 ở đâu).
     Đoán nhầm thì nhấn `f` để lật 180°.
   - Tay che bàn cờ hay camera bị xê dịch nhẹ: chương trình giữ vị trí cũ rồi tự bắt lại.
   - Phím: `r` dò lại từ đầu, `s` lưu 64 ô vào `data/raw/<thời gian>/` (dữ liệu cho tuần 2), `q` thoát.
3. **Hiệu chỉnh tay (dự phòng)** khi tự dò không ổn (ánh sáng quá tối, bàn cờ không có ô
   tương phản rõ...): `python -m scripts.calibrate` → Space để chụp → bấm 4 góc ngoài theo thứ tự
   **a1 → h1 → h8 → a8** → Enter. Sau đó chạy `python -m scripts.live_view --manual`.

Có thể thử với ảnh tĩnh thay cho camera: thêm `--image anh.jpg` vào các lệnh trên.

### Cách tự dò bàn cờ hoạt động

Không cần train model ([chess_assistant/detect.py](chess_assistant/detect.py)):
1. Tìm các đường thẳng dài (Canny + Hough; mỗi điểm cạnh chỉ bỏ phiếu cho đường vuông góc
   với gradient của nó, để không bắt nhầm đường chéo qua góc ô), chia làm 2 họ theo hướng.
2. Thử các "ô hạt giống" tạo từ 2 đường liền kề mỗi họ, mở rộng thành lưới bằng cách khớp
   thêm đường vào toạ độ nguyên và fit lại homography.
3. Chọn cửa sổ 8×8 có nhiều đường khớp nhất và màu ô xen kẽ rõ nhất.
4. Hướng: a1 là ô tối, phía có quân sáng màu là phía Trắng.

Bàn cờ phải nằm trọn trong khung hình. Mỗi lần dò mất khoảng 100ms và chạy ở luồng nền.

## Cấu trúc

```text
chess_assistant/
  board.py        nắn phẳng (homography), xác định hướng đổ của quân, cắt 64 ô có đệm
  detect.py       tự dò bàn cờ (đường thẳng -> lưới -> cửa sổ 8x8) và xác định hướng a1
  tracker.py      bám theo bàn cờ qua nhiều frame, chạy dò ở luồng nền
  calibration.py  chọn 4 góc bằng tay (dự phòng), lưu/đọc config/calibration.json
  camera.py       mở camera
  engine.py       tìm và điều khiển Stockfish
scripts/          calibrate, live_view, check_engine
tests/            pytest (test Stockfish tự bỏ qua nếu chưa cài)
engines/          đặt stockfish*.exe ở đây
```

## Ghi chú kỹ thuật

- Ảnh nắn phẳng luôn theo hướng sơ đồ chuẩn (Trắng ở dưới) và có viền rộng 1 ô quanh bàn cờ.
- Quân cao trông như "đổ" về phía xa camera. Mỗi ô được cắt thêm `pad` (mặc định 0.75 cạnh ô)
  theo hướng đó, rồi xoay sao cho đầu quân luôn hướng lên. Nhờ vậy model phân loại luôn nhận
  ảnh ô cùng tư thế, dù camera đặt ở đâu.

## Kiểm thử

```bash
python -m pytest
```
