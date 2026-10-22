# Kế hoạch: đưa Chess Assistant lên kính (bản prototype thô sơ)

> Mục tiêu: camera gắn trên gọng kính → ảnh live về mạch xử lý (chính là code hiện tại) → gợi ý hiện lên
> màn hình nhỏ gắn trên kính. Bản đầu ưu tiên **làm được, dễ sửa**, chưa cần gọn nhẹ.
>
> Đây là thiết bị cho đồ án/học tập. Không dùng khi đấu với người không biết bạn đang dùng trợ lý
> (ở giải đấu hay trên mạng đều là gian lận).

---

## 1. Kiến trúc

```text
 ┌──────────── trên kính ────────────┐          ┌──────── đeo ở thắt lưng / túi ────────┐
 │  Camera (CSI hoặc USB)  ──────────┼── cáp ──▶│  Raspberry Pi 5 (8GB)                  │
 │                                   │          │   • dò bàn cờ, YOLO, theo dõi ván      │
 │  Màn hình 1.3" 240×240 (SPI) ◀────┼── dây ───│   • Stockfish                           │
 │  (+ gương bán phản xạ, tuỳ chọn)  │          │   • vẽ sơ đồ mini có tô ô gợi ý        │
 │  Nút bấm (tuỳ chọn)  ─────────────┼── dây ──▶│  Pin dự phòng USB-C PD 5V/5A            │
 └───────────────────────────────────┘          └────────────────────────────────────────┘
```

**Vì sao không đặt mạch trên kính:** Pi 5 nặng ~50 g, toả nhiệt 5–10 W và cần quạt; đặt lên đầu thì nóng và
nặng. Bản thô sơ để mạch ở thắt lưng, trên kính chỉ có camera + màn hình + nút, nối bằng dây.

**Màn hình hiển thị gì:** màn hình 240×240 quá nhỏ để vẽ đè lên video. Kính sẽ hiện **sơ đồ bàn cờ mini**
(phía người dùng ở dưới) với **ô quân cần đi tô vàng, ô đích tô xanh lá**, kèm dòng trạng thái và điểm số.

### Chọn mạch xử lý

| Mạch | YOLO11s (ước lượng) | Ưu | Nhược |
|---|---|---|---|
| **Raspberry Pi 5 8GB** (khuyên dùng) | ~0,3–0,6 s/ảnh (CPU, NCNN, ảnh 480) | Rẻ, dễ mua, nhiều tài liệu, chân GPIO cho màn hình/nút | Chậm hơn laptop; gợi ý trễ ~1–2 s (vẫn đủ vì cờ là trò chơi chậm) |
| Pi 5 + **AI HAT+ (Hailo‑8L 13 TOPS)** | ~20–40 ms/ảnh | Nhanh như laptop | Chuyển model sang `.hef` cần Hailo Dataflow Compiler (máy Linux x86), phức tạp |
| **Jetson Orin Nano Super Dev Kit** | ~10–20 ms/ảnh (TensorRT) | Nhanh, có CUDA, gần giống môi trường laptop | Đắt hơn nhiều, to và tốn điện hơn |

**Đề xuất:** bắt đầu với **Pi 5 8GB, chạy CPU**. Chỉ nâng cấp Hailo/Jetson nếu độ trễ thật sự khó chịu.

---

## 2. Danh sách linh kiện (bản Pi 5)

Giá tham khảo, thay đổi tuỳ nơi bán.

| # | Linh kiện | Ghi chú | Giá tham khảo |
|---|---|---|---|
| 1 | Raspberry Pi 5 – 8GB | Bộ xử lý chính | ~2,3–2,7 triệu |
| 2 | Tản nhiệt chủ động chính hãng (Active Cooler) hoặc vỏ có quạt | Bắt buộc: chạy YOLO liên tục rất nóng | ~150–300 nghìn |
| 3 | Thẻ microSD 32–64 GB (A2, Class 10) | Cài hệ điều hành | ~150–250 nghìn |
| 4 | **Camera Module 3 Wide** (CSI, góc 120°) **hoặc** camera USB nhỏ (UVC, 1080p) | Gắn trên kính. USB dễ hơn (không lo cáp dẹt), CSI nhẹ và ít trễ hơn | ~0,9–1,3 triệu / ~300–600 nghìn |
| 5 | Cáp camera cho **Pi 5** (22‑pin → 15‑pin), dài **500 mm** | Chỉ khi dùng camera CSI. Pi 5 dùng cổng 22‑pin nhỏ | ~80–150 nghìn |
| 6 | Màn hình **IPS 1.3" 240×240, driver ST7789, SPI** | Gắn trên kính. Loại 7 chân (không có CS) hoặc 8 chân đều được | ~80–150 nghìn |
| 7 | Pin dự phòng **USB‑C PD 27 W (5V/5A)** | Pi 5 cần 5V/5A; pin 5V/3A vẫn chạy nhưng giới hạn dòng USB | ~500–900 nghìn |
| 8 | Cáp USB‑C → USB‑C chịu 5A | Pin → Pi | ~100 nghìn |
| 9 | Nút nhấn nhỏ (tact switch 6×6) | Tạm dừng / lật bàn / đặt lại | vài nghìn |
| 10 | Dây silicone 26–28 AWG (nhiều màu), ~1 m mỗi màu | Nối màn hình + nút từ kính xuống mạch | ~50 nghìn |
| 11 | Header cái 2×20 hoặc dây Dupont cái–cái | Cắm vào chân GPIO, khỏi hàn lên Pi | ~30 nghìn |
| 12 | Ống co nhiệt, băng keo điện, keo nến/keo silicone, dây rút | Cách điện, cố định | ~50 nghìn |
| 13 | Gọng kính rẻ (không tròng hoặc tròng trong) | Khung gắn linh kiện | ~50–100 nghìn |
| 14 | *(tuỳ chọn)* Tấm kính/acrylic bán phản xạ 30×30 mm + thấu kính hội tụ ~+10 đến +20 diop (kính lúp nhỏ, tiêu cự 50–100 mm) | Kiểu "HUD": nhìn màn hình qua phản xạ, mắt lấy nét được | ~100–200 nghìn |
| 15 | Dụng cụ: mỏ hàn (có điều chỉnh nhiệt), thiếc 0,8 mm có nhựa thông, kìm tuốt dây, đồng hồ vạn năng, súng keo | Nếu chưa có | ~400–800 nghìn |

**Tổng (không tính dụng cụ):** khoảng **4,5–6 triệu** với Pi 5; thêm ~2 triệu nếu gắn AI HAT+.

**Về việc mắt nhìn màn hình:** mắt người không lấy nét được vật cách < ~10 cm. Màn hình gắn sát mắt sẽ bị mờ.
Hai cách thô sơ:
1. **Đơn giản nhất:** gắn màn hình ở **cạnh trên/ngoài** tầm nhìn, cách mắt ~12–15 cm (trên một thanh nhựa
   chìa ra phía trước gọng kính), liếc mắt để xem.
2. **Kiểu HUD:** màn hình đặt nằm phía trên, chiếu xuống tấm bán phản xạ nghiêng 45° trước mắt; thêm thấu kính
   hội tụ giữa màn hình và tấm phản xạ để mắt lấy nét được. Hình bị lật gương — code sẽ có tuỳ chọn lật ảnh.

---

## 3. Đấu nối và hàn

### 3.1. Sơ đồ chân (Pi 5, đánh số chân vật lý trên header 40 chân)

**Màn hình ST7789 (SPI0):**

| Chân màn hình | Nối tới Pi 5 | Chân vật lý | Ghi chú |
|---|---|---|---|
| VCC | 3V3 | **1** | Không cắm 5 V (trừ khi module ghi rõ hỗ trợ) |
| GND | GND | **6** | |
| SCL / SCK | GPIO11 (SPI0 SCLK) | **23** | |
| SDA / MOSI | GPIO10 (SPI0 MOSI) | **19** | |
| RES / RST | GPIO25 | **22** | |
| DC | GPIO24 | **18** | |
| CS | GPIO8 (SPI0 CE0) | **24** | Module 7 chân không có CS → bỏ qua |
| BLK / BL | GPIO18 hoặc 3V3 | **12** (hoặc 17) | Đèn nền; nối 3V3 = luôn sáng |

**Nút nhấn:** một chân → **GPIO17 (chân 11)**, chân kia → **GND (chân 9)**. Code bật điện trở kéo lên nội.

**Camera:**
- CSI: cắm cáp dẹt vào cổng **CAM/DISP 0** của Pi 5 (mặt tiếp xúc kim loại theo hướng dẫn in trên board),
  đầu kia vào Camera Module 3. Không cần hàn.
- USB: cắm vào cổng USB 3 (xanh). Không cần hàn.

### 3.2. Các bước hàn (dây từ kính xuống mạch)

1. **Thử trước, chưa hàn:** cắm màn hình vào Pi bằng dây Dupont trên bàn, chạy thử (mục 4.4). Chỉ hàn khi
   chắc chắn màn hình và code chạy.
2. **Chuẩn bị dây:** cắt 7–8 sợi dây silicone dài ~60 cm (đủ từ kính xuống thắt lưng; **SPI không nên dài quá
   ~60–80 cm**). Tuốt 3 mm hai đầu, xoắn lõi, **tráng thiếc** trước cả hai đầu.
3. **Phía màn hình:** nếu module có sẵn hàng chân đực, hàn dây trực tiếp vào chân (hoặc tháo chân, hàn vào lỗ).
   Mỗi mối hàn ~2–3 giây ở ~320–350 °C, mối hàn phải bóng, hình nón, không vón cục. Luồn **ống co nhiệt** vào
   dây **trước khi hàn**, hàn xong kéo lên, hơ cho co.
4. **Phía Pi:** hàn dây vào **header cái 2×20** (cắm lên Pi) thay vì hàn thẳng lên Pi — tháo ra được, không làm
   hỏng Pi. Hoặc dùng đầu Dupont cái và cố định bằng keo.
5. **Bó dây:** xoắn nhẹ mỗi dây tín hiệu (SCK, MOSI) với một dây GND để giảm nhiễu; bó cả bó bằng ống co nhiệt
   lớn hoặc băng keo; buộc giữ lực kéo (strain relief) ở hai đầu bằng dây rút để mối hàn không bị giật.
6. **Nút nhấn:** hàn 2 dây vào 2 chân chéo nhau của nút; gắn nút lên càng kính.
7. **Kiểm tra bằng đồng hồ vạn năng (BẮT BUỘC, trước khi cấp điện):**
   - Đo thông mạch từng dây (đầu Pi ↔ đầu màn hình).
   - Đo **3V3 ↔ GND không bị chạm** (điện trở không được ~0 Ω).
   - Đo các dây cạnh nhau không chạm nhau.
8. **Gắn lên kính:** màn hình, camera (gần sống mũi hoặc trên càng, hướng nhìn thẳng xuống bàn cờ) cố định bằng
   keo nến/dây rút hoặc giá in 3D. Camera phải thấy **trọn bàn cờ** khi bạn nhìn xuống bàn.

### 3.3. An toàn
- Không đặt Pi và pin trên đầu (nóng). Pi luôn có tản nhiệt/quạt.
- Rút nguồn trước khi cắm/rút dây GPIO. Không cắm màn hình 3,3 V vào chân 5 V (chân 2, 4).
- Pin lithium: không gập, không đâm thủng, không để trong túi kín khi nóng.
- Giảm độ sáng màn hình nếu nhìn lâu bị chói mắt.

---

## 4. Cài đặt, nạp code, kiểm thử

### 4.1. Chuẩn bị model trên laptop
Model hiện tại (`models/pieces.pt`, YOLO11s) chạy chậm trên CPU ARM nếu để dạng PyTorch. Xuất sang **NCNN**
(tối ưu cho CPU ARM), ảnh 480 cho nhanh hơn:

```bash
yolo export model=models/pieces.pt format=ncnn imgsz=480
```
→ tạo thư mục `models/pieces_ncnn_model/`. Nếu vẫn chậm: train bản nhỏ hơn (YOLO11n) bằng notebook Colab
(đổi `BASE_MODEL = Path('yolo11n.pt')`).

### 4.2. Cài hệ điều hành cho Pi 5
1. Tải **Raspberry Pi Imager** trên laptop → chọn *Raspberry Pi 5* → *Raspberry Pi OS (64‑bit)* (bản Lite nếu
   không cần màn hình desktop) → thẻ microSD.
2. Trong *Edit settings*: đặt hostname (vd `chesspi`), user/mật khẩu, Wi‑Fi, **bật SSH**.
3. Cắm thẻ vào Pi, cấp nguồn, chờ ~1 phút. Từ laptop: `ssh <user>@chesspi.local`.
4. Cập nhật và bật SPI:
   ```bash
   sudo apt update && sudo apt full-upgrade -y
   sudo raspi-config   # Interface Options → SPI → Enable; xong reboot
   ```

### 4.3. Cài phần mềm và đưa code lên
```bash
sudo apt install -y python3-venv python3-picamera2 python3-opencv stockfish git
git clone <repo-của-bạn> ~/ChessAssistant      # hoặc scp cả thư mục từ laptop
cd ~/ChessAssistant
python3 -m venv --system-site-packages .venv   # dùng picamera2/opencv của hệ thống
source .venv/bin/activate
pip install python-chess pyyaml ultralytics ncnn luma.lcd gpiozero
```
Chép model đã xuất: `scp -r models/pieces_ncnn_model <user>@chesspi.local:~/ChessAssistant/models/`
Stockfish: `export STOCKFISH_PATH=/usr/games/stockfish` (thêm vào `~/.bashrc`).

### 4.4. Kiểm thử từng phần (đúng thứ tự)
1. **Camera:** `rpicam-hello --list-cameras` (CSI) hoặc `ls /dev/video*` (USB); chụp thử `rpicam-still -o test.jpg`.
2. **Màn hình:** chạy script thử màn hình (sẽ có: `python -m scripts.hud_test`) → phải thấy sơ đồ bàn cờ mẫu.
3. **Engine:** `python -m scripts.check_engine` → in ra nước gợi ý.
4. **Bộ test phần mềm:** `python -m pytest -q` (các test cần GPU/camera tự bỏ qua).
5. **Tốc độ:** `python -m scripts.replay_chessred --game 0 --repeat 5` → xem ms/frame; mục tiêu < ~700 ms.
6. **Đầu‑cuối trên bàn:** Pi + camera + màn hình đặt trên bàn (chưa lên kính), bày thế khởi đầu, đi vài nước.
7. **Lên kính:** lắp xong, chạy lại bước 6 khi đeo kính.

### 4.5. Tự chạy khi bật nguồn
Tạo `/etc/systemd/system/chess-glasses.service`:
```ini
[Unit]
Description=Chess Assistant glasses
After=multi-user.target

[Service]
User=<user>
WorkingDirectory=/home/<user>/ChessAssistant
Environment=STOCKFISH_PATH=/usr/games/stockfish
ExecStart=/home/<user>/ChessAssistant/.venv/bin/python -m scripts.glasses
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable --now chess-glasses
journalctl -u chess-glasses -f     # xem log khi lỗi
```

---

## 5. Phần code cần thêm/sửa (chưa làm)

| Việc | File | Mô tả |
|---|---|---|
| Vẽ HUD | `chess_assistant/hud.py` | Vẽ ảnh 240×240: sơ đồ mini (phía người dùng ở dưới), tô ô đi/ô đích, dòng trạng thái + điểm; tuỳ chọn lật gương cho kiểu HUD phản xạ |
| Xuất ra màn hình | `chess_assistant/hud.py` | 2 backend: cửa sổ OpenCV (thử trên laptop) và ST7789 qua `luma.lcd` (trên Pi) |
| Camera Pi | `chess_assistant/camera.py` | Thêm nguồn `picamera2` (CSI); USB dùng như hiện tại |
| Nạp model NCNN | `chess_assistant/pieces.py` | Hiện chỉ nhận file `.pt`; cho phép thư mục `*_ncnn_model` |
| Chương trình chính không cửa sổ | `scripts/glasses.py` | Như `play.py` nhưng không `imshow`; xuất HUD; nút GPIO17: nhấn ngắn = tạm dừng, nhấn giữ = dò lại bàn cờ |
| Tối ưu tốc độ | `pipeline.py`, `tracker.py` | Ảnh 480, dò bàn cờ thưa hơn (10 frame/lần), bỏ qua YOLO khi bàn cờ đang "moving" |
| Thử màn hình | `scripts/hud_test.py` | Hiện sơ đồ mẫu để kiểm tra dây |

**Thứ tự nên làm:** (1) `hud.py` + chạy thử trên laptop bằng cửa sổ 240×240 → (2) mua Pi + màn hình, test trên
bàn bằng dây Dupont → (3) `glasses.py` + tối ưu tốc độ trên Pi → (4) hàn, lắp lên kính → (5) tự chạy khi bật nguồn.
