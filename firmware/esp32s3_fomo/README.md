# Firmware ESP32-S3: nhận diện bàn cờ bằng FOMO

Chạy model FOMO int8 (train bằng `notebooks/train_fomo_colab.ipynb`) ngay trên ESP32-S3: chụp ảnh 240x240 →
thu về 160x160 → FOMO → tìm 4 góc bàn cờ + tâm quân → trạng thái 64 ô, in qua Serial.

> **Trạng thái:** khung code đã viết theo API của `esp-tflite-micro` 1.3 và `esp32-camera` 2.0 nhưng **chưa build/chạy
> trên mạch thật**. Phần giải mã (`fomo_decode.cc`) làm lại đúng `chess_assistant/fomo.py` (đã có test).

## Phần cứng
| Linh kiện | Ghi chú |
|---|---|
| **Seeed XIAO ESP32S3 Sense** | ESP32-S3, 8 MB PSRAM (Octal), camera OV2640 kèm theo. Mạch khác: sửa chân camera trong `main.cc` |
| Cáp USB-C có dây dữ liệu | Nạp code + xem Serial |

Không dùng ESP32 / ESP32-CAM đời đầu: không có lệnh vector (ESP-NN chạy chậm hơn nhiều) và thường thiếu PSRAM.

## 1. Cài ESP-IDF (một lần)
Cài **ESP-IDF v5.1 trở lên**: trên Windows dùng *ESP-IDF Tools Installer* (espressif.com → ESP-IDF → Get Started),
hoặc extension **ESP-IDF** của VS Code. Mở *ESP-IDF PowerShell/CMD* (đã có sẵn lệnh `idf.py`).

## 2. Chép model từ Colab
Sau khi chạy hết notebook, tải `MyDrive/ChessAssistant/fomo_esp32.zip`, giải nén 3 file vào `main/model/`:
```
main/model/fomo_model_data.cc
main/model/fomo_model_data.h
main/model/fomo_settings.h
```

## 3. Build và nạp
```bash
cd firmware/esp32s3_fomo
idf.py set-target esp32s3
idf.py build
idf.py -p COM5 flash monitor
```
(`COM5`: cổng của mạch trong Device Manager. Không nạp được thì giữ nút **BOOT**, bấm **RESET**, thả BOOT rồi nạp lại.)
Lần build đầu tự tải `esp-tflite-micro` và `esp32-camera` về `managed_components/`.

## 4. Đọc kết quả
Lúc khởi động, log in kích thước model và bộ nhớ arena dùng thật (chỉnh `kArenaSize` trong `main.cc` cho vừa).
Mỗi khung hình in một dòng:
```
BOARD RNBQKBNRPPPPPPPP................................pppppppprnbqkbnr 412 (tiền xử lý 35 ms, model 360 ms)
BOARD WWWWWWWWWWWWWWWW................................BBBBBBBBBBBBBBBB 405 (...)   ← scheme 'color'
NOBOARD 398
```
64 ký tự theo thứ tự a1, b1, …, h8. Hướng a1 chỉ là **tạm** (góc dưới-trái của ảnh): laptop/điện thoại xoay cho khớp
thế khởi đầu, theo dõi ván (luật cờ) và chạy Stockfish. Thời gian trên chỉ là ví dụ — số thật in trên Serial.

Mẹo: model học với bàn cờ chiếm 50–97% khung hình, nên đặt camera sao cho bàn cờ gần đầy khung.

## Sự cố thường gặp
| Hiện tượng | Cách xử lý |
|---|---|
| `Thiếu model` khi build | Chưa chép 3 file vào `main/model/` |
| `AllocateTensors lỗi` | Tăng `kArenaSize`, hoặc notebook in phép toán chưa có trong `resolver` → thêm `resolver.AddXxx()` |
| `Model không khớp fomo_settings.h` | Chép lại **cả 3 file** cùng một lần chạy notebook |
| Màu sai (đỏ ↔ xanh dương), luôn `NOBOARD` | Đổi thứ tự 2 byte trong `rgb565_at()` |
| Không cấp được PSRAM | `idf.py menuconfig` → Component config → ESP PSRAM → bật, chế độ **Octal** |

## Bước tiếp theo (chưa làm)
- Gửi dòng `BOARD` qua WiFi (UDP/HTTP) tới laptop thay vì Serial; nhận lại nước gợi ý.
- Vẽ sơ đồ bàn cờ + ô gợi ý lên màn ST7789 240x240 (giống `chess_assistant/hud.py`).
