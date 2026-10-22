// Chess Assistant — kính ESP32-S3: chụp ảnh, chạy FOMO (int8, TFLite Micro), ra trạng thái 64 ô.
//
// Mỗi khung hình in qua Serial:
//   BOARD <64 ký tự a1..h8> <ms>     ('.' trống, 'W'/'B' quân Trắng/Đen — hoặc FEN nếu scheme 'full')
//   NOBOARD <ms>                     (không thấy đủ 3 góc bàn cờ)
// Hướng a1 chỉ là tạm (góc dưới-trái ảnh); laptop/điện thoại xác định hướng thật bằng thế khởi đầu,
// theo dõi ván và chạy Stockfish (bước sau: gửi qua WiFi thay vì Serial).

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "fomo_decode.h"
#include "fomo_model_data.h"
#include "fomo_settings.h"

static const char *TAG = "fomo";

constexpr size_t kArenaSize = 1536 * 1024;  // trong PSRAM; log in ra số byte dùng thật để chỉnh lại
constexpr int kMaxDetections = 96;

// Chân camera của Seeed XIAO ESP32S3 Sense (mạch khác: sửa theo sơ đồ của mạch đó).
static camera_config_t camera_config() {
  camera_config_t c = {};
  c.pin_pwdn = -1;
  c.pin_reset = -1;
  c.pin_xclk = 10;
  c.pin_sccb_sda = 40;
  c.pin_sccb_scl = 39;
  c.pin_d7 = 48;
  c.pin_d6 = 11;
  c.pin_d5 = 12;
  c.pin_d4 = 14;
  c.pin_d3 = 16;
  c.pin_d2 = 18;
  c.pin_d1 = 17;
  c.pin_d0 = 15;
  c.pin_vsync = 38;
  c.pin_href = 47;
  c.pin_pclk = 13;
  c.xclk_freq_hz = 20000000;
  c.ledc_timer = LEDC_TIMER_0;
  c.ledc_channel = LEDC_CHANNEL_0;
  c.pixel_format = PIXFORMAT_RGB565;
  c.frame_size = FRAMESIZE_240X240;
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;
  return c;
}

static int8_t g_lut[256];  // giá trị pixel 0..255 -> int8 đầu vào model (đã gồm chuẩn hoá [-1, 1])

static void build_lut(float scale, int zero_point) {
  for (int p = 0; p < 256; p++) {
    long q = lroundf((p / 127.5f - 1.f) / scale) + zero_point;
    g_lut[p] = (int8_t)(q < -128 ? -128 : q > 127 ? 127 : q);
  }
}

// esp32-camera trả RGB565 byte cao trước. Nếu màu bị đảo đỏ/xanh dương, đổi thứ tự 2 byte ở đây.
static inline void rgb565_at(const uint8_t *buf, int idx, float rgb[3]) {
  uint16_t p = (uint16_t)(buf[2 * idx] << 8 | buf[2 * idx + 1]);
  rgb[0] = ((p >> 11) & 0x1F) * (255.f / 31.f);
  rgb[1] = ((p >> 5) & 0x3F) * (255.f / 63.f);
  rgb[2] = (p & 0x1F) * (255.f / 31.f);
}

// Thu ảnh camera (W x H) về FOMO_INPUT_SIZE vuông bằng nội suy song tuyến, ghi thẳng int8 RGB.
static void preprocess(const camera_fb_t *fb, int8_t *dst) {
  const int W = fb->width, H = fb->height, S = FOMO_INPUT_SIZE;
  for (int y = 0; y < S; y++) {
    float sy = (y + 0.5f) * H / S - 0.5f;
    int y0 = sy < 0 ? 0 : (int)sy, y1 = y0 + 1 < H ? y0 + 1 : H - 1;
    float fy = sy - y0 < 0 ? 0 : sy - y0;
    for (int x = 0; x < S; x++) {
      float sx = (x + 0.5f) * W / S - 0.5f;
      int x0 = sx < 0 ? 0 : (int)sx, x1 = x0 + 1 < W ? x0 + 1 : W - 1;
      float fx = sx - x0 < 0 ? 0 : sx - x0;
      float a[3], b[3], c[3], d[3];
      rgb565_at(fb->buf, y0 * W + x0, a);
      rgb565_at(fb->buf, y0 * W + x1, b);
      rgb565_at(fb->buf, y1 * W + x0, c);
      rgb565_at(fb->buf, y1 * W + x1, d);
      for (int k = 0; k < 3; k++) {
        float v = (a[k] * (1 - fx) + b[k] * fx) * (1 - fy) + (c[k] * (1 - fx) + d[k] * fx) * fy;
        dst[(y * S + x) * 3 + k] = g_lut[(int)(v + 0.5f) & 0xFF];
      }
    }
  }
}

extern "C" void app_main(void) {
  camera_config_t cam = camera_config();
  if (esp_camera_init(&cam) != ESP_OK) {
    ESP_LOGE(TAG, "Không khởi động được camera — kiểm tra cáp/chân camera");
    return;
  }

  const tflite::Model *model = tflite::GetModel(g_fomo_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Model schema %lu, thư viện cần %d", (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  // Các phép toán của MobileNetV2 cắt ngắn + đầu FOMO (danh sách thật nằm trong fomo_settings.h).
  static tflite::MicroMutableOpResolver<12> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAdd();
  resolver.AddPad();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddResizeNearestNeighbor();
  resolver.AddConcatenation();

  uint8_t *arena = (uint8_t *)heap_caps_malloc(kArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (arena == nullptr) {
    ESP_LOGE(TAG, "Không cấp được %u byte PSRAM — bật PSRAM (Octal) trong menuconfig", (unsigned)kArenaSize);
    return;
  }
  static tflite::MicroInterpreter interpreter(model, resolver, arena, kArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG, "AllocateTensors lỗi: tăng kArenaSize hoặc thiếu phép toán trong resolver");
    return;
  }
  TfLiteTensor *input = interpreter.input(0);
  TfLiteTensor *output = interpreter.output(0);
  if (input->type != kTfLiteInt8 || input->dims->data[1] != FOMO_INPUT_SIZE ||
      output->dims->data[1] != FOMO_GRID || output->dims->data[3] != FOMO_NUM_CHANNELS) {
    ESP_LOGE(TAG, "Model không khớp fomo_settings.h — chép lại cả 3 file từ notebook");
    return;
  }
  ESP_LOGI(TAG, "Model %u KB, arena dùng %u KB, lưới %dx%d, %d kênh", g_fomo_model_len / 1024,
           (unsigned)(interpreter.arena_used_bytes() / 1024), FOMO_GRID, FOMO_GRID, FOMO_NUM_CHANNELS);
  build_lut(input->params.scale, input->params.zero_point);

  static float probs[FOMO_GRID * FOMO_GRID * FOMO_NUM_CHANNELS];
  static FomoDetection dets[kMaxDetections];
  const char *symbols = FOMO_SYMBOLS;
  const float out_scale = output->params.scale;
  const int out_zp = output->params.zero_point;

  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb == nullptr) {
      vTaskDelay(pdMS_TO_TICKS(50));
      continue;
    }
    int64_t t0 = esp_timer_get_time();
    preprocess(fb, input->data.int8);
    esp_camera_fb_return(fb);
    int64_t t1 = esp_timer_get_time();
    if (interpreter.Invoke() != kTfLiteOk) {
      ESP_LOGE(TAG, "Invoke lỗi");
      continue;
    }
    int64_t t2 = esp_timer_get_time();
    for (int i = 0; i < FOMO_GRID * FOMO_GRID * FOMO_NUM_CHANNELS; i++) {
      probs[i] = (output->data.int8[i] - out_zp) * out_scale;
    }
    int n = fomo_decode(probs, FOMO_GRID, FOMO_NUM_CHANNELS, FOMO_THRESHOLD, dets, kMaxDetections);
    float corners[4][2];
    int ms = (int)((esp_timer_get_time() - t0) / 1000);
    if (!fomo_board_corners(dets, n, FOMO_CORNER_CHANNEL, corners)) {
      printf("NOBOARD %d\n", ms);
      continue;
    }
    int occ[64];
    fomo_occupancy(dets, n, corners, FOMO_CORNER_CHANNEL, occ);
    char line[65];
    for (int s = 0; s < 64; s++) line[s] = symbols[occ[s]];
    line[64] = '\0';
    printf("BOARD %s %d (tiền xử lý %d ms, model %d ms)\n", line, ms,
           (int)((t1 - t0) / 1000), (int)((t2 - t1) / 1000));
  }
}
