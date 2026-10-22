// Giải mã đầu ra FOMO -> điểm -> 64 ô. Làm lại đúng các bước của chess_assistant/fomo.py
// (decode, complete_corners, order_corners, solve_homography, occupancy) để kết quả trên ESP32
// khớp với lúc đánh giá trong notebook.
#pragma once

struct FomoDetection {
  int channel;   // kênh model (1-based, 0 = nền)
  float x, y;    // tâm, chuẩn hoá [0, 1)
  float score;
};

// probs: lưới grid x grid x channels (đã giải lượng tử). Trả về số điểm (đã sắp score giảm dần).
int fomo_decode(const float *probs, int grid, int channels, float threshold,
                FomoDetection *out, int max_out);

// 4 góc theo thứ tự tạm a1, h1, h8, a8 (dưới-trái, dưới-phải, trên-phải, trên-trái của ảnh).
// Đủ 4 góc thì lấy 4 góc score cao nhất; 3 góc thì suy góc thứ 4 theo hình bình hành.
bool fomo_board_corners(const FomoDetection *dets, int n, int corner_channel, float corners[4][2]);

// H (3x3, hàng trước) đưa 4 điểm src -> dst. false nếu suy biến.
bool fomo_homography(const float src[4][2], const float dst[4][2], float H[9]);

// occ[64] theo thứ tự a1..h8 (theo hướng tạm của corners): 0 = trống, còn lại = kênh quân.
void fomo_occupancy(const FomoDetection *dets, int n, const float corners[4][2],
                    int corner_channel, int occ[64]);
