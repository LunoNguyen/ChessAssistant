#include "fomo_decode.h"

#include <math.h>

int fomo_decode(const float *probs, int grid, int channels, float threshold,
                FomoDetection *out, int max_out) {
  int n = 0;
  for (int i = 0; i < grid; i++) {
    for (int j = 0; j < grid; j++) {
      const float *p = probs + (i * grid + j) * channels;
      int c = 0;
      for (int k = 1; k < channels; k++) {
        if (p[k] > p[c]) c = k;
      }
      if (c == 0 || p[c] < threshold) continue;
      // Đỉnh cục bộ: không thua ô nào trong 3x3 ở cùng kênh.
      bool peak = true;
      for (int di = -1; di <= 1 && peak; di++) {
        for (int dj = -1; dj <= 1; dj++) {
          int a = i + di, b = j + dj;
          if (a < 0 || b < 0 || a >= grid || b >= grid) continue;
          if (probs[(a * grid + b) * channels + c] > p[c]) { peak = false; break; }
        }
      }
      if (!peak) continue;
      FomoDetection d = {c, (j + 0.5f) / grid, (i + 0.5f) / grid, p[c]};
      // Chèn giữ thứ tự score giảm dần; đầy thì bỏ điểm yếu nhất.
      int pos = n < max_out ? n : max_out - 1;
      if (n >= max_out && out[pos].score >= d.score) continue;
      while (pos > 0 && out[pos - 1].score < d.score) { out[pos] = out[pos - 1]; pos--; }
      out[pos] = d;
      if (n < max_out) n++;
    }
  }
  return n;
}

static float angle_at(const float p[2], const float a[2], const float b[2]) {
  float ux = a[0] - p[0], uy = a[1] - p[1], vx = b[0] - p[0], vy = b[1] - p[1];
  float c = (ux * vx + uy * vy) / (sqrtf(ux * ux + uy * uy) * sqrtf(vx * vx + vy * vy) + 1e-12f);
  return acosf(fmaxf(-1.f, fminf(1.f, c)));
}

bool fomo_board_corners(const FomoDetection *dets, int n, int corner_channel, float corners[4][2]) {
  float pts[4][2];
  int m = 0;
  for (int i = 0; i < n && m < 4; i++) {  // dets đã sắp theo score
    if (dets[i].channel == corner_channel) { pts[m][0] = dets[i].x; pts[m][1] = dets[i].y; m++; }
  }
  if (m < 3) return false;
  if (m == 3) {
    int k = 0;
    float best = -1;
    for (int t = 0; t < 3; t++) {
      float a = angle_at(pts[t], pts[(t + 1) % 3], pts[(t + 2) % 3]);
      if (a > best) { best = a; k = t; }
    }
    for (int c = 0; c < 2; c++) pts[3][c] = pts[(k + 1) % 3][c] + pts[(k + 2) % 3][c] - pts[k][c];
  }
  // Sắp vòng theo góc quanh tâm, giảm dần (y hướng xuống) rồi bắt đầu từ góc dưới-trái.
  float cx = (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4;
  float cy = (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4;
  float ang[4];
  int idx[4] = {0, 1, 2, 3};
  for (int i = 0; i < 4; i++) ang[i] = atan2f(pts[i][1] - cy, pts[i][0] - cx);
  for (int i = 1; i < 4; i++) {
    int v = idx[i], j = i;
    while (j > 0 && ang[idx[j - 1]] < ang[v]) { idx[j] = idx[j - 1]; j--; }
    idx[j] = v;
  }
  int start = 0;
  for (int i = 1; i < 4; i++) {
    if (pts[idx[i]][1] - pts[idx[i]][0] > pts[idx[start]][1] - pts[idx[start]][0]) start = i;
  }
  for (int i = 0; i < 4; i++) {
    corners[i][0] = pts[idx[(start + i) % 4]][0];
    corners[i][1] = pts[idx[(start + i) % 4]][1];
  }
  return true;
}

bool fomo_homography(const float src[4][2], const float dst[4][2], float H[9]) {
  double A[8][9];  // ma trận mở rộng [A | b]
  for (int k = 0; k < 4; k++) {
    double x = src[k][0], y = src[k][1], u = dst[k][0], v = dst[k][1];
    double r1[9] = {x, y, 1, 0, 0, 0, -u * x, -u * y, u};
    double r2[9] = {0, 0, 0, x, y, 1, -v * x, -v * y, v};
    for (int c = 0; c < 9; c++) { A[2 * k][c] = r1[c]; A[2 * k + 1][c] = r2[c]; }
  }
  for (int col = 0; col < 8; col++) {  // khử Gauss, chọn trụ theo cột
    int piv = col;
    for (int r = col + 1; r < 8; r++) if (fabs(A[r][col]) > fabs(A[piv][col])) piv = r;
    if (fabs(A[piv][col]) < 1e-12) return false;
    if (piv != col) for (int c = 0; c < 9; c++) { double t = A[col][c]; A[col][c] = A[piv][c]; A[piv][c] = t; }
    for (int r = 0; r < 8; r++) {
      if (r == col) continue;
      double f = A[r][col] / A[col][col];
      for (int c = col; c < 9; c++) A[r][c] -= f * A[col][c];
    }
  }
  for (int i = 0; i < 8; i++) H[i] = (float)(A[i][8] / A[i][i]);
  H[8] = 1.f;
  return true;
}

void fomo_occupancy(const FomoDetection *dets, int n, const float corners[4][2],
                    int corner_channel, int occ[64]) {
  static const float unit[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};  // a1, h1, h8, a8
  float score[64];
  for (int s = 0; s < 64; s++) { occ[s] = 0; score[s] = 0; }
  float H[9];
  if (!fomo_homography(corners, unit, H)) return;
  for (int i = 0; i < n; i++) {
    const FomoDetection &d = dets[i];
    if (d.channel == corner_channel) continue;
    float w = H[6] * d.x + H[7] * d.y + H[8];
    float u = (H[0] * d.x + H[1] * d.y + H[2]) / w;
    float v = (H[3] * d.x + H[4] * d.y + H[5]) / w;
    int f = (int)floorf(u * 8), r = (int)floorf(v * 8);
    if (f < 0 || f > 7 || r < 0 || r > 7) continue;
    int s = r * 8 + f;
    if (d.score > score[s]) { occ[s] = d.channel; score[s] = d.score; }
  }
}
