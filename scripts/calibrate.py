"""Hiệu chỉnh bàn cờ: chọn 4 góc a1, h1, h8, a8 và lưu vào config/calibration.json.

    python -m scripts.calibrate                # camera 0
    python -m scripts.calibrate --camera 1
    python -m scripts.calibrate --image anh.jpg   # thử với ảnh tĩnh

Cửa sổ camera: nhấn Space để chụp khung hình, Esc để thoát.
"""

import argparse

import cv2

from chess_assistant.calibration import Calibration, pick_corners
from chess_assistant.camera import actual_size, open_camera


def grab_frame(camera: int, width: int, height: int):
    cap = open_camera(camera, width, height)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Không đọc được khung hình từ camera")
            preview = frame.copy()
            cv2.putText(preview, "Space: chup  |  Esc: thoat", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Camera", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return None, None
            if key == 32:
                return frame, actual_size(cap)
    finally:
        cap.release()
        cv2.destroyWindow("Camera")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--image", help="Dùng ảnh tĩnh thay cho camera")
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Không đọc được ảnh {args.image}")
        size = (frame.shape[1], frame.shape[0])
    else:
        frame, size = grab_frame(args.camera, args.width, args.height)
        if frame is None:
            return

    corners = pick_corners(frame)
    if corners is None:
        print("Đã huỷ, không lưu.")
        return
    cal = Calibration(camera_index=args.camera, width=size[0], height=size[1], corners=corners)
    cal.save()
    print(f"Đã lưu hiệu chỉnh: {cal}")


if __name__ == "__main__":
    main()
