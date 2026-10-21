import numpy as np

from chess_assistant.camera import ReconnectingCamera

FRAME = np.zeros((4, 4, 3), np.uint8)


class FakeCap:
    def __init__(self, good_frames: int):
        self.left = good_frames
        self.released = False

    def read(self):
        if self.left <= 0:
            return False, None
        self.left -= 1
        return True, FRAME

    def release(self):
        self.released = True


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_reconnect_after_failure():
    caps = [FakeCap(2), FakeCap(5)]
    calls = []

    def factory():
        calls.append(1)
        return caps[len(calls) - 1]

    clock = Clock()
    cam = ReconnectingCamera(factory, retry_every=1.0, clock=clock)
    assert cam.read() is FRAME and cam.read() is FRAME
    assert cam.read() is None and cam.connected is False
    assert caps[0].released

    clock.t = 0.5
    assert cam.read() is None and len(calls) == 1

    clock.t = 1.0
    assert cam.read() is FRAME and len(calls) == 2 and cam.connected


def test_factory_error_counts_as_disconnected():
    clock = Clock()

    def factory():
        raise RuntimeError("no camera")

    cam = ReconnectingCamera(factory, retry_every=1.0, clock=clock)
    assert cam.connected is False
    assert cam.read() is None


def test_opencv_has_gui():
    """opencv-python-headless (kéo theo bởi albumentations) ghi đè cv2 và làm mất cv2.imshow."""
    import cv2

    gui = [line for line in cv2.getBuildInformation().splitlines() if line.strip().startswith("GUI:")]
    assert gui and "NONE" not in gui[0], gui


def test_url_open_failure_explains_droidcam():
    import pytest

    from chess_assistant.camera import open_camera

    with pytest.raises(RuntimeError, match="1 kết nối"):
        open_camera("http://127.0.0.1:9/video")
