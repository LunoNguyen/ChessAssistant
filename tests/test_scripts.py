import os
import subprocess
import sys

import pytest

SCRIPTS = ["play", "live_view", "calibrate", "check_engine", "prepare_dataset", "train_pieces", "capture_labeled", "replay", "eval_pieces", "import_chessred", "replay_chessred", "capture_geometric", "export_colab", "import_chess_positions", "hud_test"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_prints_vietnamese_when_piped(script):
    """Trên Windows, stdout bị pipe dùng cp1252: chữ có dấu không được làm script crash."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    proc = subprocess.run([sys.executable, "-m", f"scripts.{script}", "--help"],
                          capture_output=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")[-500:]


def test_train_without_dataset_explains_next_step(tmp_path):
    proc = subprocess.run([sys.executable, "-m", "scripts.train_pieces", "--data", str(tmp_path / "missing.yaml")],
                          capture_output=True, timeout=60)
    err = proc.stderr.decode("utf-8", "replace")
    assert proc.returncode == 1
    assert "prepare_dataset" in err and "Traceback" not in err


def test_capture_geometric_missing_pgn_explains(tmp_path):
    (tmp_path / "van1.pgn.txt").write_text("1. e4 e5 *")  # lỗi hay gặp: Notepad tự thêm .txt
    proc = subprocess.run([sys.executable, "-m", "scripts.capture_geometric", "--set", "x", "--camera", "99",
                           "--pgn", str(tmp_path / "van1.pgn")], capture_output=True, timeout=60)
    err = proc.stderr.decode("utf-8", "replace")
    assert proc.returncode == 1
    assert "Traceback" not in err and ".txt" in err
