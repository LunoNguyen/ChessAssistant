import sys

# Trên Windows, stdout bị pipe/chuyển hướng dùng cp1252 và crash khi in tiếng Việt có dấu.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
