# Stockfish

Stockfish không cài được qua `pip`. Tải bản Windows tại https://stockfishchess.org/download/
(từ Stockfish 19 chỉ còn bản `x86-64-universal`, tự chọn tập lệnh AVX2/AVX-512 hợp với CPU),
giải nén và đặt file `.exe` vào thư mục này, ví dụ `engines/stockfish-windows-x86-64-universal.exe`.

Chương trình tìm Stockfish theo thứ tự:
1. Biến môi trường `STOCKFISH_PATH`
2. File `stockfish*` trong thư mục `engines/` (kể cả thư mục con)
3. Lệnh `stockfish` trong `PATH`

Kiểm tra: `python -m scripts.check_engine`
