# THESIS FORECAST — Hướng dẫn chạy

## Cấu trúc project
```
thesis_forecast/
├── config.py                  ← ĐIỀN SERVER/USER/PASS VÀO ĐÂY TRƯỚC
├── 01_check_connection.py     ← Test kết nối SQL Server
├── 02_build_star_schema.py    ← Tạo FACT + DIM tables
├── 03_etl_staging_to_dw.py    ← Nạp data vào Star Schema
├── 04_feature_store.py        ← Tạo đặc trưng 6 cấp độ
├── 05_train_models.py         ← Train ARIMA + LSTM + XGBoost
├── 06_evaluate.py             ← Tính RMSE, MAPE, Overstock, Stockout
├── 07_import_suggestion.py    ← Đề xuất lượng nhập hàng
├── requirements.txt
├── feature_store/             ← Tự tạo khi chạy 04
├── models/                    ← Tự tạo khi chạy 05
└── output/                    ← Kết quả cuối (CSV + Excel)
```

## Bước 0 — Cài thư viện (chạy 1 lần trong PyCharm Terminal)
```
pip install -r requirements.txt
```

Nếu lỗi pyodbc, cài thêm ODBC Driver:
https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server

## Bước 1 — Điền thông tin kết nối
Mở `config.py`, sửa:
```python
SERVER   = r"TÊN_SERVER_CỦA_BẠN"   # kết quả SELECT @@SERVERNAME trong SSMS
USERNAME = "sa"
PASSWORD = "mật_khẩu_của_bạn"
```

## Bước 2 — Chạy tuần tự
```
python 01_check_connection.py
python 02_build_star_schema.py
python 03_etl_staging_to_dw.py
python 04_feature_store.py
python 05_train_models.py
python 06_evaluate.py
python 07_import_suggestion.py
```

## Output quan trọng cho đồ án
| File | Dùng để |
|------|---------|
| `models/evaluation_results.csv` | Điền vào Bảng VI.1 (RMSE/MAPE) |
| `output/import_suggestion_by_store.xlsx` | Minh họa kết quả chương VI |
| `output/import_summary_by_province.csv` | Tổng hợp đề xuất theo tỉnh |

## Lỗi thường gặp
- **Login failed**: sai USERNAME/PASSWORD trong config.py
- **Cannot open server**: sai SERVER name, hoặc SQL Server chưa bật TCP/IP
- **Table not found**: chạy 02 và 03 trước khi chạy 04
- **TensorFlow lỗi**: thử `pip install tensorflow-cpu==2.15.0`
