# [35.2 - Nhóm 2] Ứng dụng các mô hình dự báo nhu cầu thực phẩm tươi sống

## Giới thiệu
Pipeline dự báo nhu cầu hàng tươi sống cho chuỗi siêu thị bán lẻ,
sử dụng kiến trúc Star Schema + 3 mô hình ARIMA, LSTM, XGBoost
trên 6 cấp độ tổng hợp dữ liệu.

## Cài đặt
```
pip install -r requirements.txt
```

## Cách chạy

### Cách 1 — Có SQL Server (data thực)
1. Copy config/config_template.py → config/config.py
2. Điền SERVER, USERNAME, PASSWORD vào config.py
3. Chạy lần lượt:
```
python src/feature_engineering.py
python src/train.py
python src/evaluate.py
python manage.py
```
4. Mở browser: http://localhost:5000

### Cách 2 — Dùng data mẫu (không cần SQL Server)
1. Chạy:
```
python src/train.py --sample
python manage.py
```
2. Mở browser: http://localhost:5000

## Cấu trúc project
```
config/          - Cấu hình kết nối database
forecast_api/    - API routes và logic dự báo
resources/
  models/        - Model đã train và kết quả đánh giá
  sample_data/   - Dữ liệu mẫu nhóm Rau ăn lá (subgroup 2844)
src/             - Core modules: train, evaluate, feature engineering
templates/       - Giao diện web dashboard
```

## Lưu ý
- Data thực không được public do điều khoản bảo mật
- Kết quả trong báo cáo được chạy trên 32 triệu dòng data thực
- Data mẫu chỉ dùng để kiểm tra pipeline

## Thành viên nhóm 2
- Văn Tuấn Kiệt
- Lưu Thị Lai
- Trịnh Hoàng Tuyết Trang
