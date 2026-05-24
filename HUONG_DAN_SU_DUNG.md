# Hướng dẫn sử dụng — Hệ thống Dự báo Nhu cầu & Đề xuất Nhập hàng Tươi sống

## 1. Mục đích hệ thống

Hệ thống dự báo nhu cầu tiêu thụ và đề xuất lượng nhập hàng tươi sống (rau củ, thịt, hải sản) cho chuỗi siêu thị Bách Hóa Xanh. Dựa trên dữ liệu bán hàng thực tế tháng 3/2026 và mô hình XGBoost đã được huấn luyện, hệ thống trả lời câu hỏi:

> **"Ngày mai cần nhập bao nhiêu kg/đơn vị cho từng nhóm hàng tại từng cửa hàng?"**

---

## 2. Cách khởi động

```bash
# Bước 1: Đảm bảo SQL Server đang chạy (instance localhost, DB retail_forecast)
# Bước 2: Chạy dashboard
cd "C:\tech_MBA\ADE - 35.2\thesis_forecast\thesis_forecast"
python 09_dashboard.py

# Bước 3: Mở trình duyệt
# http://localhost:5000
# Hoặc từ thiết bị khác cùng mạng: http://192.168.100.107:5000
```

---

## 3. Hướng dẫn sử dụng 5 bộ lọc

| Bộ lọc | Mô tả | Ghi chú |
|---|---|---|
| **Ngày dự báo** | Ngày cần dự báo nhu cầu | Phạm vi hợp lệ: 08/03/2026 – 15/04/2026 |
| **Thành phố** | Lọc theo tỉnh/thành phố | Chọn trống = tất cả tỉnh thành |
| **Cửa hàng** | Lọc theo cửa hàng cụ thể | Danh sách cập nhật theo Thành phố đã chọn |
| **Ngành hàng** | Lọc theo ngành (Rau Củ, Thịt, Hải sản...) | Chọn trống = tất cả ngành hàng |
| **Nhóm hàng** | Lọc chi tiết hơn trong ngành | Danh sách cập nhật theo Ngành hàng đã chọn |

**Quy trình sử dụng thông thường:**
1. Chọn ngày dự báo (mặc định: 31/03/2026)
2. Chọn tỉnh thành → danh sách cửa hàng tự cập nhật
3. Chọn ngành hàng → danh sách nhóm hàng tự cập nhật
4. Nhấn **Dự báo**

---

## 4. Giải thích các chỉ số hiển thị

### 4.1 Chỉ số tổng hợp (4 ô metric trên cùng)

| Chỉ số | Ý nghĩa | Đơn vị |
|---|---|---|
| **Dự báo doanh số** (`predicted_qty`) | Sản lượng dự kiến bán được trong ngày được chọn | kg hoặc đơn vị quy đổi |
| **Tồn kho đầu ngày** (`opening_inventory`) | Lượng hàng còn trong kho tính đến đầu ngày dự báo. Giá trị âm (hợp lệ): hàng đã bán vượt tồn, giao hàng chưa về kịp | kg/đv |
| **Đề xuất nhập** (`import_suggestion`) | Lượng hàng cần đặt mua thêm = `predicted_qty × (1 + safety_factor) − opening_inventory`. Bằng 0 nếu tồn kho đã đủ | kg/đv |
| **Safety buffer** (`safety_factor`) | Hệ số dự phòng động, tính theo độ biến động lịch sử (CV). Dao động 10%–40%. Nhóm hàng biến động cao (hải sản tươi) → buffer lớn hơn | % |

### 4.2 Biểu đồ xu hướng 7 ngày

- **Cột xanh**: doanh số thực tế 7 ngày gần nhất
- **Cột cam**: lượng nhập hàng trong 7 ngày gần nhất
- Dùng để nhận biết xu hướng tăng/giảm trước khi ra quyết định nhập

### 4.3 Bảng chi tiết theo nhóm hàng

| Trạng thái | Ý nghĩa |
|---|---|
| ✓ Đủ hàng | Tồn kho hiện tại đủ bù đắp nhu cầu dự báo |
| → Cần nhập thêm | Cần nhập thêm dưới 50% lượng dự báo |
| ⚠ Cần nhập nhiều | Cần nhập trên 50% lượng dự báo — ưu tiên xử lý |

---

## 5. Dự báo ngoài phạm vi data (01/04 – 15/04/2026)

Khi chọn ngày sau 31/03/2026, hệ thống chuyển sang chế độ **Rolling Forecast**:

- Dự báo tuần tự từ 01/04 đến ngày được chọn
- Mỗi ngày dùng kết quả dự báo ngày trước làm đầu vào (`lag_1`) cho ngày tiếp theo
- Dashboard hiển thị cảnh báo màu vàng: *"Dự báo ngoài phạm vi data thực — độ chính xác giảm dần"*
- Sai số tích lũy ước tính tăng ~2% mỗi ngày dự báo thêm

**Khuyến nghị**: Sử dụng rolling forecast cho kế hoạch nhập hàng định hướng (1–2 tuần tới), không dùng cho quyết định nhập hàng chính xác từng ngày.

---

## 6. Giới hạn của hệ thống

| Giới hạn | Chi tiết |
|---|---|
| Phạm vi data | Chỉ có dữ liệu thực tháng 3/2026 (01/03 – 31/03). Dự báo trong tháng 3 chính xác nhất |
| Độ chính xác mô hình | XGBoost MAPE trung bình ~22.6% trên tập test. Cấp Ngành × Tỉnh tốt nhất (MAPE 12.6%) |
| Ngày dự báo tối thiểu | Cần ít nhất 7 ngày lịch sử → ngày sớm nhất có thể dự báo: 08/03/2026 |
| Ngày dự báo tối đa | 15/04/2026 (2 tuần sau data thực). Ngoài phạm vi này cần cập nhật data mới |
| Không hỗ trợ | Dự báo theo SKU đơn lẻ (chỉ theo nhóm hàng trở lên) |

---

## 7. Cấu trúc file liên quan

```
thesis_forecast/
├── 09_dashboard.py          # File chạy web dashboard này
├── config.py                # Cấu hình kết nối SQL Server
├── models/
│   ├── xgb_*.pkl            # 6 mô hình XGBoost đã train
│   └── evaluation_results.csv  # Kết quả đánh giá → Bảng VI.1 đồ án
├── output/
│   ├── import_suggestion_by_store.xlsx  # Lệnh nhập chi tiết (Excel)
│   └── import_summary_by_province.csv  # Tổng hợp theo tỉnh thành
└── feature_store/
    └── level_*.pkl          # Feature đã tổng hợp 6 cấp độ
```

---

## 8. Liên hệ & hỗ trợ

Hệ thống được phát triển trong khuôn khổ đồ án môn **Advanced Data Engineering (ADE)** — MBA 35.2, UEH.

Tác giả: **Trịnh Hoàng Tuyết Trang** — MSSV 525202181070
