# ============================================================
#  04_feature_store.py  —  Tạo đặc trưng từ Star Schema
#  Tổng hợp 6 cấp độ + lag features + rolling statistics
#  Nghiệp vụ: mỗi cấp độ = một góc nhìn quyết định nhập hàng
# ============================================================

import pandas as pd
import numpy as np
import pyodbc
import pickle
import os
from config import CONNECTION_STRING, LAG_DAYS

OUTPUT_DIR = "feature_store"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_aggregated_data(conn, group_cols):
    """
    GROUP BY ngay trong SQL Server theo group_cols + full_date.
    Không kéo 32M dòng raw về pandas — chỉ pull kết quả đã tổng hợp.
    """
    # Map tên cột logic → biểu thức SQL trong DIM tương ứng
    col_sql = {
        'maingroup_id':   'dp.maingroup_id',
        'maingroup_name': 'dp.maingroup_name',
        'subgroup_id':    'dp.subgroup_id',
        'subgroup_name':  'dp.subgroup_name',
        'store_id':       'ds.store_id',
        'province_id':    'ds.province_id',
        'province_name':  'ds.province_name',
    }

    select_part  = ',\n            '.join(col_sql[c] for c in group_cols if c in col_sql)
    groupby_part = ', '.join(col_sql[c] for c in group_cols if c in col_sql)

    sql = f"""
        SELECT
            dd.full_date,
            dd.day_of_week,
            dd.is_weekend,
            dd.week_of_month,
            {select_part},
            SUM(f.sales_qty)          AS sales_qty,
            SUM(f.sales_revenue)      AS sales_revenue,
            SUM(f.import_qty)         AS import_qty,
            SUM(f.opening_inventory)  AS opening_inventory
        FROM FACT_DAILY_SALES f
        JOIN DIM_DATE    dd ON dd.date_key    = f.date_key
        JOIN DIM_PRODUCT dp ON dp.product_key = f.product_key
        JOIN DIM_STORE   ds ON ds.store_key   = f.store_key
        GROUP BY
            dd.full_date, dd.day_of_week, dd.is_weekend, dd.week_of_month,
            {groupby_part}
        ORDER BY dd.full_date, {groupby_part}
    """
    df = pd.read_sql(sql, conn, parse_dates=['full_date'])
    print(f"    SQL GROUP BY → {len(df):,} dòng")
    return df


def add_lag_features(df_agg, group_cols, target='sales_qty'):
    """
    Thêm đặc trưng lag và rolling cho mỗi chuỗi.
    Nghiệp vụ: doanh số hôm qua (lag_1) và cùng ngày tuần trước (lag_7)
    là tín hiệu dự báo mạnh nhất cho hàng tươi sống.
    """
    df = df_agg.sort_values(group_cols + ['full_date']).copy()

    for lag in range(1, LAG_DAYS + 1):
        df[f'lag_{lag}'] = df.groupby(group_cols)[target].shift(lag)

    # Rolling statistics (7 ngày): bắt lấy xu hướng ngắn hạn
    df['rolling_mean_7'] = (
        df.groupby(group_cols)[target]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    )
    df['rolling_std_7'] = (
        df.groupby(group_cols)[target]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).std().fillna(0))
    )
    # Hệ số biến động (dùng cho safety factor khi đề xuất nhập)
    df['cv_7'] = df['rolling_std_7'] / (df['rolling_mean_7'] + 1e-6)

    # Log transform mục tiêu (giảm ảnh hưởng outlier)
    df['log_sales_qty'] = np.log1p(df[target])

    return df.dropna(subset=['lag_7'])  # cần ít nhất 7 ngày để có lag đầy đủ


def build_level(conn, level_name, group_cols):
    """Pull dữ liệu đã GROUP BY từ SQL, tạo lag/rolling features, lưu file"""
    df_agg = get_aggregated_data(conn, group_cols)

    # Thêm đặc trưng lag và rolling
    df_feat = add_lag_features(df_agg, group_cols)

    # Encode categorical thành số
    for col in ['store_id', 'province_id', 'subgroup_id', 'maingroup_id']:
        if col in df_feat.columns:
            df_feat[f'{col}_enc'] = pd.factorize(df_feat[col])[0]

    # Lưu ra file
    path = os.path.join(OUTPUT_DIR, f"level_{level_name}.pkl")
    df_feat.to_pickle(path)
    print(f"  ✓ Cấp {level_name:25s}: {len(df_feat):>8,} dòng  →  {path}")
    return df_feat


def main():
    print("=" * 55)
    print("  FEATURE STORE — 6 CẤP ĐỘ TỔNG HỢP")
    print("  Mỗi cấp = một góc nhìn quyết định nhập hàng")
    print("=" * 55)

    conn = pyodbc.connect(CONNECTION_STRING)

    print("\n--- Tạo 6 cấp độ Feature Store ---")

    # Cấp 1: Ngành hàng × Toàn quốc
    build_level(conn, "1_category_national",
                group_cols=['maingroup_id', 'maingroup_name'])

    # Cấp 2: Ngành hàng × Tỉnh thành
    build_level(conn, "2_category_province",
                group_cols=['maingroup_id', 'maingroup_name', 'province_id', 'province_name'])

    # Cấp 3: Ngành hàng × Cửa hàng
    build_level(conn, "3_category_store",
                group_cols=['maingroup_id', 'maingroup_name', 'store_id'])

    # Cấp 4: Nhóm hàng × Toàn quốc
    build_level(conn, "4_subgroup_national",
                group_cols=['subgroup_id', 'subgroup_name'])

    # Cấp 5: Nhóm hàng × Tỉnh thành  ← thường là cấp tối ưu nhất
    build_level(conn, "5_subgroup_province",
                group_cols=['subgroup_id', 'subgroup_name', 'province_id', 'province_name'])

    # Cấp 6: Nhóm hàng × Cửa hàng
    build_level(conn, "6_subgroup_store",
                group_cols=['subgroup_id', 'subgroup_name', 'store_id'])

    conn.close()

    print(f"\n✓ 6 file feature đã lưu vào thư mục: {OUTPUT_DIR}/")
    print("  → Chạy tiếp: python 05_train_models.py")


if __name__ == "__main__":
    main()
