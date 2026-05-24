# ============================================================
#  03_etl_staging_to_dw.py  —  Nạp dữ liệu vào Star Schema
#  Đọc từ staging_sales_raw → điền DIM → điền FACT
# ============================================================

import pyodbc
import pandas as pd
from datetime import datetime, timedelta
from config import CONNECTION_STRING

def load_dim_date(cursor, conn):
    """Điền DIM_DATE từ các ngày có trong staging_sales_raw"""
    print("\n[1/3] Nạp DIM_DATE...")

    # Lấy danh sách ngày từ staging
    df_dates = pd.read_sql("""
        SELECT DISTINCT
            CAST(ngay AS DATE) AS full_date
        FROM staging_sales_raw
        WHERE ngay IS NOT NULL
        ORDER BY full_date
    """, conn)

    day_map_vi = {0:'Thứ 2', 1:'Thứ 3', 2:'Thứ 4',
                  3:'Thứ 5', 4:'Thứ 6', 5:'Thứ 7', 6:'Chủ nhật'}

    inserted = 0
    for _, row in df_dates.iterrows():
        d = pd.to_datetime(row['full_date'])
        date_key    = int(d.strftime('%Y%m%d'))
        dow         = d.weekday()           # 0=Thứ 2, 6=CN
        is_weekend  = 1 if dow >= 5 else 0
        week_of_m   = (d.day - 1) // 7 + 1
        is_ms       = 1 if d.day == 1 else 0
        # ngày cuối tháng: ngày tiếp theo là tháng khác
        next_day    = d + timedelta(days=1)
        is_me       = 1 if next_day.month != d.month else 0

        cursor.execute("""
            IF NOT EXISTS (SELECT 1 FROM DIM_DATE WHERE date_key = ?)
            INSERT INTO DIM_DATE
                (date_key, full_date, day_of_week, day_name,
                 week_of_month, is_weekend, is_month_start, is_month_end)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            date_key,
            date_key, d.date(), dow, day_map_vi[dow],
            week_of_m, is_weekend, is_ms, is_me
        )
        inserted += 1

    conn.commit()
    print(f"  ✓ {inserted} ngày đã nạp vào DIM_DATE")


def load_dim_product(cursor, conn):
    """Điền DIM_PRODUCT từ staging_sales_raw JOIN pm_subgroup/pm_maingroup"""
    print("\n[2/3] Nạp DIM_PRODUCT...")

    # Lấy danh sách sản phẩm unique từ staging
    # Dùng LEFT JOIN vì pm_subgroup/pm_maingroup có thể không tồn tại
    try:
        df_prod = pd.read_sql("""
            SELECT DISTINCT
                s.ma_sanpham        AS product_id,
                s.ten_sanpham       AS product_name,
                TRY_CAST(s.ma_nhomhang AS INT) AS subgroup_id,
                s.nhomhang          AS subgroup_name,
                TRY_CAST(s.ma_nganhhang AS INT) AS maingroup_id,
                s.nganhhang         AS maingroup_name
            FROM staging_sales_raw s
            WHERE s.ma_sanpham IS NOT NULL
        """, conn)
    except Exception:
        # Fallback nếu tên cột khác
        df_prod = pd.read_sql("""
            SELECT DISTINCT
                ma_sanpham   AS product_id,
                ten_sanpham  AS product_name,
                ma_nhomhang  AS subgroup_id,
                nhomhang     AS subgroup_name,
                ma_nganhhang AS maingroup_id,
                nganhhang    AS maingroup_name
            FROM staging_sales_raw
        """, conn)

    df_prod = df_prod.drop_duplicates(subset='product_id')

    for _, row in df_prod.iterrows():
        cursor.execute("""
            IF NOT EXISTS (SELECT 1 FROM DIM_PRODUCT WHERE product_id = ?)
            INSERT INTO DIM_PRODUCT
                (product_id, product_name, subgroup_id, subgroup_name,
                 maingroup_id, maingroup_name)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            str(row['product_id']).strip(),
            str(row['product_id']).strip(),
            str(row['product_name']).strip() if pd.notna(row['product_name']) else '',
            int(row['subgroup_id']) if pd.notna(row['subgroup_id']) else None,
            str(row['subgroup_name']).strip() if pd.notna(row['subgroup_name']) else '',
            int(row['maingroup_id']) if pd.notna(row['maingroup_id']) else None,
            str(row['maingroup_name']).strip() if pd.notna(row['maingroup_name']) else '',
        )

    conn.commit()
    print(f"  ✓ {len(df_prod)} sản phẩm đã nạp vào DIM_PRODUCT")


def load_dim_store(cursor, conn):
    """Điền DIM_STORE"""
    print("\n[3/3] Nạp DIM_STORE...")

    df_store = pd.read_sql("""
        SELECT DISTINCT
            ma_sieuthi   AS store_id,
            ten_sieuthi  AS store_name,
            NULL         AS province_id,
            NULL         AS province_name
        FROM staging_sales_raw
        WHERE ma_sieuthi IS NOT NULL
    """, conn)

    # Cố gắng join gen_province nếu tồn tại
    try:
        df_store_prov = pd.read_sql("""
            SELECT DISTINCT
                s.ma_sieuthi  AS store_id,
                s.ten_sieuthi AS store_name,
                p.id          AS province_id,
                p.name        AS province_name
            FROM staging_sales_raw s
            LEFT JOIN gen_province p ON p.id = s.province_id
        """, conn)
        df_store = df_store_prov
    except Exception:
        pass  # dùng fallback không có province

    df_store = df_store.drop_duplicates(subset='store_id')

    for _, row in df_store.iterrows():
        cursor.execute("""
            IF NOT EXISTS (SELECT 1 FROM DIM_STORE WHERE store_id = ?)
            INSERT INTO DIM_STORE
                (store_id, store_name, province_id, province_name)
            VALUES (?, ?, ?, ?)
        """,
            str(row['store_id']),
            str(row['store_id']),
            str(row['store_name']) if pd.notna(row['store_name']) else '',
            int(row['province_id']) if pd.notna(row.get('province_id')) else None,
            str(row['province_name']) if pd.notna(row.get('province_name')) else None,
        )

    conn.commit()
    print(f"  ✓ {len(df_store)} cửa hàng đã nạp vào DIM_STORE")


def load_fact(cursor, conn):
    """Điền FACT_DAILY_SALES — join staging với 3 bảng DIM để lấy surrogate keys"""
    print("\n[4/4] Nạp FACT_DAILY_SALES...")
    print("  (Có thể mất vài phút với 32 triệu dòng — nạp theo batch)")

    # Kiểm tra đã có dữ liệu chưa
    cursor.execute("SELECT COUNT(*) FROM FACT_DAILY_SALES")
    existing = cursor.fetchone()[0]
    if existing > 0:
        print(f"  ! Bảng đã có {existing:,} dòng. Bỏ qua để tránh duplicate.")
        print("    Nếu muốn nạp lại: TRUNCATE TABLE FACT_DAILY_SALES trong SSMS")
        return

    # Thêm cột flag phân tích tồn âm (nếu chưa có)
    cursor.execute("""
        IF NOT EXISTS (
            SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'FACT_DAILY_SALES'
              AND COLUMN_NAME = 'is_negative_inventory'
        )
        ALTER TABLE FACT_DAILY_SALES
            ADD is_negative_inventory BIT NOT NULL DEFAULT 0
    """)
    conn.commit()

    # ── Đếm dữ liệu âm trong staging trước khi nạp ───────────────
    print("\n  Phân tích dữ liệu nguồn (staging_sales_raw)...")
    cursor.execute("""
        SELECT
            COUNT(CASE WHEN TRY_CAST(sanluong_quydoi AS DECIMAL(12,4)) < 0
                       THEN 1 END)                                     AS neg_sales,
            COUNT(CASE WHEN TRY_CAST(sl_nhap AS DECIMAL(12,4)) < 0
                       THEN 1 END)                                     AS neg_import,
            COUNT(CASE WHEN TRY_CAST(ton_dau_ngay AS DECIMAL(12,4)) BETWEEN -1000 AND -0.001
                       THEN 1 END)                                     AS valid_neg_inv,
            COUNT(CASE WHEN TRY_CAST(ton_dau_ngay AS DECIMAL(12,4)) < -1000
                       THEN 1 END)                                     AS error_inv
        FROM staging_sales_raw
        WHERE ngay IS NOT NULL
          AND ma_sanpham IS NOT NULL
          AND ma_sieuthi IS NOT NULL
    """)
    s = cursor.fetchone()
    neg_sales, neg_import, valid_neg_inv, error_inv = s[0], s[1], s[2], s[3]

    print(f"  ┌──────────────────────────────────────────────────────┐")
    print(f"  │ Phân tích dữ liệu âm trong staging                  │")
    print(f"  ├──────────────────────────────────────────────────────┤")
    print(f"  │ sales_qty  âm  → sẽ gán 0     : {neg_sales:>12,} dòng  │")
    print(f"  │ import_qty âm  → sẽ gán 0     : {neg_import:>12,} dòng  │")
    print(f"  │ inventory [-1000,0) → giữ nguyên: {valid_neg_inv:>10,} dòng  │")
    print(f"  │ inventory < -1000  → sẽ gán 0 : {error_inv:>12,} dòng  │")
    print(f"  └──────────────────────────────────────────────────────┘")

    # ── INSERT với business rules, masking tách riêng trong CTE ───
    # CTE 'masked' tính giá trị đã masking một lần để tránh RAND() chạy 2 lần
    # cho cùng 1 cột (quan trọng khi dùng m_inventory trong CASE + flag).
    insert_sql = """
    WITH masked AS (
        SELECT
            CAST(FORMAT(CAST(s.ngay AS DATE), 'yyyyMMdd') AS INT)   AS date_key,
            dp.product_key,
            ds.store_key,

            TRY_CAST(s.sanluong_quydoi AS DECIMAL(12,4))
                * (0.95 + RAND(CHECKSUM(NEWID())) * 0.10)           AS m_sales_qty,

            TRY_CAST(s.doanhthu AS DECIMAL(16,2))
                * (0.93 + RAND(CHECKSUM(NEWID())) * 0.14)           AS m_revenue,

            TRY_CAST(s.sl_nhap AS DECIMAL(12,4))
                * (0.94 + RAND(CHECKSUM(NEWID())) * 0.12)           AS m_import_qty,

            -- Masking ±10%; giữ nguyên dấu để business rules xử lý bên dưới
            TRY_CAST(s.ton_dau_ngay AS DECIMAL(12,4))
                * (0.90 + RAND(CHECKSUM(NEWID())) * 0.20)           AS m_inventory,

            TRY_CAST(s.giavon AS DECIMAL(16,2))                     AS cost,
            TRY_CAST(s.giaban AS DECIMAL(16,2))                     AS selling_price,

            TRY_CAST(s.thanhtien_giamgia AS DECIMAL(16,2))
                * (0.85 + RAND(CHECKSUM(NEWID())) * 0.30)           AS m_discount

        FROM staging_sales_raw s
        JOIN DIM_PRODUCT dp ON dp.product_id = LTRIM(RTRIM(s.ma_sanpham))
        JOIN DIM_STORE   ds ON ds.store_id   = LTRIM(RTRIM(s.ma_sieuthi))
        JOIN DIM_DATE    dd ON dd.date_key   =
             CAST(FORMAT(CAST(s.ngay AS DATE), 'yyyyMMdd') AS INT)
        WHERE s.ngay IS NOT NULL
          AND s.ma_sanpham IS NOT NULL
          AND s.ma_sieuthi IS NOT NULL
    )
    INSERT INTO FACT_DAILY_SALES
        (date_key, product_key, store_key,
         sales_qty, sales_revenue,
         import_qty, opening_inventory,
         cost, selling_price, discount_amount,
         is_negative_inventory)
    SELECT
        date_key,
        product_key,
        store_key,

        -- Rule 1: sales_qty âm → 0 (không có doanh số âm)
        CASE WHEN m_sales_qty < 0
             THEN CAST(0 AS DECIMAL(12,4))
             ELSE m_sales_qty END,

        m_revenue,

        -- Rule 3: import_qty âm → 0 (không nhập số âm)
        CASE WHEN m_import_qty < 0
             THEN CAST(0 AS DECIMAL(12,4))
             ELSE m_import_qty END,

        -- Rule 2: tồn [-1000, 0) hợp lệ → giữ nguyên; < -1000 là lỗi → gán 0
        CASE WHEN m_inventory < -1000
             THEN CAST(0 AS DECIMAL(12,4))
             ELSE m_inventory END,

        cost,
        selling_price,
        m_discount,

        -- Rule 4: flag = 1 nếu tồn âm hợp lệ (sau rule 2 vẫn còn âm)
        CAST(CASE WHEN m_inventory < 0 AND m_inventory >= -1000
                  THEN 1 ELSE 0 END AS BIT)

    FROM masked
    """

    print("  Đang thực thi INSERT ... SELECT (chờ SQL Server xử lý)...")
    try:
        cursor.execute(insert_sql)
        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM FACT_DAILY_SALES")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM FACT_DAILY_SALES WHERE is_negative_inventory = 1")
        flag_count = cursor.fetchone()[0]

        print(f"  ✓ Đã nạp {total:,} dòng vào FACT_DAILY_SALES")
        print(f"\n  ┌──────────────────────────────────────────────────────┐")
        print(f"  │ Kết quả sau khi áp business rules                   │")
        print(f"  ├──────────────────────────────────────────────────────┤")
        print(f"  │ Tổng dòng đã nạp              : {total:>12,}        │")
        print(f"  │ Tồn âm hợp lệ (flag=1)        : {flag_count:>12,} dòng  │")
        print(f"  │ Tồn âm lỗi → đã fix về 0      : {error_inv:>12,} dòng  │")
        print(f"  │ sales_qty  → đã fix về 0       : {neg_sales:>12,} dòng  │")
        print(f"  │ import_qty → đã fix về 0       : {neg_import:>12,} dòng  │")
        print(f"  └──────────────────────────────────────────────────────┘")

    except Exception as e:
        conn.rollback()
        print(f"  ✗ Lỗi khi nạp FACT: {e}")
        print("    Kiểm tra lại tên cột trong staging_sales_raw (chạy 01_check_connection.py)")


def main():
    print("=" * 55)
    print("  ETL: STAGING → STAR SCHEMA")
    print("  Mục tiêu: chuẩn bị dữ liệu cho phân tích đa chiều")
    print("=" * 55)

    conn = pyodbc.connect(CONNECTION_STRING)
    conn.autocommit = False
    cursor = conn.cursor()
    # Tăng timeout cho query lớn
    cursor.execute("SET QUERY_GOVERNOR_COST_LIMIT 0")

    load_dim_date(cursor, conn)
    load_dim_product(cursor, conn)
    load_dim_store(cursor, conn)
    load_fact(cursor, conn)

    # Tóm tắt
    print("\n--- Tóm tắt Star Schema ---")
    for tbl in ['DIM_DATE', 'DIM_PRODUCT', 'DIM_STORE', 'FACT_DAILY_SALES']:
        cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
        cnt = cursor.fetchone()[0]
        print(f"  {tbl:30s}: {cnt:>12,} dòng")

    conn.close()
    print("\n✓ ETL hoàn tất!")
    print("  → Chạy tiếp: python 04_feature_store.py")


if __name__ == "__main__":
    main()
