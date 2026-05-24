# ============================================================
#  02_build_star_schema.py  —  Tạo cấu trúc Star Schema
#  Mục đích nghiệp vụ: tổ chức dữ liệu để có thể phân tích
#  theo 3 chiều đồng thời: SẢN PHẨM × CỬA HÀNG × NGÀY
# ============================================================

import pyodbc
from config import CONNECTION_STRING

# DDL cho từng bảng — thứ tự quan trọng (DIM trước, FACT sau)
DDL_STATEMENTS = [

    # ── DIM_DATE ──────────────────────────────────────────────
    # Nghiệp vụ: phân tích doanh số theo ngày trong tuần,
    # cuối tuần vs đầu tuần (phát hiện từ EDA: +25-40% cuối tuần)
    ("""
    IF OBJECT_ID('DIM_DATE', 'U') IS NULL
    CREATE TABLE DIM_DATE (
        date_key        INT          NOT NULL PRIMARY KEY,  -- YYYYMMDD
        full_date       DATE         NOT NULL,
        day_of_week     TINYINT      NOT NULL,   -- 0=Thứ 2 ... 6=CN
        day_name        NVARCHAR(10) NOT NULL,   -- Tên ngày tiếng Việt
        week_of_month   TINYINT      NOT NULL,   -- tuần thứ mấy trong tháng
        is_weekend      BIT          NOT NULL,   -- 1 = Thứ 7 hoặc CN
        is_month_start  BIT          NOT NULL,   -- 1 = ngày đầu tháng
        is_month_end    BIT          NOT NULL    -- 1 = ngày cuối tháng
    )
    """, "DIM_DATE"),

    # ── DIM_PRODUCT ───────────────────────────────────────────
    # Nghiệp vụ: drill-down từ Ngành hàng → Nhóm hàng → SKU
    # để biết nhóm nào cần nhập nhiều hơn
    ("""
    IF OBJECT_ID('DIM_PRODUCT', 'U') IS NULL
    CREATE TABLE DIM_PRODUCT (
        product_key     INT IDENTITY(1,1) PRIMARY KEY,
        product_id      NVARCHAR(50)  NOT NULL,
        product_name    NVARCHAR(255) NOT NULL,
        subgroup_id     INT,
        subgroup_name   NVARCHAR(255),
        maingroup_id    INT,
        maingroup_name  NVARCHAR(255)
    )
    """, "DIM_PRODUCT"),

    # ── DIM_STORE ─────────────────────────────────────────────
    # Nghiệp vụ: so sánh nhu cầu giữa các cửa hàng / tỉnh thành
    ("""
    IF OBJECT_ID('DIM_STORE', 'U') IS NULL
    CREATE TABLE DIM_STORE (
        store_key       INT IDENTITY(1,1) PRIMARY KEY,
        store_id        NVARCHAR(50)  NOT NULL,
        store_name      NVARCHAR(255) NOT NULL,
        province_id     INT,
        province_name   NVARCHAR(255)
    )
    """, "DIM_STORE"),

    # ── FACT_DAILY_SALES ──────────────────────────────────────
    # Nghiệp vụ: bảng trung tâm chứa TẤT CẢ chỉ số cần thiết
    # để trả lời: "hôm nay bán bao nhiêu, tồn bao nhiêu, nhập bao nhiêu?"
    ("""
    IF OBJECT_ID('FACT_DAILY_SALES', 'U') IS NULL
    CREATE TABLE FACT_DAILY_SALES (
        fact_id             BIGINT IDENTITY(1,1) PRIMARY KEY,
        date_key            INT           NOT NULL REFERENCES DIM_DATE(date_key),
        product_key         INT           NOT NULL REFERENCES DIM_PRODUCT(product_key),
        store_key           INT           NOT NULL REFERENCES DIM_STORE(store_key),

        -- Chỉ số doanh số (đã masking ±5-7%)
        sales_qty           DECIMAL(12,4),   -- sản lượng bán quy đổi
        sales_revenue       DECIMAL(16,2),   -- doanh thu

        -- Chỉ số nhập kho & tồn kho (cốt lõi cho bài toán nhập hàng)
        import_qty          DECIMAL(12,4),   -- số lượng nhập trong ngày
        opening_inventory   DECIMAL(12,4),   -- tồn kho đầu ngày
        closing_inventory   AS (opening_inventory + import_qty - sales_qty),  -- tồn cuối ngày (computed)

        -- Chỉ số giá
        cost                DECIMAL(16,2),
        selling_price       DECIMAL(16,2),
        discount_amount     DECIMAL(16,2)
    )
    """, "FACT_DAILY_SALES"),

    # ── Indexes để tăng tốc query Feature Store ───────────────
    ("""
    IF NOT EXISTS (
        SELECT 1 FROM sys.indexes
        WHERE name = 'IX_FACT_date_store_product'
        AND object_id = OBJECT_ID('FACT_DAILY_SALES')
    )
    CREATE INDEX IX_FACT_date_store_product
    ON FACT_DAILY_SALES (date_key, store_key, product_key)
    """, "INDEX_1"),

    ("""
    IF NOT EXISTS (
        SELECT 1 FROM sys.indexes
        WHERE name = 'IX_FACT_product_date'
        AND object_id = OBJECT_ID('FACT_DAILY_SALES')
    )
    CREATE INDEX IX_FACT_product_date
    ON FACT_DAILY_SALES (product_key, date_key)
    INCLUDE (sales_qty, import_qty, opening_inventory)
    """, "INDEX_2"),
]


def build_star_schema():
    print("=" * 55)
    print("  XÂY DỰNG STAR SCHEMA")
    print("  Mục tiêu: phân tích đa chiều Sản phẩm × Cửa hàng × Ngày")
    print("=" * 55)

    conn = pyodbc.connect(CONNECTION_STRING)
    conn.autocommit = True
    cursor = conn.cursor()

    for sql, name in DDL_STATEMENTS:
        try:
            cursor.execute(sql)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    # Kiểm tra kết quả
    print("\n--- Xác nhận các bảng đã tạo ---")
    cursor.execute("""
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_NAME IN ('DIM_DATE','DIM_PRODUCT','DIM_STORE','FACT_DAILY_SALES')
        ORDER BY TABLE_NAME
    """)
    for row in cursor.fetchall():
        print(f"  ✓ {row.TABLE_NAME}")

    conn.close()
    print("\n✓ Star Schema sẵn sàng!")
    print("  → Chạy tiếp: python 03_etl_staging_to_dw.py")


if __name__ == "__main__":
    build_star_schema()
