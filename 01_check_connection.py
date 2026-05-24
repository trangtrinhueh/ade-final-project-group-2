# ============================================================
#  01_check_connection.py  —  Kiểm tra kết nối SQL Server
#  Chạy file này ĐẦU TIÊN để xác nhận kết nối OK
# ============================================================

import pyodbc
import pandas as pd
from config import CONNECTION_STRING, DATABASE, STAGING_SALES_RAW

def check_connection():
    print("=" * 55)
    print("  KIỂM TRA KẾT NỐI SQL SERVER")
    print("=" * 55)

    # 1. Test kết nối cơ bản
    try:
        conn = pyodbc.connect(CONNECTION_STRING, timeout=10)
        print("✓ Kết nối SQL Server thành công!")
    except Exception as e:
        print(f"✗ Lỗi kết nối: {e}")
        print("\nHướng xử lý:")
        print("  - Kiểm tra SERVER trong config.py (chạy: SELECT @@SERVERNAME trong SSMS)")
        print("  - Kiểm tra USERNAME / PASSWORD")
        print("  - Bật TCP/IP trong SQL Server Configuration Manager")
        return

    # 2. Kiểm tra database
    cursor = conn.cursor()
    cursor.execute("SELECT DB_NAME() AS current_db, @@SERVERNAME AS server_name")
    row = cursor.fetchone()
    print(f"✓ Database hiện tại : {row.current_db}")
    print(f"✓ Server name       : {row.server_name}")

    # 3. Liệt kê các bảng staging hiện có
    print("\n--- Các bảng trong database ---")
    cursor.execute("""
        SELECT TABLE_NAME, 
               (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS c 
                WHERE c.TABLE_NAME = t.TABLE_NAME) AS col_count
        FROM INFORMATION_SCHEMA.TABLES t
        WHERE TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
    """)
    tables = cursor.fetchall()
    for t in tables:
        print(f"  {t.TABLE_NAME:45s}  ({t.col_count} cột)")

    # 4. Kiểm tra bảng staging_sales_raw
    print(f"\n--- Preview {STAGING_SALES_RAW} ---")
    try:
        df = pd.read_sql(
            f"SELECT TOP 5 * FROM {STAGING_SALES_RAW}",
            conn
        )
        print(f"✓ Tìm thấy bảng, shape: {df.shape}")
        print(df.to_string(index=False))

        # Thống kê nhanh
        df_stats = pd.read_sql(f"""
            SELECT 
                COUNT(*)          AS total_rows,
                COUNT(DISTINCT ma_sieuthi)  AS stores,
                COUNT(DISTINCT ma_sanpham)  AS products,
                MIN(ngay)         AS date_min,
                MAX(ngay)         AS date_max
            FROM {STAGING_SALES_RAW}
        """, conn)
        print(f"\n--- Thống kê tổng quan ---")
        print(df_stats.to_string(index=False))

    except Exception as e:
        print(f"✗ Không đọc được bảng: {e}")

    conn.close()
    print("\n✓ Đóng kết nối. Sẵn sàng chạy bước tiếp theo!")
    print("  → Chạy tiếp: python 02_build_star_schema.py")

if __name__ == "__main__":
    check_connection()
