import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import pyodbc

from config.config import CONNECTION_STRING

OUTPUT_DIR = os.path.join(_ROOT, 'resources', 'sample_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUBGROUP_ID = 2844


def export(conn, name, sql, params=None):
    df = pd.read_sql(sql, conn, params=params)
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  {name}: {len(df):,} dòng  →  {path}")
    return df


def main():
    print("=" * 55)
    print("  EXPORT SAMPLE DATA — subgroup_id = 2844 (Rau ăn lá)")
    print("=" * 55)

    conn = pyodbc.connect(CONNECTION_STRING)

    export(conn, 'fact_sample.csv', f"""
        SELECT TOP 50000
            f.*,
            dd.full_date, dd.day_of_week, dd.is_weekend, dd.week_of_month,
            dp.product_id, dp.subgroup_id, dp.subgroup_name,
            dp.maingroup_id, dp.maingroup_name,
            ds.store_id, ds.store_name, ds.province_id, ds.province_name
        FROM FACT_DAILY_SALES f
        JOIN DIM_DATE    dd ON dd.date_key    = f.date_key
        JOIN DIM_PRODUCT dp ON dp.product_key = f.product_key
        JOIN DIM_STORE   ds ON ds.store_key   = f.store_key
        WHERE dp.subgroup_id = {SUBGROUP_ID}
    """)

    export(conn, 'dim_date_sample.csv', "SELECT * FROM DIM_DATE")

    export(conn, 'dim_product_sample.csv',
           f"SELECT * FROM DIM_PRODUCT WHERE subgroup_id = {SUBGROUP_ID}")

    export(conn, 'dim_store_sample.csv', "SELECT * FROM DIM_STORE")

    conn.close()

    readme = os.path.join(OUTPUT_DIR, 'README.md')
    with open(readme, 'w', encoding='utf-8') as f:
        f.write(
            "Dữ liệu mẫu nhóm hàng Rau ăn lá (subgroup_id=2844)\n"
            "Dùng để test pipeline khi không có SQL Server\n"
            "Cách dùng: import vào SQL Server hoặc đọc trực tiếp bằng pandas"
        )
    print(f"  README.md  →  {readme}")
    print("\n✓ Hoàn tất! Dữ liệu mẫu lưu tại: resources/sample_data/")


if __name__ == "__main__":
    main()
