# ============================================================
#  10_update_store_province.py
#  Bước 1: Import PM_STORE_202605221619.csv → PM_STORE_MAPPING
#  Bước 2: UPDATE DIM_STORE.province_id / province_name
#  Bước 3: Báo cáo số store cập nhật / không tìm thấy mapping
# ============================================================

import os
import sys
import pandas as pd
import pyodbc
from config import CONNECTION_STRING

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

CSV_PATH = r"C:\BI\DS BÁO CÁO\PM_STORE_202605221619.csv"
MAP_TABLE = "PM_STORE_MAPPING"


# ── 1. Đọc CSV ────────────────────────────────────────────────
def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]
    for col in ["STOREID", "PROVINCEID", "DISTRICTID", "WARDID"]:
        if col not in df.columns:
            raise ValueError(f"CSV thiếu cột: {col}")
    df = df[["STOREID", "PROVINCEID", "DISTRICTID", "WARDID"]].drop_duplicates()
    print(f"  ✓ Đọc CSV: {len(df):,} dòng (sau dedup STOREID)")
    return df


# ── 2. Tạo / nạp PM_STORE_MAPPING ────────────────────────────
def import_mapping(conn, df):
    cur = conn.cursor()

    # DROP + CREATE để đảm bảo schema mới nhất
    cur.execute(f"IF OBJECT_ID('{MAP_TABLE}','U') IS NOT NULL DROP TABLE {MAP_TABLE}")
    cur.execute(f"""
        CREATE TABLE {MAP_TABLE} (
            STOREID    NVARCHAR(50)  NOT NULL,
            PROVINCEID INT,
            DISTRICTID INT,
            WARDID     INT
        )
    """)
    conn.commit()
    print(f"  ✓ Bảng {MAP_TABLE} tạo mới")

    # Bulk insert với fast_executemany
    cur.fast_executemany = True
    rows = [
        (
            str(r["STOREID"]).strip(),
            int(r["PROVINCEID"]) if pd.notna(r["PROVINCEID"]) else None,
            int(r["DISTRICTID"]) if pd.notna(r["DISTRICTID"]) else None,
            int(r["WARDID"])     if pd.notna(r["WARDID"])     else None,
        )
        for _, r in df.iterrows()
    ]
    cur.executemany(
        f"INSERT INTO {MAP_TABLE} (STOREID, PROVINCEID, DISTRICTID, WARDID) VALUES (?,?,?,?)",
        rows,
    )
    conn.commit()
    print(f"  ✓ Nạp {len(rows):,} dòng vào {MAP_TABLE}")

    # Index hỗ trợ JOIN
    cur.execute(f"""
        IF NOT EXISTS (
            SELECT 1 FROM sys.indexes
            WHERE name = 'IX_PM_STORE_STOREID'
            AND object_id = OBJECT_ID('{MAP_TABLE}')
        )
        CREATE INDEX IX_PM_STORE_STOREID ON {MAP_TABLE} (STOREID)
    """)
    conn.commit()


# ── 3. UPDATE DIM_STORE ───────────────────────────────────────
def update_dim_store(conn):
    cur = conn.cursor()

    # Kiểm tra trước: bao nhiêu store hiện chưa có province_id
    cur.execute("SELECT COUNT(*) FROM DIM_STORE WHERE province_id IS NULL")
    null_before = cur.fetchone()[0]

    cur.execute(f"""
        UPDATE ds
        SET
            ds.province_id   = gp.province_id,
            ds.province_name = gp.province_name
        FROM DIM_STORE ds
        INNER JOIN {MAP_TABLE}  pm ON pm.STOREID    = ds.store_id
        INNER JOIN gen_province gp ON gp.province_id = pm.PROVINCEID
    """)
    updated = cur.rowcount
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM DIM_STORE WHERE province_id IS NULL")
    null_after = cur.fetchone()[0]

    return updated, null_before, null_after


# ── 4. Báo cáo chi tiết ───────────────────────────────────────
def report(conn, updated, null_before, null_after):
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM DIM_STORE")
    total_store = cur.fetchone()[0]

    # Store không có trong mapping
    cur.execute(f"""
        SELECT COUNT(*)
        FROM DIM_STORE ds
        LEFT JOIN {MAP_TABLE} pm ON pm.STOREID = ds.store_id
        WHERE pm.STOREID IS NULL
    """)
    no_mapping = cur.fetchone()[0]

    # Store có mapping nhưng PROVINCEID không khớp gen_province
    cur.execute(f"""
        SELECT COUNT(*)
        FROM DIM_STORE ds
        INNER JOIN {MAP_TABLE}  pm ON pm.STOREID    = ds.store_id
        LEFT  JOIN gen_province gp ON gp.province_id = pm.PROVINCEID
        WHERE gp.province_id IS NULL
    """)
    no_province = cur.fetchone()[0]

    print("\n" + "=" * 55)
    print("  KẾT QUẢ CẬP NHẬT PROVINCE")
    print("=" * 55)
    print(f"  Tổng số store trong DIM_STORE : {total_store:>6,}")
    print(f"  Store được UPDATE province   : {updated:>6,}")
    print(f"  Store NULL trước UPDATE      : {null_before:>6,}")
    print(f"  Store NULL sau  UPDATE       : {null_after:>6,}")
    print(f"  Store không có trong mapping : {no_mapping:>6,}")
    print(f"  Store có mapping, PROVINCEID")
    print(f"    không khớp gen_province    : {no_province:>6,}")
    print("=" * 55)

    if no_mapping > 0:
        print(f"\n  [!] {no_mapping} store_id không có trong PM_STORE_MAPPING:")
        cur.execute(f"""
            SELECT TOP 20 ds.store_id, ds.store_name
            FROM DIM_STORE ds
            LEFT JOIN {MAP_TABLE} pm ON pm.STOREID = ds.store_id
            WHERE pm.STOREID IS NULL
        """)
        for row in cur.fetchall():
            print(f"      {row[0]:15s}  {row[1]}")
        if no_mapping > 20:
            print(f"      ... (còn {no_mapping - 20} store nữa)")

    if no_province > 0:
        print(f"\n  [!] {no_province} store có PROVINCEID không tồn tại trong gen_province:")
        cur.execute(f"""
            SELECT TOP 10 ds.store_id, pm.PROVINCEID
            FROM DIM_STORE ds
            INNER JOIN {MAP_TABLE}  pm ON pm.STOREID    = ds.store_id
            LEFT  JOIN gen_province gp ON gp.province_id = pm.PROVINCEID
            WHERE gp.province_id IS NULL
        """)
        for row in cur.fetchall():
            print(f"      store_id={row[0]:15s}  PROVINCEID={row[1]}")


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  IMPORT MAPPING & CẬP NHẬT PROVINCE CHO DIM_STORE")
    print("=" * 55)

    if not os.path.exists(CSV_PATH):
        print(f"✗ Không tìm thấy file: {CSV_PATH}")
        return

    df = load_csv(CSV_PATH)

    conn = pyodbc.connect(CONNECTION_STRING)
    try:
        print("\n[Bước 1] Import CSV → SQL Server ...")
        import_mapping(conn, df)

        print("\n[Bước 2] UPDATE DIM_STORE ...")
        updated, null_before, null_after = update_dim_store(conn)
        print(f"  ✓ Cập nhật {updated:,} dòng trong DIM_STORE")

        print("\n[Bước 3] Báo cáo ...")
        report(conn, updated, null_before, null_after)

    finally:
        conn.close()

    print("\n✓ Hoàn tất!")


if __name__ == "__main__":
    main()
