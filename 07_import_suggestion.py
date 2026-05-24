# ============================================================
#  07_import_suggestion.py  —  Đề xuất lượng nhập hàng
#  Nghiệp vụ: dùng dự báo XGBoost tốt nhất → tính lượng nhập
#  tối ưu theo công thức: nhập = dự báo × (1 + safety) − tồn
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os
import pyodbc
from config import CONNECTION_STRING, SAFETY_FACTOR, LEAD_TIME_DAYS

MODEL_DIR   = "models"
FEATURE_DIR = "feature_store"
OUTPUT_DIR  = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEVEL_CONFIGS = {
    "5_subgroup_province":  ['subgroup_id', 'subgroup_name', 'province_id', 'province_name'],
    "6_subgroup_store":     ['subgroup_id', 'subgroup_name', 'store_id'],
    "3_category_store":     ['maingroup_id', 'maingroup_name', 'store_id'],
}


def compute_safety_factor(cv):
    """
    Safety factor động dựa trên hệ số biến động (CV).
    Nghiệp vụ: nhóm thủy hải sản biến động cao → buffer lớn hơn.
    Công thức: safety = base + CV × scale, tối đa 40%.
    """
    base  = SAFETY_FACTOR         # 0.15 mặc định
    scale = 0.25
    return np.clip(base + cv * scale, 0.10, 0.40)


def suggest_import(predicted_qty, opening_inventory, cv, lead_time=LEAD_TIME_DAYS):
    """
    Công thức đề xuất nhập hàng:
        demand_covered = predicted_qty × (1 + safety_factor) × lead_time
        import_needed  = max(0, demand_covered − opening_inventory)

    Nghiệp vụ:
    - predicted_qty: dự báo doanh số ngày mai (từ XGBoost)
    - opening_inventory: tồn kho hiện tại (từ FACT_DAILY_SALES)
    - cv: hệ số biến động (từ Feature Store) → điều chỉnh buffer
    - lead_time: số ngày cần đặt trước (mặc định 1 ngày)
    """
    safety = compute_safety_factor(np.array(cv))
    demand_covered = predicted_qty * (1 + safety) * lead_time
    import_needed  = np.maximum(0, demand_covered - np.array(opening_inventory))
    return import_needed, safety


def run_suggestion_for_level(level_name, group_cols):
    """Tạo bảng đề xuất nhập hàng cho một cấp độ"""

    feat_path  = os.path.join(FEATURE_DIR, f"level_{level_name}.pkl")
    model_path = os.path.join(MODEL_DIR,   f"xgb_{level_name}.pkl")

    if not os.path.exists(feat_path):
        print(f"  ! Bỏ qua {level_name}: feature file không tồn tại")
        return None
    if not os.path.exists(model_path):
        print(f"  ! Bỏ qua {level_name}: model file không tồn tại")
        return None

    df = pd.read_pickle(feat_path)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Lấy ngày cuối cùng (ngày 31 hoặc max) → dự báo cho ngày tiếp theo
    last_day = df['full_date'].max()
    df_last  = df[df['full_date'] == last_day].copy()

    if len(df_last) == 0:
        return None

    # Chuẩn bị features
    from sklearn.preprocessing import LabelEncoder
    LAG_DAYS = 7
    feat_cols_base = (
        [f'lag_{i}' for i in range(1, LAG_DAYS + 1)] +
        ['rolling_mean_7', 'rolling_std_7', 'cv_7',
         'day_of_week', 'is_weekend', 'week_of_month']
    )

    for col in group_cols:
        if df_last[col].dtype == object:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            df_last[f'{col}_enc'] = le.transform(df_last[col].astype(str))

    enc_cols  = [c for c in df_last.columns if c.endswith('_enc')]
    feat_cols = [c for c in feat_cols_base + enc_cols if c in df_last.columns]

    X = df_last[feat_cols].fillna(0)
    pred_log = model.predict(X)
    pred_qty = np.expm1(pred_log)

    # Tính đề xuất nhập
    opening_inv = df_last['opening_inventory'].fillna(0).values
    cv_vals     = df_last['cv_7'].fillna(0).values
    import_needed, safety = suggest_import(pred_qty, opening_inv, cv_vals)

    # Xây bảng output
    result = df_last[group_cols + ['full_date', 'sales_qty', 'opening_inventory']].copy()
    result['forecast_date']      = last_day + pd.Timedelta(days=1)
    result['predicted_qty']      = pred_qty.round(2)
    result['opening_inventory']  = opening_inv.round(2)
    result['safety_factor (%)']  = (safety * 100).round(1)
    result['import_suggestion']  = import_needed.round(2)
    result['action'] = result['import_suggestion'].apply(
        lambda x: '🟢 Không cần nhập' if x == 0 else f'🔴 Nhập {x:.1f} kg/đv'
    )

    return result


def main():
    print("=" * 60)
    print("  ĐỀ XUẤT LƯỢNG NHẬP HÀNG")
    print("  Input : dự báo XGBoost + tồn kho đầu ngày")
    print("  Output: lệnh nhập tối ưu theo nhóm hàng × cửa hàng")
    print("=" * 60)

    all_suggestions = []

    for level_name, group_cols in LEVEL_CONFIGS.items():
        print(f"\n  Xử lý cấp: {level_name}")
        result = run_suggestion_for_level(level_name, group_cols)
        if result is not None:
            result['level'] = level_name
            all_suggestions.append(result)
            print(f"  ✓ {len(result)} dòng đề xuất")

            # In mẫu
            preview_cols = [c for c in group_cols if c in result.columns]
            preview_cols += ['forecast_date', 'predicted_qty',
                             'opening_inventory', 'safety_factor (%)',
                             'import_suggestion', 'action']
            print(result[preview_cols].head(8).to_string(index=False))

    if not all_suggestions:
        print("\n✗ Không có dữ liệu đề xuất. Chạy 05_train_models.py trước.")
        return

    # Lưu kết quả chi tiết nhất (cấp Subgroup × Cửa hàng)
    detail = [s for s in all_suggestions if (s['level'] == '6_subgroup_store').any()]
    if detail:
        out = detail[0]
        csv_path  = os.path.join(OUTPUT_DIR, "import_suggestion_by_store.csv")
        xlsx_path = os.path.join(OUTPUT_DIR, "import_suggestion_by_store.xlsx")
        out.to_csv(csv_path,  index=False, encoding='utf-8-sig')
        out.to_excel(xlsx_path, index=False)
        print(f"\n✓ Bảng đề xuất chi tiết: {csv_path}")
        print(f"✓ Excel để trình bày   : {xlsx_path}")

    # Tổng hợp theo nhóm hàng (cấp 5 — tỉnh thành)
    prov = [s for s in all_suggestions if (s['level'] == '5_subgroup_province').any()]
    if prov:
        summary = (
            prov[0]
            .groupby(['subgroup_name', 'province_name'])
            .agg(
                predicted_qty_total   = ('predicted_qty',    'sum'),
                import_suggestion_total = ('import_suggestion', 'sum'),
            )
            .reset_index()
            .sort_values('import_suggestion_total', ascending=False)
        )
        summary_path = os.path.join(OUTPUT_DIR, "import_summary_by_province.csv")
        summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"✓ Tổng hợp theo tỉnh   : {summary_path}")

        print("\n--- Top 10 nhóm hàng cần nhập nhiều nhất ---")
        print(summary.head(10).to_string(index=False))

    print("\n✓ Hoàn tất! Kết quả trong thư mục output/")
    print("  Dùng file Excel để trình bày trong đồ án chương VI.")


if __name__ == "__main__":
    main()
