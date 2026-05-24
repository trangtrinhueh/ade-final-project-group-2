# ============================================================
#  06_evaluate.py  —  Đánh giá mô hình
#  Chỉ số kỹ thuật: RMSE, MAPE, MAE, PRED(25), MBRE, MIBRE
#  Chỉ số nghiệp vụ: Overstock Rate, Stockout Rate
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os

MODEL_DIR    = "models"
FEATURE_DIR  = "feature_store"
RESULTS_FILE = os.path.join(MODEL_DIR, "all_results.pkl")


def rmse(actual, predicted):
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    return np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))


def mape(actual, predicted, eps=1e-6):
    """MAPE — bỏ qua các quan sát có actual ≈ 0 (tránh chia 0)"""
    mask = actual > eps
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def mae(actual, predicted):
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    return np.mean(np.abs(actual[mask] - predicted[mask]))


def pred25(actual, predicted, eps=1e-6):
    """PRED(25): % quan sát có |actual - predicted| / actual < 0.25"""
    mask = actual > eps
    if mask.sum() == 0:
        return np.nan
    within = np.abs(actual[mask] - predicted[mask]) / actual[mask] < 0.25
    return within.mean() * 100


def mbre(actual, predicted, eps=1e-6):
    """MBRE: mean((predicted - actual) / actual) — đo độ lệch có hướng"""
    mask = actual > eps
    if mask.sum() == 0:
        return np.nan
    return np.mean((predicted[mask] - actual[mask]) / actual[mask]) * 100


def mibre(actual, predicted, eps=1e-6):
    """MIBRE: mean(|predicted - actual| / ((|actual| + |predicted|) / 2)) — improved bias"""
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denom > eps
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(predicted[mask] - actual[mask]) / denom[mask]) * 100


def overstock_rate(actual, predicted):
    """
    Tỷ lệ dự báo vượt quá thực tế > 20%.
    Nghiệp vụ: dự báo quá cao → nhập thừa → hàng hư.
    """
    mask = actual > 0
    if mask.sum() == 0:
        return np.nan
    over = ((predicted[mask] - actual[mask]) / actual[mask]) > 0.20
    return over.mean() * 100


def stockout_rate(actual, predicted):
    """
    Tỷ lệ dự báo thấp hơn thực tế > 20%.
    Nghiệp vụ: dự báo quá thấp → nhập thiếu → hết hàng.
    """
    mask = actual > 0
    if mask.sum() == 0:
        return np.nan
    under = ((actual[mask] - predicted[mask]) / actual[mask]) > 0.20
    return under.mean() * 100


def evaluate_all(all_results):
    rows = []
    for level_name, level_results in all_results.items():
        for model_name, res in level_results.items():
            actual    = np.array(res['actual'],    dtype=float)
            predicted = np.array(res['predicted'], dtype=float)

            r  = rmse(actual, predicted)
            m  = mape(actual, predicted)
            ma = mae(actual, predicted)
            p  = pred25(actual, predicted)
            mb = mbre(actual, predicted)
            mi = mibre(actual, predicted)
            o  = overstock_rate(actual, predicted)
            s  = stockout_rate(actual, predicted)

            rows.append({
                'Cấp độ':             level_name,
                'Mô hình':            model_name,
                'RMSE':               round(r,  4),
                'MAPE (%)':           round(m,  2),
                'MAE':                round(ma, 4),
                'PRED(25) (%)':       round(p,  2),
                'MBRE (%)':           round(mb, 2),
                'MIBRE (%)':          round(mi, 2),
                'Overstock Rate (%)': round(o,  1),
                'Stockout Rate (%)':  round(s,  1),
                'N test':             len(actual),
            })

    return pd.DataFrame(rows)


def print_summary(df_eval):
    print("\n" + "=" * 85)
    print("  KẾT QUẢ ĐÁNH GIÁ — 6 CẤP ĐỘ × 3 MÔ HÌNH")
    print("  (Tập test: ngày 27–31 trong tháng)")
    print("=" * 85)
    print(df_eval.to_string(index=False))

    # Xếp hạng mô hình theo MAPE trung bình
    print("\n--- Xếp hạng theo MAPE trung bình ---")
    ranking = (
        df_eval.groupby('Mô hình')['MAPE (%)']
        .mean()
        .sort_values()
        .reset_index()
    )
    ranking.columns = ['Mô hình', 'MAPE trung bình (%)']
    ranking['MAPE trung bình (%)'] = ranking['MAPE trung bình (%)'].round(2)
    print(ranking.to_string(index=False))

    # Cấp độ tốt nhất (theo MAPE của mô hình tốt nhất)
    best_model = ranking.iloc[0]['Mô hình']
    print(f"\n--- Cấp độ tối ưu (dùng {best_model}) ---")
    best_level = (
        df_eval[df_eval['Mô hình'] == best_model]
        .sort_values('MAPE (%)')
        [['Cấp độ', 'MAPE (%)', 'Overstock Rate (%)', 'Stockout Rate (%)']]
        .reset_index(drop=True)
    )
    print(best_level.to_string(index=False))

    # Feature importance tổng hợp (từ XGBoost)
    print("\n--- Tầm quan trọng đặc trưng (XGBoost — trung bình 6 cấp) ---")
    fi_all = {}
    for level_name, level_results in all_results.items():
        if 'XGBoost' in level_results and 'feature_importance' in level_results['XGBoost']:
            fi = level_results['XGBoost']['feature_importance']
            for feat, val in fi.items():
                fi_all[feat] = fi_all.get(feat, []) + [val]
    fi_mean = {k: np.mean(v) for k, v in fi_all.items()}
    fi_df = pd.Series(fi_mean).sort_values(ascending=False).head(10).reset_index()
    fi_df.columns = ['Đặc trưng', 'Importance trung bình']
    fi_df['Importance trung bình'] = fi_df['Importance trung bình'].round(4)
    print(fi_df.to_string(index=False))


def main():
    print("=" * 55)
    print("  ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 55)

    if not os.path.exists(RESULTS_FILE):
        print(f"✗ Không tìm thấy {RESULTS_FILE}")
        print("  Chạy 05_train_models.py trước!")
        return

    global all_results
    with open(RESULTS_FILE, 'rb') as f:
        all_results = pickle.load(f)

    df_eval = evaluate_all(all_results)
    print_summary(df_eval)

    # Lưu bảng kết quả ra CSV (dùng điền vào đồ án)
    out_path = os.path.join(MODEL_DIR, "evaluation_results.csv")
    df_eval.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ Bảng kết quả lưu tại: {out_path}")
    print("  (Dùng file này để điền vào Bảng VI.1 trong đồ án Word)")
    print("\n  → Chạy tiếp: python 07_import_suggestion.py")


if __name__ == "__main__":
    main()
