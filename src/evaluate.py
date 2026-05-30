#  Đánh giá mô hình
#  Chỉ số: RMSE, MAPE, MAE, PRED(25), MBRE, MIBRE
#  Chỉ số nghiệp vụ: MBRE, MIBRE

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import numpy as np
import pickle

MODEL_DIR    = os.path.join(_ROOT, 'resources', 'models')
RESULTS_FILE = os.path.join(MODEL_DIR, "all_results.pkl")


def rmse(actual, predicted):
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    return np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))


def mape(actual, predicted, eps=1e-6):
    mask = actual > eps
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def mae(actual, predicted):
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    return np.mean(np.abs(actual[mask] - predicted[mask]))


def pred25(actual, predicted, eps=1e-6):
    mask = actual > eps
    if mask.sum() == 0:
        return np.nan
    return (np.abs(actual[mask] - predicted[mask]) / actual[mask] < 0.25).mean() * 100


def mbre(actual, predicted, eps=1e-6):
    mask = actual > eps
    if mask.sum() == 0:
        return np.nan
    return np.mean((predicted[mask] - actual[mask]) / actual[mask]) * 100


def mibre(actual, predicted, eps=1e-6):
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    mask  = denom > eps
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(predicted[mask] - actual[mask]) / denom[mask]) * 100


def evaluate_all(all_results):
    rows = []
    for level_name, level_results in all_results.items():
        for model_name, res in level_results.items():
            actual    = np.array(res['actual'],    dtype=float)
            predicted = np.array(res['predicted'], dtype=float)
            rows.append({
                'Cấp độ':             level_name,
                'Mô hình':            model_name,
                'RMSE':               round(rmse(actual, predicted),        4),
                'MAPE (%)':           round(mape(actual, predicted),        2),
                'MAE':                round(mae(actual, predicted),         4),
                'PRED(25) (%)':       round(pred25(actual, predicted),      2),
                'MBRE (%)':           round(mbre(actual, predicted),        2),
                'MIBRE (%)':          round(mibre(actual, predicted),       2),
                'N test':             len(actual),
            })
    return pd.DataFrame(rows)


def main():
    print("=" * 55)
    print("  ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 55)

    if not os.path.exists(RESULTS_FILE):
        print(f"Không tìm thấy {RESULTS_FILE}")
        print("  Chạy src/train.py trước!")
        return

    with open(RESULTS_FILE, 'rb') as f:
        all_results = pickle.load(f)

    df_eval = evaluate_all(all_results)
    print(df_eval.to_string(index=False))

    best_model = df_eval.groupby('Mô hình')['MAPE (%)'].mean().sort_values().index[0]
    print(f"\nMô hình tốt nhất theo MAPE trung bình: {best_model}")

    out_path = os.path.join(MODEL_DIR, "evaluation_results.csv")
    df_eval.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\nKết quả lưu tại: {out_path}")


if __name__ == "__main__":
    main()
