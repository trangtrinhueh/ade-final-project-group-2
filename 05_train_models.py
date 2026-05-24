# ============================================================
#  05_train_models.py  —  Huấn luyện ARIMA / LSTM / XGBoost
#  Train ngày 1–21 | Val ngày 22–26 | Test ngày 27–31
#  Nghiệp vụ: tìm mô hình dự báo tốt nhất cho từng nhóm hàng
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from config import TRAIN_END_DAY, VAL_END_DAY, LAG_DAYS

FEATURE_DIR = "feature_store"
MODEL_DIR   = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Đặc trưng đầu vào cho XGBoost và LSTM
FEATURE_COLS = (
    [f'lag_{i}' for i in range(1, LAG_DAYS + 1)] +
    ['rolling_mean_7', 'rolling_std_7', 'cv_7',
     'day_of_week', 'is_weekend', 'week_of_month']
)

# ─── SPLIT THEO NGÀY TRONG THÁNG ─────────────────────────────
def split_by_day(df):
    """
    Chia dữ liệu theo ngày trong tháng.
    Train: ngày 1–21 | Val: ngày 22–26 | Test: ngày 27–31
    """
    day = df['full_date'].dt.day
    train = df[day <= TRAIN_END_DAY].copy()
    val   = df[(day > TRAIN_END_DAY) & (day <= VAL_END_DAY)].copy()
    test  = df[day > VAL_END_DAY].copy()
    return train, val, test


# ─── ARIMA ────────────────────────────────────────────────────
def train_arima_series(series_train, series_test):
    """
    Huấn luyện ARIMA cho một chuỗi đơn.
    Bậc d tự động xác định qua ADF test.
    """
    # Xác định d qua ADF test
    d = 0
    try:
        p_val = adfuller(series_train.dropna())[1]
        if p_val > 0.05:
            d = 1
    except Exception:
        d = 1

    # Thử các (p,q) phổ biến, chọn AIC thấp nhất
    best_aic = np.inf
    best_model = None
    for p in [0, 1, 2]:
        for q in [0, 1, 2]:
            try:
                m = ARIMA(series_train, order=(p, d, q)).fit()
                if m.aic < best_aic:
                    best_aic = m.aic
                    best_model = m
            except Exception:
                continue

    if best_model is None:
        # Fallback: naive forecast (dùng giá trị cuối train)
        preds = np.full(len(series_test), series_train.iloc[-1])
    else:
        preds = best_model.forecast(steps=len(series_test))

    return np.array(preds), best_model


def run_arima_level(df, group_cols, level_name):
    """Chạy ARIMA cho tất cả chuỗi trong một cấp độ"""
    print(f"    ARIMA  [{level_name}]...")
    results = []

    for keys, grp in df.groupby(group_cols):
        grp = grp.sort_values('full_date')
        train, val, test = split_by_day(grp)

        if len(train) < 7 or len(test) == 0:
            continue

        # Dự báo trên val để chọn tham số, sau đó test
        train_val = pd.concat([train, val])
        preds_test, _ = train_arima_series(
            train_val['log_sales_qty'],
            test['log_sales_qty']
        )

        actual = test['sales_qty'].values
        # Inverse log transform
        pred_qty = np.expm1(preds_test)

        results.append({
            'keys': keys, 'actual': actual, 'predicted': pred_qty,
            'model': 'ARIMA'
        })

    return results


# ─── XGBOOST ──────────────────────────────────────────────────
def train_xgboost_level(df, group_cols, level_name):
    """
    Huấn luyện XGBoost — một mô hình duy nhất cho tất cả chuỗi
    trong cấp độ (global model), dùng group encoding làm đặc trưng.
    Nghiệp vụ: hiệu quả hơn khi mỗi chuỗi chỉ có ~21 ngày train.
    """
    print(f"    XGBoost [{level_name}]...")

    # Encode group keys
    df_work = df.copy()
    for col in group_cols:
        if df_work[col].dtype == object or str(df_work[col].dtype) == 'category':
            le = LabelEncoder()
            df_work[f'{col}_enc'] = le.fit_transform(df_work[col].astype(str))

    enc_cols = [c for c in df_work.columns if c.endswith('_enc')]
    feat_cols = FEATURE_COLS + enc_cols

    # Lọc cột thực sự tồn tại
    feat_cols = [c for c in feat_cols if c in df_work.columns]

    train_df, val_df, test_df = split_by_day(df_work)

    X_train = train_df[feat_cols].fillna(0)
    y_train = train_df['log_sales_qty']
    X_val   = val_df[feat_cols].fillna(0)
    y_val   = val_df['log_sales_qty']
    X_test  = test_df[feat_cols].fillna(0)

    if len(X_train) == 0 or len(X_test) == 0:
        return None, None, None

    # Grid search đơn giản (3 bộ tham số)
    param_grid = [
        {'n_estimators': 500,  'max_depth': 3, 'learning_rate': 0.05},
        {'n_estimators': 1000, 'max_depth': 5, 'learning_rate': 0.01},
        {'n_estimators': 500,  'max_depth': 5, 'learning_rate': 0.1},
    ]
    best_val_rmse = np.inf
    best_model = None

    for params in param_grid:
        model = xgb.XGBRegressor(
            **params,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbosity=0
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        if rmse < best_val_rmse:
            best_val_rmse = rmse
            best_model = model

    preds_log = best_model.predict(X_test)
    preds_qty = np.expm1(preds_log)
    actual    = np.expm1(test_df['log_sales_qty'].values)

    # Feature importance
    fi = pd.Series(best_model.feature_importances_, index=feat_cols).sort_values(ascending=False)

    # Lưu model
    model_path = os.path.join(MODEL_DIR, f"xgb_{level_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    return actual, preds_qty, fi


# ─── LSTM ─────────────────────────────────────────────────────
def train_lstm_level(df, group_cols, level_name):
    """
    LSTM global model cho cấp độ.
    Với 1 tháng dữ liệu, LSTM khó học tốt → kết quả thường
    kém hơn XGBoost (điều này có giá trị so sánh cho đồ án).
    """
    print(f"    LSTM   [{level_name}]...")

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    train_df, val_df, test_df = split_by_day(df)

    # Ép sang float32 — tránh lỗi "Invalid dtype: object" với Keras 3
    # (pyodbc trả BIT columns dưới dạng bool, cần cast tường minh)
    X_train = train_df[feat_cols].fillna(0).astype(np.float32).values
    y_train = train_df['log_sales_qty'].values.astype(np.float32)
    X_val   = val_df[feat_cols].fillna(0).astype(np.float32).values
    y_val   = val_df['log_sales_qty'].values.astype(np.float32)
    X_test  = test_df[feat_cols].fillna(0).astype(np.float32).values

    if len(X_train) < 10 or len(X_test) == 0:
        return None, None

    # Reshape cho LSTM: (samples, timesteps=1, features)
    X_train_3d = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val_3d   = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    X_test_3d  = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # Thử 3 kiến trúc, chọn val loss tốt nhất
    configs = [
        {'units': [32], 'dropout': 0.1},
        {'units': [64, 32], 'dropout': 0.2},
        {'units': [32, 16], 'dropout': 0.1},
    ]
    best_val_loss = np.inf
    best_model = None

    for cfg in configs:
        model = Sequential()
        for i, units in enumerate(cfg['units']):
            return_seq = (i < len(cfg['units']) - 1)
            if i == 0:
                model.add(LSTM(units, return_sequences=return_seq,
                               input_shape=(1, X_train.shape[1])))
            else:
                model.add(LSTM(units, return_sequences=return_seq))
            model.add(Dropout(cfg['dropout']))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        es = EarlyStopping(monitor='val_loss', patience=10,
                           restore_best_weights=True, verbose=0)
        history = model.fit(
            X_train_3d, y_train,
            validation_data=(X_val_3d, y_val),
            epochs=100, batch_size=64,
            callbacks=[es], verbose=0
        )
        val_loss = min(history.history['val_loss'])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

    preds_log = best_model.predict(X_test_3d, verbose=0).flatten()
    preds_qty = np.expm1(preds_log)
    actual    = np.expm1(test_df['log_sales_qty'].values)

    return actual, preds_qty


# ─── MAIN ─────────────────────────────────────────────────────
LEVEL_CONFIGS = {
    "1_category_national":  ['maingroup_id', 'maingroup_name'],
    "2_category_province":  ['maingroup_id', 'maingroup_name', 'province_id', 'province_name'],
    "3_category_store":     ['maingroup_id', 'maingroup_name', 'store_id'],
    "4_subgroup_national":  ['subgroup_id', 'subgroup_name'],
    "5_subgroup_province":  ['subgroup_id', 'subgroup_name', 'province_id', 'province_name'],
    "6_subgroup_store":     ['subgroup_id', 'subgroup_name', 'store_id'],
}

def main():
    print("=" * 55)
    print("  HUẤN LUYỆN MÔ HÌNH — 6 CẤP ĐỘ × 3 THUẬT TOÁN")
    print("  Split: Train ngày 1-21 | Val 22-26 | Test 27-31")
    print("=" * 55)

    all_results = {}

    for level_name, group_cols in LEVEL_CONFIGS.items():
        path = os.path.join(FEATURE_DIR, f"level_{level_name}.pkl")
        if not os.path.exists(path):
            print(f"\n! Bỏ qua {level_name}: file không tồn tại (chạy 04_feature_store.py trước)")
            continue

        print(f"\n{'─'*50}")
        print(f"  Cấp độ: {level_name}")
        print(f"{'─'*50}")

        df = pd.read_pickle(path)
        print(f"  Dữ liệu: {len(df):,} dòng | {df['full_date'].min().date()} → {df['full_date'].max().date()}")

        level_results = {}

        # XGBoost (quan trọng nhất — chạy trước)
        actual_xgb, pred_xgb, fi = train_xgboost_level(df, group_cols, level_name)
        if actual_xgb is not None:
            level_results['XGBoost'] = {'actual': actual_xgb, 'predicted': pred_xgb, 'feature_importance': fi}
            print(f"      Top 5 features: {fi.head(5).index.tolist()}")

        # LSTM
        actual_lstm, pred_lstm = train_lstm_level(df, group_cols, level_name)
        if actual_lstm is not None:
            level_results['LSTM'] = {'actual': actual_lstm, 'predicted': pred_lstm}

        # ARIMA (chậm nhất vì per-series)
        arima_results = run_arima_level(df, group_cols, level_name)
        if arima_results:
            # Gộp tất cả chuỗi lại để tính metric tổng hợp
            all_actual = np.concatenate([r['actual'] for r in arima_results])
            all_pred   = np.concatenate([r['predicted'] for r in arima_results])
            level_results['ARIMA'] = {'actual': all_actual, 'predicted': all_pred}

        all_results[level_name] = level_results

    # Lưu kết quả để 06_evaluate.py đọc
    results_path = os.path.join(MODEL_DIR, "all_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\n✓ Kết quả mô hình lưu tại: {results_path}")
    print("  → Chạy tiếp: python 06_evaluate.py")


if __name__ == "__main__":
    main()
