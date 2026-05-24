#  Train ngày 1–21 | Val ngày 22–26 | Test ngày 27–31
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from config.config import TRAIN_END_DAY, VAL_END_DAY, LAG_DAYS

FEATURE_DIR = os.path.join(_ROOT, 'resources', 'feature_store')
MODEL_DIR   = os.path.join(_ROOT, 'resources', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = (
    [f'lag_{i}' for i in range(1, LAG_DAYS + 1)] +
    ['rolling_mean_7', 'rolling_std_7', 'cv_7',
     'day_of_week', 'is_weekend', 'week_of_month']
)


def split_by_day(df):
    day = df['full_date'].dt.day
    train = df[day <= TRAIN_END_DAY].copy()
    val   = df[(day > TRAIN_END_DAY) & (day <= VAL_END_DAY)].copy()
    test  = df[day > VAL_END_DAY].copy()
    return train, val, test


def train_arima_series(series_train, series_test):
    d = 0
    try:
        if adfuller(series_train.dropna())[1] > 0.05:
            d = 1
    except Exception:
        d = 1

    best_aic, best_model = np.inf, None
    for p in [0, 1, 2]:
        for q in [0, 1, 2]:
            try:
                m = ARIMA(series_train, order=(p, d, q)).fit()
                if m.aic < best_aic:
                    best_aic, best_model = m.aic, m
            except Exception:
                continue

    if best_model is None:
        return np.full(len(series_test), series_train.iloc[-1]), None
    return np.array(best_model.forecast(steps=len(series_test))), best_model


def run_arima_level(df, group_cols, level_name):
    print(f"    ARIMA  [{level_name}]...")
    results = []
    for keys, grp in df.groupby(group_cols):
        grp = grp.sort_values('full_date')
        train, val, test = split_by_day(grp)
        if len(train) < 7 or len(test) == 0:
            continue
        train_val = pd.concat([train, val])
        preds_test, _ = train_arima_series(train_val['log_sales_qty'], test['log_sales_qty'])
        results.append({
            'keys': keys,
            'actual':    test['sales_qty'].values,
            'predicted': np.expm1(preds_test),
        })
    return results


def train_xgboost_level(df, group_cols, level_name):
    print(f"    XGBoost [{level_name}]...")
    df_work = df.copy()
    for col in group_cols:
        if df_work[col].dtype == object or str(df_work[col].dtype) == 'category':
            le = LabelEncoder()
            df_work[f'{col}_enc'] = le.fit_transform(df_work[col].astype(str))

    enc_cols  = [c for c in df_work.columns if c.endswith('_enc')]
    feat_cols = [c for c in FEATURE_COLS + enc_cols if c in df_work.columns]

    train_df, val_df, test_df = split_by_day(df_work)
    X_train = train_df[feat_cols].fillna(0)
    y_train = train_df['log_sales_qty']
    X_val   = val_df[feat_cols].fillna(0)
    y_val   = val_df['log_sales_qty']
    X_test  = test_df[feat_cols].fillna(0)

    if len(X_train) == 0 or len(X_test) == 0:
        return None, None, None

    param_grid = [
        {'n_estimators': 500,  'max_depth': 3, 'learning_rate': 0.05},
        {'n_estimators': 1000, 'max_depth': 5, 'learning_rate': 0.01},
        {'n_estimators': 500,  'max_depth': 5, 'learning_rate': 0.1},
    ]
    best_val_rmse, best_model = np.inf, None
    for params in param_grid:
        model = xgb.XGBRegressor(
            **params, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            eval_metric='rmse', early_stopping_rounds=50, verbosity=0
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
        if rmse < best_val_rmse:
            best_val_rmse, best_model = rmse, model

    preds_qty = np.expm1(best_model.predict(X_test))
    actual    = np.expm1(test_df['log_sales_qty'].values)
    fi        = pd.Series(best_model.feature_importances_, index=feat_cols).sort_values(ascending=False)

    model_path = os.path.join(MODEL_DIR, f"xgb_{level_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    return actual, preds_qty, fi


def train_lstm_level(df, group_cols, level_name):
    print(f"    LSTM   [{level_name}]...")
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    train_df, val_df, test_df = split_by_day(df)

    X_train = train_df[feat_cols].fillna(0).astype(np.float32).values
    y_train = train_df['log_sales_qty'].values.astype(np.float32)
    X_val   = val_df[feat_cols].fillna(0).astype(np.float32).values
    y_val   = val_df['log_sales_qty'].values.astype(np.float32)
    X_test  = test_df[feat_cols].fillna(0).astype(np.float32).values

    if len(X_train) < 10 or len(X_test) == 0:
        return None, None

    X_train_3d = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val_3d   = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    X_test_3d  = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    configs = [
        {'units': [32],      'dropout': 0.1},
        {'units': [64, 32],  'dropout': 0.2},
        {'units': [32, 16],  'dropout': 0.1},
    ]
    best_val_loss, best_model = np.inf, None
    for cfg in configs:
        model = Sequential()
        for i, units in enumerate(cfg['units']):
            ret_seq = (i < len(cfg['units']) - 1)
            if i == 0:
                model.add(LSTM(units, return_sequences=ret_seq, input_shape=(1, X_train.shape[1])))
            else:
                model.add(LSTM(units, return_sequences=ret_seq))
            model.add(Dropout(cfg['dropout']))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        history = model.fit(
            X_train_3d, y_train,
            validation_data=(X_val_3d, y_val),
            epochs=100, batch_size=64, callbacks=[es], verbose=0
        )
        val_loss = min(history.history['val_loss'])
        if val_loss < best_val_loss:
            best_val_loss, best_model = val_loss, model

    preds_qty = np.expm1(best_model.predict(X_test_3d, verbose=0).flatten())
    actual    = np.expm1(test_df['log_sales_qty'].values)
    return actual, preds_qty


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
    print("=" * 55)

    all_results = {}
    for level_name, group_cols in LEVEL_CONFIGS.items():
        path = os.path.join(FEATURE_DIR, f"level_{level_name}.pkl")
        if not os.path.exists(path):
            print(f"\n! Bỏ qua {level_name}: chạy src/feature_engineering.py trước")
            continue

        print(f"\n{'─'*50}\n  Cấp độ: {level_name}\n{'─'*50}")
        df = pd.read_pickle(path)
        print(f"  Dữ liệu: {len(df):,} dòng")

        level_results = {}

        actual_xgb, pred_xgb, fi = train_xgboost_level(df, group_cols, level_name)
        if actual_xgb is not None:
            level_results['XGBoost'] = {'actual': actual_xgb, 'predicted': pred_xgb, 'feature_importance': fi}

        actual_lstm, pred_lstm = train_lstm_level(df, group_cols, level_name)
        if actual_lstm is not None:
            level_results['LSTM'] = {'actual': actual_lstm, 'predicted': pred_lstm}

        arima_results = run_arima_level(df, group_cols, level_name)
        if arima_results:
            all_actual = np.concatenate([r['actual'] for r in arima_results])
            all_pred   = np.concatenate([r['predicted'] for r in arima_results])
            level_results['ARIMA'] = {'actual': all_actual, 'predicted': all_pred}

        all_results[level_name] = level_results

    results_path = os.path.join(MODEL_DIR, "all_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\nKết quả lưu tại: {results_path}")

if __name__ == "__main__":
    main()
