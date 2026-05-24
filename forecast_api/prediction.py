import os
import sys

_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, 'resources', 'models')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pickle
import numpy as np
import pandas as pd
import pyodbc
from flask import jsonify

from config.config import CONNECTION_STRING, LAG_DAYS

DATA_CUTOFF = pd.Timestamp('2026-03-31')


def get_conn():
    return pyodbc.connect(CONNECTION_STRING)


def build_features(series_window, target_date):
    feats = {}
    for i in range(1, LAG_DAYS + 1):
        feats[f'lag_{i}'] = float(series_window[-i]) if len(series_window) >= i else 0.0
    feats['rolling_mean_7'] = (float(np.mean(series_window[-7:]))
                               if len(series_window) >= 7
                               else float(np.mean(series_window) if len(series_window) else 0))
    feats['rolling_std_7']  = float(np.std(series_window[-7:])) if len(series_window) >= 7 else 0.0
    feats['cv_7']           = feats['rolling_std_7'] / (feats['rolling_mean_7'] + 1e-6)
    feats['day_of_week']    = target_date.weekday()
    feats['is_weekend']     = 1 if target_date.weekday() >= 5 else 0
    feats['week_of_month']  = (target_date.day - 1) // 7 + 1
    return feats


def api_predict_logic(req):
    forecast_date = req.get('forecast_date')
    province_id   = req.get('province_id')
    store_id      = req.get('store_id')
    maingroup_id  = req.get('maingroup_id')
    subgroup_id   = req.get('subgroup_id')

    try:
        conn = get_conn()

        forecast_dt  = pd.to_datetime(forecast_date)
        is_rolling   = forecast_dt > DATA_CUTOFF
        rollout_days = int((forecast_dt - DATA_CUTOFF).days) if is_rolling else 0

        query_before = ((DATA_CUTOFF + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                        if is_rolling else forecast_date)

        conditions = ["dd.full_date < ?"]
        params = [query_before]
        if store_id:
            conditions.append("ds.store_id = ?");    params.append(store_id)
        if province_id:
            conditions.append("ds.province_id = ?"); params.append(province_id)
        if subgroup_id:
            conditions.append("dp.subgroup_id = ?"); params.append(subgroup_id)
        elif maingroup_id:
            conditions.append("dp.maingroup_id = ?"); params.append(maingroup_id)

        where = " AND ".join(conditions)

        df = pd.read_sql(f"""
            SELECT TOP 300
                CAST(dd.full_date AS VARCHAR) AS date,
                dd.day_of_week, dd.is_weekend, dd.week_of_month,
                dp.subgroup_id, dp.subgroup_name,
                dp.maingroup_id, dp.maingroup_name,
                SUM(f.sales_qty)         AS sales_qty,
                SUM(f.opening_inventory) AS opening_inventory,
                SUM(f.import_qty)        AS import_qty
            FROM FACT_DAILY_SALES f
            JOIN DIM_DATE    dd ON dd.date_key    = f.date_key
            JOIN DIM_PRODUCT dp ON dp.product_key = f.product_key
            JOIN DIM_STORE   ds ON ds.store_key   = f.store_key
            WHERE {where}
            GROUP BY dd.full_date, dd.day_of_week, dd.is_weekend, dd.week_of_month,
                     dp.subgroup_id, dp.subgroup_name, dp.maingroup_id, dp.maingroup_name
            ORDER BY dd.full_date DESC
        """, conn, params=params)
        conn.close()

        if len(df) == 0:
            return jsonify({"error": "Không có dữ liệu cho bộ lọc này"}), 400

        df_daily = df.groupby('date').agg(
            sales_qty         = ('sales_qty', 'sum'),
            opening_inventory = ('opening_inventory', 'sum'),
            import_qty        = ('import_qty', 'sum'),
            day_of_week       = ('day_of_week', 'first'),
            is_weekend        = ('is_weekend', 'first'),
            week_of_month     = ('week_of_month', 'first'),
        ).reset_index().sort_values('date')

        models_available = [f for f in os.listdir(MODEL_DIR)
                            if f.startswith('xgb_') and f.endswith('.pkl')]
        if not models_available:
            return jsonify({"error": "Chưa có model. Chạy src/train.py trước."}), 404

        preferred = ['xgb_5_subgroup_province.pkl', 'xgb_4_subgroup_national.pkl',
                     'xgb_2_category_province.pkl', 'xgb_1_category_national.pkl']
        model_file = next((m for m in preferred if m in models_available), models_available[0])
        with open(os.path.join(MODEL_DIR, model_file), 'rb') as fh:
            model = pickle.load(fh)

        series = list(df_daily['sales_qty'].values)

        if not is_rolling:
            features = build_features(series, forecast_dt)
            X = np.array([[features.get(c, 0) for c in model.feature_names_in_]])
            pred_qty = float(np.expm1(model.predict(X)[0]))
        else:
            rolling_series = series.copy()
            for step in range(rollout_days):
                step_date = DATA_CUTOFF + pd.Timedelta(days=step + 1)
                feats = build_features(rolling_series, step_date)
                X     = np.array([[feats.get(c, 0) for c in model.feature_names_in_]])
                step_pred = float(np.expm1(model.predict(X)[0]))
                rolling_series.append(step_pred)
            pred_qty = rolling_series[-1]
            features = build_features(rolling_series[:-1], forecast_dt)

        opening_inv = float(df_daily['opening_inventory'].iloc[-1])
        safety      = min(0.40, 0.15 + features['cv_7'] * 0.25)
        import_need = max(0, pred_qty * (1 + safety) - opening_inv)
        if opening_inv > pred_qty * 3:
            import_need = 0

        history = df_daily.tail(7)[['date', 'sales_qty', 'import_qty']].to_dict('records')

        subgroup_details = []
        for sg_id, sg_df in df.groupby('subgroup_id'):
            sg_series = sg_df.sort_values('date').groupby('date')['sales_qty'].sum().values
            sg_inv    = float(sg_df['opening_inventory'].sum() / max(len(sg_df['date'].unique()), 1))
            sg_cv     = (float(np.std(sg_series[-7:]) / (np.mean(sg_series[-7:]) + 1e-6))
                         if len(sg_series) >= 3 else 0.15)
            sg_pred   = float(sg_series[-1]) * (1 + 0.05) if len(sg_series) > 0 else 0
            sg_safety = min(0.40, 0.15 + sg_cv * 0.25)
            sg_import = max(0, sg_pred * (1 + sg_safety) - sg_inv)
            if sg_inv > sg_pred * 3:
                sg_import = 0
            subgroup_details.append({
                'subgroup_id':       int(sg_id) if sg_id else None,
                'subgroup_name':     sg_df['subgroup_name'].iloc[0],
                'maingroup_name':    sg_df['maingroup_name'].iloc[0],
                'predicted_qty':     round(sg_pred, 2),
                'opening_inventory': round(sg_inv, 2),
                'import_suggestion': round(sg_import, 2),
                'safety_factor_pct': round(sg_safety * 100, 1),
            })

        return jsonify({
            "forecast_date":     forecast_date,
            "predicted_qty":     round(pred_qty, 2),
            "opening_inventory": round(opening_inv, 2),
            "safety_factor_pct": round(safety * 100, 1),
            "import_suggestion": round(import_need, 2),
            "model_used":        model_file,
            "is_rolling":        is_rolling,
            "rollout_days":      rollout_days,
            "history":           history,
            "details":           subgroup_details,
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
