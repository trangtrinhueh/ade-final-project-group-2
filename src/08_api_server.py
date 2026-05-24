# ============================================================
#  08_api_server.py  —  FastAPI Backend
#  Thiết kế: KHÔNG load toàn bộ data lên RAM
#  Mỗi request chỉ query SQL Server theo filter → trả kết quả
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pyodbc
import pandas as pd
import numpy as np
import pickle
import os
import sys

sys.path.append(os.path.dirname(__file__))
from config import CONNECTION_STRING, LAG_DAYS

app = FastAPI(title="Thesis Forecast API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
FEATURE_DIR = os.path.join(os.path.dirname(__file__), "feature_store")

# ── Cache nhẹ: chỉ lưu danh sách filter (không lưu data) ────
_filter_cache = {}

def get_conn():
    return pyodbc.connect(CONNECTION_STRING)

# ─── FILTER ENDPOINTS ────────────────────────────────────────

@app.get("/filters/provinces")
def get_provinces():
    """Lấy danh sách tỉnh/thành phố"""
    if "provinces" not in _filter_cache:
        conn = get_conn()
        df = pd.read_sql("""
            SELECT DISTINCT province_id, province_name
            FROM DIM_STORE
            WHERE province_name IS NOT NULL
            ORDER BY province_name
        """, conn)
        conn.close()
        _filter_cache["provinces"] = df.to_dict("records")
    return _filter_cache["provinces"]

@app.get("/filters/stores")
def get_stores(province_id: Optional[int] = None):
    """Lấy danh sách cửa hàng, lọc theo tỉnh nếu có"""
    conn = get_conn()
    if province_id:
        df = pd.read_sql("""
            SELECT store_id, store_name, province_id, province_name
            FROM DIM_STORE
            WHERE province_id = ?
            ORDER BY store_name
        """, conn, params=[province_id])
    else:
        df = pd.read_sql("""
            SELECT store_id, store_name, province_id, province_name
            FROM DIM_STORE ORDER BY store_name
        """, conn)
    conn.close()
    return df.to_dict("records")

@app.get("/filters/maingroups")
def get_maingroups():
    """Lấy danh sách ngành hàng"""
    if "maingroups" not in _filter_cache:
        conn = get_conn()
        df = pd.read_sql("""
            SELECT DISTINCT maingroup_id, maingroup_name
            FROM DIM_PRODUCT
            WHERE maingroup_name IS NOT NULL
            ORDER BY maingroup_name
        """, conn)
        conn.close()
        _filter_cache["maingroups"] = df.to_dict("records")
    return _filter_cache["maingroups"]

@app.get("/filters/subgroups")
def get_subgroups(maingroup_id: Optional[int] = None):
    """Lấy danh sách nhóm hàng, lọc theo ngành nếu có"""
    conn = get_conn()
    if maingroup_id:
        df = pd.read_sql("""
            SELECT DISTINCT subgroup_id, subgroup_name
            FROM DIM_PRODUCT
            WHERE maingroup_id = ? AND subgroup_name IS NOT NULL
            ORDER BY subgroup_name
        """, conn, params=[maingroup_id])
    else:
        df = pd.read_sql("""
            SELECT DISTINCT subgroup_id, subgroup_name
            FROM DIM_PRODUCT
            WHERE subgroup_name IS NOT NULL
            ORDER BY subgroup_name
        """, conn)
    conn.close()
    return df.to_dict("records")

@app.get("/filters/dates")
def get_available_dates():
    """Lấy danh sách ngày có dữ liệu"""
    conn = get_conn()
    df = pd.read_sql("""
        SELECT DISTINCT full_date
        FROM DIM_DATE
        ORDER BY full_date
    """, conn)
    conn.close()
    return [str(d) for d in df['full_date']]

# ─── PREDICT ENDPOINT ────────────────────────────────────────

class PredictRequest(BaseModel):
    forecast_date: str          # "2026-03-28"
    province_id:   Optional[int]   = None
    store_id:      Optional[str]   = None
    maingroup_id:  Optional[int]   = None
    subgroup_id:   Optional[int]   = None
    level:         Optional[str]   = "5_subgroup_province"  # cấp độ mặc định

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Dự báo doanh số và đề xuất nhập hàng.
    Query SQL Server theo filter → tính features → load model → predict.
    KHÔNG load toàn bộ data.
    """
    # 1. Query lịch sử 14 ngày gần nhất để tính lag
    conn = get_conn()

    # Xây dựng điều kiện WHERE linh hoạt
    conditions = ["dd.full_date < ?"]
    params = [req.forecast_date]

    if req.store_id:
        conditions.append("ds.store_id = ?")
        params.append(req.store_id)
    if req.province_id:
        conditions.append("ds.province_id = ?")
        params.append(req.province_id)
    if req.subgroup_id:
        conditions.append("dp.subgroup_id = ?")
        params.append(req.subgroup_id)
    elif req.maingroup_id:
        conditions.append("dp.maingroup_id = ?")
        params.append(req.maingroup_id)

    where_clause = " AND ".join(conditions)

    # Xác định group by theo cấp độ
    if req.store_id:
        group_by = "ds.store_id, ds.store_name, ds.province_id, ds.province_name"
        select_group = "ds.store_id, ds.store_name, ds.province_id, ds.province_name"
    elif req.province_id:
        group_by = "ds.province_id, ds.province_name"
        select_group = "ds.province_id, ds.province_name"
    else:
        group_by = "'ALL' AS region"
        select_group = "'ALL' AS region"

    if req.subgroup_id:
        group_by += ", dp.subgroup_id, dp.subgroup_name"
        select_group += ", dp.subgroup_id, dp.subgroup_name"
    elif req.maingroup_id:
        group_by += ", dp.maingroup_id, dp.maingroup_name"
        select_group += ", dp.maingroup_id, dp.maingroup_name"

    query = f"""
        SELECT TOP 500
            dd.full_date,
            dd.day_of_week,
            dd.is_weekend,
            {select_group},
            SUM(f.sales_qty)          AS sales_qty,
            SUM(f.opening_inventory)  AS opening_inventory,
            SUM(f.import_qty)         AS import_qty
        FROM FACT_DAILY_SALES f
        JOIN DIM_DATE    dd ON dd.date_key    = f.date_key
        JOIN DIM_PRODUCT dp ON dp.product_key = f.product_key
        JOIN DIM_STORE   ds ON ds.store_key   = f.store_key
        WHERE {where_clause}
        GROUP BY dd.full_date, dd.day_of_week, dd.is_weekend, {group_by}
        ORDER BY dd.full_date DESC
    """

    try:
        df = pd.read_sql(query, conn, params=params, parse_dates=['full_date'])
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")
    conn.close()

    if len(df) < LAG_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"Không đủ dữ liệu lịch sử (cần {LAG_DAYS} ngày, có {len(df)})"
        )

    # 2. Tính features từ lịch sử
    df = df.sort_values('full_date')
    last_row = df.iloc[-1]

    features = {}
    for i in range(1, LAG_DAYS + 1):
        idx = -(i)
        features[f'lag_{i}'] = float(df['sales_qty'].iloc[idx]) if len(df) >= i else 0

    features['rolling_mean_7'] = float(df['sales_qty'].tail(7).mean())
    features['rolling_std_7']  = float(df['sales_qty'].tail(7).std() or 0)
    features['cv_7'] = features['rolling_std_7'] / (features['rolling_mean_7'] + 1e-6)

    # Lấy thông tin ngày dự báo
    import datetime
    fdate = pd.to_datetime(req.forecast_date)
    features['day_of_week']   = fdate.weekday()
    features['is_weekend']    = 1 if fdate.weekday() >= 5 else 0
    features['week_of_month'] = (fdate.day - 1) // 7 + 1

    # 3. Load model phù hợp
    model_path = os.path.join(MODEL_DIR, f"xgb_{req.level}.pkl")
    if not os.path.exists(model_path):
        # Fallback: dùng model cấp quốc gia
        available = [f for f in os.listdir(MODEL_DIR) if f.startswith("xgb_") and f.endswith(".pkl")]
        if not available:
            raise HTTPException(status_code=404, detail="Chưa có model. Chạy 05_train_models.py trước.")
        model_path = os.path.join(MODEL_DIR, available[0])

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # 4. Predict
    feat_cols = [c for c in model.feature_names_in_ if c in features]
    X = np.array([[features.get(c, 0) for c in model.feature_names_in_]])

    pred_log = model.predict(X)[0]
    pred_qty = float(np.expm1(pred_log))

    # 5. Import suggestion
    opening_inv  = float(last_row['opening_inventory']) if 'opening_inventory' in last_row else 0
    safety       = min(0.40, 0.15 + features['cv_7'] * 0.25)
    import_needed = max(0, pred_qty * (1 + safety) - opening_inv)

    return {
        "forecast_date":      req.forecast_date,
        "predicted_qty":      round(pred_qty, 2),
        "opening_inventory":  round(opening_inv, 2),
        "safety_factor_pct":  round(safety * 100, 1),
        "import_suggestion":  round(import_needed, 2),
        "history_days_used":  len(df),
        "model_used":         os.path.basename(model_path),
        "filters": {
            "province_id":  req.province_id,
            "store_id":     req.store_id,
            "maingroup_id": req.maingroup_id,
            "subgroup_id":  req.subgroup_id,
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "models_available": os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else []}

if __name__ == "__main__":
    import uvicorn
    print("API server chạy tại: http://localhost:8000")
    print("Docs tại: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
