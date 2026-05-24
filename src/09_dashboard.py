# ============================================================
#  09_dashboard.py  —  Flask Dashboard
#  5 filter: Ngày | Thành phố | Cửa hàng | Ngành hàng | Nhóm hàng
#  Query SQL Server theo filter → KHÔNG load toàn bộ data
# ============================================================

from flask import Flask, render_template_string, jsonify, request
import pyodbc
import pandas as pd
import numpy as np
import pickle
import os
import sys

sys.path.append(os.path.dirname(__file__))
from config import CONNECTION_STRING, LAG_DAYS, SAFETY_FACTOR

app = Flask(__name__)
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
DATA_CUTOFF = pd.Timestamp('2026-03-31')   # ngày cuối có data thực

def get_conn():
    return pyodbc.connect(CONNECTION_STRING)

# ─── HTML TEMPLATE ────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hệ thống Dự báo Nhập hàng Tươi sống</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #f0f4f8; color: #1a202c; }

  .header {
    background: linear-gradient(135deg, #1a365d, #2b6cb0);
    color: white; padding: 20px 32px;
    display: flex; align-items: center; gap: 16px;
  }
  .header h1 { font-size: 20px; font-weight: 600; }
  .header p  { font-size: 13px; opacity: 0.8; margin-top: 2px; }

  .container { max-width: 1200px; margin: 0 auto; padding: 24px; }

  .card {
    background: white; border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 24px; margin-bottom: 20px;
  }
  .card-title { font-size: 15px; font-weight: 600; color: #2d3748; margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px; }

  .filter-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;
  }
  .filter-group label { display: block; font-size: 12px; font-weight: 600;
    color: #4a5568; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }
  .filter-group select, .filter-group input {
    width: 100%; padding: 10px 12px; border: 1.5px solid #e2e8f0;
    border-radius: 8px; font-size: 14px; background: #f7fafc;
    transition: border-color 0.2s; outline: none;
  }
  .filter-group select:focus, .filter-group input:focus { border-color: #3182ce; background: white; }

  .btn-predict {
    background: #3182ce; color: white; border: none;
    padding: 12px 32px; border-radius: 8px; font-size: 15px; font-weight: 600;
    cursor: pointer; transition: background 0.2s; margin-top: 8px;
  }
  .btn-predict:hover { background: #2c5282; }
  .btn-predict:disabled { background: #a0aec0; cursor: not-allowed; }

  .result-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px;
    margin-bottom: 20px;
  }
  .metric-card {
    background: #f7fafc; border-radius: 10px; padding: 16px;
    border-left: 4px solid #3182ce; text-align: center;
  }
  .metric-card.green  { border-left-color: #38a169; }
  .metric-card.orange { border-left-color: #dd6b20; }
  .metric-card.purple { border-left-color: #805ad5; }
  .metric-label { font-size: 11px; color: #718096; text-transform: uppercase;
    font-weight: 600; letter-spacing: 0.05em; margin-bottom: 8px; }
  .metric-value { font-size: 28px; font-weight: 700; color: #2d3748; }
  .metric-unit  { font-size: 12px; color: #718096; margin-top: 2px; }
  .metric-context { font-size: 11px; color: #718096; margin-top: 6px; font-style: italic; }

  .result-section { display: none; }
  .result-section.visible { display: block; }

  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { background: #2b6cb0; color: white; padding: 10px 14px; text-align: left;
    font-weight: 600; font-size: 12px; text-transform: uppercase; }
  td { padding: 10px 14px; border-bottom: 1px solid #e2e8f0; }
  tr:hover td { background: #ebf8ff; }
  .tag-ok   { background: #c6f6d5; color: #276749; padding: 3px 10px;
    border-radius: 20px; font-size: 11px; font-weight: 600; }
  .tag-warn { background: #fefcbf; color: #744210; padding: 3px 10px;
    border-radius: 20px; font-size: 11px; font-weight: 600; }
  .tag-alert{ background: #fed7d7; color: #742a2a; padding: 3px 10px;
    border-radius: 20px; font-size: 11px; font-weight: 600; }
  .table-note { font-size: 12px; color: #718096; margin-top: 14px;
    padding: 10px 14px; background: #f7fafc; border-radius: 6px; line-height: 1.6; }

  .loading { text-align: center; padding: 40px; color: #718096; }
  .spinner { display: inline-block; width: 32px; height: 32px;
    border: 3px solid #e2e8f0; border-top-color: #3182ce;
    border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .error-box { background: #fff5f5; border: 1px solid #fc8181;
    border-radius: 8px; padding: 14px 18px; color: #c53030; font-size: 14px; }

  .bar-chart { display: flex; align-items: flex-end; gap: 6px;
    height: 140px; margin-top: 12px; padding: 0 4px; }
  .bar-wrap  { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px; }
  .bar       { width: 100%; background: #3182ce; border-radius: 4px 4px 0 0;
    transition: height 0.4s ease; min-height: 4px; }
  .bar.import-bar { background: #dd6b20; }
  .bar-label { font-size: 10px; color: #718096; text-align: center; }
  .chart-legend { display: flex; gap: 16px; margin-top: 8px; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #4a5568; }
  .legend-dot  { width: 10px; height: 10px; border-radius: 2px; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>Hệ thống Dự báo Nhu cầu và Đề xuất Nhập hàng Tươi sống</h1>
    <p>Chọn các tiêu chí bên dưới và bấm Dự báo để xem kết quả</p>
  </div>
</div>

<div class="container">

  <!-- FILTER CARD -->
  <div class="card">
    <div class="card-title">Bộ lọc dự báo</div>
    <div class="filter-grid">
      <div class="filter-group">
        <label>Ngày dự báo</label>
        <input type="date" id="forecast_date"
               min="2026-03-08" max="2026-04-15" />
        <div style="font-size:11px;color:#718096;margin-top:4px">
          Data thực: 08/03 – 31/03 · Dự báo ngoài: 01/04 – 15/04
        </div>
      </div>
      <div class="filter-group">
        <label>Thành phố</label>
        <select id="province_id" onchange="loadStores()">
          <option value="">-- Tất cả --</option>
        </select>
      </div>
      <div class="filter-group">
        <label>Cửa hàng</label>
        <select id="store_id">
          <option value="">-- Tất cả --</option>
        </select>
      </div>
      <div class="filter-group">
        <label>Ngành hàng</label>
        <select id="maingroup_id" onchange="loadSubgroups()">
          <option value="">-- Tất cả --</option>
        </select>
      </div>
      <div class="filter-group">
        <label>Nhóm hàng</label>
        <select id="subgroup_id">
          <option value="">-- Tất cả --</option>
        </select>
      </div>
    </div>
    <br>
    <button class="btn-predict" onclick="runForecast()">Du bao</button>
  </div>

  <!-- ROLLING FORECAST WARNING -->
  <div id="rolling_warning" style="display:none" class="card">
    <div style="background:#fffbeb;border:1px solid #f6ad55;border-radius:8px;
                padding:14px 18px;color:#7b341e;font-size:14px;line-height:1.6">
      <b>Dự báo ngoài phạm vi data thực — độ chính xác giảm dần theo số ngày</b><br>
      <span id="rolling_info" style="font-size:13px;opacity:0.85"></span>
    </div>
  </div>

  <!-- LOADING -->
  <div id="loading" style="display:none" class="card loading">
    <div class="spinner"></div>
    <p style="margin-top:12px">Đang truy vấn dữ liệu và tính toán dự báo...</p>
  </div>

  <!-- ERROR -->
  <div id="error_box" style="display:none" class="card">
    <div class="error-box" id="error_msg"></div>
  </div>

  <!-- KẾT QUẢ -->
  <div id="result_section" class="result-section">

    <!-- METRICS -->
    <div class="result-grid">
      <div class="metric-card">
        <div class="metric-label">Dự báo sản lượng bán hôm nay</div>
        <div class="metric-value" id="m_qty">—</div>
        <div class="metric-unit">kg</div>
        <div class="metric-context" id="m_qty_ctx"></div>
      </div>
      <div class="metric-card green">
        <div class="metric-label">Hàng còn trong kho</div>
        <div class="metric-value" id="m_inv">—</div>
        <div class="metric-unit">kg</div>
        <div class="metric-context" id="m_inv_ctx"></div>
      </div>
      <div class="metric-card orange">
        <div class="metric-label">Cần nhập thêm hôm nay</div>
        <div class="metric-value" id="m_import">—</div>
        <div class="metric-unit">kg</div>
        <div class="metric-context" id="m_import_ctx"></div>
      </div>
      <div class="metric-card purple">
        <div class="metric-label">Dự phòng an toàn</div>
        <div class="metric-value" id="m_safety">—</div>
        <div class="metric-unit">%</div>
        <div class="metric-context">Tu dong theo muc bien dong</div>
      </div>
    </div>

    <!-- CHART + TABLE -->
    <div class="card">
      <div class="card-title">Xu hướng 7 ngày gần nhất</div>
      <div class="bar-chart" id="bar_chart"></div>
      <div class="chart-legend">
        <div class="legend-item"><div class="legend-dot" style="background:#3182ce"></div>Doanh số thực</div>
        <div class="legend-item"><div class="legend-dot" style="background:#dd6b20"></div>Đề xuất nhập</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Chi tiết theo nhóm hàng</div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Nhóm hàng</th>
              <th>Dự báo (kg)</th>
              <th>Hàng còn trong kho</th>
              <th>Cần nhập thêm</th>
              <th>Dự phòng %</th>
              <th>Trạng thái</th>
            </tr>
          </thead>
          <tbody id="result_table"></tbody>
        </table>
      </div>
      <div class="table-note">
        San luong tinh theo don vi quy doi. De xuat nhap = 0 nghia la
        ton kho hien tai du dap ung nhu cau du bao co tinh du phong.
      </div>
    </div>

  </div>
</div>

<script>
// ── Khởi tạo filter khi load trang ──
window.onload = async () => {
  document.getElementById('forecast_date').value = '2026-03-31';
  await Promise.all([loadProvinces(), loadMaingroups()]);
};

async function loadProvinces() {
  const res = await fetch('/api/provinces');
  const data = await res.json();
  const sel = document.getElementById('province_id');
  data.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.province_id || '';
    opt.text  = p.province_name || 'N/A';
    sel.appendChild(opt);
  });
}

async function loadStores() {
  const provId = document.getElementById('province_id').value;
  const url = provId ? `/api/stores?province_id=${provId}` : '/api/stores';
  const res  = await fetch(url);
  const data = await res.json();
  const sel  = document.getElementById('store_id');
  sel.innerHTML = '<option value="">-- Tất cả --</option>';
  data.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.store_id;
    opt.text  = s.store_name;
    sel.appendChild(opt);
  });
}

async function loadMaingroups() {
  const res  = await fetch('/api/maingroups');
  const data = await res.json();
  const sel  = document.getElementById('maingroup_id');
  data.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.maingroup_id || '';
    opt.text  = m.maingroup_name || 'N/A';
    sel.appendChild(opt);
  });
}

async function loadSubgroups() {
  const mgId = document.getElementById('maingroup_id').value;
  const url  = mgId ? `/api/subgroups?maingroup_id=${mgId}` : '/api/subgroups';
  const res  = await fetch(url);
  const data = await res.json();
  const sel  = document.getElementById('subgroup_id');
  sel.innerHTML = '<option value="">-- Tất cả --</option>';
  data.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.subgroup_id || '';
    opt.text  = s.subgroup_name || 'N/A';
    sel.appendChild(opt);
  });
}

// ── Gọi dự báo ──
async function runForecast() {
  const btn = document.querySelector('.btn-predict');
  btn.disabled = true;
  btn.textContent = 'Dang xu ly...';

  document.getElementById('loading').style.display = 'block';
  document.getElementById('error_box').style.display = 'none';
  document.getElementById('result_section').classList.remove('visible');

  const payload = {
    forecast_date: document.getElementById('forecast_date').value,
    province_id:   parseInt(document.getElementById('province_id').value) || null,
    store_id:      document.getElementById('store_id').value || null,
    maingroup_id:  parseInt(document.getElementById('maingroup_id').value) || null,
    subgroup_id:   parseInt(document.getElementById('subgroup_id').value) || null,
  };

  try {
    const res  = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || 'Lỗi không xác định');
    }

    renderResults(data);
  } catch(e) {
    document.getElementById('error_msg').textContent = e.message;
    document.getElementById('error_box').style.display = 'block';
  } finally {
    document.getElementById('loading').style.display = 'none';
    btn.disabled = false;
    btn.textContent = 'Du bao';
  }
}

function renderResults(data) {
  // Rolling forecast warning
  const warn = document.getElementById('rolling_warning');
  const info = document.getElementById('rolling_info');
  if (data.is_rolling) {
    warn.style.display = 'block';
    info.textContent   = `Du bao cuon ${data.rollout_days} ngay tu data thuc `
                       + `(31/03/2026 → ${data.forecast_date}). `
                       + `Sai so tich luy tang ~${(data.rollout_days * 2).toFixed(0)}% so voi du bao 1 ngay.`;
  } else {
    warn.style.display = 'none';
  }

  // Metrics
  const qty = data.predicted_qty || 0;
  const inv = data.opening_inventory || 0;
  const imp = data.import_suggestion || 0;

  document.getElementById('m_qty').textContent    = qty.toLocaleString('vi-VN');
  document.getElementById('m_inv').textContent    = inv.toLocaleString('vi-VN');
  document.getElementById('m_import').textContent = imp.toLocaleString('vi-VN');
  document.getElementById('m_safety').textContent = data.safety_factor_pct || '—';

  // Context lines
  document.getElementById('m_qty_ctx').textContent = `Du bao cho ngay ${data.forecast_date}`;

  if (inv > 0 && qty > 0) {
    const days = Math.round(inv / qty);
    document.getElementById('m_inv_ctx').textContent = `Du dung khoang ${days} ngay`;
  } else {
    document.getElementById('m_inv_ctx').textContent = 'Ton kho am - can nhap ngay';
  }

  document.getElementById('m_import_ctx').textContent = imp === 0 ? 'Khong can nhap hom nay' : '';

  // Bar chart (lịch sử 7 ngày)
  const chart = document.getElementById('bar_chart');
  chart.innerHTML = '';
  if (data.history && data.history.length > 0) {
    const maxVal = Math.max(...data.history.map(h => Math.max(h.sales_qty, h.import_qty || 0)), 1);
    data.history.forEach(h => {
      const salesH  = Math.round((h.sales_qty / maxVal) * 120);
      const importH = Math.round(((h.import_qty || 0) / maxVal) * 120);
      chart.innerHTML += `
        <div class="bar-wrap">
          <div style="display:flex;align-items:flex-end;gap:2px;height:120px">
            <div class="bar" style="height:${salesH}px;width:14px" title="Doanh so: ${h.sales_qty}"></div>
            <div class="bar import-bar" style="height:${importH}px;width:14px" title="Nhap: ${h.import_qty}"></div>
          </div>
          <div class="bar-label">${h.date ? h.date.slice(5) : ''}</div>
        </div>`;
    });
  }

  // Table
  const tbody = document.getElementById('result_table');
  tbody.innerHTML = '';
  const rows = data.details || [data];
  rows.forEach(r => {
    const rimp = r.import_suggestion || 0;
    const rqty = r.predicted_qty || 0;
    let tag = '<span class="tag-ok">Du hang</span>';
    if (rimp > rqty * 0.5) tag = '<span class="tag-alert">Thieu hang - can nhap ngay</span>';
    else if (rimp > 0)      tag = '<span class="tag-warn">Sap thieu - nen nhap them</span>';

    tbody.innerHTML += `
      <tr>
        <td>${r.subgroup_name || r.maingroup_name || 'Tổng hợp'}</td>
        <td><b>${rqty.toLocaleString('vi-VN')}</b></td>
        <td>${(r.opening_inventory || 0).toLocaleString('vi-VN')}</td>
        <td><b style="color:#dd6b20">${rimp.toLocaleString('vi-VN')}</b></td>
        <td>${r.safety_factor_pct || '—'}%</td>
        <td>${tag}</td>
      </tr>`;
  });

  document.getElementById('result_section').classList.add('visible');
}
</script>
</body>
</html>
"""

# ─── FLASK ROUTES ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/provinces')
def api_provinces():
    try:
        conn = get_conn()
        df = pd.read_sql("""
            SELECT DISTINCT province_id, province_name FROM DIM_STORE
            WHERE province_name IS NOT NULL ORDER BY province_name
        """, conn)
        conn.close()
        return jsonify(df.fillna('').to_dict('records'))
    except Exception as e:
        return jsonify([]), 200

@app.route('/api/stores')
def api_stores():
    province_id = request.args.get('province_id')
    try:
        conn = get_conn()
        if province_id:
            df = pd.read_sql("""
                SELECT store_id, store_name FROM DIM_STORE
                WHERE province_id = ? ORDER BY store_name
            """, conn, params=[province_id])
        else:
            df = pd.read_sql("SELECT store_id, store_name FROM DIM_STORE ORDER BY store_name", conn)
        conn.close()
        return jsonify(df.fillna('').to_dict('records'))
    except Exception as e:
        return jsonify([])

@app.route('/api/maingroups')
def api_maingroups():
    try:
        conn = get_conn()
        df = pd.read_sql("""
            SELECT DISTINCT dp.maingroup_id, dp.maingroup_name
            FROM DIM_PRODUCT dp
            WHERE dp.maingroup_id IN (
                SELECT DISTINCT dp2.maingroup_id
                FROM FACT_DAILY_SALES f
                JOIN DIM_PRODUCT dp2 ON dp2.product_key = f.product_key
            )
            AND dp.maingroup_name IS NOT NULL
            AND dp.maingroup_name NOT LIKE '%[0-9]%'
            ORDER BY dp.maingroup_name
        """, conn)
        conn.close()
        return jsonify(df.fillna('').to_dict('records'))
    except Exception as e:
        return jsonify([])

@app.route('/api/subgroups')
def api_subgroups():
    maingroup_id = request.args.get('maingroup_id')
    try:
        conn = get_conn()
        if maingroup_id:
            df = pd.read_sql("""
                SELECT DISTINCT dp.subgroup_id, dp.subgroup_name
                FROM DIM_PRODUCT dp
                WHERE dp.maingroup_id = ?
                AND dp.subgroup_id IN (
                    SELECT DISTINCT dp2.subgroup_id
                    FROM FACT_DAILY_SALES f
                    JOIN DIM_PRODUCT dp2 ON dp2.product_key = f.product_key
                )
                AND dp.subgroup_name IS NOT NULL
                AND dp.subgroup_name NOT LIKE '%[0-9]%'
                ORDER BY dp.subgroup_name
            """, conn, params=[maingroup_id])
        else:
            df = pd.read_sql("""
                SELECT DISTINCT dp.subgroup_id, dp.subgroup_name
                FROM DIM_PRODUCT dp
                WHERE dp.subgroup_id IN (
                    SELECT DISTINCT dp2.subgroup_id
                    FROM FACT_DAILY_SALES f
                    JOIN DIM_PRODUCT dp2 ON dp2.product_key = f.product_key
                )
                AND dp.subgroup_name IS NOT NULL
                AND dp.subgroup_name NOT LIKE '%[0-9]%'
                ORDER BY dp.subgroup_name
            """, conn)
        conn.close()
        return jsonify(df.fillna('').to_dict('records'))
    except Exception as e:
        return jsonify([])

@app.route('/api/predict', methods=['POST'])
def api_predict():
    req = request.json
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

        # Với rolling forecast, query đến DATA_CUTOFF; bình thường query < forecast_date
        query_before = (DATA_CUTOFF + pd.Timedelta(days=1)).strftime('%Y-%m-%d') if is_rolling else forecast_date

        # Build WHERE
        conditions = ["dd.full_date < ?"]
        params = [query_before]
        if store_id:
            conditions.append("ds.store_id = ?"); params.append(store_id)
        if province_id:
            conditions.append("ds.province_id = ?"); params.append(province_id)
        if subgroup_id:
            conditions.append("dp.subgroup_id = ?"); params.append(subgroup_id)
        elif maingroup_id:
            conditions.append("dp.maingroup_id = ?"); params.append(maingroup_id)

        where = " AND ".join(conditions)

        # Query lịch sử
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

        # Tổng hợp theo ngày để tính lag
        df_daily = df.groupby('date').agg(
            sales_qty        = ('sales_qty', 'sum'),
            opening_inventory= ('opening_inventory', 'sum'),
            import_qty       = ('import_qty', 'sum'),
            day_of_week      = ('day_of_week', 'first'),
            is_weekend       = ('is_weekend', 'first'),
            week_of_month    = ('week_of_month', 'first'),
        ).reset_index().sort_values('date')

        # Load model
        models_available = [f for f in os.listdir(MODEL_DIR) if f.startswith('xgb_') and f.endswith('.pkl')]
        if not models_available:
            return jsonify({"error": "Chưa có model. Chạy 05_train_models.py trước."}), 404

        preferred  = ['xgb_5_subgroup_province.pkl', 'xgb_4_subgroup_national.pkl',
                      'xgb_2_category_province.pkl', 'xgb_1_category_national.pkl']
        model_file = next((m for m in preferred if m in models_available), models_available[0])
        with open(os.path.join(MODEL_DIR, model_file), 'rb') as f:
            model = pickle.load(f)

        def build_features(series_window, target_date):
            feats = {}
            for i in range(1, LAG_DAYS + 1):
                feats[f'lag_{i}'] = float(series_window[-i]) if len(series_window) >= i else 0.0
            feats['rolling_mean_7'] = float(np.mean(series_window[-7:])) if len(series_window) >= 7 else float(np.mean(series_window) if len(series_window) else 0)
            feats['rolling_std_7']  = float(np.std(series_window[-7:]))  if len(series_window) >= 7 else 0.0
            feats['cv_7']           = feats['rolling_std_7'] / (feats['rolling_mean_7'] + 1e-6)
            feats['day_of_week']    = target_date.weekday()
            feats['is_weekend']     = 1 if target_date.weekday() >= 5 else 0
            feats['week_of_month']  = (target_date.day - 1) // 7 + 1
            return feats

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
        # Nếu tồn kho đủ dùng hơn 3 ngày thì không cần nhập
        if opening_inv > pred_qty * 3:
            import_need = 0

        # Lịch sử 7 ngày cho chart
        history = df_daily.tail(7)[['date','sales_qty','import_qty']].to_dict('records')

        # Chi tiết theo subgroup
        subgroup_details = []
        for sg_id, sg_df in df.groupby('subgroup_id'):
            sg_series = sg_df.sort_values('date').groupby('date')['sales_qty'].sum().values
            sg_inv    = float(sg_df['opening_inventory'].sum() / max(len(sg_df['date'].unique()), 1))
            sg_cv     = float(np.std(sg_series[-7:]) / (np.mean(sg_series[-7:]) + 1e-6)) if len(sg_series) >= 3 else 0.15
            sg_pred   = float(sg_series[-1]) * (1 + 0.05) if len(sg_series) > 0 else 0
            sg_safety = min(0.40, 0.15 + sg_cv * 0.25)
            sg_import = max(0, sg_pred * (1 + sg_safety) - sg_inv)
            # Nếu tồn kho đủ dùng hơn 3 ngày thì không cần nhập
            if sg_inv > sg_pred * 3:
                sg_import = 0
            subgroup_details.append({
                'subgroup_id':      int(sg_id) if sg_id else None,
                'subgroup_name':    sg_df['subgroup_name'].iloc[0],
                'maingroup_name':   sg_df['maingroup_name'].iloc[0],
                'predicted_qty':    round(sg_pred, 2),
                'opening_inventory':round(sg_inv, 2),
                'import_suggestion':round(sg_import, 2),
                'safety_factor_pct':round(sg_safety * 100, 1),
            })

        return jsonify({
            "forecast_date":       forecast_date,
            "predicted_qty":       round(pred_qty, 2),
            "opening_inventory":   round(opening_inv, 2),
            "safety_factor_pct":   round(safety * 100, 1),
            "import_suggestion":   round(import_need, 2),
            "model_used":          model_file,
            "is_rolling":          is_rolling,
            "rollout_days":        rollout_days,
            "history":             history,
            "details":             subgroup_details,
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("  Dashboard chạy tại: http://localhost:5000")
    print("  Ctrl+C để dừng")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=5000)
