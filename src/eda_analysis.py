#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
11_eda_analysis.py
EDA — 4 biểu đồ phân tích khám phá dữ liệu
  1. eda_distribution.png      — Phân phối sales_qty trước/sau log1p
  2. eda_negative_inventory.png — Tỷ lệ tồn kho âm theo ngành hàng
  3. eda_daily_trend.png        — Xu hướng 31 ngày tháng 3/2026
  4. eda_correlation.png        — Heatmap tương quan đặc trưng cấp 1
"""

import io
import os
import sys
import warnings
import pickle
import numpy as np
import pandas as pd
import pyodbc

warnings.filterwarnings("ignore")

# Force UTF-8 stdout (Windows terminal)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy.stats import gaussian_kde

# ── Root & paths ─────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
from config.config import CONNECTION_STRING

OUT_DIR          = os.path.join(_ROOT, "output", "charts")
FEATURE_STORE_L1 = os.path.join(_ROOT, "resources", "feature_store",
                                "level_1_category_national.pkl")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Vietnamese-friendly font (Arial có sẵn trên Windows 11) ──────────────────
plt.rcParams.update({
    "font.family":        ["Arial", "DejaVu Sans", "Tahoma", "sans-serif"],
    "axes.unicode_minus": False,
    "figure.dpi":         300,
})

COLORS = ["#2E86AB", "#E84855", "#3BB273"]   # xanh dương, đỏ, xanh lá


# ─────────────────────────────────────────────────────────────────────────────
# Helper: kết nối DB và trả về DataFrame (dùng cursor, tránh warning pandas)
# ─────────────────────────────────────────────────────────────────────────────
def _query(sql: str, parse_dates: list = None) -> pd.DataFrame:
    conn   = pyodbc.connect(CONNECTION_STRING, timeout=60)
    cursor = conn.cursor()
    cursor.execute(sql)
    cols = [c[0] for c in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    df = pd.DataFrame.from_records(rows, columns=cols)
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Chart 1 — Phân phối sales_qty (histogram + KDE, trước/sau log1p)
# ═════════════════════════════════════════════════════════════════════════════
def chart_distribution():
    print("  [1/4] Đang query sales_qty distribution …")
    sql = """
        SELECT TOP 100000 sales_qty
        FROM   FACT_DAILY_SALES
        WHERE  sales_qty > 0 AND sales_qty < 1000
        ORDER BY NEWID()
    """
    df   = _query(sql)
    vals = df["sales_qty"].dropna().astype(float).values
    log_vals = np.log1p(vals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phân phối sales_qty trước và sau biến đổi log1p",
                 fontsize=14, fontweight="bold", y=1.01)

    def _draw_hist(ax, data, title, xlabel, color):
        ax.hist(data, bins=60, color=color, alpha=0.65, density=True,
                edgecolor="white", linewidth=0.4, label="Histogram")

        # KDE
        kde = gaussian_kde(data, bw_method="scott")
        x_grid = np.linspace(data.min(), data.max(), 400)
        ax.plot(x_grid, kde(x_grid), color="black", lw=1.8, label="KDE")

        mean_v   = np.mean(data)
        median_v = np.median(data)
        ax.axvline(mean_v,   color="#E84855", lw=1.6, linestyle="--",
                   label=f"Mean = {mean_v:.2f}")
        ax.axvline(median_v, color="#3BB273", lw=1.6, linestyle="-.",
                   label=f"Median = {median_v:.2f}")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Mật độ xác suất", fontsize=10)
        ax.legend(fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)

    _draw_hist(axes[0], vals,     "Trước biến đổi (gốc)",  "sales_qty",        "#2E86AB")
    _draw_hist(axes[1], log_vals, "Sau biến đổi log1p",    "log1p(sales_qty)", "#E84855")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "eda_distribution.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Đã lưu: {out}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Chart 2 — Tỷ lệ tồn kho âm theo ngành hàng (horizontal bar)
# ═════════════════════════════════════════════════════════════════════════════
def chart_negative_inventory():
    print("  [2/4] Đang query tồn kho âm …")
    sql = """
        SELECT
            dp.maingroup_name,
            SUM(CAST(f.is_negative_inventory AS INT)) AS neg_count,
            COUNT(*)                                   AS total_count
        FROM FACT_DAILY_SALES  f
        JOIN DIM_PRODUCT        dp ON dp.product_key = f.product_key
        GROUP BY dp.maingroup_name
    """
    df = _query(sql)
    df["neg_count"]   = df["neg_count"].astype(float)
    df["total_count"] = df["total_count"].astype(float)
    df["pct_neg"]     = df["neg_count"] / df["total_count"] * 100
    df = df.sort_values("pct_neg", ascending=True)

    cmap      = plt.cm.get_cmap("RdYlGn_r", len(df))
    bar_colors = [cmap(i) for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(11, max(5, len(df) * 0.55)))
    bars = ax.barh(df["maingroup_name"], df["pct_neg"],
                   color=bar_colors, edgecolor="white", height=0.65)

    # Nhãn % trên mỗi bar
    for bar, pct in zip(bars, df["pct_neg"]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", ha="left", fontsize=9,
                fontweight="bold", color="#333333")

    ax.set_title("Tỷ lệ tồn kho âm theo ngành hàng (%)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Tỷ lệ tồn kho âm (%)", fontsize=10)
    ax.set_xlim(0, df["pct_neg"].max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "eda_negative_inventory.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Đã lưu: {out}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Chart 3 — Xu hướng sản lượng bán ngày tháng 3/2026 (line + weekend shade)
# ═════════════════════════════════════════════════════════════════════════════
def chart_daily_trend():
    print("  [3/4] Đang query xu hướng tháng 3/2026 …")
    sql = """
        SELECT
            dd.full_date,
            dd.is_weekend,
            dp.maingroup_name,
            SUM(f.sales_qty) AS sales_qty
        FROM FACT_DAILY_SALES  f
        JOIN DIM_DATE          dd ON dd.date_key    = f.date_key
        JOIN DIM_PRODUCT       dp ON dp.product_key = f.product_key
        WHERE dd.full_date >= '2026-03-01'
          AND dd.full_date <= '2026-03-31'
        GROUP BY dd.full_date, dd.is_weekend, dp.maingroup_name
        ORDER BY dd.full_date, dp.maingroup_name
    """
    df = _query(sql, parse_dates=["full_date"])
    if df.empty:
        print("  ! Không có dữ liệu tháng 3/2026, bỏ qua chart 3.")
        return None

    df["sales_qty"] = df["sales_qty"].astype(float)

    # Lấy top 3 ngành hàng theo tổng doanh số
    top3 = (df.groupby("maingroup_name")["sales_qty"]
              .sum().nlargest(3).index.tolist())
    df3   = df[df["maingroup_name"].isin(top3)].copy()
    pivot = df3.pivot_table(index="full_date", columns="maingroup_name",
                            values="sales_qty", aggfunc="sum")

    # Ngày cuối tuần trong tháng 3/2026
    all_dates    = pd.date_range("2026-03-01", "2026-03-31")
    weekend_dates = all_dates[all_dates.dayofweek >= 5]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Tô vùng xám nhạt cho cuối tuần
    for d in weekend_dates:
        ax.axvspan(d - pd.Timedelta(hours=12),
                   d + pd.Timedelta(hours=12),
                   color="lightgray", alpha=0.35, linewidth=0)

    for i, col in enumerate(pivot.columns):
        ax.plot(pivot.index, pivot[col], marker="o", markersize=3.5,
                linewidth=1.8, color=COLORS[i % len(COLORS)], label=col)

    # Chú thích vùng cuối tuần
    weekend_patch = mpatches.Patch(color="lightgray", alpha=0.5,
                                   label="Cuối tuần")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [weekend_patch], labels + ["Cuối tuần"],
              fontsize=9, loc="upper left", framealpha=0.8)

    ax.set_title("Xu hướng sản lượng bán theo ngày tháng 3/2026",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Ngày", fontsize=10)
    ax.set_ylabel("Tổng sales_qty", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(
        matplotlib.dates.DateFormatter("%d/%m"))
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "eda_daily_trend.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Đã lưu: {out}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Chart 4 — Heatmap ma trận tương quan đặc trưng (cấp 1)
# ═════════════════════════════════════════════════════════════════════════════
def chart_correlation():
    print("  [4/4] Đang đọc feature store cấp 1 …")
    if not os.path.exists(FEATURE_STORE_L1):
        print(f"  ! Không tìm thấy {FEATURE_STORE_L1}, bỏ qua chart 4.")
        return None

    df = pd.read_pickle(FEATURE_STORE_L1)

    feat_cols = (
        [f"lag_{i}" for i in range(1, 8)]
        + ["rolling_mean_7", "rolling_std_7", "cv_7",
           "day_of_week", "is_weekend"]
    )
    # Chỉ giữ lại cột tồn tại trong df
    feat_cols = [c for c in feat_cols if c in df.columns]
    corr = df[feat_cols].astype(float).corr()

    # Nhãn thân thiện cho bảng heatmap
    label_map = {
        **{f"lag_{i}": f"lag_{i}" for i in range(1, 8)},
        "rolling_mean_7": "roll_mean7",
        "rolling_std_7":  "roll_std7",
        "cv_7":           "cv_7",
        "day_of_week":    "day_of_wk",
        "is_weekend":     "is_weekend",
    }
    corr.index   = [label_map.get(c, c) for c in corr.index]
    corr.columns = [label_map.get(c, c) for c in corr.columns]

    n = len(corr)
    fig, ax = plt.subplots(figsize=(n * 0.85 + 1.5, n * 0.75 + 1.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # ẩn tam giác trên

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 8},
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title("Ma trận tương quan giữa các đặc trưng (cấp 1)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "eda_correlation.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Đã lưu: {out}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  EDA — 4 BIỂU ĐỒ PHÂN TÍCH KHÁM PHÁ DỮ LIỆU")
    print("=" * 60)

    results = {}
    try:
        results["distribution"]       = chart_distribution()
    except Exception as e:
        print(f"  [LỖI chart 1] {e}")

    try:
        results["negative_inventory"] = chart_negative_inventory()
    except Exception as e:
        print(f"  [LỖI chart 2] {e}")

    try:
        results["daily_trend"]        = chart_daily_trend()
    except Exception as e:
        print(f"  [LỖI chart 3] {e}")

    try:
        results["correlation"]        = chart_correlation()
    except Exception as e:
        print(f"  [LỖI chart 4] {e}")

    print("\n" + "=" * 60)
    print("  KẾT QUẢ — đường dẫn file đã lưu:")
    print("=" * 60)
    for name, path in results.items():
        status = path if path else "KHÔNG TẠO ĐƯỢC"
        print(f"  {name:<25} → {status}")
    print()


if __name__ == "__main__":
    main()
