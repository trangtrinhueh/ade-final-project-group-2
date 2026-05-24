# ============================================================
#  10_visualization.py — Visualize kết quả đồ án
#  3 biểu đồ: MAPE comparison, Feature Importance, Predicted vs Actual
#  Output: output/charts/*.png  (dpi=300)
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import pandas as pd

# Font hỗ trợ tiếng Việt (Arial/Tahoma có sẵn trên Windows)
plt.rcParams['font.sans-serif'] = ['Arial', 'Tahoma', 'DejaVu Sans']
plt.rcParams['font.family']     = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'charts')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  CHART 1 — MAPE Comparison: 6 cấp độ × 3 mô hình
# ══════════════════════════════════════════════════════════════
def chart_mape():
    levels = [
        'Cấp 1\nCategory×QG',
        'Cấp 2\nCategory×TP',
        'Cấp 3\nCategory×ST',
        'Cấp 4\nSubgroup×QG',
        'Cấp 5\nSubgroup×TP',
        'Cấp 6\nSubgroup×ST',
    ]
    mape_data = {
        'ARIMA':   [14.32, 17.86, 24.52, 40.16, 32.89, 40.07],
        'LSTM':    [99.92, 153.51, 19.75, 225.61, 64.73, 34.89],
        'XGBoost': [ 6.51,  12.56, 18.12,  30.12, 33.78, 34.45],
    }
    colors = {'ARIMA': '#4299e1', 'LSTM': '#fc8181', 'XGBoost': '#68d391'}
    Y_CAP  = 60   # cắt trục y tại 60% vì LSTM cấp 1,2,4 vượt 100%

    x     = np.arange(len(levels))
    width = 0.25
    offsets = {'ARIMA': -width, 'LSTM': 0, 'XGBoost': width}

    fig, ax = plt.subplots(figsize=(14, 6.5))

    for model, values in mape_data.items():
        clipped = [min(v, Y_CAP) for v in values]
        bars    = ax.bar(x + offsets[model], clipped, width,
                         label=model, color=colors[model],
                         edgecolor='white', linewidth=0.8, zorder=3)

        for bar, real_val in zip(bars, values):
            if real_val > Y_CAP:
                # Giá trị thực bị cắt → annotation màu đỏ phía trên cột
                ax.annotate(
                    f'{real_val:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, Y_CAP),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=8.5, color='#c53030', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#c53030',
                                   lw=0.8, shrinkA=0),
                    annotation_clip=False,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.6,
                    f'{real_val:.1f}',
                    ha='center', va='bottom', fontsize=7.5, color='#2d3748',
                )

    # Đường tham chiếu MAPE = 20%
    ax.axhline(y=20, color='#a0aec0', linestyle='--', linewidth=1, zorder=2)
    ax.text(5.62, 21, 'MAPE = 20%', fontsize=8, color='#718096')

    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=10.5)
    ax.set_ylim(0, Y_CAP + 12)
    ax.set_ylabel('MAPE (%)', fontsize=11)
    ax.set_title(
        'So sánh MAPE theo Cấp độ Tổng hợp × Mô hình\n'
        '(Tập test: ngày 27–31/03/2026  |  Trục Y cắt tại 60%;  số đỏ = giá trị thực vượt ngưỡng)',
        fontsize=12, pad=14
    )
    ax.legend(fontsize=10.5, framealpha=0.9, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linestyle='--', zorder=0)
    ax.set_axisbelow(True)

    out = os.path.join(OUTPUT_DIR, 'mape_comparison.png')
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out}')


# ══════════════════════════════════════════════════════════════
#  CHART 2 — Feature Importance: 6 subplot (2 hàng × 3 cột)
# ══════════════════════════════════════════════════════════════
def chart_feature_importance():
    level_files = [
        ('Cấp 1: Category × Quốc gia',  'xgb_1_category_national.pkl'),
        ('Cấp 2: Category × Tỉnh thành', 'xgb_2_category_province.pkl'),
        ('Cấp 3: Category × Cửa hàng',  'xgb_3_category_store.pkl'),
        ('Cấp 4: Subgroup × Quốc gia',  'xgb_4_subgroup_national.pkl'),
        ('Cấp 5: Subgroup × Tỉnh thành', 'xgb_5_subgroup_province.pkl'),
        ('Cấp 6: Subgroup × Cửa hàng',  'xgb_6_subgroup_store.pkl'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat  = axes.flatten()
    base_color = '#3182ce'

    for idx, (title, fname) in enumerate(level_files):
        ax   = axes_flat[idx]
        path = os.path.join(MODEL_DIR, fname)

        if not os.path.exists(path):
            ax.text(0.5, 0.5, 'File không tồn tại', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='#718096')
            ax.set_title(title, fontsize=9, fontweight='bold')
            continue

        with open(path, 'rb') as f:
            model = pickle.load(f)

        fi = (
            pd.Series(model.feature_importances_, index=model.feature_names_in_)
            .sort_values(ascending=False)
            .head(5)
            .sort_values(ascending=True)    # horizontal: thấp nhất ở trên
        )

        # Gradient màu theo importance
        norm_vals  = (fi.values - fi.values.min()) / ((fi.values.max() - fi.values.min()) + 1e-9)
        bar_colors = [matplotlib.colors.to_hex(
                          plt.cm.Blues(0.35 + 0.55 * v)) for v in norm_vals]

        bars = ax.barh(fi.index, fi.values, color=bar_colors,
                       edgecolor='white', linewidth=0.6)

        for bar, val in zip(bars, fi.values):
            ax.text(val + fi.values.max() * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=8, color='#2d3748')

        ax.set_title(title, fontsize=9.5, fontweight='bold', pad=7)
        ax.set_xlabel('Importance', fontsize=8.5)
        ax.set_xlim(0, fi.values.max() * 1.22)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=9)
        ax.xaxis.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    fig.suptitle(
        'Feature Importance — XGBoost  (Top 5 đặc trưng quan trọng nhất theo từng cấp độ)',
        fontsize=13, fontweight='bold', y=1.02
    )
    out = os.path.join(OUTPUT_DIR, 'feature_importance.png')
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out}')


# ══════════════════════════════════════════════════════════════
#  CHART 3 — Predicted vs Actual: Cấp 1 Category × QG
# ══════════════════════════════════════════════════════════════
def chart_predicted_vs_actual():
    results_path = os.path.join(MODEL_DIR, 'all_results.pkl')
    if not os.path.exists(results_path):
        print('  ! Bỏ qua chart 3: all_results.pkl không tìm thấy')
        return

    with open(results_path, 'rb') as f:
        all_results = pickle.load(f)

    level_key = '1_category_national'
    if level_key not in all_results or 'XGBoost' not in all_results[level_key]:
        print(f'  ! Bỏ qua chart 3: không tìm thấy {level_key}/XGBoost')
        return

    res    = all_results[level_key]['XGBoost']
    actual = np.array(res['actual'],    dtype=float)
    pred   = np.array(res['predicted'], dtype=float)
    n      = len(actual)

    # Tổng hợp 20 mẫu → 5 ngày test (4 maingroups × 5 ngày)
    # Dữ liệu sắp xếp: group 1 hết rồi đến group 2 → reshape (n_groups, 5)
    n_days = 5
    if n % n_days == 0:
        n_groups     = n // n_days
        actual_daily = actual.reshape(n_groups, n_days).sum(axis=0)
        pred_daily   = pred.reshape(n_groups, n_days).sum(axis=0)
        dates        = ['27/03', '28/03', '29/03', '30/03', '31/03']
        xlabel       = 'Ngày (tháng 3/2026)'
    else:
        actual_daily = actual
        pred_daily   = pred
        dates        = [f'Mẫu {i+1}' for i in range(n)]
        xlabel       = 'Mẫu test'

    x_idx = np.arange(len(dates))

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Vùng tô sai số giữa 2 đường
    ax.fill_between(x_idx, actual_daily, pred_daily,
                    alpha=0.13, color='#3182ce', label='_nolegend_')

    # Đường actual và predicted
    ax.plot(x_idx, actual_daily, color='#2b6cb0', linewidth=2.4,
            marker='o', markersize=8, zorder=4, label='Thực tế (Actual)')
    ax.plot(x_idx, pred_daily,   color='#dd6b20', linewidth=2.1,
            marker='s', markersize=7, linestyle='--', zorder=3,
            label='Dự báo XGBoost')

    # MAPE từng ngày
    for i, (a, p) in enumerate(zip(actual_daily, pred_daily)):
        mape_pt = abs(a - p) / (a + 1e-6) * 100
        y_pos   = max(a, p) + (actual_daily.max() - actual_daily.min()) * 0.04
        ax.annotate(
            f'err={mape_pt:.1f}%',
            xy=(i, y_pos),
            ha='center', va='bottom',
            fontsize=8.5, color='#718096',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none')
        )

    ax.set_xticks(x_idx)
    ax.set_xticklabels(dates, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Tổng sản lượng bán (đv quy đổi)', fontsize=11)
    ax.set_title(
        'Predicted vs Actual — Cấp 1: Category × Quốc gia\n'
        '(XGBoost  |  Tập test: ngày 27–31/03/2026  |  Tổng hợp tất cả ngành hàng)',
        fontsize=12, pad=12
    )
    ax.legend(fontsize=11, framealpha=0.9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linestyle='--')
    ax.set_axisbelow(True)

    # Chú thích vùng tô
    mid_x = len(dates) // 2
    mid_y = (actual_daily[mid_x] + pred_daily[mid_x]) / 2
    ax.annotate('Vùng sai số', xy=(mid_x, mid_y),
                fontsize=8, color='#4299e1', ha='center',
                style='italic', alpha=0.8)

    out = os.path.join(OUTPUT_DIR, 'predicted_vs_actual.png')
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out}')


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=' * 55)
    print('  VISUALIZATION — 3 biểu đồ đồ án')
    print(f'  Output: {OUTPUT_DIR}')
    print('=' * 55)

    print('\n[1/3] Vẽ MAPE Comparison...')
    chart_mape()

    print('[2/3] Vẽ Feature Importance...')
    chart_feature_importance()

    print('[3/3] Vẽ Predicted vs Actual...')
    chart_predicted_vs_actual()

    print(f'\n✓ Hoàn tất! 3 file PNG đã lưu vào: output/charts/')
