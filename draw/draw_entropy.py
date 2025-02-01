import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

# 設定科研風格
sns.set(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

def plot_entropy_ranges(df, y_column, title, output_dir='./results/img'):
    """整合两个数据区间到单一图表并优化显示"""
    # 合并数据筛选条件
    mask_small = (df['Data Length'] >= 50) & (df['Data Length'] <= 1000)
    mask_large = (df['Data Length'] >= 5000) & (df['Data Length'] <= 500000)
    df_small = df[mask_small]
    df_large = df[mask_large]

    if df_small.empty and df_large.empty:
        print("Error: Not enough data points.")
        return

    # 创建图表和主轴
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴

    # 使用对数坐标轴处理跨度大的数据长度
    ax1.set_xscale('log')
    ax2.set_xscale('log')

    # 可视化参数设置
    algorithms = df['Algorithm'].unique()
    palette = sns.color_palette("Set2", n_colors=len(algorithms))  # 改为色盲友好的色调
    line_styles = ['-', '-', '-', ':']
    markers = ['o', 's', 'D', '^']

    # 在主轴绘制小数据区间
    for i, algorithm in enumerate(algorithms):
        data_small = df_small[df_small['Algorithm'] == algorithm].sort_values('Data Length')
        if not data_small.empty:
            ax1.plot('Data Length', y_column,
                     data=data_small,
                     color=palette[i],
                     linestyle=line_styles[i % len(line_styles)],
                     marker=markers[i % len(markers)],
                     markersize=4,
                     linewidth=1.5,
                     label=f'{algorithm}')

    # 在次轴绘制大数据区间
    for i, algorithm in enumerate(algorithms):
        data_large = df_large[df_large['Algorithm'] == algorithm].sort_values('Data Length')
        if not data_large.empty:
            ax2.plot('Data Length', y_column,
                     data=data_large,
                     color=palette[i],
                     linestyle=line_styles[i % len(line_styles)],
                     marker=markers[i % len(markers)],
                     markersize=4,
                     linewidth=1.5,
                     label='')

    # 添加区间标注
    ax1.axvspan(50, 1000, alpha=0.05, color='blue', label='Small Data Range')
    ax2.axvspan(5000, 500000, alpha=0.05, color='orange', label='Large Data Range')

    # 设置y轴标签
    ax1.set_ylabel('Entropy (Small Range)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Entropy (Large Range)', fontweight='bold', fontsize=12)

    # 设置x轴标签
    ax1.set_xlabel('Data Length (Bytes) - Log Scale', fontweight='bold', fontsize=12)

    # 设置标题
    plt.title(title, fontweight='bold', fontsize=14, pad=15)

    # 优化坐标轴刻度
    xticks = [50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
    xlabels = ['50', '100', '500', '1K', '5K', '10K', '50K', '100K', '500K']
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels)

    # 组合图例
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    unique_labels = dict(zip(labels, handles))  # 去重
    ax1.legend(unique_labels.values(), unique_labels.keys(),
               title='Legend', frameon=True, loc='lower right', fontsize=10, title_fontsize=12)

    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax2.grid(True, which="both", ls="-", alpha=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_result.png'), dpi=1200, bbox_inches='tight')
    plt.show()

def main(input_path, output_dir='./results/img'):
    """主函數：讀取數據並繪製圖表"""
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print("Error: The specified file does not exist.")
        return

    # 檢查必要列
    required_columns = ['Algorithm', 'Data Length', 'Average Entropy']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing column '{col}' in the dataset.")
            return

    # 創建結果目錄
    os.makedirs(output_dir, exist_ok=True)

    # 繪製整合後的熵圖
    plot_entropy_ranges(df, 'Average Entropy', 'Entropy Analysis')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot entropy results.')
    parser.add_argument('--input', type=str, default='./results/csv/shang/shang_2.csv',
                       help='Path to the input CSV file.')
    parser.add_argument('--output', type=str, default='./results/img',
                       help='Output directory for saving the plot.')
    args = parser.parse_args()
    main(args.input, args.output)