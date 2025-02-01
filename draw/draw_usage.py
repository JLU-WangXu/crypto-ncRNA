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

def remove_outliers(df, group_columns, target_columns):
    """按分组删除异常值（基于IQR方法）"""
    mask = pd.Series([True] * len(df), index=df.index)
    
    # 遍历每个分组
    for name, group in df.groupby(group_columns):
        # 遍历每个目标列
        for col in target_columns:
            # 确保有足够的数据点计算分位数
            if len(group) >= 3:
                q1 = group[col].quantile(0.25)
                q3 = group[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # 更新该组的掩码
                group_mask = (group[col] >= lower_bound) & (group[col] <= upper_bound)
                mask.loc[group.index] &= group_mask
                
    return df[mask].copy()

def plot_memory_usage(ax, df, y_column, title):
    """繪製內存使用折線圖，並確保 x 軸均勻分佈，並加入均值線"""
    algorithms = df['Algorithm'].unique()
    palette = sns.color_palette("husl", n_colors=len(algorithms))
    line_styles = ['-', '-', '-', ':']
    markers = ['o', 's', 'D', '^']

    # 把 Data Length 轉為字串類別，以便均勻分佈
    df['Data Length'] = df['Data Length'].astype(str)

    for i, algorithm in enumerate(algorithms):
        data = df[df['Algorithm'] == algorithm]

        # 繪製曲線
        sns.lineplot(
            x='Data Length',  
            y=y_column,
            data=data,
            ax=ax,
            color=palette[i],
            linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)],
            label=algorithm 
        )

        # 計算該算法的平均值
        mean_value = data[y_column].mean()

        # 添加均值線，顏色與曲線相同但增加透明度
        ax.axhline(mean_value, color=palette[i], linestyle='-', alpha=0.3, linewidth=1.5)

    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel('Data Length (Bytes)', fontweight='bold')
    ax.set_ylabel('Memory Usage (KiB)', fontweight='bold')
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.grid(axis="x", alpha=0)
    ax.legend(title='Algorithm', loc='upper left', fontsize=10, frameon=True)
    ax.set_xticks(range(len(df['Data Length'].unique())))  
    ax.set_xticklabels(df['Data Length'].unique())

def main(input_path, output_dir='./results/img'):
    """主函數：讀取數據並繪製圖表"""
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print("Error: The specified file does not exist.")
        return 

    # 檢查必要列
    required_columns = ['Algorithm', 'Data Length', 'Encryption Memory Usage (MB)', 'Decryption Memory Usage (MB)']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing column '{col}' in the dataset.")
            return 

    # 删除异常值（按算法和数据长度分组处理）
    df = remove_outliers(
        df,
        group_columns=['Algorithm', 'Data Length'],
        target_columns=['Encryption Memory Usage (MB)', 'Decryption Memory Usage (MB)']
    )

    # 转换单位：MB -> KiB
    df['Encryption Memory Usage (KiB)'] = df['Encryption Memory Usage (MB)'] * 1024 
    df['Decryption Memory Usage (KiB)'] = df['Decryption Memory Usage (MB)'] * 1024 

    # 創建圖像視窗
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # 繪製圖表
    plot_memory_usage(ax1, df, 'Encryption Memory Usage (KiB)', 'Encryption Memory Usage')
    plot_memory_usage(ax2, df, 'Decryption Memory Usage (KiB)', 'Decryption Memory Usage')

    # 調整佈局並保存
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'memory_usage_result.png'), dpi=1200, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot memory usage results.')
    parser.add_argument('--input', type=str, default='./results/csv/usage/usage_4.csv',
                       help='Path to the input CSV file.')
    parser.add_argument('--output', type=str, default='./results/img',
                       help='Output directory for saving the plot.')
    args = parser.parse_args()
    main(args.input, args.output)