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

def plot_throughput(ax, df, y_column, title):
    """繪製吞吐量折線圖，並確保 x 軸均勻分佈，並加入均值線"""
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
    ax.set_ylabel('Throughput (KiB/S)', fontweight='bold')
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.grid(axis="x",alpha=0)

    # **把 legend 放進圖內**
    ax.legend(title='Algorithm', loc='upper left', fontsize=10, frameon=True)

    # 設定 x 軸的刻度，使其均勻分佈
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
    required_columns = ['Algorithm', 'Data Length', 'Encryption Throughput (Bytes/Second)', 'Decryption Throughput (Bytes/Second)']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing column '{col}' in the dataset.")
            return 

    # 转换单位：Bytes/Second -> KiB/S
    df['Encryption Throughput (KiB/S)'] = df['Encryption Throughput (Bytes/Second)'] / 1024 
    df['Decryption Throughput (KiB/S)'] = df['Decryption Throughput (Bytes/Second)'] / 1024 

    # 創建圖像視窗（size (10,5)）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # 繪製圖表
    plot_throughput(ax1, df, 'Encryption Throughput (KiB/S)', 'Encryption Throughput')
    plot_throughput(ax2, df, 'Decryption Throughput (KiB/S)', 'Decryption Throughput')

    # 設置y軸範圍
    max_encryption = df['Encryption Throughput (KiB/S)'].max()
    max_decryption = df['Decryption Throughput (KiB/S)'].max()

    ax1.set_ylim(0, max_encryption * 1.1)
    ax2.set_ylim(0, max_decryption * 1.1)

    # 調整佈局並保存
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'throughput_result.png'), dpi=1200, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot throughput results.')
    parser.add_argument('--input', type=str, default='./results/csv/throughput/throughput_3.csv',
                       help='Path to the input CSV file.')
    parser.add_argument('--output', type=str, default='./results/img',
                       help='Output directory for saving the plot.')
    args = parser.parse_args()
    main(args.input, args.output)