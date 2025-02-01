import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
import os 
 
# 設定科研風格 
sns.set(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False 
 
# 讀取數據 
df = pd.read_csv('./results/csv/time/time_2.csv')
 
# 創建並列圖像視窗 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
 
# 自定義顏色和透明度 
palette = sns.color_palette("muted", n_colors=len(df['Algorithm'].unique()))  # 高對比調色板 
scatter_alpha = 0.7  # 散點透明度 
 
# --------------------------
# 左圖：加密時間分析（散點）
# --------------------------
 
# 疊加原始數據散點 
sns.stripplot(data=df, x='Data Length', y='Encryption Time', hue='Algorithm',
              palette=palette, dodge=True, alpha=scatter_alpha, ax=ax1,
              jitter=0.2, edgecolor='gray', linewidth=0.2, size=5)
 
ax1.set_title('Encryption Time Distribution (Max 1s)', fontweight='bold', pad=15)
ax1.set_xlabel('Data Length', fontweight='bold')
ax1.set_ylabel('Time (s)', fontweight='bold')
ax1.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
ax1.legend_.remove()
 
# --------------------------
# 右圖：解密時間分析（散點）
# --------------------------
 
# 疊加原始數據散點 
sns.stripplot(data=df, x='Data Length', y='Decryption Time', hue='Algorithm',
              palette=palette, dodge=True, alpha=scatter_alpha, ax=ax2,
              jitter=0.2, edgecolor='gray', linewidth=0.2, size=5)
 
ax2.set_title('Decryption Time Distribution', fontweight='bold', pad=15)
ax2.set_xlabel('Data Length', fontweight='bold')
ax2.set_ylabel('')
ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
 
# --------------------------
# 統一調整 
# --------------------------
y_max = 1
ax1.set_ylim(0, y_max)
ax2.set_ylim(0, 0.5)
ax1.set_yticks(np.linspace(0, y_max, 21))
ax2.set_yticks(np.linspace(0, 0.5, 21))
 
# 添加全局圖例 
handles, labels = ax1.get_legend_handles_labels()
unique_labels = list(df['Algorithm'].unique())
unique_handles = [handles[labels.index(label)] for label in unique_labels]
 
# 調整佈局並保存 
plt.tight_layout()
output_dir = './results/img'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'time_result.png'),
           dpi=1200, bbox_inches='tight')
plt.show()