"""
Generate visualization charts for model comparison results (compatible with NTUST_ML codebase)
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load results from baseline_metrics.csv
metrics_df = pd.read_csv('baseline_metrics.csv')

# For compatibility, create synthetic columns for missing advanced metrics
metrics_df['Model'] = metrics_df['model']
metrics_df['Mean Haversine (km)'] = np.nan  # Placeholder
metrics_df['Median Haversine (km)'] = np.nan
metrics_df['P90 Haversine (km)'] = np.nan
metrics_df['Within 1km (%)'] = np.nan
metrics_df['Within 2km (%)'] = np.nan
metrics_df['Lon R²'] = metrics_df['r2_train']
metrics_df['Lat R²'] = metrics_df['r2_val']
metrics_df['Training Time (s)'] = np.nan

# Use R² train/val as proxies for Lon/Lat R²
# If you have more advanced metrics, you can merge them here

# For best model, use the one with highest r2_val
best_idx = metrics_df['r2_val'].idxmax()
best_model = metrics_df.loc[best_idx, 'Model']

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# 1. R² Score Comparison (since Haversine is not available)
ax1 = plt.subplot(2, 3, 1)
bars1 = ax1.bar(metrics_df['Model'], metrics_df['Lon R²'], label='Train R²', alpha=0.8, color='#f39c12')
bars2 = ax1.bar(metrics_df['Model'], metrics_df['Lat R²'], label='Val R²', alpha=0.8, color='#1abc9c', bottom=metrics_df['Lon R²'])
ax1.set_title('R² Scores (Higher is Better)', fontsize=12, fontweight='bold')
ax1.set_ylabel('R² Score')
ax1.set_xlabel('Model')
ax1.tick_params(axis='x', rotation=45)
ax1.legend()
ax1.set_ylim([0, 2])
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 2. Placeholder for accuracy (not available)
ax2 = plt.subplot(2, 3, 2)
ax2.text(0.5, 0.5, 'Accuracy metrics not available', ha='center', va='center', fontsize=14)
ax2.set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
ax2.axis('off')

# 3. Placeholder for Haversine (not available)
ax3 = plt.subplot(2, 3, 3)
ax3.text(0.5, 0.5, 'Haversine metrics not available', ha='center', va='center', fontsize=14)
ax3.set_title('Mean Haversine Distance', fontsize=12, fontweight='bold')
ax3.axis('off')

# 4. Placeholder for training time (not available)
ax4 = plt.subplot(2, 3, 4)
ax4.text(0.5, 0.5, 'Training time not available', ha='center', va='center', fontsize=14)
ax4.set_title('Training Time', fontsize=12, fontweight='bold')
ax4.axis('off')

# 5. Placeholder for Median vs P90 Haversine
ax5 = plt.subplot(2, 3, 5)
ax5.text(0.5, 0.5, 'Median/P90 Haversine not available', ha='center', va='center', fontsize=14)
ax5.set_title('Distance Distribution', fontsize=12, fontweight='bold')
ax5.axis('off')

# 6. Overall Performance Score (use r2_val as proxy)
ax6 = plt.subplot(2, 3, 6)
performance_score = metrics_df['r2_val'] * 100
colors_perf = ['#2ecc71' if model == best_model else '#3498db' for model in metrics_df['Model']]
bars = ax6.bar(metrics_df['Model'], performance_score, color=colors_perf, alpha=0.8)
ax6.set_title('Overall Performance Score (R² Val)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Score (0-100)')
ax6.set_xlabel('Model')
ax6.tick_params(axis='x', rotation=45)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison_charts.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison_charts.png")

# Create a detailed comparison table image
fig2, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['Model', 'Train R²', 'Val R²'])
for _, row in metrics_df.iterrows():
    table_data.append([
        row['Model'],
        f"{row['Lon R²']:.4f}",
        f"{row['Lat R²']:.4f}",
    ])

# Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.3, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(3):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best model row
best_row = best_idx + 1
for i in range(3):
    table[(best_row, i)].set_facecolor('#d5f4e6')

# Alternate row colors
for i in range(1, len(table_data)):
    if i != best_row:
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

plt.title('Model Performance Comparison Table', fontsize=16, fontweight='bold', pad=20)
plt.savefig('model_comparison_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison_table.png")

# Create a simple bar chart showing best model
fig3, ax = plt.subplots(figsize=(10, 6))
metrics = ['Train R²', 'Val R²']
best_scores = [metrics_df.loc[best_idx, 'Lon R²'] * 100, metrics_df.loc[best_idx, 'Lat R²'] * 100]
colors_radar = ['#f39c12', '#1abc9c']
bars = ax.barh(metrics, best_scores, color=colors_radar, alpha=0.8)
ax.set_xlabel('Score (0-100)', fontsize=12)
ax.set_title(f'Best Model: {best_model} - Performance Breakdown', fontsize=14, fontweight='bold')
ax.set_xlim([0, 105])
for i, (bar, score) in enumerate(zip(bars, best_scores)):
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
            f'{score:.1f}', ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('best_model_breakdown.png', dpi=300, bbox_inches='tight')
print("✓ Saved: best_model_breakdown.png")

print("\n✓ All visualizations generated successfully!")
print(f"\nBest Model: {best_model}")
print(f"Val R²: {metrics_df.loc[best_idx, 'Lat R²']:.4f}")