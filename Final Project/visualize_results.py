"""
Generate visualization charts for model comparison results
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load results
with open('model_comparison_report.json', 'r') as f:
    report = json.load(f)

df = pd.DataFrame(report['models'])

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# 1. Mean Haversine Distance Comparison (Bar Chart)
ax1 = plt.subplot(2, 3, 1)
colors = ['#2ecc71' if model == report['best_model'] else '#3498db' for model in df['Model']]
bars = ax1.bar(df['Model'], df['Mean Haversine (km)'], color=colors, alpha=0.8)
ax1.set_title('Mean Haversine Distance (Lower is Better)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Distance (km)')
ax1.set_xlabel('Model')
ax1.tick_params(axis='x', rotation=45)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 2. Accuracy Comparison (Within 1km and 2km)
ax2 = plt.subplot(2, 3, 2)
x = np.arange(len(df['Model']))
width = 0.35
bars1 = ax2.bar(x - width/2, df['Within 1km (%)'], width, label='Within 1km', alpha=0.8, color='#e74c3c')
bars2 = ax2.bar(x + width/2, df['Within 2km (%)'], width, label='Within 2km', alpha=0.8, color='#9b59b6')
ax2.set_title('Prediction Accuracy (Higher is Better)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)')
ax2.set_xlabel('Model')
ax2.set_xticks(x)
ax2.set_xticklabels(df['Model'], rotation=45)
ax2.legend()
ax2.set_ylim([95, 100])

# 3. R² Score Comparison
ax3 = plt.subplot(2, 3, 3)
x = np.arange(len(df['Model']))
bars1 = ax3.bar(x - width/2, df['Lon R²'], width, label='Longitude R²', alpha=0.8, color='#f39c12')
bars2 = ax3.bar(x + width/2, df['Lat R²'], width, label='Latitude R²', alpha=0.8, color='#1abc9c')
ax3.set_title('R² Scores (Higher is Better)', fontsize=12, fontweight='bold')
ax3.set_ylabel('R² Score')
ax3.set_xlabel('Model')
ax3.set_xticks(x)
ax3.set_xticklabels(df['Model'], rotation=45)
ax3.legend()
ax3.set_ylim([0.6, 0.9])

# 4. Training Time Comparison
ax4 = plt.subplot(2, 3, 4)
colors_time = ['#e67e22' if time < 35 else '#95a5a6' for time in df['Training Time (s)']]
bars = ax4.bar(df['Model'], df['Training Time (s)'], color=colors_time, alpha=0.8)
ax4.set_title('Training Time (Lower is Better)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Time (seconds)')
ax4.set_xlabel('Model')
ax4.tick_params(axis='x', rotation=45)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}s', ha='center', va='bottom', fontsize=9)

# 5. Median vs P90 Haversine Distance
ax5 = plt.subplot(2, 3, 5)
x = np.arange(len(df['Model']))
bars1 = ax5.bar(x - width/2, df['Median Haversine (km)'], width, label='Median', alpha=0.8, color='#3498db')
bars2 = ax5.bar(x + width/2, df['P90 Haversine (km)'], width, label='P90', alpha=0.8, color='#e74c3c')
ax5.set_title('Distance Distribution (Median vs P90)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Distance (km)')
ax5.set_xlabel('Model')
ax5.set_xticks(x)
ax5.set_xticklabels(df['Model'], rotation=45)
ax5.legend()

# 6. Overall Performance Score (Combined Metric)
ax6 = plt.subplot(2, 3, 6)
# Normalize metrics (lower Haversine is better, higher accuracy is better)
haversine_norm = 1 - (df['Mean Haversine (km)'] / df['Mean Haversine (km)'].max())
accuracy_norm = df['Within 2km (%)'] / 100
r2_norm = (df['Lon R²'] + df['Lat R²']) / 2
performance_score = (haversine_norm * 0.4 + accuracy_norm * 0.4 + r2_norm * 0.2) * 100

colors_perf = ['#2ecc71' if model == report['best_model'] else '#3498db' for model in df['Model']]
bars = ax6.bar(df['Model'], performance_score, color=colors_perf, alpha=0.8)
ax6.set_title('Overall Performance Score', fontsize=12, fontweight='bold')
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
table_data.append(['Model', 'Mean Haversine\n(km)', 'Within 1km\n(%)', 'Within 2km\n(%)', 
                   'Lon R²', 'Lat R²', 'Training Time\n(s)'])

for _, row in df.iterrows():
    table_data.append([
        row['Model'],
        f"{row['Mean Haversine (km)']:.4f}",
        f"{row['Within 1km (%)']:.2f}",
        f"{row['Within 2km (%)']:.2f}",
        f"{row['Lon R²']:.4f}",
        f"{row['Lat R²']:.4f}",
        f"{row['Training Time (s)']:.1f}"
    ])

# Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(7):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best model row
best_row = df[df['Model'] == report['best_model']].index[0] + 1
for i in range(7):
    table[(best_row, i)].set_facecolor('#d5f4e6')

# Alternate row colors
for i in range(1, len(table_data)):
    if i != best_row:
        for j in range(7):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

plt.title('Model Performance Comparison Table', fontsize=16, fontweight='bold', pad=20)
plt.savefig('model_comparison_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison_table.png")

# Create a simple bar chart showing best model
fig3, ax = plt.subplots(figsize=(10, 6))
metrics = ['Mean Haversine', 'Within 1km%', 'Within 2km%', 'Avg R²', 'Speed Score']
best_idx = df[df['Model'] == report['best_model']].index[0]

# Normalize all metrics to 0-100 scale for comparison
best_scores = [
    100 * (1 - df.loc[best_idx, 'Mean Haversine (km)'] / df['Mean Haversine (km)'].max()),
    df.loc[best_idx, 'Within 1km (%)'],
    df.loc[best_idx, 'Within 2km (%)'],
    100 * ((df.loc[best_idx, 'Lon R²'] + df.loc[best_idx, 'Lat R²']) / 2),
    100 * (1 - df.loc[best_idx, 'Training Time (s)'] / df['Training Time (s)'].max())
]

colors_radar = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
bars = ax.barh(metrics, best_scores, color=colors_radar, alpha=0.8)
ax.set_xlabel('Score (0-100)', fontsize=12)
ax.set_title(f'Best Model: {report["best_model"]} - Performance Breakdown', 
             fontsize=14, fontweight='bold')
ax.set_xlim([0, 105])

for i, (bar, score) in enumerate(zip(bars, best_scores)):
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
            f'{score:.1f}', ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('best_model_breakdown.png', dpi=300, bbox_inches='tight')
print("✓ Saved: best_model_breakdown.png")

print("\n✓ All visualizations generated successfully!")
print(f"\nBest Model: {report['best_model']}")
print(f"Mean Haversine Distance: {report['best_haversine_km']:.4f} km")
