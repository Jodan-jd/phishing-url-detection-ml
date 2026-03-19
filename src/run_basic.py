"""
Fixed version that auto-detects column names
"""
import os
# Set BEFORE importing TensorFlow
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['TF_NUM_INTRAOP_THREADS'] = '24'
os.environ['TF_NUM_INTEROP_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '24'

import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime

# Configure TensorFlow to use all threads
tf.config.threading.set_intra_op_parallelism_threads(24)
tf.config.threading.set_inter_op_parallelism_threads(12)

from phishing_detector import PhishingDetector

print("="*80)
print("FIXED VERSION - Auto-detects column names")
print("="*80)

# Load data and inspect columns
print("\n[Step 1] Loading and inspecting dataset...")
df = pd.read_csv('phishing_dataset.csv')
print(f"Dataset loaded: {len(df)} records")
print(f"Columns found: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head(3))

# Auto-detect URL column
url_col = None
for col in df.columns:
    if 'url' in col.lower():
        url_col = col
        break
if url_col is None:
    url_col = df.columns[0]
    print(f"\nWarning: No 'url' column found, using first column: '{url_col}'")
else:
    print(f"\nDetected URL column: '{url_col}'")

# Auto-detect label column
label_col = None
for col in df.columns:
    if 'label' in col.lower():
        label_col = col
        break
if label_col is None:
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['class', 'target', 'type']):
            label_col = col
            break
if label_col is None:
    label_col = df.columns[1]
    print(f"Warning: No label column found, using second column: '{label_col}'")
else:
    print(f"Detected label column: '{label_col}'")

# Show label distribution
print(f"\nLabel distribution:")
print(df[label_col].value_counts())

# Rename columns for compatibility
df_renamed = df[[url_col, label_col]].copy()
df_renamed.columns = ['url', 'label']

# Convert labels to binary if needed
unique_labels = df_renamed['label'].unique()
print(f"\nUnique labels: {unique_labels}")

if len(unique_labels) != 2:
    raise ValueError(f"Expected 2 classes, found {len(unique_labels)}")

# Ensure labels are 0 and 1
if set(unique_labels) != {0, 1}:
    print(f"Converting labels to 0/1...")
    # Assume first unique value is legitimate (0), second is phishing (1)
    sorted_labels = sorted(unique_labels)
    df_renamed['label'] = (df_renamed['label'] == sorted_labels[1]).astype(int)
    print(f"Mapped {sorted_labels[0]} -> 0 (legitimate)")
    print(f"Mapped {sorted_labels[1]} -> 1 (phishing)")

print(f"\nFinal label distribution:")
print(f"  Legitimate (0): {(df_renamed['label'] == 0).sum()}")
print(f"  Phishing (1): {(df_renamed['label'] == 1).sum()}")

# Save the cleaned dataset
df_renamed.to_csv('phishing_dataset_cleaned.csv', index=False)
print(f"\nSaved cleaned dataset to 'phishing_dataset_cleaned.csv'")

# Now run the detector
print("\n" + "="*80)
print("[Step 2] Training models...")
print("="*80)

detector = PhishingDetector()
X_train, X_test, y_train, y_test, _ = detector.preprocess_data(
    df_renamed, url_column='url', label_column='label'
)

# Train all basic models
print("\n[1/4] Training Feedforward NN...")
start = datetime.now()
detector.train_deep_learning(X_train, X_test, y_train, y_test, 'feedforward')
print(f"   Completed in: {(datetime.now() - start).total_seconds():.0f} seconds")

print("\n[2/4] Training Deep NN...")
start = datetime.now()
detector.train_deep_learning(X_train, X_test, y_train, y_test, 'deep')
print(f"   Completed in: {(datetime.now() - start).total_seconds():.0f} seconds")

print("\n[3/4] Training SVM...")
start = datetime.now()
detector.train_svm(X_train, X_test, y_train, y_test)
print(f"   Completed in: {(datetime.now() - start).total_seconds():.0f} seconds")

print("\n[4/4] Training Logistic Regression...")
start = datetime.now()
detector.train_logistic_regression(X_train, X_test, y_train, y_test)
print(f"   Completed in: {(datetime.now() - start).total_seconds():.0f} seconds")

# Results
results_df = detector.print_results()
detector.plot_results()

# Save
results_df.to_csv('complete_results.csv')
print("\n✓ Results saved to complete_results.csv")
print("✓ Chart saved to phishing_detection_results.png")

# Create F1 Score Comparison Chart
print("\nGenerating F1 Score comparison chart...")
import matplotlib.pyplot as plt
import seaborn as sns

# Debug: Print column names
print(f"Available columns: {results_df.columns.tolist()}")

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# Extract F1 scores - handle different possible column names
models = list(results_df.index)
if 'F1-Score' in results_df.columns:
    f1_scores = results_df['F1-Score'].values
elif 'f1_score' in results_df.columns:
    f1_scores = results_df['f1_score'].values
elif 'F1' in results_df.columns:
    f1_scores = results_df['F1'].values
else:
    raise KeyError(f"Could not find F1 score column. Available columns: {results_df.columns.tolist()}")

# Create color map (best model gets special color)
colors = ['#2ecc71' if f1 == max(f1_scores) else '#3498db' for f1 in f1_scores]

# Create bar chart
bars = plt.bar(models, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on top of bars
for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{score:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Customize chart
plt.title('F1-Score Comparison Across Models', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylim([min(f1_scores) - 0.01, max(f1_scores) + 0.01])

# Add grid
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add horizontal line at 0.99 for reference
plt.axhline(y=0.99, color='red', linestyle='--', linewidth=1, alpha=0.5, label='0.99 threshold')

# Add legend
best_model = models[f1_scores.argmax()]
plt.legend([f'Best Model: {best_model}', '0.99 threshold'], loc='lower right')

# Tight layout
plt.tight_layout()

# Save
plt.savefig('f1_score_comparison.png', dpi=300, bbox_inches='tight')
print("✓ F1-Score chart saved to f1_score_comparison.png")
plt.close()

# Create a second detailed comparison chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Chart 1: All metrics comparison for each model
ax1 = axes[0]
# Use actual column names from results_df
metrics_to_plot = [col for col in results_df.columns if col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'accuracy', 'precision', 'recall', 'f1_score']]
x = np.arange(len(models))
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    offset = (i - 1.5) * width
    values = results_df[metric].values
    ax1.bar(x + offset, values, width, label=metric.capitalize(), alpha=0.8)

ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
ax1.set_title('All Metrics Comparison', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.legend(loc='lower right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([min(results_df[metrics_to_plot].min()) - 0.01, 1.0])

# Chart 2: F1-Score ranking with error bars showing precision/recall spread
ax2 = axes[1]
sorted_indices = f1_scores.argsort()[::-1]  # Sort descending
sorted_models = [models[i] for i in sorted_indices]
sorted_f1 = [f1_scores[i] for i in sorted_indices]

# Get precision and recall with correct column names
precision_col = 'Precision' if 'Precision' in results_df.columns else 'precision'
recall_col = 'Recall' if 'Recall' in results_df.columns else 'recall'
sorted_precision = [results_df[precision_col].values[i] for i in sorted_indices]
sorted_recall = [results_df[recall_col].values[i] for i in sorted_indices]

# Calculate error (distance from F1 to precision and recall)
yerr_lower = [f1 - min(p, r) for f1, p, r in zip(sorted_f1, sorted_precision, sorted_recall)]
yerr_upper = [max(p, r) - f1 for f1, p, r in zip(sorted_f1, sorted_precision, sorted_recall)]

bars2 = ax2.barh(sorted_models, sorted_f1, 
                 color=['#2ecc71' if i == 0 else '#3498db' for i in range(len(sorted_models))],
                 alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, score) in enumerate(zip(bars2, sorted_f1)):
    width = bar.get_width()
    ax2.text(width + 0.003, bar.get_y() + bar.get_height()/2.,
             f'{score:.4f}',
             ha='left', va='center', fontsize=10, fontweight='bold')

ax2.set_xlabel('F1-Score', fontsize=11, fontweight='bold')
ax2.set_title('F1-Score Ranking (Best to Worst)', fontsize=13, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.set_xlim([min(sorted_f1) - 0.01, max(sorted_f1) + 0.015])

plt.tight_layout()
plt.savefig('detailed_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Detailed metrics chart saved to detailed_metrics_comparison.png")
plt.close()

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. complete_results.csv - All metrics in table format")
print("  2. phishing_detection_results.png - Original 4-panel comparison")
print("  3. f1_score_comparison.png - F1-Score bar chart")
print("  4. detailed_metrics_comparison.png - Comprehensive metrics view")
print("="*80)
