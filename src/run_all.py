"""
Complete Experimental Pipeline - FIXED VERSION
Includes robust label conversion handling
"""
import os

# Use most of your 32 logical CPUs
THREADS = "24"  # or "28" if the system stays responsive

os.environ["OMP_NUM_THREADS"] = THREADS        # NumPy/Scikit-learn (MKL/OpenMP)
os.environ["MKL_NUM_THREADS"] = THREADS        # Explicitly cap MKL too
os.environ["NUMEXPR_NUM_THREADS"] = THREADS    # If numexpr is used

# TensorFlow CPU thread pools (matters mostly if you run TF on CPU)
os.environ["TF_NUM_INTRAOP_THREADS"] = THREADS
os.environ["TF_NUM_INTEROP_THREADS"] = "2"     # keep inter-op small; 1–4 is typical

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

from phishing_detector import PhishingDetector
from phishing_lstm import AdvancedPhishingDetector


def convert_labels_robust(labels):
    """
    Robustly convert labels to binary format
    Handles various input formats
    """
    print("\n[DEBUG] Label Conversion")
    print(f"  Original dtype: {labels.dtype}")
    print(f"  Original unique values: {np.unique(labels)}")
    print(f"  Sample values: {labels[:5]}")
    
    # Get unique values
    unique_vals = np.unique(labels)
    n_unique = len(unique_vals)
    
    print(f"  Number of unique values: {n_unique}")
    
    if n_unique != 2:
        raise ValueError(f"Expected 2 classes, found {n_unique}: {unique_vals}")
    
    # Sort to ensure consistent mapping
    unique_vals = sorted(unique_vals)
    
    # Determine which is phishing
    # Check if labels are already 0/1
    if set(unique_vals) == {0, 1}:
        print("  Labels are already 0/1")
        binary_labels = labels.astype(int)
    # Check if labels contain 'phishing' string
    elif any('phish' in str(val).lower() for val in unique_vals):
        print("  Detected string labels containing 'phish'")
        phishing_val = [val for val in unique_vals if 'phish' in str(val).lower()][0]
        print(f"  Mapping '{phishing_val}' -> 1 (phishing)")
        print(f"  Mapping other value -> 0 (legitimate)")
        binary_labels = (labels == phishing_val).astype(int)
    else:
        # Default: map second unique value to phishing (1)
        print(f"  Using default mapping:")
        print(f"  Mapping '{unique_vals[0]}' -> 0 (legitimate)")
        print(f"  Mapping '{unique_vals[1]}' -> 1 (phishing)")
        binary_labels = (labels == unique_vals[1]).astype(int)
    
    # Verify the conversion
    converted_unique = np.unique(binary_labels)
    print(f"  After conversion unique values: {converted_unique}")
    
    if len(converted_unique) != 2:
        raise ValueError(f"Conversion failed! Got {len(converted_unique)} classes")
    
    class_counts = np.bincount(binary_labels)
    print(f"  Class 0 (legitimate): {class_counts[0]}")
    print(f"  Class 1 (phishing): {class_counts[1]}")
    print(f"  [OK] Label conversion successful!")
    
    return binary_labels


def run_complete_pipeline(data_path='phishing_dataset.csv'):
    """
    Run complete experimental pipeline including:
    - Traditional ML (SVM, Logistic Regression)
    - Feedforward Neural Networks
    - Advanced DL (LSTM, CNN, Hybrid)
    - Generate all visualizations and result files
    """
    
    print("="*80)
    print("PHISHING DETECTION - COMPLETE EXPERIMENTAL PIPELINE")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\n[1/5] Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"   Total URLs: {len(df)}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Determine column names
    print("\n[DEBUG] Identifying columns...")
    
    # Find URL column
    url_col = None
    for col in df.columns:
        if 'url' in col.lower():
            url_col = col
            break
    if url_col is None:
        url_col = df.columns[0]
        print(f"  Warning: No 'url' column found, using first column: '{url_col}'")
    
    # Find label column (check 'label' first, it's most specific)
    label_col = None
    for col in df.columns:
        if 'label' in col.lower():
            label_col = col
            break
    if label_col is None:
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['class', 'target']):
                label_col = col
                break
    if label_col is None:
        label_col = df.columns[1]
        print(f"  Warning: No label column found, using second column: '{label_col}'")
    
    print(f"  [OK] Using URL column: '{url_col}'")
    print(f"  [OK] Using Label column: '{label_col}'")
    
    # Extract data
    urls = df[url_col].values
    raw_labels = df[label_col].values
    
    # Convert labels to binary with robust method
    try:
        labels = convert_labels_robust(raw_labels)
    except Exception as e:
        print(f"\n[ERROR] Label conversion failed: {e}")
        print("\nDebugging information:")
        print(f"  Raw label sample: {raw_labels[:10]}")
        print(f"  Raw label dtype: {raw_labels.dtype}")
        print(f"  Unique raw labels: {np.unique(raw_labels)}")
        raise
    
    # Part 1: Traditional ML and Basic Deep Learning
    print("\n[2/5] Running Traditional ML and Feedforward NN...")
    detector = PhishingDetector()
    
    # Create a temporary dataframe with corrected labels
    temp_df = pd.DataFrame({
        url_col: urls,
        label_col: labels
    })
    
    X_train, X_test, y_train, y_test, _ = detector.preprocess_data(
        temp_df, url_column=url_col, label_column=label_col
    )
    
    # Verify labels in train/test sets
    print(f"\n[DEBUG] Train/Test Label Distribution")
    print(f"  y_train unique: {np.unique(y_train)}")
    print(f"  y_train distribution: {np.bincount(y_train)}")
    print(f"  y_test unique: {np.unique(y_test)}")
    print(f"  y_test distribution: {np.bincount(y_test)}")
    
    # Train models
    detector.train_deep_learning(X_train, X_test, y_train, y_test, model_type='feedforward')
    detector.train_deep_learning(X_train, X_test, y_train, y_test, model_type='deep')
    detector.train_svm(X_train, X_test, y_train, y_test)
    detector.train_logistic_regression(X_train, X_test, y_train, y_test)
    
    basic_results = detector.print_results()
    detector.plot_results()
    
    # Part 2: Advanced Deep Learning
    print("\n[3/5] Running Advanced Deep Learning Models...")
    adv_detector = AdvancedPhishingDetector(max_url_length=200)
    advanced_results_dict = adv_detector.run_all_experiments(urls, labels)
    advanced_results = adv_detector.print_comparison()
    
    # Part 3: Combine all results
    print("\n[4/5] Combining and saving results...")
    
    all_results = pd.concat([basic_results, advanced_results])
    all_results.to_csv('complete_results.csv')
    print("   Saved: complete_results.csv")
    
    # Create comprehensive comparison plot
    create_comprehensive_plots(detector.results, advanced_results_dict)
    
    # Part 4: Generate summary statistics
    print("\n[5/5] Generating summary report...")
    summary = generate_summary(all_results, detector, adv_detector)
    
    with open('experiment_summary.txt', 'w') as f:
        f.write(summary)
    print("   Saved: experiment_summary.txt")
    
    # Save results as JSON
    results_json = {
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'total_samples': len(df),
            'legitimate': int(np.sum(labels == 0)),
            'phishing': int(np.sum(labels == 1)),
            'training_samples': len(y_train),
            'testing_samples': len(y_test)
        },
        'results': all_results.to_dict()
    }
    
    with open('results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print("   Saved: results.json")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated Files:")
    print("  - complete_results.csv: All model metrics")
    print("  - phishing_detection_results.png: Basic model comparison")
    print("  - comprehensive_comparison.png: All models comparison")
    print("  - experiment_summary.txt: Text summary for report")
    print("  - results.json: Structured results data")
    
    return all_results


def create_comprehensive_plots(basic_results, advanced_results):
    """Create comprehensive comparison plots"""
    
    # Combine all results
    all_models = {}
    all_models.update(basic_results)
    all_models.update(advanced_results)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Overall metrics comparison
    ax1 = plt.subplot(2, 3, 1)
    metrics_df = pd.DataFrame({
        model: {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score']
        }
        for model, metrics in all_models.items()
    })
    metrics_df.T.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_ylim([0.94, 1.0])
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. ROC-AUC comparison
    ax2 = plt.subplot(2, 3, 2)
    roc_scores = pd.Series({model: metrics['roc_auc'] for model, metrics in all_models.items()})
    roc_scores.sort_values(ascending=False).plot(kind='barh', ax=ax2, color='steelblue')
    ax2.set_title('ROC-AUC Scores', fontsize=12, fontweight='bold')
    ax2.set_xlabel('ROC-AUC Score')
    ax2.set_xlim([0.98, 1.0])
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. F1-Score ranking
    ax3 = plt.subplot(2, 3, 3)
    f1_scores = pd.Series({model: metrics['f1_score'] for model, metrics in all_models.items()})
    f1_scores.sort_values(ascending=True).plot(kind='barh', ax=ax3, color='coral')
    ax3.set_title('F1-Score Ranking', fontsize=12, fontweight='bold')
    ax3.set_xlabel('F1-Score')
    ax3.set_xlim([0.94, 1.0])
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Precision vs Recall scatter
    ax4 = plt.subplot(2, 3, 4)
    precisions = [metrics['precision'] for metrics in all_models.values()]
    recalls = [metrics['recall'] for metrics in all_models.values()]
    model_names = list(all_models.keys())
    
    scatter = ax4.scatter(precisions, recalls, s=100, alpha=0.6, c=range(len(model_names)), cmap='viridis')
    for i, name in enumerate(model_names):
        ax4.annotate(name, (precisions[i], recalls[i]), fontsize=7, 
                    xytext=(5, 5), textcoords='offset points')
    ax4.set_title('Precision vs Recall Trade-off', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Precision')
    ax4.set_ylabel('Recall')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0.94, 1.0])
    ax4.set_ylim([0.94, 1.0])
    
    # 5. Model category comparison
    ax5 = plt.subplot(2, 3, 5)
    categories = {
        'Traditional ML': ['svm', 'logistic_regression'],
        'Feedforward NN': ['feedforward', 'deep'],
        'Sequential DL': ['LSTM', 'BiLSTM', 'CNN'],
        'Hybrid': ['Hybrid']
    }
    
    category_scores = {}
    for cat, models in categories.items():
        scores = [all_models[m]['f1_score'] for m in models if m in all_models]
        if scores:
            category_scores[cat] = np.mean(scores)
    
    pd.Series(category_scores).plot(kind='bar', ax=ax5, color='lightgreen')
    ax5.set_title('Average F1-Score by Model Category', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Average F1-Score')
    ax5.set_ylim([0.96, 1.0])
    ax5.grid(True, alpha=0.3, axis='y')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 6. Error rates comparison
    ax6 = plt.subplot(2, 3, 6)
    error_rates = pd.DataFrame({
        model: {
            'False Positive Rate': 1 - metrics['precision'],
            'False Negative Rate': 1 - metrics['recall']
        }
        for model, metrics in all_models.items()
    })
    error_rates.T.plot(kind='bar', ax=ax6, width=0.8)
    ax6.set_title('Error Rates Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Error Rate')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print("   Saved: comprehensive_comparison.png")
    plt.close()


def generate_summary(results_df, basic_detector, adv_detector):
    """Generate text summary for the report"""
    
    summary = []
    summary.append("="*80)
    summary.append("EXPERIMENTAL RESULTS SUMMARY")
    summary.append("="*80)
    summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")
    
    # Best models
    summary.append("BEST PERFORMING MODELS")
    summary.append("-"*80)
    
    best_f1 = results_df['f1_score'].idxmax()
    best_acc = results_df['accuracy'].idxmax()
    best_roc = results_df['roc_auc'].idxmax()
    
    summary.append(f"Best F1-Score: {best_f1} ({results_df.loc[best_f1, 'f1_score']:.4f})")
    summary.append(f"Best Accuracy: {best_acc} ({results_df.loc[best_acc, 'accuracy']:.4f})")
    summary.append(f"Best ROC-AUC: {best_roc} ({results_df.loc[best_roc, 'roc_auc']:.4f})")
    summary.append("")
    
    # Detailed results table
    summary.append("DETAILED RESULTS")
    summary.append("-"*80)
    summary.append(results_df.to_string())
    summary.append("")
    
    # Performance improvements
    summary.append("PERFORMANCE IMPROVEMENTS OVER BASELINES")
    summary.append("-"*80)
    
    if 'logistic_regression' in results_df.index and best_f1 != 'logistic_regression':
        baseline_f1 = results_df.loc['logistic_regression', 'f1_score']
        best_f1_score = results_df.loc[best_f1, 'f1_score']
        improvement = ((best_f1_score - baseline_f1) / baseline_f1) * 100
        summary.append(f"{best_f1} vs Logistic Regression:")
        summary.append(f"  F1-Score improvement: {improvement:.2f}%")
        summary.append(f"  Absolute difference: {best_f1_score - baseline_f1:.4f}")
    
    if 'svm' in results_df.index and best_f1 != 'svm':
        baseline_f1 = results_df.loc['svm', 'f1_score']
        best_f1_score = results_df.loc[best_f1, 'f1_score']
        improvement = ((best_f1_score - baseline_f1) / baseline_f1) * 100
        summary.append(f"\n{best_f1} vs SVM:")
        summary.append(f"  F1-Score improvement: {improvement:.2f}%")
        summary.append(f"  Absolute difference: {best_f1_score - baseline_f1:.4f}")
    
    summary.append("")
    summary.append("="*80)
    
    return "\n".join(summary)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = 'phishing_dataset.csv'
        print(f"No data path provided, using default: {data_path}")
        print("Usage: python run_all_fixed.py <path_to_dataset.csv>")
        print()
    
    try:
        results = run_complete_pipeline(data_path)
        print("\n[OK] All experiments completed successfully!")
        print("\nNext steps:")
        print("1. Review 'experiment_summary.txt' for report content")
        print("2. Check 'comprehensive_comparison.png' for visualizations")
        print("3. Use 'complete_results.csv' for creating tables in your report")
        
    except FileNotFoundError:
        print(f"\n[ERROR] Error: Could not find dataset at '{data_path}'")
        print("\nPlease ensure you have downloaded the StealthPhisher dataset")
        
    except Exception as e:
        print(f"\n[ERROR] Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTry running debug_labels.py first to diagnose the issue:")
        print(f"  python debug_labels.py {data_path}")
