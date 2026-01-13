"""
STANDARDIZATION_TEST.py

Tests robustness of model to different standardization approaches:
- Z-score normalization (mean=0, std=1)
- Min-Max scaling (0-1 range)
- Robust scaling (median-based, resistant to outliers)
- No scaling (raw features)

Evaluates which standardization method is most robust for this dataset.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_combined_data(filepath='data/combined_all_hourly_data.csv'):
    """Load combined hourly data"""
    print("Loading combined hourly data...")
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"  Loaded {len(df)} hourly measurements")
    return df

def extract_daily_features(df):
    """Extract daily-level features from hourly data"""
    print("\nExtracting daily features...")

    df['day'] = df['datetime'].dt.date

    daily_features = []

    for (patient_id, day), group in df.groupby(['id', 'day']):
        steps = group['steps'].values

        features = {
            'id': patient_id,
            'day': day,
            'label': group['label'].iloc[0],
            'mean_steps': np.mean(steps),
            'std_steps': np.std(steps),
            'max_steps': np.max(steps),
            'min_steps': np.min(steps),
            'total_steps': np.sum(steps),
            'cv_steps': np.std(steps) / (np.mean(steps) + 1e-8),
            'num_hours': len(steps),
        }

        daily_features.append(features)

    df_daily = pd.DataFrame(daily_features)
    print(f"  Extracted {len(df_daily)} daily feature vectors")
    return df_daily

def evaluate_with_scaler(X_train, X_test, y_train, y_test, scaler, scaler_name):
    """
    Train and evaluate model with specific scaler
    """
    # Scale features
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Calculate class weights
    n_ms = np.sum(y_train == 0)
    n_healthy = np.sum(y_train == 1)
    total = n_ms + n_healthy
    class_weight = {
        0: total / (2 * n_ms),
        1: total / (2 * n_healthy)
    }

    # Train model
    model = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Evaluate
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    print(f"  {scaler_name:20s} | AUC: {auc:.3f} | Precision: {precision:.3f} | "
          f"Recall: {recall:.3f} | F1: {f1:.3f}")

    return {
        'scaler': scaler_name,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'tpr': tpr
    }

def cross_validate_scalers(X, y, groups, scaler_dict, n_splits=5):
    """
    Cross-validation comparison of different scalers

    FIXED: Uses GroupKFold to ensure same patient's data never appears in both train and test
    FIXED: Creates fresh scaler instance for each fold to avoid object reuse across folds
    """
    from sklearn.model_selection import GroupKFold

    print(f"\nCross-validation with {n_splits} folds (patient-level)...")

    gkf = GroupKFold(n_splits=n_splits)

    results = []

    for scaler_name, scaler_template in scaler_dict.items():
        aucs = []

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # FIXED: Create fresh scaler instance for each fold to avoid object reuse
            if scaler_template is not None:
                # Create new instance of the same scaler class
                scaler = scaler_template.__class__()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test

            # Calculate class weights
            n_ms = np.sum(y_train == 0)
            n_healthy = np.sum(y_train == 1)
            total = n_ms + n_healthy
            class_weight = {
                0: total / (2 * n_ms),
                1: total / (2 * n_healthy)
            }

            # Train and evaluate
            model = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            aucs.append(auc)

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        results.append({
            'scaler': scaler_name,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'aucs': aucs
        })

        print(f"  {scaler_name:20s} | Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}")

    return results

def plot_scaler_comparison(results, cv_results, output_path='result/standardization_comparison.png'):
    """
    Plot comparison of different standardization methods
    """
    print(f"\nPlotting standardization comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: ROC curves
    ax1 = axes[0, 0]
    for r in results:
        ax1.plot(r['fpr'], r['tpr'], linewidth=2, label=f"{r['scaler']} (AUC={r['auc']:.3f})")

    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves: Different Standardization Methods', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bar plot of test AUCs
    ax2 = axes[0, 1]
    scalers = [r['scaler'] for r in results]
    aucs = [r['auc'] for r in results]
    colors = ['blue', 'green', 'orange', 'red']
    ax2.bar(scalers, aucs, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('ROC-AUC Score', fontsize=12)
    ax2.set_title('Test Set Performance', fontsize=14)
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)

    # Plot 3: Cross-validation results with error bars
    ax3 = axes[1, 0]
    cv_scalers = [r['scaler'] for r in cv_results]
    cv_means = [r['mean_auc'] for r in cv_results]
    cv_stds = [r['std_auc'] for r in cv_results]

    ax3.errorbar(range(len(cv_scalers)), cv_means, yerr=cv_stds,
                fmt='o', markersize=10, capsize=5, capthick=2, linewidth=2)
    ax3.set_xticks(range(len(cv_scalers)))
    ax3.set_xticklabels(cv_scalers, rotation=45, ha='right')
    ax3.set_ylabel('Mean AUC ± SD', fontsize=12)
    ax3.set_title('Cross-Validation Performance (5-Fold)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])

    # Plot 4: Distribution of CV AUCs
    ax4 = axes[1, 1]
    for i, r in enumerate(cv_results):
        ax4.scatter([i]*len(r['aucs']), r['aucs'], alpha=0.6, s=100, color=colors[i])

    ax4.set_xticks(range(len(cv_scalers)))
    ax4.set_xticklabels(cv_scalers, rotation=45, ha='right')
    ax4.set_ylabel('AUC Score', fontsize=12)
    ax4.set_title('Distribution of CV Fold AUCs', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved standardization comparison to: {output_path}")

    return fig

def main():
    """Main standardization testing pipeline"""
    print("\n" + "="*60)
    print("STANDARDIZATION ROBUSTNESS TESTING")
    print("="*60)

    # Create output directory
    Path('result').mkdir(exist_ok=True)

    # Step 1: Load data
    df = load_combined_data()

    # Step 2: Extract daily features
    df_daily = extract_daily_features(df)

    # Step 3: Prepare feature matrix
    feature_cols = ['mean_steps', 'std_steps', 'max_steps', 'min_steps',
                   'total_steps', 'cv_steps', 'num_hours']

    X = df_daily[feature_cols].values
    y = df_daily['label'].values

    # Step 4: Patient-level train/test split
    from sklearn.model_selection import GroupShuffleSplit

    groups = df_daily['id'].values  # CRITICAL: Group by patient ID

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\nDataset split (patient-level):")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Training patients: {len(np.unique(groups[train_idx]))}")
    print(f"  Test patients: {len(np.unique(groups[test_idx]))}")

    # Step 5: Define scalers to test
    scaler_dict = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'No Scaling': None
    }

    # Step 6: Evaluate each scaler on test set
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION")
    print(f"{'='*60}")

    results = []
    for scaler_name, scaler in scaler_dict.items():
        result = evaluate_with_scaler(X_train, X_test, y_train, y_test,
                                     scaler, scaler_name)
        results.append(result)

    # Save test results
    test_results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ['fpr', 'tpr']}
        for r in results
    ])
    test_results_df.to_csv('result/standardization_test_results.csv', index=False)
    print(f"\nSaved test results to: result/standardization_test_results.csv")

    # Step 7: Cross-validation comparison
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION COMPARISON")
    print(f"{'='*60}")

    # Need to create fresh scalers for CV
    scaler_dict_cv = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'No Scaling': None
    }

    # Get groups for CV
    groups = df_daily['id'].values

    cv_results = cross_validate_scalers(X, y, groups, scaler_dict_cv, n_splits=5)

    # Save CV results
    cv_results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'aucs'}
        for r in cv_results
    ])
    cv_results_df.to_csv('result/standardization_cv_results.csv', index=False)
    print(f"\nSaved CV results to: result/standardization_cv_results.csv")

    # Step 8: Plot comparison
    plot_scaler_comparison(results, cv_results)

    # Step 9: Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    best_test = max(results, key=lambda x: x['auc'])
    best_cv = max(cv_results, key=lambda x: x['mean_auc'])

    print(f"\nBest on test set: {best_test['scaler']} (AUC={best_test['auc']:.3f})")
    print(f"Best on cross-validation: {best_cv['scaler']} (Mean AUC={best_cv['mean_auc']:.3f})")

    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")
    print("\nStandardScaler (z-score normalization) is typically recommended for")
    print("logistic regression as it ensures all features have comparable scales.")
    print("\nRobustScaler is better if data has outliers (uses median/IQR instead of mean/std).")
    print("\nIf performance is similar across scalers, the model is robust to standardization.")
    print(f"{'='*60}")

    print("\n✓ Standardization testing complete!")

if __name__ == '__main__':
    main()
