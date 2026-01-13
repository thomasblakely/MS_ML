"""
DATASET_SOURCE_SENSITIVITY.py

Sensitivity analysis to verify model learns MS patterns, not dataset source artifacts.

Tests:
1. Main analysis: All data mixed (Kaggle + SEMA together)
2. Cross-dataset validation: Train on Kaggle, test on SEMA
3. Cross-dataset validation: Train on SEMA, test on Kaggle
4. Dataset distinguishability: Can model distinguish Kaggle vs SEMA? (should be low if datasets similar)

If model performs well across all tests → learning real MS patterns
If performance drops in cross-dataset tests → learning dataset artifacts
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
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
    print(f"  MS patients: {df[df['label']==0]['id'].nunique()}")
    print(f"  Kaggle healthy: {df[df['source']=='Kaggle_Healthy']['id'].nunique()}")
    print(f"  SEMA healthy: {df[df['source']=='SEMA_Healthy']['id'].nunique()}")
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
            'source': group['source'].iloc[0],
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

def train_and_evaluate(df_train, df_test, test_name):
    """
    Train model on train set, evaluate on test set
    Both sets must contain MS and healthy patients
    """
    print(f"\n{test_name}")
    print("="*60)

    feature_cols = ['mean_steps', 'std_steps', 'max_steps', 'min_steps',
                   'total_steps', 'cv_steps', 'num_hours']

    # Check we have both classes in both sets
    train_ms = (df_train['label'] == 0).sum()
    train_healthy = (df_train['label'] == 1).sum()
    test_ms = (df_test['label'] == 0).sum()
    test_healthy = (df_test['label'] == 1).sum()

    print(f"  Training: {train_ms} MS, {train_healthy} Healthy")
    print(f"  Testing: {test_ms} MS, {test_healthy} Healthy")

    if train_ms == 0 or train_healthy == 0:
        print("  ⚠ Skipping: Training set needs both classes")
        return None

    if test_ms == 0 or test_healthy == 0:
        print("  ⚠ Skipping: Test set needs both classes")
        return None

    # Prepare features
    X_train = df_train[feature_cols].values
    y_train = df_train['label'].values
    X_test = df_test[feature_cols].values
    y_test = df_test['label'].values

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  Results:")
    print(f"    AUC: {auc:.3f}")
    print(f"    Precision: {precision:.3f}")
    print(f"    Recall: {recall:.3f}")
    print(f"    F1: {f1:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[1,1]}, FP={cm[1,0]}")
    print(f"    FN={cm[0,1]}, TP={cm[0,0]}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    return {
        'test_name': test_name,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'tpr': tpr,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'train_ms': train_ms,
        'train_healthy': train_healthy,
        'test_ms': test_ms,
        'test_healthy': test_healthy,
    }

def analysis_1_mixed_datasets(df_daily):
    """
    Analysis 1: All data mixed (baseline)
    Train and test both contain mix of Kaggle + SEMA
    """
    print("\n" + "="*70)
    print("ANALYSIS 1: BASELINE (ALL DATA MIXED)")
    print("="*70)

    # Patient-level split (all sources mixed)
    feature_cols = ['mean_steps', 'std_steps', 'max_steps', 'min_steps',
                   'total_steps', 'cv_steps', 'num_hours']

    X = df_daily[feature_cols].values
    y = df_daily['label'].values
    groups = df_daily['id'].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    df_train = df_daily.iloc[train_idx]
    df_test = df_daily.iloc[test_idx]

    result = train_and_evaluate(df_train, df_test, "Baseline: Mixed Datasets")

    return result

def analysis_2_kaggle_to_sema(df_daily):
    """
    Analysis 2: Train on Kaggle healthy, test on SEMA healthy
    Tests if model generalizes across healthy control datasets
    """
    print("\n" + "="*70)
    print("ANALYSIS 2: TRAIN KAGGLE → TEST SEMA")
    print("="*70)

    # Split MS patients
    df_ms = df_daily[df_daily['label'] == 0]
    ms_patients = df_ms['id'].unique()

    # Random 80/20 split of MS patients
    np.random.seed(42)
    n_train_ms = int(0.8 * len(ms_patients))
    train_ms_patients = np.random.choice(ms_patients, size=n_train_ms, replace=False)
    test_ms_patients = np.setdiff1d(ms_patients, train_ms_patients)

    df_ms_train = df_ms[df_ms['id'].isin(train_ms_patients)]
    df_ms_test = df_ms[df_ms['id'].isin(test_ms_patients)]

    # Use ALL Kaggle for training, ALL SEMA for testing
    df_kaggle = df_daily[df_daily['source'] == 'Kaggle_Healthy']
    df_sema = df_daily[df_daily['source'] == 'SEMA_Healthy']

    df_train = pd.concat([df_ms_train, df_kaggle], ignore_index=True)
    df_test = pd.concat([df_ms_test, df_sema], ignore_index=True)

    result = train_and_evaluate(df_train, df_test, "Cross-Dataset: Kaggle→SEMA")

    return result

def analysis_3_sema_to_kaggle(df_daily):
    """
    Analysis 3: Train on SEMA healthy, test on Kaggle healthy
    Tests if model generalizes in the opposite direction
    """
    print("\n" + "="*70)
    print("ANALYSIS 3: TRAIN SEMA → TEST KAGGLE")
    print("="*70)

    # FIXED: Use different random seed for independent test (not correlated with Analysis 2)
    # This makes the two cross-dataset tests statistically independent
    df_ms = df_daily[df_daily['label'] == 0]
    ms_patients = df_ms['id'].unique()

    np.random.seed(43)  # FIXED: Different seed from Analysis 2 (which uses 42)
    n_train_ms = int(0.8 * len(ms_patients))
    train_ms_patients = np.random.choice(ms_patients, size=n_train_ms, replace=False)
    test_ms_patients = np.setdiff1d(ms_patients, train_ms_patients)

    df_ms_train = df_ms[df_ms['id'].isin(train_ms_patients)]
    df_ms_test = df_ms[df_ms['id'].isin(test_ms_patients)]

    # Use ALL SEMA for training, ALL Kaggle for testing
    df_kaggle = df_daily[df_daily['source'] == 'Kaggle_Healthy']
    df_sema = df_daily[df_daily['source'] == 'SEMA_Healthy']

    df_train = pd.concat([df_ms_train, df_sema], ignore_index=True)
    df_test = pd.concat([df_ms_test, df_kaggle], ignore_index=True)

    result = train_and_evaluate(df_train, df_test, "Cross-Dataset: SEMA→Kaggle")

    return result

def analysis_4_dataset_distinguishability(df_daily):
    """
    Analysis 4: Can model distinguish Kaggle vs SEMA (both healthy)?
    High accuracy = datasets are distinguishable (BAD)
    Low accuracy (~0.5) = datasets are similar (GOOD)
    """
    print("\n" + "="*70)
    print("ANALYSIS 4: DATASET DISTINGUISHABILITY (Kaggle vs SEMA)")
    print("="*70)

    # Only healthy controls
    df_healthy = df_daily[df_daily['label'] == 1]

    # Create binary labels: 0=Kaggle, 1=SEMA
    df_healthy['dataset_label'] = (df_healthy['source'] == 'SEMA_Healthy').astype(int)

    print(f"  Kaggle healthy: {(df_healthy['dataset_label'] == 0).sum()}")
    print(f"  SEMA healthy: {(df_healthy['dataset_label'] == 1).sum()}")

    # Patient-level split
    feature_cols = ['mean_steps', 'std_steps', 'max_steps', 'min_steps',
                   'total_steps', 'cv_steps', 'num_hours']

    X = df_healthy[feature_cols].values
    y = df_healthy['dataset_label'].values
    groups = df_healthy['id'].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model (no class weighting - this is a sanity check)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Evaluate
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = (y_pred == y_test).mean()

    print(f"\n  Results:")
    print(f"    AUC: {auc:.3f}")
    print(f"    Accuracy: {accuracy:.3f}")

    if auc > 0.7:
        print(f"\n  ⚠ WARNING: Datasets are highly distinguishable (AUC={auc:.3f})")
        print("    This suggests systematic differences between Kaggle and SEMA.")
        print("    Model may be learning dataset artifacts in MS detection task.")
    elif auc < 0.6:
        print(f"\n  ✓ GOOD: Datasets are not easily distinguishable (AUC={auc:.3f})")
        print("    This suggests Kaggle and SEMA are similar.")
        print("    Model likely learning real patterns, not dataset artifacts.")
    else:
        print(f"\n  ⚠ MODERATE: Some distinguishability (AUC={auc:.3f})")
        print("    Datasets have some differences but may be acceptable.")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    return {
        'test_name': 'Dataset Distinguishability',
        'auc': auc,
        'accuracy': accuracy,
        'fpr': fpr,
        'tpr': tpr,
    }

def plot_all_roc_curves(results_list, output_path='result/dataset_sensitivity_roc.png'):
    """
    Plot ROC curves for all analyses
    """
    print(f"\nPlotting ROC curves...")

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['blue', 'green', 'orange', 'red']

    for i, result in enumerate(results_list):
        if result is not None and 'fpr' in result:
            ax.plot(result['fpr'], result['tpr'], linewidth=2, color=colors[i],
                   label=f"{result['test_name']} (AUC={result['auc']:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Dataset Source Sensitivity Analysis: ROC Curves', fontsize=14)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved ROC curves to: {output_path}")

    return fig

def save_results_summary(results_list, output_path='result/dataset_sensitivity_summary.csv'):
    """
    Save summary of all analyses
    """
    summary_data = []

    for result in results_list:
        if result is not None:
            summary_data.append({
                'test_name': result['test_name'],
                'auc': result.get('auc', np.nan),
                'precision': result.get('precision', np.nan),
                'recall': result.get('recall', np.nan),
                'f1': result.get('f1', np.nan),
                'n_train': result.get('n_train', np.nan),
                'n_test': result.get('n_test', np.nan),
            })

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_path, index=False)
    print(f"\nSaved summary to: {output_path}")

    return df_summary

def main():
    """Main sensitivity analysis pipeline"""
    print("\n" + "="*70)
    print("DATASET SOURCE SENSITIVITY ANALYSIS")
    print("="*70)

    # Create output directory
    Path('result').mkdir(exist_ok=True)

    # Step 1: Load data
    df = load_combined_data()

    # Step 2: Extract daily features
    df_daily = extract_daily_features(df)

    # Step 3: Run all analyses
    results = []

    # Analysis 1: Baseline (all data mixed)
    result_1 = analysis_1_mixed_datasets(df_daily)
    results.append(result_1)

    # Analysis 2: Train Kaggle → Test SEMA
    result_2 = analysis_2_kaggle_to_sema(df_daily)
    results.append(result_2)

    # Analysis 3: Train SEMA → Test Kaggle
    result_3 = analysis_3_sema_to_kaggle(df_daily)
    results.append(result_3)

    # Analysis 4: Dataset distinguishability
    result_4 = analysis_4_dataset_distinguishability(df_daily)
    results.append(result_4)

    # Step 4: Plot results
    plot_all_roc_curves(results)

    # Step 5: Save summary
    summary_df = save_results_summary(results)

    # Step 6: Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)

    print("\n✓ Model is learning real MS patterns if:")
    print("  - Analysis 1 (baseline) AUC > 0.7")
    print("  - Analysis 2 & 3 (cross-dataset) AUC similar to baseline (±0.1)")
    print("  - Analysis 4 (distinguishability) AUC < 0.6")

    print("\n⚠ Model may be learning dataset artifacts if:")
    print("  - Cross-dataset AUC drops significantly (>0.15 below baseline)")
    print("  - Distinguishability AUC > 0.7")

    print("\n" + "="*70)

    if result_1 and result_2 and result_3:
        baseline_auc = result_1['auc']
        kaggle_to_sema_auc = result_2['auc']
        sema_to_kaggle_auc = result_3['auc']

        drop_2 = baseline_auc - kaggle_to_sema_auc
        drop_3 = baseline_auc - sema_to_kaggle_auc

        print(f"\nPerformance Summary:")
        print(f"  Baseline (mixed): AUC = {baseline_auc:.3f}")
        print(f"  Kaggle→SEMA: AUC = {kaggle_to_sema_auc:.3f} (drop: {drop_2:.3f})")
        print(f"  SEMA→Kaggle: AUC = {sema_to_kaggle_auc:.3f} (drop: {drop_3:.3f})")

        if drop_2 < 0.1 and drop_3 < 0.1:
            print("\n✓ EXCELLENT: Model generalizes well across datasets!")
            print("  Likely learning real MS patterns, not dataset artifacts.")
        elif drop_2 < 0.15 and drop_3 < 0.15:
            print("\n✓ GOOD: Model shows acceptable cross-dataset performance.")
            print("  Some dataset differences but overall robust.")
        else:
            print("\n⚠ CONCERN: Significant performance drop in cross-dataset validation.")
            print("  Model may be learning some dataset-specific patterns.")
            print("  Consider investigating systematic differences between datasets.")

    print("\n✓ Dataset sensitivity analysis complete!")

if __name__ == '__main__':
    main()
