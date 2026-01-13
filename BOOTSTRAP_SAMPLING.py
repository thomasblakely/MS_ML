"""
BOOTSTRAP_SAMPLING.py

Implements bootstrap sampling analysis to estimate robustness and variance:
- Randomly sample different subsets of days/participants
- Calculate mean ROC-AUC and standard deviation error bars
- Evaluate variance across different sample sizes
- Tests with 6 days, 30 days per participant

Key analysis:
"Six days of data points, 30 days for 1 participant, randomly sample mean roc and sd error bar"
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
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

def bootstrap_sample_days(df_daily, n_days, n_bootstrap=100, test_size=0.2, random_state=None):
    """
    Bootstrap sampling: randomly sample N days per participant, repeat n_bootstrap times
    Calculate mean AUC and standard deviation

    FIXED: Uses GroupShuffleSplit to ensure same patient's data never appears in both train and test
    FIXED: Iteration-dependent random state varies patient splits across bootstrap iterations
    FIXED: Skips patients with insufficient days instead of sampling with replacement
    FIXED: Uses RandomState object for reproducible day sampling
    """
    from sklearn.model_selection import GroupShuffleSplit

    print(f"\nBootstrap sampling with {n_days} days per participant ({n_bootstrap} iterations)...")

    # FIXED: Create RandomState object for reproducible sampling
    if random_state is not None:
        rng = np.random.RandomState(random_state + 1000)  # Offset to avoid overlap with GroupShuffleSplit
    else:
        rng = np.random

    feature_cols = ['mean_steps', 'std_steps', 'max_steps', 'min_steps',
                   'total_steps', 'cv_steps', 'num_hours']

    aucs = []
    roc_curves = []
    n_skipped_patients = 0

    for i in range(n_bootstrap):
        if (i + 1) % 10 == 0:
            print(f"  Bootstrap iteration {i + 1}/{n_bootstrap}...", end='\r')

        # Sample days per participant
        sampled_data = []

        for patient_id, patient_data in df_daily.groupby('id'):
            patient_days = patient_data['day'].unique()

            # FIXED: Skip patients with insufficient days to avoid sampling with replacement
            # Sampling with replacement creates temporal data leakage when same patient
            # appears in both train and test with duplicated days
            if len(patient_days) < n_days:
                n_skipped_patients += 1
                continue  # Skip this patient

            # Sample WITHOUT replacement (guaranteed to work now)
            # Use rng instead of np.random for reproducibility
            sampled_days = rng.choice(patient_days, size=n_days, replace=False)

            # Get data for sampled days
            sampled_patient_data = patient_data[patient_data['day'].isin(sampled_days)]
            sampled_data.append(sampled_patient_data)

        df_sampled = pd.concat(sampled_data, ignore_index=True)

        # Check if we have both classes
        if len(df_sampled['label'].unique()) < 2:
            continue

        # Prepare features
        X = df_sampled[feature_cols].values
        y = df_sampled['label'].values
        groups = df_sampled['id'].values  # CRITICAL: Group by patient ID

        # Patient-level train/test split
        # FIXED: Use iteration-dependent random state to vary patient splits across iterations
        # This ensures bootstrap captures variance from both day selection AND patient selection
        iteration_seed = random_state + i if random_state is not None else None
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=iteration_seed)

        try:
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        except:
            continue

        # Check class balance in train and test
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

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
        model = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=random_state)
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
            aucs.append(auc)

            # Store ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_curves.append({'fpr': fpr, 'tpr': tpr, 'auc': auc})
        except:
            continue

    print(f"\n  Completed {len(aucs)}/{n_bootstrap} successful bootstrap iterations")

    if n_skipped_patients > 0:
        avg_skipped = n_skipped_patients / n_bootstrap
        print(f"  Average {avg_skipped:.1f} patients/iteration skipped (insufficient days)")

    if len(aucs) == 0:
        print("  ⚠ WARNING: All bootstrap iterations failed")
        return None, None

    if len(aucs) < n_bootstrap * 0.5:
        print(f"  ⚠ WARNING: Only {len(aucs)}/{n_bootstrap} iterations succeeded (<50%)")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ci_lower = np.percentile(aucs, 2.5)
    ci_upper = np.percentile(aucs, 97.5)

    print(f"  Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

    return {
        'n_days': n_days,
        'n_bootstrap': len(aucs),
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'aucs': aucs
    }, roc_curves

def plot_bootstrap_results(results_list, output_path='result/bootstrap_auc_comparison.png'):
    """
    Plot bootstrap results for different day counts
    """
    print(f"\nPlotting bootstrap comparison...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: AUC with error bars
    n_days_list = [r['n_days'] for r in results_list]
    mean_aucs = [r['mean_auc'] for r in results_list]
    std_aucs = [r['std_auc'] for r in results_list]

    ax1.errorbar(
        n_days_list,
        mean_aucs,
        yerr=std_aucs,
        marker='o',
        linestyle='-',
        capsize=5,
        capthick=2,
        linewidth=2,
        markersize=10,
        label='Mean AUC ± SD'
    )

    ax1.set_xlabel('Number of Days Sampled per Participant', fontsize=12)
    ax1.set_ylabel('ROC-AUC Score', fontsize=12)
    ax1.set_title('Bootstrap Sampling: AUC vs Sample Size', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (AUC=0.5)')
    ax1.legend(fontsize=10)

    # Plot 2: Distribution of AUCs
    for r in results_list:
        ax2.hist(r['aucs'], bins=20, alpha=0.5, label=f"{r['n_days']} days")

    ax2.set_xlabel('ROC-AUC Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Bootstrap AUCs', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved bootstrap comparison to: {output_path}")

    return fig

def plot_mean_roc_curve(roc_curves, n_days, output_path='result/bootstrap_mean_roc_curve.png'):
    """
    Plot mean ROC curve with confidence intervals from bootstrap samples
    """
    print(f"\nPlotting mean ROC curve for {n_days} days...")

    # Interpolate all ROC curves to common FPR points
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for roc in roc_curves:
        tpr_interp = np.interp(mean_fpr, roc['fpr'], roc['tpr'])
        tprs.append(tpr_interp)

    tprs = np.array(tprs)
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)

    # Calculate mean AUC
    mean_auc = np.mean([r['auc'] for r in roc_curves])
    std_auc = np.std([r['auc'] for r in roc_curves])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot mean ROC curve
    ax.plot(mean_fpr, mean_tpr, color='blue', linewidth=2,
           label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')

    # Plot confidence interval (mean ± 1 SD)
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    color='blue', alpha=0.2, label='± 1 SD')

    # Plot random classifier
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random (AUC = 0.5)')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'Bootstrap Mean ROC Curve ({n_days} days sampled)', fontsize=14)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved mean ROC curve to: {output_path}")

    return fig

def main():
    """Main bootstrap sampling pipeline"""
    print("\n" + "="*60)
    print("BOOTSTRAP SAMPLING ANALYSIS")
    print("="*60)

    # Create output directory
    Path('result').mkdir(exist_ok=True)

    # Step 1: Load data
    df = load_combined_data()

    # Step 2: Extract daily features
    df_daily = extract_daily_features(df)

    # Step 3: Bootstrap sampling for different day counts
    day_counts = [6, 10, 15, 30]
    n_bootstrap = 100

    all_results = []
    all_roc_curves = {}

    for n_days in day_counts:
        print(f"\n{'='*60}")
        print(f"BOOTSTRAP ANALYSIS: {n_days} DAYS")
        print(f"{'='*60}")

        result, roc_curves = bootstrap_sample_days(
            df_daily,
            n_days=n_days,
            n_bootstrap=n_bootstrap,
            test_size=0.2,
            random_state=42
        )

        if result is not None:
            all_results.append(result)
            all_roc_curves[n_days] = roc_curves

            # Save individual results
            pd.DataFrame({
                'auc': result['aucs']
            }).to_csv(f'result/bootstrap_{n_days}days_aucs.csv', index=False)

    # Step 4: Plot comparison
    if len(all_results) > 0:
        plot_bootstrap_results(all_results)

        # Save summary
        summary_df = pd.DataFrame([
            {
                'n_days': r['n_days'],
                'mean_auc': r['mean_auc'],
                'std_auc': r['std_auc'],
                'n_bootstrap': r['n_bootstrap']
            }
            for r in all_results
        ])
        summary_df.to_csv('result/bootstrap_summary.csv', index=False)
        print(f"\nSaved bootstrap summary to: result/bootstrap_summary.csv")

    # Step 5: Plot mean ROC curves for each day count
    for n_days, roc_curves in all_roc_curves.items():
        plot_mean_roc_curve(
            roc_curves,
            n_days=n_days,
            output_path=f'result/bootstrap_mean_roc_{n_days}days.png'
        )

    print("\n✓ Bootstrap sampling complete!")

if __name__ == '__main__':
    main()
