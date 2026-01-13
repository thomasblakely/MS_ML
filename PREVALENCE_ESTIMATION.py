"""
PREVALENCE_ESTIMATION.py

Estimates MS prevalence in healthy control population:
- Treats healthy controls as population sample
- Uses model predictions to estimate MS prevalence
- Compares to known 1:400 population prevalence
- Evaluates if model can distinguish MS from population controls

Key concept:
"Get into all healthy controls, estimate prevalence, treat like population controls"
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
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
    print(f"  MS patients: {df[df['label']==0]['id'].nunique()}")
    print(f"  Healthy controls: {df[df['label']==1]['id'].nunique()}")
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

def train_classifier(df_daily, test_size=0.2, random_state=42):
    """
    Train classifier on MS vs healthy controls

    FIXED: Uses GroupShuffleSplit to ensure same patient's data never appears in both train and test
    """
    from sklearn.model_selection import GroupShuffleSplit

    print("\nTraining MS vs Healthy classifier...")

    feature_cols = ['mean_steps', 'std_steps', 'max_steps', 'min_steps',
                   'total_steps', 'cv_steps', 'num_hours']

    X = df_daily[feature_cols].values
    y = df_daily['label'].values
    groups = df_daily['id'].values  # CRITICAL: Group by patient ID

    # Patient-level train/test split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

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

    print(f"  Training samples: {len(X_train)} (MS: {n_ms}, Healthy: {n_healthy})")

    # Train model
    model = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n  Test Set Performance:")
    print(f"    AUC: {auc:.3f}")
    print(f"    Precision: {precision:.3f}")
    print(f"    Recall: {recall:.3f}")
    print(f"    F1: {f1:.3f}")

    return model, scaler, feature_cols

def estimate_prevalence_in_healthy(df_daily, model, scaler, feature_cols):
    """
    Apply trained model to all healthy controls to estimate MS prevalence
    """
    print("\n" + "="*60)
    print("PREVALENCE ESTIMATION IN HEALTHY CONTROLS")
    print("="*60)

    # Get all healthy control data
    df_healthy = df_daily[df_daily['label'] == 1].copy()

    print(f"\nHealthy control population:")
    print(f"  Individuals: {df_healthy['id'].nunique()}")
    print(f"  Daily observations: {len(df_healthy)}")

    # Prepare features
    X_healthy = df_healthy[feature_cols].values
    X_healthy_scaled = scaler.transform(X_healthy)

    # Predict MS probability
    y_pred_proba = model.predict_proba(X_healthy_scaled)[:, 0]  # Probability of MS (label=0)
    y_pred = model.predict(X_healthy_scaled)

    # Add predictions to dataframe
    df_healthy['predicted_ms_prob'] = y_pred_proba
    df_healthy['predicted_label'] = y_pred

    # Estimate prevalence by participant (aggregate daily predictions)
    participant_predictions = df_healthy.groupby('id').agg({
        'predicted_ms_prob': 'mean',  # Mean MS probability across all days
        'predicted_label': lambda x: (x == 0).mean()  # Proportion of days predicted as MS
    }).reset_index()

    participant_predictions.columns = ['id', 'mean_ms_prob', 'prop_days_ms']

    # Classification: consider someone MS if mean probability > 0.5
    participant_predictions['classified_as_ms'] = participant_predictions['mean_ms_prob'] > 0.5

    # Estimate prevalence
    n_healthy = len(participant_predictions)
    n_predicted_ms = participant_predictions['classified_as_ms'].sum()
    estimated_prevalence = n_predicted_ms / n_healthy

    print(f"\nPrevalence Estimation:")
    print(f"  Total healthy individuals: {n_healthy}")
    print(f"  Classified as MS by model: {n_predicted_ms}")
    print(f"  Estimated prevalence: {estimated_prevalence:.4f} ({estimated_prevalence*100:.2f}%)")
    print(f"  Expected prevalence (1:400): 0.0025 (0.25%)")

    # Distribution statistics
    print(f"\nMS Probability Distribution:")
    print(f"  Mean: {participant_predictions['mean_ms_prob'].mean():.4f}")
    print(f"  Median: {participant_predictions['mean_ms_prob'].median():.4f}")
    print(f"  Std: {participant_predictions['mean_ms_prob'].std():.4f}")
    print(f"  Min: {participant_predictions['mean_ms_prob'].min():.4f}")
    print(f"  Max: {participant_predictions['mean_ms_prob'].max():.4f}")

    return participant_predictions

def plot_prevalence_distributions(participant_predictions, output_path='result/prevalence_distribution.png'):
    """
    Plot distributions of MS probability predictions in healthy controls
    """
    print(f"\nPlotting prevalence distributions...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Histogram of mean MS probability
    ax1 = axes[0, 0]
    ax1.hist(participant_predictions['mean_ms_prob'], bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Classification threshold (0.5)')
    ax1.set_xlabel('Mean MS Probability', fontsize=12)
    ax1.set_ylabel('Number of Participants', fontsize=12)
    ax1.set_title('Distribution of MS Probability in Healthy Controls', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Boxplot
    ax2 = axes[0, 1]
    ax2.boxplot([participant_predictions['mean_ms_prob']], labels=['Healthy Controls'])
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax2.set_ylabel('Mean MS Probability', fontsize=12)
    ax2.set_title('MS Probability Distribution (Box Plot)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Proportion of days classified as MS
    ax3 = axes[1, 0]
    ax3.hist(participant_predictions['prop_days_ms'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax3.set_xlabel('Proportion of Days Classified as MS', fontsize=12)
    ax3.set_ylabel('Number of Participants', fontsize=12)
    ax3.set_title('Proportion of Days Predicted as MS per Participant', fontsize=14)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Scatter plot
    ax4 = axes[1, 1]
    colors = ['red' if x else 'blue' for x in participant_predictions['classified_as_ms']]
    ax4.scatter(range(len(participant_predictions)), participant_predictions['mean_ms_prob'],
               c=colors, alpha=0.6)
    ax4.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax4.set_xlabel('Participant Index', fontsize=12)
    ax4.set_ylabel('Mean MS Probability', fontsize=12)
    ax4.set_title('MS Probability per Participant (Red = Classified as MS)', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved prevalence distributions to: {output_path}")

    return fig

def sensitivity_analysis(participant_predictions, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    Analyze how estimated prevalence changes with different classification thresholds
    """
    print(f"\n" + "="*60)
    print("SENSITIVITY ANALYSIS: Prevalence vs Classification Threshold")
    print("="*60)

    results = []

    for threshold in thresholds:
        n_predicted_ms = (participant_predictions['mean_ms_prob'] > threshold).sum()
        prevalence = n_predicted_ms / len(participant_predictions)

        results.append({
            'threshold': threshold,
            'n_predicted_ms': n_predicted_ms,
            'estimated_prevalence': prevalence,
            'prevalence_percent': prevalence * 100
        })

        print(f"  Threshold {threshold:.2f}: {n_predicted_ms} predicted MS, "
              f"prevalence = {prevalence:.4f} ({prevalence*100:.2f}%)")

    results_df = pd.DataFrame(results)
    results_df.to_csv('result/prevalence_sensitivity_analysis.csv', index=False)
    print(f"\nSaved sensitivity analysis to: result/prevalence_sensitivity_analysis.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df['threshold'], results_df['prevalence_percent'], marker='o',
           linewidth=2, markersize=8)
    ax.axhline(y=0.25, color='red', linestyle='--', linewidth=2,
              label='Expected prevalence (1:400 = 0.25%)')
    ax.set_xlabel('Classification Threshold', fontsize=12)
    ax.set_ylabel('Estimated Prevalence (%)', fontsize=12)
    ax.set_title('Sensitivity Analysis: MS Prevalence vs Threshold', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('result/prevalence_sensitivity.png', dpi=300, bbox_inches='tight')
    print(f"Saved sensitivity plot to: result/prevalence_sensitivity.png")

    return results_df

def main():
    """Main prevalence estimation pipeline"""
    print("\n" + "="*60)
    print("MS PREVALENCE ESTIMATION IN HEALTHY CONTROLS")
    print("="*60)

    # Create output directory
    Path('result').mkdir(exist_ok=True)

    # Step 1: Load data
    df = load_combined_data()

    # Step 2: Extract daily features
    df_daily = extract_daily_features(df)

    # Step 3: Train classifier
    model, scaler, feature_cols = train_classifier(df_daily, test_size=0.2, random_state=42)

    # Step 4: Estimate prevalence in healthy controls
    participant_predictions = estimate_prevalence_in_healthy(df_daily, model, scaler, feature_cols)

    # Save predictions
    participant_predictions.to_csv('result/healthy_controls_ms_predictions.csv', index=False)
    print(f"\nSaved participant predictions to: result/healthy_controls_ms_predictions.csv")

    # Step 5: Plot distributions
    plot_prevalence_distributions(participant_predictions)

    # Step 6: Sensitivity analysis
    sensitivity_results = sensitivity_analysis(participant_predictions)

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("\nThis analysis estimates MS prevalence by applying the trained model")
    print("to healthy controls (treated as population sample).")
    print("\nIf estimated prevalence is much higher than 0.25% (1:400),")
    print("it may indicate:")
    print("  1. Model has high false positive rate")
    print("  2. Need to adjust classification threshold")
    print("  3. Model needs better calibration")
    print("\nIf prevalence is close to 0.25%, the model is well-calibrated")
    print("for population screening.")
    print("="*60)

    print("\n✓ Prevalence estimation complete!")

if __name__ == '__main__':
    main()
