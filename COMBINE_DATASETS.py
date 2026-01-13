"""
COMBINE_DATASETS.py

Combines hourly Kaggle FitBit data with new Zeno SEMA hourly fitbit data for healthy controls.
Includes proper class weighting for MS patients (20) vs healthy controls (30 Kaggle + 71 SEMA = 101).

Output:
- combined_all_hourly_data.csv: Full combined dataset with proper labels
- class_weights.txt: Computed class weights for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_ms_data(filepath='data/FitMRI_fitbit_intraday_steps_trainingData.csv'):
    """
    Load MS patient data from FitMRI
    Returns: DataFrame with id, datetime, steps, label
    """
    print("Loading MS patient data...")
    df = pd.read_csv(filepath)

    # Create datetime from date and time columns
    df['datetime'] = pd.to_datetime(
        df['measured_date'] + ' ' + df['measured_time'],
        format='%d-%b-%y %H:%M:%S'
    )

    # Rename fitmri_id to id for consistency
    df = df.rename(columns={'fitmri_id': 'id'})

    # Keep only needed columns
    df = df[['id', 'datetime', 'steps']]

    # Add label: 0 = MS patient
    df['label'] = 0
    df['source'] = 'FitMRI_MS'

    print(f"  Loaded {len(df)} measurements from {df['id'].nunique()} MS patients")
    return df

def load_kaggle_healthy(filepath1='hourlySteps_merged_31216_41116.csv',
                        filepath2='hourlySteps_merged_41216_51216.csv'):
    """
    Load healthy control data from Kaggle
    Returns: DataFrame with id, datetime, steps, label
    """
    print("Loading Kaggle healthy control data...")

    # Load both time periods
    df1 = pd.read_csv(filepath1)
    df2 = pd.read_csv(filepath2)

    # Combine
    df = pd.concat([df1, df2], ignore_index=True)

    # Parse datetime
    df['datetime'] = pd.to_datetime(df['ActivityHour'])

    # Rename columns
    df = df.rename(columns={'Id': 'id', 'StepTotal': 'steps'})

    # Keep only needed columns
    df = df[['id', 'datetime', 'steps']]

    # Add label: 1 = Healthy control
    df['label'] = 1
    df['source'] = 'Kaggle_Healthy'

    print(f"  Loaded {len(df)} measurements from {df['id'].nunique()} healthy individuals")
    return df

def load_sema_healthy(filepath='hourly_fitbit_sema_df_unprocessed.csv'):
    """
    Load SEMA healthy control data
    Returns: DataFrame with id, datetime, steps, label
    """
    print("Loading SEMA healthy control data...")

    df = pd.read_csv(filepath)

    # Create datetime from date and hour columns
    df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')

    # Keep only needed columns
    df = df[['id', 'datetime', 'steps']].copy()

    # Add label: 1 = Healthy control
    df['label'] = 1
    df['source'] = 'SEMA_Healthy'

    # Drop rows with missing steps
    df = df.dropna(subset=['steps'])

    print(f"  Loaded {len(df)} measurements from {df['id'].nunique()} SEMA healthy individuals")
    return df

def round_to_nearest_hour(df):
    """
    Round all timestamps to the nearest hour and aggregate steps
    """
    print("Rounding to nearest hour and aggregating steps...")

    # Round datetime to nearest hour
    df['hour'] = df['datetime'].dt.round('H')

    # Group by id, hour, label, and source; sum the steps
    df_hourly = df.groupby(['id', 'hour', 'label', 'source'], as_index=False)['steps'].sum()

    # Rename hour back to datetime
    df_hourly = df_hourly.rename(columns={'hour': 'datetime'})

    print(f"  Aggregated to {len(df_hourly)} hourly measurements")
    return df_hourly

def filter_quality_users(df, min_hours=12):
    """
    Filter users by data quality (minimum hours of data)
    Keep all MS patients (no filtering)
    Keep all healthy controls that meet minimum quality threshold

    No artificial caps - use ALL available quality data with proper class weighting
    """
    print(f"\nFiltering by data quality (min {min_hours} hours)...")

    # Separate MS and healthy by source
    df_ms = df[df['source'] == 'FitMRI_MS'].copy()
    df_kaggle = df[df['source'] == 'Kaggle_Healthy'].copy()
    df_sema = df[df['source'] == 'SEMA_Healthy'].copy()

    # Keep ALL MS patients (no filtering)
    print(f"  MS: Keeping all {df_ms['id'].nunique()} patients (no filtering)")

    # Count hours per Kaggle user and filter by quality
    kaggle_counts = df_kaggle.groupby('id').size().reset_index(name='num_hours')
    kaggle_quality = kaggle_counts[kaggle_counts['num_hours'] >= min_hours]

    print(f"  Kaggle: {len(kaggle_quality)}/{len(kaggle_counts)} users meet quality threshold "
          f"(min={kaggle_quality['num_hours'].min()}, max={kaggle_quality['num_hours'].max()}, "
          f"mean={kaggle_quality['num_hours'].mean():.0f} hours)")

    # Count hours per SEMA user and filter by quality
    sema_counts = df_sema.groupby('id').size().reset_index(name='num_hours')
    sema_quality = sema_counts[sema_counts['num_hours'] >= min_hours]

    print(f"  SEMA: {len(sema_quality)}/{len(sema_counts)} users meet quality threshold "
          f"(min={sema_quality['num_hours'].min()}, max={sema_quality['num_hours'].max()}, "
          f"mean={sema_quality['num_hours'].mean():.0f} hours)")

    # Filter healthy data to quality users (NO ARTIFICIAL CAPS)
    df_kaggle_filtered = df_kaggle[df_kaggle['id'].isin(kaggle_quality['id'])]
    df_sema_filtered = df_sema[df_sema['id'].isin(sema_quality['id'])]

    # Combine all
    df_combined = pd.concat([df_ms, df_kaggle_filtered, df_sema_filtered], ignore_index=True)

    print(f"\n  Final dataset: {len(df_combined)} hourly measurements")
    print(f"  MS patients: {df_ms['id'].nunique()}")
    print(f"  Kaggle healthy: {df_kaggle_filtered['id'].nunique()}")
    print(f"  SEMA healthy: {df_sema_filtered['id'].nunique()}")
    print(f"  Total healthy: {df_kaggle_filtered['id'].nunique() + df_sema_filtered['id'].nunique()}")
    print(f"  Imbalance ratio: 1:{(df_kaggle_filtered['id'].nunique() + df_sema_filtered['id'].nunique()) / df_ms['id'].nunique():.1f}")

    # Save exclusion log
    exclusion_log = {
        'kaggle_excluded': len(kaggle_counts) - len(kaggle_quality),
        'kaggle_excluded_ids': kaggle_counts[~kaggle_counts['id'].isin(kaggle_quality['id'])]['id'].tolist(),
        'sema_excluded': len(sema_counts) - len(sema_quality),
        'sema_excluded_ids': sema_counts[~sema_counts['id'].isin(sema_quality['id'])]['id'].tolist(),
    }

    return df_combined, exclusion_log

def calculate_class_weights(df):
    """
    Calculate class weights for handling imbalance
    """
    n_ms = df[df['label'] == 0]['id'].nunique()
    n_healthy = df[df['label'] == 1]['id'].nunique()

    # Weight for positive class (healthy) in BCEWithLogitsLoss: n_ms / n_healthy
    pos_weight = n_ms / n_healthy

    # Weight for sklearn models: inverse frequency
    total = n_ms + n_healthy
    weight_ms = total / (2 * n_ms)
    weight_healthy = total / (2 * n_healthy)

    print(f"\n" + "="*60)
    print("CLASS WEIGHTS")
    print("="*60)
    print(f"MS patients (label=0): {n_ms} individuals")
    print(f"Healthy controls (label=1): {n_healthy} individuals")
    print(f"\nFor PyTorch BCEWithLogitsLoss:")
    print(f"  pos_weight = {pos_weight:.4f}")
    print(f"\nFor sklearn class_weight parameter:")
    print(f"  class_weight = {{0: {weight_ms:.4f}, 1: {weight_healthy:.4f}}}")
    print("="*60)

    return {
        'pos_weight': pos_weight,
        'sklearn_weights': {0: weight_ms, 1: weight_healthy},
        'n_ms': n_ms,
        'n_healthy': n_healthy
    }

def save_datasets(df, weights, exclusion_log, output_dir='data'):
    """
    Save the processed datasets, class weights, and exclusion log
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save full combined dataset
    full_path = output_path / 'combined_all_hourly_data.csv'
    df.to_csv(full_path, index=False)
    print(f"\nSaved combined dataset to: {full_path}")

    # Save class weights
    weights_path = output_path / 'class_weights.txt'
    with open(weights_path, 'w') as f:
        f.write("CLASS WEIGHTS FOR MODEL TRAINING\n")
        f.write("="*60 + "\n\n")
        f.write(f"MS patients (label=0): {weights['n_ms']} individuals\n")
        f.write(f"Healthy controls (label=1): {weights['n_healthy']} individuals\n")
        f.write(f"Imbalance ratio: 1:{weights['n_healthy']/weights['n_ms']:.1f}\n\n")
        f.write("PyTorch BCEWithLogitsLoss:\n")
        f.write(f"  pos_weight = torch.tensor([{weights['pos_weight']:.4f}])\n\n")
        f.write("Sklearn models:\n")
        f.write(f"  class_weight = {weights['sklearn_weights']}\n")

    print(f"Saved class weights to: {weights_path}")

    # Save exclusion log
    exclusion_path = output_path / 'exclusion_log.txt'
    with open(exclusion_path, 'w') as f:
        f.write("DATA EXCLUSION LOG\n")
        f.write("="*60 + "\n\n")
        f.write(f"Kaggle users excluded: {exclusion_log['kaggle_excluded']}\n")
        f.write(f"  IDs: {exclusion_log['kaggle_excluded_ids']}\n\n")
        f.write(f"SEMA users excluded: {exclusion_log['sema_excluded']}\n")
        f.write(f"  IDs (first 10): {exclusion_log['sema_excluded_ids'][:10]}\n")
        f.write(f"  ... and {max(0, len(exclusion_log['sema_excluded_ids']) - 10)} more\n\n")
        f.write("Reason: Did not meet minimum data quality threshold (12 hours)\n")

    print(f"Saved exclusion log to: {exclusion_path}")

    # Save summary statistics
    summary = {
        'total_measurements': len(df),
        'total_patients': df['id'].nunique(),
        'ms_patients': df[df['label'] == 0]['id'].nunique(),
        'healthy_patients': df[df['label'] == 1]['id'].nunique(),
        'ms_measurements': len(df[df['label'] == 0]),
        'healthy_measurements': len(df[df['label'] == 1]),
    }

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("="*60)

def main():
    """
    Main preprocessing pipeline
    """
    print("\n" + "="*60)
    print("COMBINED DATASET PREPROCESSING PIPELINE")
    print("="*60 + "\n")

    # Step 1: Load all data sources
    df_ms = load_ms_data()
    df_kaggle = load_kaggle_healthy()
    df_sema = load_sema_healthy()

    # Step 2: Combine datasets
    print("\nCombining datasets...")
    df_combined = pd.concat([df_ms, df_kaggle, df_sema], ignore_index=True)
    print(f"  Combined: {len(df_combined)} total measurements")

    # Step 3: Round to nearest hour and aggregate
    df_hourly = round_to_nearest_hour(df_combined)

    # Step 4: Filter by data quality (no artificial caps)
    df_final, exclusion_log = filter_quality_users(df_hourly, min_hours=12)

    # Step 5: Calculate class weights
    weights = calculate_class_weights(df_final)

    # Step 6: Save outputs
    save_datasets(df_final, weights, exclusion_log)

    print("\n✓ Preprocessing complete!")

if __name__ == '__main__':
    main()
