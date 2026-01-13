"""
RUN_ALL_ANALYSES.py

Master script that runs all analyses in sequence:
1. COMBINE_DATASETS.py - Combine MS + Kaggle + SEMA data with proper weighting
2. STANDARDIZATION_TEST.py - Test robustness to different standardization methods
3. TEMPORAL_VALIDATION.py - Train on N days, test on next day (asymptote graph)
4. BOOTSTRAP_SAMPLING.py - Bootstrap sampling with error bars
5. PREVALENCE_ESTIMATION.py - Estimate MS prevalence in healthy controls

This provides a comprehensive analysis pipeline for the MS detection study.
"""

import subprocess
import sys
from pathlib import Path
import time

def run_script(script_name, description):
    """
    Run a Python script and capture output
    """
    print("\n" + "="*70)
    print(f"RUNNING: {script_name}")
    print(f"Description: {description}")
    print("="*70 + "\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )

        elapsed_time = time.time() - start_time
        print(f"\n✓ {script_name} completed successfully in {elapsed_time:.1f} seconds")
        return True

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ {script_name} failed after {elapsed_time:.1f} seconds")
        print(f"Error: {e}")
        return False

def check_dependencies():
    """
    Check if all required packages are installed
    """
    print("\n" + "="*70)
    print("CHECKING DEPENDENCIES")
    print("="*70)

    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'scipy'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (MISSING)")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False

    print("\n✓ All dependencies installed")
    return True

def check_data_files():
    """
    Check if required data files exist
    """
    print("\n" + "="*70)
    print("CHECKING DATA FILES")
    print("="*70)

    required_files = [
        'data/FitMRI_fitbit_intraday_steps_trainingData.csv',
        'hourlySteps_merged_31216_41116.csv',
        'hourlySteps_merged_41216_51216.csv',
        'hourly_fitbit_sema_df_unprocessed.csv'
    ]

    missing_files = []

    for filepath in required_files:
        if Path(filepath).exists():
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} (MISSING)")
            missing_files.append(filepath)

    if missing_files:
        print(f"\n⚠ Missing files: {missing_files}")
        return False

    print("\n✓ All data files found")
    return True

def create_output_directories():
    """
    Create necessary output directories
    """
    print("\n" + "="*70)
    print("CREATING OUTPUT DIRECTORIES")
    print("="*70)

    directories = [
        'data',
        'result',
        'output',
        'output/ready'
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True, parents=True)
        print(f"  ✓ {directory}")

    print("\n✓ All directories ready")

def main():
    """
    Main pipeline execution
    """
    print("\n" + "="*70)
    print("MS DETECTION ANALYSIS - MASTER PIPELINE")
    print("="*70)
    print("\nThis script will run the following analyses:")
    print("  1. Combine datasets (MS + Kaggle + SEMA)")
    print("  2. Test standardization robustness")
    print("  3. Temporal validation (train N days → test next day)")
    print("  4. Bootstrap sampling with error bars")
    print("  5. Prevalence estimation in healthy controls")
    print("  6. Dataset source sensitivity analysis")
    print("\n  NOTE: Scripts 2, 4, 5 have been FIXED for patient-level data leakage")
    print("\n" + "="*70)

    # Check prerequisites
    if not check_dependencies():
        print("\n✗ Please install missing dependencies before continuing")
        return

    if not check_data_files():
        print("\n✗ Please ensure all data files are in place before continuing")
        return

    create_output_directories()

    # Define analysis pipeline
    analyses = [
        {
            'script': 'COMBINE_DATASETS.py',
            'description': 'Combine MS patient data with Kaggle and SEMA healthy controls'
        },
        {
            'script': 'STANDARDIZATION_TEST.py',
            'description': 'Test robustness to different feature standardization methods (FIXED: patient-level splits)'
        },
        {
            'script': 'TEMPORAL_VALIDATION.py',
            'description': 'Temporal validation: train on N days, test on next day'
        },
        {
            'script': 'BOOTSTRAP_SAMPLING.py',
            'description': 'Bootstrap sampling analysis with confidence intervals (FIXED: patient-level splits)'
        },
        {
            'script': 'PREVALENCE_ESTIMATION.py',
            'description': 'Estimate MS prevalence in healthy control population (FIXED: patient-level splits)'
        },
        {
            'script': 'DATASET_SOURCE_SENSITIVITY.py',
            'description': 'Dataset source sensitivity: verify model learns MS patterns, not dataset artifacts'
        }
    ]

    # Track results
    results = []
    total_start_time = time.time()

    # Run each analysis
    for i, analysis in enumerate(analyses, 1):
        print(f"\n\n{'#'*70}")
        print(f"STEP {i}/{len(analyses)}")
        print(f"{'#'*70}")

        success = run_script(analysis['script'], analysis['description'])
        results.append({
            'step': i,
            'script': analysis['script'],
            'success': success
        })

        if not success:
            print(f"\n⚠ Warning: {analysis['script']} failed")
            user_input = input("Continue with remaining analyses? (y/n): ")
            if user_input.lower() != 'y':
                print("\nPipeline stopped by user")
                break

    # Summary
    total_elapsed = time.time() - total_start_time

    print("\n\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)

    for result in results:
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"  Step {result['step']}: {result['script']:30s} {status}")

    n_success = sum(1 for r in results if r['success'])
    n_total = len(results)

    print(f"\n  Total: {n_success}/{n_total} analyses completed successfully")
    print(f"  Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")

    print("\n" + "="*70)
    print("OUTPUT FILES")
    print("="*70)
    print("\nKey output files to review:")
    print("  • data/combined_all_hourly_data.csv - Combined dataset")
    print("  • data/class_weights.txt - Class weights for models")
    print("  • data/exclusion_log.txt - Participant exclusion log")
    print("  • result/standardization_comparison.png - Standardization robustness")
    print("  • result/temporal_learning_curve.png - Learning curve (asymptote)")
    print("  • result/bootstrap_auc_comparison.png - Bootstrap sampling results")
    print("  • result/bootstrap_mean_roc_*.png - Mean ROC curves with error bars")
    print("  • result/prevalence_distribution.png - MS prevalence estimation")
    print("  • result/prevalence_sensitivity.png - Sensitivity analysis")
    print("  • result/dataset_sensitivity_roc.png - Cross-dataset validation")
    print("  • result/dataset_sensitivity_summary.csv - Sensitivity analysis results")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Review all generated plots in the result/ directory")
    print("2. Check CSV files for detailed numerical results")
    print("3. Examine class_weights.txt for proper model weighting")
    print("4. Consider fixing methodological issues documented in")
    print("   COMPREHENSIVE_METHODOLOGICAL_ISSUES.md")
    print("\n" + "="*70)

    if n_success == n_total:
        print("\n✓ All analyses completed successfully!")
    else:
        print(f"\n⚠ {n_total - n_success} analysis(es) failed - review output above")

if __name__ == '__main__':
    main()
