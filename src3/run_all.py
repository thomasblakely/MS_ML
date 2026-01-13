import subprocess

scripts = [
    'merge_data.py',
    'standardize_test.py',
    'bootstrap_stability.py',
    'expected_fp.py'
]

for script in scripts:
    print(f"\n{'='*60}")
    print(f"Running {script}")
    print('='*60)
    subprocess.run(['python', f'src3/{script}'])
