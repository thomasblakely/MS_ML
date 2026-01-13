import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, roc_auc_score

# Config
DATA_PATH = "src3/src3_results/combined_data.csv"
K_DAYS = 4
N_SPLITS = 200
TEST_SIZE = 0.25
RANDOM_STATE = 42
PI_TARGET = 1 / 400
SCREEN_N = 100000
SPEC_TARGETS = [0.80 ,0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
FEATURE_COLS = ["mean_steps", "std_steps", "max_steps", "min_steps", "total_steps", "cv_steps", "num_hours"]

# Load data
df = pd.read_csv(DATA_PATH, low_memory=False)
df["id"] = df["id"].astype(str)
df["datetime"] = pd.to_datetime(df["datetime"])
df["steps"] = pd.to_numeric(df["steps"], errors="coerce").fillna(0.0)
df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)
df["day"] = df["datetime"].dt.date

# FIX 1: Invert labels so MS = 1 (positive class)
df["label_ms"] = 1 - df["label"]  # Now: 1=MS, 0=Healthy

print(f"Loaded {len(df)} hourly rows")
print(f"Label convention: 1=MS, 0=Healthy")

# Build daily features
daily_features = []
for (pid, day), g in df.groupby(["id", "day"]):
    steps = g["steps"].values
    daily_features.append({
        "id": pid,
        "day": day,
        "label_ms": int(g["label_ms"].iloc[0]),
        "mean_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps, ddof=0)),
        "max_steps": float(np.max(steps)),
        "min_steps": float(np.min(steps)),
        "total_steps": float(np.sum(steps)),
        "cv_steps": float(np.std(steps, ddof=0) / (np.mean(steps) + 1e-8)),
        "num_hours": int(len(steps)),
    })

df_daily = pd.DataFrame(daily_features)
X_all = df_daily[FEATURE_COLS].values
y_all = df_daily["label_ms"].values
groups_all = df_daily["id"].values

n_patients = df_daily["id"].nunique()
n_ms = (df_daily[["id", "label_ms"]].drop_duplicates()["label_ms"] == 1).sum()
n_healthy = (df_daily[["id", "label_ms"]].drop_duplicates()["label_ms"] == 0).sum()

print(f"Daily rows: {len(df_daily)} | Patients: {n_patients} (MS: {n_ms}, Healthy: {n_healthy})")

# FIX 2: Patient-weighted class weighting
def patient_weighted_class_weight(df_daily, y_train_indices):
    """Compute class weights at patient level, not day level"""
    train_patients = df_daily.iloc[y_train_indices]
    patient_labels = train_patients.groupby('id')['label_ms'].first()
    n_ms_patients = (patient_labels == 1).sum()
    n_healthy_patients = (patient_labels == 0).sum()
    total_patients = n_ms_patients + n_healthy_patients
    return {
        1: total_patients / (2 * n_ms_patients),
        0: total_patients / (2 * n_healthy_patients)
    }

# FIX 3: Patient scores with explicit P(MS)
def patient_scores_first_k_days(df_days, model, scaler, feature_cols, k):
    """
    Returns:
        labels: array of patient labels (1=MS, 0=Healthy)
        scores_ms: array of P(MS) scores
        excluded: number of patients excluded due to <K days
    """
    labels, scores_ms = [], []
    excluded = 0

    for pid, g in df_days.sort_values("day").groupby("id"):
        if len(g) < k:
            excluded += 1
            continue
        gk = g.head(k)
        Xp = scaler.transform(gk[feature_cols].values)
        p_ms = model.predict_proba(Xp)[:, 1].mean()  # P(MS) = P(class=1)
        scores_ms.append(p_ms)
        labels.append(int(gk["label_ms"].iloc[0]))

    return np.array(labels), np.array(scores_ms), excluded

# FIX 4: ROC with MS as positive class
def thresholds_from_train(train_labels_ms, train_scores_ms, spec_targets):
    """
    Select thresholds for MS detection
    train_labels_ms: 1=MS, 0=Healthy
    train_scores_ms: P(MS)
    """
    fpr, tpr, thr = roc_curve(train_labels_ms, train_scores_ms, pos_label=1)
    spec = 1 - fpr  # Specificity for MS detection

    out = {}
    for tspec in spec_targets:
        idx = np.where(spec >= tspec)[0]
        if len(idx) == 0:
            continue
        best_idx = idx[np.argmax(tpr[idx])]  # Maximize sensitivity at this specificity
        out[tspec] = float(thr[best_idx])
    return out

# FIX 5: Threshold evaluation with MS as positive
def eval_thresholds_on_test(test_labels_ms, test_scores_ms, thresholds, pi_target, screen_n):
    """
    Evaluate MS screening performance
    test_labels_ms: 1=MS, 0=Healthy
    test_scores_ms: P(MS)
    """
    rows = []
    for tspec, thr in thresholds.items():
        pred_ms = (test_scores_ms >= thr).astype(int)  # Predict MS if score >= threshold

        # Confusion matrix for MS as positive class
        tp = np.sum((pred_ms == 1) & (test_labels_ms == 1))  # Correctly identified MS
        fp = np.sum((pred_ms == 1) & (test_labels_ms == 0))  # Healthy called MS
        tn = np.sum((pred_ms == 0) & (test_labels_ms == 0))  # Correctly identified Healthy
        fn = np.sum((pred_ms == 0) & (test_labels_ms == 1))  # Missed MS

        sens = tp / (tp + fn + 1e-12)  # Sensitivity for MS
        spec = tn / (tn + fp + 1e-12)  # Specificity for MS

        # Screening burden at population prevalence
        tp_100k = screen_n * pi_target * sens
        fp_100k = screen_n * (1 - pi_target) * (1 - spec)
        ppv = (sens * pi_target) / (sens * pi_target + (1 - spec) * (1 - pi_target) + 1e-12)

        rows.append({
            "specificity_target": float(tspec),
            "threshold_train": float(thr),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "PPV_at_pi": float(ppv),
            "TP_per_100k": float(tp_100k),
            "FP_per_100k": float(fp_100k),
            "referrals_per_100k": float(tp_100k + fp_100k),
        })

    return pd.DataFrame(rows)

# Main analysis loop
gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)

split_rows = []
auc_rows = []
skipped = 0
exclusion_log = []

for split_i, (train_idx, test_idx) in enumerate(gss.split(X_all, y_all, groups=groups_all), start=1):
    train_patients = df_daily.iloc[train_idx]["id"].unique()
    test_patients = df_daily.iloc[test_idx]["id"].unique()

    train_data = df_daily[df_daily["id"].isin(train_patients)].copy()
    test_data = df_daily[df_daily["id"].isin(test_patients)].copy()

    # Patient-weighted class weights
    class_weight = patient_weighted_class_weight(df_daily, train_idx)

    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_data[FEATURE_COLS].values)
    y_train = train_data["label_ms"].values

    model = LogisticRegression(class_weight=class_weight, max_iter=2000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    train_labels_ms, train_scores_ms, train_excl = patient_scores_first_k_days(train_data, model, scaler, FEATURE_COLS, K_DAYS)
    test_labels_ms, test_scores_ms, test_excl = patient_scores_first_k_days(test_data, model, scaler, FEATURE_COLS, K_DAYS)

    exclusion_log.append({
        "split": split_i,
        "train_excluded": train_excl,
        "test_excluded": test_excl,
        "train_patients": len(train_labels_ms),
        "test_patients": len(test_labels_ms)
    })

    if len(np.unique(train_labels_ms)) < 2 or len(np.unique(test_labels_ms)) < 2:
        skipped += 1
        continue

    auc = roc_auc_score(test_labels_ms, test_scores_ms)
    auc_rows.append({
        "split": split_i,
        "patient_auc": float(auc),
        "n_test_patients": int(len(test_labels_ms)),
        "n_test_ms": int(np.sum(test_labels_ms == 1)),
        "n_test_healthy": int(np.sum(test_labels_ms == 0))
    })

    thr_map = thresholds_from_train(train_labels_ms, train_scores_ms, SPEC_TARGETS)
    if len(thr_map) == 0:
        skipped += 1
        continue

    sweep_df = eval_thresholds_on_test(test_labels_ms, test_scores_ms, thr_map, PI_TARGET, SCREEN_N)
    sweep_df["split"] = split_i
    split_rows.append(sweep_df)

# Results
auc_splits = pd.DataFrame(auc_rows)
screen_splits = pd.concat(split_rows, ignore_index=True) if len(split_rows) else pd.DataFrame()
exclusion_df = pd.DataFrame(exclusion_log)

print(f"\nSplits: {N_SPLITS} requested | {len(auc_splits)} completed | {skipped} skipped")
print(f"Mean AUC: {auc_splits['patient_auc'].mean():.3f} ± {auc_splits['patient_auc'].std(ddof=0):.3f}")
print(f"\nExclusions (mean): Train={exclusion_df['train_excluded'].mean():.1f}, Test={exclusion_df['test_excluded'].mean():.1f}")

Path("src3/src3_results").mkdir(parents=True, exist_ok=True)
auc_splits.to_csv("src3/src3_results/screening_auc_corrected.csv", index=False)
screen_splits.to_csv("src3/src3_results/screening_sweep_corrected.csv", index=False)
exclusion_df.to_csv("src3/src3_results/screening_exclusions.csv", index=False)

if not screen_splits.empty:
    summary = screen_splits.groupby("specificity_target", as_index=False).agg(
        mean_sensitivity=("sensitivity", "mean"),
        sd_sensitivity=("sensitivity", lambda x: np.std(x, ddof=0)),
        mean_specificity=("specificity", "mean"),
        sd_specificity=("specificity", lambda x: np.std(x, ddof=0)),
        mean_FP_per_100k=("FP_per_100k", "mean"),
        mean_TP_per_100k=("TP_per_100k", "mean"),
        mean_referrals_per_100k=("referrals_per_100k", "mean"),
        mean_PPV=("PPV_at_pi", "mean"),
        n_splits=("split", "nunique")
    )

    summary.to_csv("src3/src3_results/screening_summary_corrected.csv", index=False)
    print("\nScreening Summary:")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.errorbar(summary["specificity_target"], summary["mean_sensitivity"],
                yerr=summary["sd_sensitivity"], marker="o", capsize=4)
    ax.set_xlabel("Specificity Target")
    ax.set_ylabel("Sensitivity (MS Detection)")
    ax.set_title("Sensitivity vs Specificity Target")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(summary["specificity_target"], summary["mean_FP_per_100k"], marker="o")
    ax.set_xlabel("Specificity Target")
    ax.set_ylabel("False Positives per 100k")
    ax.set_title("Screening Burden")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(summary["specificity_target"], summary["mean_referrals_per_100k"], marker="o")
    ax.set_xlabel("Specificity Target")
    ax.set_ylabel("Total Referrals per 100k")
    ax.set_title("Total Screening Referrals")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(summary["specificity_target"], summary["mean_PPV"], marker="o")
    ax.set_xlabel("Specificity Target")
    ax.set_ylabel("Positive Predictive Value")
    ax.set_title("PPV at Population Prevalence (1:400)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("src3/src3_results/screening_analysis_corrected.png", dpi=300)
    plt.close()

print("\nDone. Results saved to src3/src3_results/")
