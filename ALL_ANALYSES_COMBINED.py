import pandas as pd
import numpy as np
from pathlib import Path

# FitMRI dataset
df_FitMRI = pd.read_csv('data/FitMRI_fitbit_intraday_steps_trainingData.csv')
df_FitMRI['datetime'] = pd.to_datetime(
    df_FitMRI['measured_date'] + ' ' + df_FitMRI['measured_time'],
    format='%d-%b-%y %H:%M:%S'
)
df_FitMRI = df_FitMRI.rename(columns={'fitmri_id': 'id'})
df_FitMRI.drop(columns=['measured_date', 'measured_time'], inplace=True)
df_FitMRI = (
    df_FitMRI
      .set_index('datetime')
      .groupby('id')['steps']
      .resample('h')
      .sum()
      .reset_index()
)
df_FitMRI['steps'] = df_FitMRI['steps'].fillna(0).astype(int)
df_FitMRI['label'] = 0
df_FitMRI['source'] = 'FitMRI'
df_FitMRI = df_FitMRI[['id', 'datetime', 'steps', 'label', 'source']]

# Kaggle dataset
df_Kaggle1 = pd.read_csv('hourlySteps_merged_31216_41116.csv')
df_Kaggle2 = pd.read_csv('hourlySteps_merged_41216_51216.csv')
df_Kaggle = pd.concat([df_Kaggle1, df_Kaggle2], ignore_index=True)
df_Kaggle = df_Kaggle.rename(columns={
    'Id': 'id', 
    'ActivityHour': 'datetime', 
    'StepTotal': 'steps'
    })
df_Kaggle['datetime'] = pd.to_datetime(
    df_Kaggle['datetime'],
    format='%m/%d/%Y %I:%M:%S %p'
)
df_Kaggle = (
    df_Kaggle
      .set_index('datetime')
      .groupby('id')['steps']
      .resample('h')
      .sum()
      .reset_index()
)
df_Kaggle['steps'] = df_Kaggle['steps'].fillna(0).astype(int)
df_Kaggle['label'] = 1
df_Kaggle['source'] = 'Kaggle_Healthy'
df_Kaggle = df_Kaggle[['id', 'datetime', 'steps', 'label', 'source']]

hours_per_user = df_Kaggle.groupby('id')['datetime'].nunique()
kaggle_quality = hours_per_user[hours_per_user >= 12].index
df_Kaggle = df_Kaggle[df_Kaggle['id'].isin(kaggle_quality)]

# Sema dataset
df_sema = pd.read_csv(
    'hourly_fitbit_sema_df_unprocessed.csv',
    usecols=['id', 'date', 'hour', 'steps'],
    low_memory=False
)
df_sema['datetime'] = pd.to_datetime(df_sema['date']) + pd.to_timedelta(df_sema['hour'], unit='h')
df_sema.drop(columns=['date', 'hour'], inplace=True)
df_sema = (
    df_sema
      .set_index('datetime')
      .groupby('id')['steps']
      .resample('h')
      .sum()
      .reset_index()
)
df_sema['steps'] = df_sema['steps'].fillna(0).astype(int)
df_sema['label'] = 1
df_sema['source'] = 'SEMA_Healthy'
df_sema = df_sema[['id', 'datetime', 'steps', 'label', 'source']]

hours_per_user = df_sema.groupby('id')['datetime'].nunique()
sema_quality = hours_per_user[hours_per_user >= 12].index
df_sema = df_sema[df_sema['id'].isin(sema_quality)]

df_all = pd.concat([df_FitMRI, df_Kaggle, df_sema], ignore_index=True)

n_ms = df_FitMRI['id'].nunique()
n_kaggle = df_Kaggle['id'].nunique()
n_sema = df_sema['id'].nunique()
n0 = np.sum(df_all['label'] == 0)
n1 = np.sum(df_all['label'] == 1)
total = n0 + n1
cw_0 = total / (2 * n0)
cw_1 = total / (2 * n1)
out_path = Path('src3/src3_results')
out_path.mkdir(parents=True, exist_ok=True)
df_all.to_csv(out_path / 'combined_data.csv', index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit, GroupKFold


df = pd.read_csv('data/combined_all_hourly_data.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"Loaded {len(df)} hourly measurements")

df['day'] = df['datetime'].dt.date

daily_features = []
for (pid, day), g in df.groupby(['id', 'day']):
    steps = g['steps'].values
    daily_features.append({
        'id': pid,
        'day': day,
        'label': g['label'].iloc[0],
        'mean_steps': np.mean(steps),
        'std_steps': np.std(steps),
        'max_steps': np.max(steps),
        'min_steps': np.min(steps),
        'total_steps': np.sum(steps),
        'cv_steps': np.std(steps) / (np.mean(steps) + 1e-8),
        'num_hours': len(steps),
    })

df_daily = pd.DataFrame(daily_features)
print(f"Extracted {len(df_daily)} daily feature vectors")

feature_cols = [
    'mean_steps', 'std_steps', 'max_steps',
    'min_steps', 'total_steps', 'cv_steps', 'num_hours'
]

df_daily[feature_cols].hist(bins=50, figsize=(14, 10))
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.savefig('src3/src3_results/feature_distributions.png', dpi=300)
plt.close()

X = df_daily[feature_cols].values
y = df_daily['label'].values
groups = df_daily['id'].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

scaler_dict = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'No Scaling': None
}

results = []

for name, scaler in scaler_dict.items():

    if scaler is not None:
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
    else:
        X_tr, X_te = X_train, X_test

    n0 = np.sum(y_train == 0)
    n1 = np.sum(y_train == 1)
    total = n0 + n1
    class_weight = {0: total / (2 * n0), 1: total / (2 * n1)}

    model = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)
    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    print(f"{name:20s} | AUC={auc:.3f} | P={precision:.3f} | R={recall:.3f} | F1={f1:.3f}")

    results.append({
        'scaler': name,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'tpr': tpr
    })

pd.DataFrame([{k: v for k, v in r.items() if k not in ['fpr', 'tpr']} for r in results]) \
  .to_csv('src3/src3_results/standardization_test_results.csv', index=False)

gkf = GroupKFold(n_splits=5)
cv_results = []

for name, scaler_template in scaler_dict.items():
    aucs = []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if scaler_template is not None:
            scaler = scaler_template.__class__()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        n0 = np.sum(y_tr == 0)
        n1 = np.sum(y_tr == 1)
        total = n0 + n1
        class_weight = {0: total / (2 * n0), 1: total / (2 * n1)}

        model = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)
        model.fit(X_tr, y_tr)
        aucs.append(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))

    cv_results.append({
        'scaler': name,
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs),
        'aucs': aucs
    })

pd.DataFrame([{k: v for k, v in r.items() if k != 'aucs'} for r in cv_results]) \
  .to_csv('src3/src3_results/standardization_cv_results.csv', index=False)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
for r in results:
    ax.plot(r['fpr'], r['tpr'], label=f"{r['scaler']} ({r['auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--')
ax.legend()

ax = axes[0, 1]
ax.bar([r['scaler'] for r in results], [r['auc'] for r in results])
ax.set_ylim(0, 1)

ax = axes[1, 0]
ax.boxplot([r['aucs'] for r in cv_results])
ax.set_xticklabels([r['scaler'] for r in cv_results], rotation=45, ha='right')
ax.set_ylim(0, 1)

ax = axes[1, 1]
baseline = next(r for r in cv_results if r['scaler'] == 'StandardScaler')['mean_auc']
deltas = [r['mean_auc'] - baseline for r in cv_results]

ax.bar([r['scaler'] for r in cv_results], deltas)
ax.axhline(0, color='k', linestyle='--')
ax.set_ylabel('Δ AUC vs StandardScaler')


plt.tight_layout()
plt.savefig('src3/src3_results/standardization_comparison.png', dpi=300)
plt.close()

best_test = max(results, key=lambda x: x['auc'])
best_cv = max(cv_results, key=lambda x: x['mean_auc'])

print(f"Best test: {best_test['scaler']} ({best_test['auc']:.3f})")
print(f"Best CV: {best_cv['scaler']} ({best_cv['mean_auc']:.3f})")
print("Done.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

df = pd.read_csv('data/combined_all_hourly_data.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"Loaded {len(df)} hourly measurements")

df['day'] = df['datetime'].dt.date

daily_features = []
for (pid, day), g in df.groupby(['id', 'day']):
    steps = g['steps'].values
    daily_features.append({
        'id': pid,
        'day': day,
        'label': g['label'].iloc[0],
        'mean_steps': np.mean(steps),
        'std_steps': np.std(steps, ddof=0),
        'max_steps': np.max(steps),
        'min_steps': np.min(steps),
        'total_steps': np.sum(steps),
        'cv_steps': np.std(steps, ddof=0) / (np.mean(steps) + 1e-8),
        'num_hours': len(steps),
    })

df_daily = pd.DataFrame(daily_features)
print(f"Extracted {len(df_daily)} daily feature vectors")

feature_cols = [
    'mean_steps', 'std_steps', 'max_steps',
    'min_steps', 'total_steps', 'cv_steps', 'num_hours'
]

X = df_daily[feature_cols].values
y = df_daily['label'].values
groups = df_daily['id'].values

K_list = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]

days_results = []

for i, K in enumerate(K_list):
    gkf = GroupKFold(n_splits=4)
    fold_aucs = []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        train_patients = df_daily.iloc[train_idx]['id'].unique()
        test_patients  = df_daily.iloc[test_idx]['id'].unique()

        train_data = df_daily[df_daily['id'].isin(train_patients)].copy()
        test_data  = df_daily[df_daily['id'].isin(test_patients)].copy()

        X_train = train_data[feature_cols].values
        y_train = train_data['label'].values

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        n0 = np.sum(y_train == 0)
        n1 = np.sum(y_train == 1)
        total = n0 + n1
        class_weight = {0: total / (2 * n0), 1: total / (2 * n1)}

        model = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)

        patient_scores = []
        patient_labels = []

        for pid, g in test_data.sort_values('day').groupby('id'):
            if len(g) < K:
                continue
            gK = g.head(K)

            Xp = gK[feature_cols].values
            Xp_scaled = scaler.transform(Xp)
            prob_days = model.predict_proba(Xp_scaled)[:, 1]

            patient_scores.append(prob_days.mean())
            patient_labels.append(gK['label'].iloc[0])

        if len(set(patient_labels)) < 2:
            continue

        fold_aucs.append(roc_auc_score(patient_labels, patient_scores))

    if len(fold_aucs) > 0:
        days_results.append({
            'K_days': K,
            'mean_auc': float(np.mean(fold_aucs)),
            'std_auc': float(np.std(fold_aucs, ddof=0)),
            'n_folds': len(fold_aucs)
        })

days_df = pd.DataFrame(days_results)


days_df.to_csv('src3/src3_results/days_vs_auc.csv', index=False)

plt.figure(figsize=(8, 5))
plt.errorbar(days_df['K_days'], days_df['mean_auc'], yerr=days_df['std_auc'], marker='o')
plt.ylim(0, 1)
plt.xlabel('Days per patient (K)')
plt.ylabel('Patient-level ROC-AUC')
plt.title('How many days are sufficient?')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('src3/src3_results/days_vs_auc.png', dpi=300)
plt.close()

print(days_df)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, roc_auc_score

# ----------------------------
# Config
# ----------------------------
DATA_PATH = "src3/src3_results/combined_data.csv"

K_DAYS = 4
N_SPLITS = 200
TEST_SIZE = 0.25
RANDOM_STATE = 42

PI_TARGET = 1 / 400
SCREEN_N = 100000
SPEC_TARGETS = [0.95, 0.97, 0.98, 0.99, 0.995, 0.999]

FEATURE_COLS = [
    "mean_steps", "std_steps", "max_steps",
    "min_steps", "total_steps", "cv_steps", "num_hours"
]

# ----------------------------
# Load + clean
# ----------------------------
df = pd.read_csv(DATA_PATH, low_memory=False)

# stabilize types
df["id"] = df["id"].astype(str)
df["datetime"] = pd.to_datetime(df["datetime"])

# ensure numeric steps
df["steps"] = pd.to_numeric(df["steps"], errors="coerce").fillna(0.0)

df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(int)

df["day"] = df["datetime"].dt.date

print(f"Loaded hourly rows: {len(df)}")
print("Columns:", list(df.columns))

# ----------------------------
# Build daily feature table
# ----------------------------
daily_features = []
for (pid, day), g in df.groupby(["id", "day"]):
    steps = g["steps"].values
    daily_features.append({
        "id": pid,
        "day": day,
        "label": int(g["label"].iloc[0]),
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
y_all = df_daily["label"].values
groups_all = df_daily["id"].values

n_patients = df_daily["id"].nunique()
n_ms = (df_daily[["id", "label"]].drop_duplicates()["label"] == 0).sum()
n_healthy = (df_daily[["id", "label"]].drop_duplicates()["label"] == 1).sum()

print(f"Built daily rows: {len(df_daily)}")
print(f"Patients: {n_patients} | MS: {n_ms} | Healthy: {n_healthy}")

# ----------------------------
# Helpers
# ----------------------------
def patient_scores_first_k_days(df_days, model, scaler, feature_cols, k):
    labels, scores = [], []

    for pid, g in df_days.sort_values("day").groupby("id"):
        gk = g.head(k)
        if len(gk) < k:
            continue
        Xp = scaler.transform(gk[feature_cols].values)
        scores.append(model.predict_proba(Xp)[:, 1].mean())
        labels.append(int(gk["label"].iloc[0]))

    return np.array(labels), np.array(scores)

def class_weight_from_y(y):
    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    total = n0 + n1
    return {0: total / (2 * n0), 1: total / (2 * n1)}

def thresholds_from_train(train_labels, train_scores, spec_targets):
    fpr, tpr, thr = roc_curve(train_labels, train_scores)
    spec = 1 - fpr

    out = {}
    for tspec in spec_targets:
        idx = np.where(spec >= tspec)[0]
        if len(idx) == 0:
            continue
        best_idx = idx[np.argmax(tpr[idx])]
        out[tspec] = float(thr[best_idx])
    return out

def eval_thresholds_on_test(test_labels, test_scores, thresholds, pi_target, screen_n):
    rows = []
    for tspec, thr in thresholds.items():
        pred_healthy = (test_scores >= thr).astype(int)

        tp = np.sum((pred_healthy == 0) & (test_labels == 0))
        fp = np.sum((pred_healthy == 0) & (test_labels == 1))
        tn = np.sum((pred_healthy == 1) & (test_labels == 1))
        fn = np.sum((pred_healthy == 1) & (test_labels == 0))

        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)

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

# ----------------------------
# Repeated patient-level evaluation
# ----------------------------
gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)

split_rows = []
auc_rows = []
skipped = 0

for split_i, (train_idx, test_idx) in enumerate(gss.split(X_all, y_all, groups=groups_all), start=1):
    train_patients = df_daily.iloc[train_idx]["id"].unique()
    test_patients  = df_daily.iloc[test_idx]["id"].unique()

    train_data = df_daily[df_daily["id"].isin(train_patients)].copy()
    test_data  = df_daily[df_daily["id"].isin(test_patients)].copy()

    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_data[FEATURE_COLS].values)
    y_train = train_data["label"].values

    model = LogisticRegression(
        class_weight=class_weight_from_y(y_train),
        max_iter=2000,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    train_labels_pat, train_scores_pat = patient_scores_first_k_days(train_data, model, scaler, FEATURE_COLS, K_DAYS)
    test_labels_pat,  test_scores_pat  = patient_scores_first_k_days(test_data,  model, scaler, FEATURE_COLS, K_DAYS)

    if len(np.unique(train_labels_pat)) < 2 or len(np.unique(test_labels_pat)) < 2:
        skipped += 1
        continue

    auc_rows.append({
        "split": split_i,
        "patient_auc": float(roc_auc_score(test_labels_pat, test_scores_pat)),
        "n_test_patients": int(len(test_labels_pat))
    })

    thr_map = thresholds_from_train(train_labels_pat, train_scores_pat, SPEC_TARGETS)
    if len(thr_map) == 0:
        skipped += 1
        continue

    sweep_df = eval_thresholds_on_test(test_labels_pat, test_scores_pat, thr_map, PI_TARGET, SCREEN_N)
    sweep_df["split"] = split_i
    split_rows.append(sweep_df)

auc_splits = pd.DataFrame(auc_rows)
screen_splits = pd.concat(split_rows, ignore_index=True) if len(split_rows) else pd.DataFrame()

print(f"Splits requested: {N_SPLITS} | Completed: {len(auc_splits)} | Skipped: {skipped}")
if not auc_splits.empty:
    print(f"Mean patient-level AUC over splits: {auc_splits['patient_auc'].mean():.3f} ± {auc_splits['patient_auc'].std(ddof=0):.3f}")

auc_splits.to_csv("src3/src3_results/screening_patient_auc_splits.csv", index=False)
screen_splits.to_csv("src3/src3_results/screening_threshold_sweep_splits.csv", index=False)

if not screen_splits.empty:
    summary = (screen_splits
               .groupby(["specificity_target"], as_index=False)
               .agg(
                   mean_sensitivity=("sensitivity", "mean"),
                   sd_sensitivity=("sensitivity", lambda x: float(np.std(x, ddof=0))),
                   mean_specificity=("specificity", "mean"),
                   sd_specificity=("specificity", lambda x: float(np.std(x, ddof=0))),
                   mean_FP_per_100k=("FP_per_100k", "mean"),
                   mean_TP_per_100k=("TP_per_100k", "mean"),
                   mean_referrals_per_100k=("referrals_per_100k", "mean"),
                   n_splits=("split", "nunique"),
               ))

    summary.to_csv("src3/src3_results/screening_summary_option1.csv", index=False)
    print("\nScreening summary (averaged across splits):")
    print(summary)

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        summary["specificity_target"],
        summary["mean_sensitivity"],
        yerr=summary["sd_sensitivity"],
        marker="o",
        capsize=4
    )
    plt.ylim(0, 1)
    plt.xlabel("Specificity target (chosen on train; evaluated on test)")
    plt.ylabel("Sensitivity on test (mean ± SD across splits)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("src3/src3_results/screening_sensitivity_at_specificity.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(summary["specificity_target"], summary["mean_FP_per_100k"], marker="o")
    plt.xlabel("Specificity target (chosen on train; evaluated on test)")
    plt.ylabel("Expected false positives per 100k screened")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("src3/src3_results/screening_fp_burden_per_100k.png", dpi=300)
    plt.close()

print("\nDone.")
