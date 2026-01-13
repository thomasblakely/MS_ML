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

test_results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['fpr', 'tpr']} for r in results])
test_results_df.to_csv('src3/src3_results/standardization_test_results.csv', index=False)

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

cv_results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'aucs'} for r in cv_results])
cv_results_df.to_csv('src3/src3_results/standardization_cv_results.csv', index=False)

# Combine test and CV results for comprehensive output
combined_results = test_results_df.merge(cv_results_df, on='scaler', suffixes=('_test', '_cv'))
combined_results.to_csv('src3/src3_results/standardization_combined_results.csv', index=False)
print("\nStandardization Results:")
print(combined_results.to_string(index=False))

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

for r in results:
    ax.plot(r['fpr'], r['tpr'], label=f"{r['scaler']} (AUC={r['auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves: Standardization Methods Comparison')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# ax = axes[0, 1]
# ax.bar([r['scaler'] for r in results], [r['auc'] for r in results])
# ax.set_ylim(0, 1)

# ax = axes[1, 0]
# ax.boxplot([r['aucs'] for r in cv_results])
# ax.set_xticklabels([r['scaler'] for r in cv_results], rotation=45, ha='right')
# ax.set_ylim(0, 1)

# ax = axes[1, 1]
# baseline = next(r for r in cv_results if r['scaler'] == 'StandardScaler')['mean_auc']
# deltas = [r['mean_auc'] - baseline for r in cv_results]
# ax.bar([r['scaler'] for r in cv_results], deltas)
# ax.axhline(0, color='k', linestyle='--')
# ax.set_ylabel('Δ AUC vs StandardScaler')


plt.tight_layout()
plt.savefig('src3/src3_results/standardization_comparison.png', dpi=300)
plt.close()

best_test = max(results, key=lambda x: x['auc'])
best_cv = max(cv_results, key=lambda x: x['mean_auc'])

print(f"Best test: {best_test['scaler']} ({best_test['auc']:.3f})")
print(f"Best CV: {best_cv['scaler']} ({best_cv['mean_auc']:.3f})")
print("Done.")
