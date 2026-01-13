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
