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
