import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv('unsupervised/MS&health_data.csv')
    return df

def make_windows(df, window, interval):
    windows = []
    for pid, sub in df.groupby('id'):
        steps = sub['steps'].values
        label = sub['label'].iloc[0]

        n = len(steps)
        for i in range(0, n - window + 1, interval):
            segment = steps[i : i + window]
            windows.append((pid, segment, label))
    df_windows = pd.DataFrame(windows, columns=['id','steps', 'label'])
    print('This is the windows dataframe')
    print(df_windows.head())
    return df_windows

def extract_features(series, low_th, high_th):
    mean_val = np.mean(series)
    std_val = np.std(series)
    cv_val = std_val / (mean_val + 1e-8)

    x = np.arange(len(series))
    # ********ISSUE*********
    # Fitting x vs x (always slope=1). Should be np.polyfit(x, series, 1)[0] to get trend of steps over time
    slope = np.polyfit(x, x, 1)[0]

    # ********ISSUE*********
    # ACF calculation error: correlating index positions (x) instead of step values (series)
    # Should be: acf7 = np.corrcoef(series[:-7], series[7:])[0, 1]
    if len(series) > 7:
        acf7 = np.corrcoef(x[:-7], x[7:])[0, 1]
    else:
        acf7 = np.nan

    low_ratio  = np.mean(series < low_th)
    high_ratio = np.mean(series > high_th)

    return {
        'mean': mean_val,
        'std': std_val,
        'cv': cv_val,
        'trend': slope,
        'acf': acf7,
        'low_ratio': low_ratio,
        'high_ratio': high_ratio
    }

def build_windows(df, window, interval):
    df_windows = make_windows(df, window = window, interval = interval)

    # ********ISSUE*********
    # DATA LEAKAGE: Thresholds calculated on ALL data before train/test split
    # Should calculate only on training data after split
    # ********ISSUE*********
    # LABEL INCONSISTENCY: label==0 is MS patients, not healthy. Should be label==1 for healthy controls
    health_steps = df[df['label'] == 0]['steps']
    low_th = np.percentile(health_steps, 10)
    high_th = np.percentile(health_steps, 90)

    features_list = df_windows['steps'].apply(lambda arr: extract_features(arr, low_th, high_th))
    df_feat = pd.DataFrame(features_list.tolist())

    df_features = pd.concat([df_windows[['id', 'label']], df_feat], axis = 1)
    print('This is the features dataframe')
    print (df_features.head())
    return df_features

def save_data(df_features):
    df_features.to_csv('unsupervised/features.csv', index = False)
    print('This is the saved data')
    return df_features



def main():
    dfo = load_data()
    df_features = build_windows(dfo, 14, 7)
    save_data(df_features)
if __name__ == '__main__':
    main()