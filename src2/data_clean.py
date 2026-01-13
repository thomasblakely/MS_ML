import numpy as np
import pandas as pd

def load_data():
    df = pd.read_csv('data/daily_fitbit_sema_df_unprocessed.csv')
    print('Healthy data loaded')
    return df

def check_dates(df):
    df = df.dropna(subset=['steps'])
    summary = df.groupby('id')['steps'].nunique().reset_index(name='num_days')
    max_day = summary.loc[summary['num_days'].idxmax()]
    min_day = summary.loc[summary['num_days'].idxmin()]
    print('Summary')
    print(summary)
    print(f"max_id: {max_day}")
    print(f"min_id: {min_day}")
    return summary

def sort_by_dates(summary):
    # ********ISSUE*********
    # MISLEADING VARIABLE NAME: Variable named "top_20" but selects 30 patients
    # Print says "top 20" but actually shows 30 (functional code is correct, just confusing naming)
    top_20 = summary.query("num_days >= 12").nlargest(30, 'num_days').reset_index(drop=True)
    print("Here's the top 20 patients:")
    print(top_20)
    return top_20

def clean_data(df, top_20):
    top_id = top_20['id']
    df_clean = df[df['id'].isin(top_id)].reset_index(drop=True)
    df_clean = df_clean.loc[:,['id','steps']]
    print(f"This is cleaned df:{df_clean}")
    print(df_clean.head())
    return df_clean

def fill_na(df_clean):
    for id in df_clean['id'].unique():
        sub_df = df_clean[df_clean['id'] == id]
        mean_sub = sub_df['steps'].mean()
        df_clean.loc[df_clean['id'] == id, 'steps'] = sub_df['steps'].fillna(mean_sub).round(2)
    return df_clean

def add_label(df_clean):
    df_clean['label'] = 1
    return df_clean

def save_data(df_clean):
    df_clean.to_csv('unsupervised/cleaned_data.csv', index=False)
    print('Data saved')

def main():
    df = load_data()
    summary = check_dates(df)
    save_id = sort_by_dates(summary)
    df_c = clean_data(df, save_id)
    df_c = fill_na(df_c)
    df_c = add_label(df_c)
    save_data(df_c)

if __name__ == '__main__':
    main()