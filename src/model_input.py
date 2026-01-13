import numpy as np
import pandas as pd

def load_data():
    df = pd.read_csv('output/analysis_result.csv')
    return df

def split_data(df, train_ratio):
    # ********ISSUE*********
    # TEMPORAL SPLIT ONLY FOR MS PATIENTS: This splits by time for MS patients (fitmri_id)
    # Healthy controls from Kaggle don't have this field, so asymmetric treatment
    # Only MS patients get temporal 80/20 split, healthy controls handled elsewhere
    train_df = []
    test_df = []
    for pid, col in df.groupby('fitmri_id'):
        group = col.sort_values('measured_date')
        group = group[['fitmri_id', 'total_steps_normalized']]
        idx = int(len(group) * train_ratio)
        train_df.append(group[:idx])
        test_df.append(group[idx:])

    train_df = pd.concat(train_df, ignore_index=True)
    test_df = pd.concat(test_df, ignore_index=True)
    return train_df, test_df

def making_label(df, window, threshold):
    x_list = []
    y_list = []
    g_list = []
    for pid, col in df.groupby('fitmri_id'):
        group = col
        value = group['total_steps_normalized'].values

        for i in range(len(value) - 2 * window):
            past = value[i : i + window]
            future = value[i + window : i + 2 * window]

            mean_past = np.mean(past)
            mean_future = np.mean(future)
            rate = (mean_future - mean_past) / mean_past
            if rate < -threshold:
                label = 0
            elif rate > threshold:
                label = 1
            else:
                label = 0
            x_list.append(past)
            y_list.append(label)
            g_list.append(pid)
    x = np.array(x_list)
    y = np.array(y_list)
    return x, y

def save_data(train_df, test_df):
    train_df.to_csv('output/train.csv', index=False)
    test_df.to_csv('output/test.csv', index=False)
    print('Train and Test data saved')

def save_x_y(x_train, y_train, x_test, y_test):
    np.save("output/ready/x_train.npy", x_train)
    np.save("output/ready/y_train.npy", y_train)
    np.save("output/ready/x_test.npy", x_test)
    np.save("output/ready/y_test.npy", y_test)
    print('x and y data saved')

def main():
    df = load_data()
    train_df, test_df = split_data(df, 0.8)   # <------ if anyone wants to change the dataset ratio.
    save_data(train_df, test_df)

    x_train, y_train = making_label(train_df, 30,0.1)  # <------needs to be refined in the future. The relationship between the threshold and rate.
    x_test, y_test = making_label(test_df, 30,0.1)     # <------train and test, these two must be aligned.
    save_x_y(x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    main()