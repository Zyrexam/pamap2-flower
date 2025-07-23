import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


activityIDdict = {
    0: 'transient', 1: 'lying', 2: 'sitting', 3: 'standing', 4: 'walking',
    5: 'running', 6: 'cycling', 7: 'Nordic_walking', 9: 'watching_TV',
    10: 'computer_work', 11: 'car driving', 12: 'ascending_stairs',
    13: 'descending_stairs', 16: 'vacuum_cleaning', 17: 'ironing',
    18: 'folding_laundry', 19: 'house_cleaning', 20: 'playing_soccer', 24: 'rope_jumping'
}


colNames = ["timestamp", "activityID","heartrate"]

IMUhand = [
    'handTemperature', 'handAcc16_1', 'handAcc16_2', 'handAcc16_3',
    'handAcc6_1', 'handAcc6_2', 'handAcc6_3', 'handGyro1', 'handGyro2', 'handGyro3',
    'handMagne1', 'handMagne2', 'handMagne3', 'handOrientation1', 'handOrientation2',
    'handOrientation3', 'handOrientation4'
]

IMUchest = [
    'chestTemperature', 'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3',
    'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3', 'chestGyro1', 'chestGyro2', 'chestGyro3',
    'chestMagne1', 'chestMagne2', 'chestMagne3', 'chestOrientation1', 'chestOrientation2',
    'chestOrientation3', 'chestOrientation4'
]

IMUankle = [
    'ankleTemperature', 'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3',
    'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3', 'ankleGyro1', 'ankleGyro2', 'ankleGyro3',
    'ankleMagne1', 'ankleMagne2', 'ankleMagne3', 'ankleOrientation1', 'ankleOrientation2',
    'ankleOrientation3', 'ankleOrientation4'
]

columns = colNames + IMUhand + IMUchest + IMUankle


def data_cleaning(df):
    """
    Cleans raw PAMAP2 dataframe: drops unwanted columns, removes activityID==0, fills NaNs.
    """
    df = df.drop([
        "timestamp", "heartrate",
        'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4',
        'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4',
        'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4',
        'handTemperature','chestTemperature','ankleTemperature',
        'handAcc6_1', 'handAcc6_2', 'handAcc6_3',
        'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3',
        'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3',
        'handMagne1', 'handMagne2', 'handMagne3',
        'chestMagne1', 'chestMagne2', 'chestMagne3',
        'ankleMagne1', 'ankleMagne2', 'ankleMagne3',
    ], axis=1)
    df = df[df["activityID"] != 0]  # remove transient rows
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.interpolate()  # fill NaNs
    return df


def standard_scale(df):
    """
    Standard scales sensor features; keeps activityID unchanged.
    """
    features = df.drop(columns=["activityID"])
    labels = df["activityID"]
    features_z = (features - features.mean()) / (features.std(ddof=0) + 1e-8)
    return pd.concat([features_z, labels], axis=1)


def data_preprocessing(df):
    """
    Applies standard scaling and moves activityID column to the end.
    """
    scaled_df = standard_scale(df)
    cols = [col for col in scaled_df.columns if col != "activityID"] + ["activityID"]
    return scaled_df[cols]


def create_fixed_windows(df, window_size=50, shift=25):
    """
    Splits scaled dataframe into overlapping windows.
    Label = mode of window activityID.
    Returns: list of np.arrays (windows), list of int labels
    """
    windows, labels = [], []
    N = len(df)
    for start in range(0, N - window_size + 1, shift):
        window = df.iloc[start:start+window_size]
        window_label = int(window["activityID"].mode()[0])
        window_features = window.drop(columns=["activityID"]).to_numpy(dtype=np.float32)
        windows.append(window_features)
        labels.append(window_label)
    return windows, labels


class IMUDataset(Dataset):
    """
    Simple PyTorch Dataset: returns (window_tensor, label).
    """
    def __init__(self, windows, labels):
        self.windows = windows
        self.labels = labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx], dtype=torch.float32)
        y = self.labels[idx]
        return x, y
