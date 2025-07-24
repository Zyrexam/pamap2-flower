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

# ✅ number of features after cleaning:
# remove unwanted columns → keep handAcc16, chestAcc16 etc.
# save names to split correctly later
clean_hand_cols = ['handAcc16_1', 'handAcc16_2', 'handAcc16_3', 'handGyro1', 'handGyro2', 'handGyro3']
clean_chest_cols = ['chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 'chestGyro1', 'chestGyro2', 'chestGyro3']
clean_ankle_cols = ['ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 'ankleGyro1', 'ankleGyro2', 'ankleGyro3']

# in total: 6+6+6=18 features
final_feature_cols = clean_hand_cols + clean_chest_cols + clean_ankle_cols

def data_cleaning(df):
    """
    Cleans dataframe: drop unwanted columns, remove activityID==0, fill NaNs.
    """
    # drop all except needed columns
    to_keep = ["activityID"] + final_feature_cols
    df = df[to_keep]
    df = df[df["activityID"] != 0]
    df = df.apply(pd.to_numeric, errors='coerce').interpolate()
    return df

def standard_scale(df):
    """
    Standard scale sensor features; keep activityID unchanged.
    """
    features = df.drop(columns=["activityID"])
    labels = df["activityID"]
    features_z = (features - features.mean()) / (features.std(ddof=0) + 1e-8)
    return pd.concat([features_z, labels], axis=1)

def data_preprocessing(df):
    scaled_df = standard_scale(df)
    cols = [col for col in scaled_df.columns if col != "activityID"] + ["activityID"]
    return scaled_df[cols]

def create_fixed_windows(df, window_size=50, shift=25):
    """
    Create windows. Each window is shape (time, features)
    """
    windows, labels = [], []
    N = len(df)
    for start in range(0, N - window_size + 1, shift):
        window = df.iloc[start:start+window_size]
        label = int(window["activityID"].mode()[0])
        window_features = window.drop(columns=["activityID"]).to_numpy(dtype=np.float32)
        windows.append(window_features)
        labels.append(label)
    return windows, labels

class IMUDataset(Dataset):
    """
    Dataset returns (window, label)
    window: (time, features)
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
