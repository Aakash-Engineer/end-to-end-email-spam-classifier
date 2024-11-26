import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path




class DataIngestion:
    def __init__(self):
        pass

    def main(self, data_path: Path = Path('../aftifacts/data.csv')):
        try:
            df = pd.read_csv(data_path)
            # train test split
            x_train, x_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.05, random_state=42)
            # save train and test data
            pd.concat([x_train, y_train], axis=1).to_csv(Path('../artifacts/training/train.csv'), index=False)
            pd.concat([x_test, y_test], axis=1).to_csv(Path('../artifacts/training/test.csv'), index=False)
        except Exception as e:
            print(e)
            raise Exception("Error in data ingestion")
