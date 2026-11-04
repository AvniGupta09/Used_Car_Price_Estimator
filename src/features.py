

from pathlib import Path

import pandas as pd
from src.data_cleaning import DataCleaning

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed.csv"



class FeatureEngineering:

    def __init__(self, data_cleaned):
        self.data_cleaned = data_cleaned
        self.processed = None

    def build(self):
        df = self.data_cleaned.copy()

        # keep the target as the last column
        target = "selling_price"
        columns = [c for c in df.columns if c != target] + [target]
        self.processed = df[columns]
        return self.processed

    def save(self, path=PROCESSED_PATH):
        self.processed.to_csv(path, index=False)
        print(f"Saved {path} with shape {self.processed.shape}")


def main():
    raw = pd.read_csv(PROJECT_ROOT / "data" / "raw.csv")
    cleaner = DataCleaning(raw)
    data_cleaned = cleaner.run()          # cleaned data kept in memory

    fe = FeatureEngineering(data_cleaned)
    fe.build()
    fe.save()

 
if __name__ == "__main__":
    main()