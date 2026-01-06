import pandas as pd
import numpy as np
import os

def preprocess():
    input_path = "data/raw/heart.csv"
    output_path = "data/processed/heart_clean.csv"

    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv(input_path)

    # Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric)

    # Convert target to binary (0 = no disease, 1 = disease)
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    df.to_csv(output_path, index=False)
    print("✅ Data preprocessing completed")

if __name__ == "__main__":
    preprocess()
