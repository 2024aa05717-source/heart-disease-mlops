import pandas as pd
import os

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

def main():
    os.makedirs("data/raw", exist_ok=True)
    df = pd.read_csv(URL, names=COLUMNS)
    df.to_csv("data/raw/heart.csv", index=False)
    print("✅ Heart disease dataset downloaded")

if __name__ == "__main__":
    main()
