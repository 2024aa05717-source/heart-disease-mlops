import pandas as pd

def test_processed_data_exists():
    df = pd.read_csv("data/processed/heart_clean.csv")
    assert not df.empty

def test_target_is_binary():
    df = pd.read_csv("data/processed/heart_clean.csv")
    assert set(df["target"].unique()).issubset({0, 1})
