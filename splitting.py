import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/cleaned.csv")
OUTPUT_DIR = Path("splits")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv(INPUT_FILE)

    #we sort the data based on their date columns
    print("Sorting rows by chronological order...")
    df = df.sort_values(by=["year", "month", "day", "hour"]).reset_index(drop=True)

    total_rows = len(df)
    print(f"Total rows: {total_rows}")

    train_end = int(total_rows * 0.35)
    val_end = int(total_rows * 0.70)

    print(f"Train end index: {train_end}")
    print(f"Validation end index: {val_end}")

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "validate.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\nSaved splits to 'splits/' directory:")
    print(" - train.csv")
    print(" - validate.csv")
    print(" - test.csv")


if __name__ == "__main__":
    main()
