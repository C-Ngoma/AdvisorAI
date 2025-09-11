import pandas as pd
import os

# Path to your CSV file
file_path = r"C:\Desktop\PROJECTT\datacollection\ai_job_dataset_merged.csv"

# Check if the file exists first
if not os.path.exists(file_path):
    print(f"❌ File not found at {file_path}. Check your path!")
else:
    # Preview first 5 rows
    df = pd.read_csv(file_path, nrows=5)
    print("\n✅ Columns in dataset:")
    print(df.columns.tolist())

    print("\n✅ Sample Data (first 5 rows):")
    print(df.head())

    # Full dataset info
    df_full = pd.read_csv(file_path)
    print("\n=== Data Types ===")
    print(df_full.dtypes)

    print("\n=== Missing Values ===")
    print(df_full.isnull().sum())

