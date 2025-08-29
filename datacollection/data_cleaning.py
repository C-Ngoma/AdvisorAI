import pandas as pd

# === Load datasets ===
df1 = pd.read_csv("datacollection/ai_job_dataset_raw.csv")
df2 = pd.read_csv("datacollection/ai_job_dataset1_raw.csv")

print("✅ Raw datasets loaded")
print("Dataset 1 shape:", df1.shape)
print("Dataset 2 shape:", df2.shape)

# === Clean Dataset 1 ===
df1.drop_duplicates(inplace=True)
if "salary_usd" in df1.columns:
    df1["salary_usd"] = df1["salary_usd"].fillna(df1["salary_usd"].mean())

# === Clean Dataset 2 ===
df2.drop_duplicates(inplace=True)
if "salary_usd" in df2.columns:
    df2["salary_usd"] = df2["salary_usd"].fillna(df2["salary_usd"].mean())
if "salary_local" in df2.columns:
    df2["salary_local"] = df2["salary_local"].fillna(df2["salary_local"].mean())

# === Save cleaned individual files ===
df1.to_csv("datacollection/ai_job_dataset1_cleaned.csv", index=False)
df2.to_csv("datacollection/ai_job_dataset2_cleaned.csv", index=False)
print("✅ Individual cleaned datasets saved")

# === Merge datasets on job_id ===
# We only take the extra column(s) from df2 (to avoid duplication of everything else)
if "salary_local" in df2.columns:
    merged_df = pd.merge(df1, df2[["job_id", "salary_local"]], on="job_id", how="left")
else:
    merged_df = df1.copy()

# === Save merged dataset ===
merged_df.to_csv("datacollection/ai_job_dataset_merged.csv", index=False)

print("✅ Final merged dataset saved!")
print("Merged dataset shape:", merged_df.shape)
print(merged_df.head())
