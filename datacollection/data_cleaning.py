import pandas as pd

# Load both datasets
df1 = pd.read_csv("datacollection/ai_job_dataset_raw.csv")
df2 = pd.read_csv("datacollection/ai_job_dataset1_raw.csv")

# Preview each
print("Dataset 1 Preview:")
print(df1.head())
print(df1.info())

print("\nDataset 2 Preview:")
print(df2.head())
print(df2.info())

# Clean Dataset 1
df1.drop_duplicates(inplace=True)
if "salary" in df1.columns:
    df1['salary'] = df1['salary'].fillna(df1['salary'].mean())

# Clean Dataset 2
df2.drop_duplicates(inplace=True)
if "salary" in df2.columns:
    df2['salary'] = df2['salary'].fillna(df2['salary'].mean())

# Save cleaned versions
df1.to_csv("datacollection/ai_job_dataset_cleaned.csv", index=False)
df2.to_csv("datacollection/ai_job_dataset1_cleaned.csv", index=False)

print("✅ Both datasets cleaned and saved!")
# Check for missing values
print("\nMissing values in Dataset 1:")
