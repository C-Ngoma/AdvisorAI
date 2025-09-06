import pandas as pd

# === Load merged dataset ===
df = pd.read_csv("datacollection/ai_job_dataset_merged.csv")
print("✅ Merged dataset loaded:", df.shape)

# === Select relevant columns ===
resume_df = df[['job_id', 'job_title', 'required_skills', 'years_experience', 'salary_usd']].copy()

# === Clean skills column ===
# - Convert to lowercase
# - Remove extra spaces
resume_df['required_skills'] = resume_df['required_skills'].str.lower().str.strip()

# === Handle missing values (if any) ===
resume_df['required_skills'].fillna("not specified", inplace=True)
resume_df['years_experience'].fillna(0, inplace=True)
resume_df['salary_usd'].fillna(resume_df['salary_usd'].mean(), inplace=True)

# === Save prepared dataset ===
resume_df.to_csv("datacollection/resume_matching_dataset.csv", index=False)

print("✅ Resume matching dataset prepared and saved!")
print(resume_df.head())
# Prepare dataset for resume matching
# - Select relevant columns