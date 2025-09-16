import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('datacollection/ai_job_dataset_merged.csv')

# Parse the posting_date column
df['posting_date'] = pd.to_datetime(df['posting_date'])

# Group by date to get daily job demand (number of postings per day)
daily_demand = df.groupby('posting_date').size().reset_index(name='job_postings')

print("Job postings per day:")
print(daily_demand)

# Total number of job postings (entire dataset)
total_postings = len(df)
print(f"\nTotal job postings in dataset: {total_postings}")

# Plot daily job demand (number of postings per day)
plt.figure(figsize=(12,5))
plt.plot(daily_demand['posting_date'], daily_demand['job_postings'], marker='o')
plt.title('Job Demand: Number of Job Postings Per Day')
plt.xlabel('Date')
plt.ylabel('Number of Postings')
plt.tight_layout()

plt.show()