import pandas as pd

# Load the dataset
df = pd.read_csv('datacollection/ai_job_dataset_merged.csv')

# Output all rows and columns to a text file
df.to_string('full_dataset_output.txt')

print("Dataset written to full_dataset_output.txt")