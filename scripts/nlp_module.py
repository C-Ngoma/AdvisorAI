import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load your dataset
df = pd.read_csv(r"C:\Desktop\PROJECTT\datacollection\ai_job_dataset_merged.csv")

# Initialize spaCy for entity extraction
nlp = spacy.load("en_core_web_sm")

# Function to extract skills (simple example)
def extract_skills(text):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "SKILL"]]  # can adjust labels
    return skills

# Prepare TF-IDF for job_title (for intent detection)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['job_title'])

# For now, use 'industry' as the intent label
y = df['industry']

# Train a simple intent detection model
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

# Example user query
query = "Looking for an AI research position requiring Python and ML"

# Transform query for intent detection
query_vec = vectorizer.transform([query])
predicted_intent = clf.predict(query_vec)[0]

print("Predicted Intent:", predicted_intent)
print("Extracted Skills:", extract_skills(query))
