import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv(r"C:\Desktop\PROJECTT\datacollection\ai_job_dataset_merged.csv")

# Initialize spaCy for entity extraction
nlp = spacy.load("en_core_web_sm")

# -------------------
# ENTITY EXTRACTION
# -------------------
def extract_skills(text):
    doc = nlp(text)
    # For simplicity, extract all proper nouns and known skill keywords
    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "SKILL"]]
    return skills

def extract_entities(query):
    doc = nlp(query)
    entities = {
        "skills": extract_skills(query),
        "education": [],
        "experience": [],
        "industry": [],
        "location": []
    }
    
    # Simple keyword search for other entities
    education_keywords = df['education_required'].dropna().unique().tolist()
    for word in education_keywords:
        if word.lower() in query.lower():
            entities['education'].append(word)
    
    experience_keywords = df['experience_level'].dropna().unique().tolist()
    for word in experience_keywords:
        if word.lower() in query.lower():
            entities['experience'].append(word)
    
    industry_keywords = df['industry'].dropna().unique().tolist()
    for word in industry_keywords:
        if word.lower() in query.lower():
            entities['industry'].append(word)
    
    location_keywords = df['company_location'].dropna().unique().tolist()
    for word in location_keywords:
        if word.lower() in query.lower():
            entities['location'].append(word)
    
    return entities

# -------------------
# INTENT DETECTION
# -------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['job_title'])
y = df['industry']  # using industry as intent

clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

def detect_intent(query):
    query_vec = vectorizer.transform([query])
    return clf.predict(query_vec)[0]

# -------------------
# JOB MATCHING
# -------------------
def match_jobs(query, top_n=5):
    intent = detect_intent(query)
    entities = extract_entities(query)
    
    results = df.copy()
    
    # Filter by extracted entities
    if entities['skills']:
        results = results[results['required_skills'].apply(lambda x: any(skill.lower() in x.lower() for skill in entities['skills']))]
    
    if entities['education']:
        results = results[results['education_required'].isin(entities['education'])]
    
    if entities['experience']:
        results = results[results['experience_level'].isin(entities['experience'])]
    
    if entities['industry']:
        results = results[results['industry'].isin(entities['industry'])]
    else:
        # fallback to intent
        results = results[results['industry'] == intent]
    
    if entities['location']:
        results = results[results['company_location'].isin(entities['location'])]
    
    # Return top N jobs by salary
    results = results.sort_values(by='salary_usd', ascending=False)
    return results.head(top_n)

# -------------------
# EXAMPLE QUERY
# -------------------
query = "Looking for an AI Software Engineer role requiring Python and ML in USA for entry level"
matched_jobs = match_jobs(query)

print("Extracted Entities & Intent:\n", extract_entities(query), "\nIntent:", detect_intent(query))
print("\nTop Matching Jobs:\n", matched_jobs[['job_title','company_name','salary_usd','required_skills','education_required','experience_level','industry','company_location']])

