import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from fuzzywuzzy import fuzz

# Load dataset
df = pd.read_csv(r"C:\Desktop\PROJECTT\datacollection\ai_job_dataset_merged.csv")

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# -------------------
# ENTITY EXTRACTION
# -------------------
def extract_skills(query, skill_list):
    query = query.lower()
    matched_skills = []
    for skill in skill_list:
        if fuzz.partial_ratio(skill.lower(), query) > 80:
            matched_skills.append(skill)
    return matched_skills

def extract_entities(query):
    entities = {"skills": [], "education": [], "experience": [], "industry": [], "location": []}

    # Skills
    skill_list = []
    for skills_str in df['required_skills'].dropna():
        skill_list.extend([s.strip() for s in skills_str.split(',')])
    entities['skills'] = extract_skills(query, list(set(skill_list)))

    # Education
    for edu in df['education_required'].dropna().unique():
        if fuzz.partial_ratio(edu.lower(), query.lower()) > 80:
            entities['education'].append(edu)

    # Experience
    for exp in df['experience_level'].dropna().unique():
        if fuzz.partial_ratio(exp.lower(), query.lower()) > 80:
            entities['experience'].append(exp)

    # Industry
    for ind in df['industry'].dropna().unique():
        if fuzz.partial_ratio(ind.lower(), query.lower()) > 80:
            entities['industry'].append(ind)

    # Location
    for loc in df['company_location'].dropna().unique():
        if fuzz.partial_ratio(loc.lower(), query.lower()) > 80:
            entities['location'].append(loc)

    return entities

# -------------------
# INTENT DETECTION
# -------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['job_title'])
y = df['industry']
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

def detect_intent(query):
    query_vec = vectorizer.transform([query])
    return clf.predict(query_vec)[0]

# -------------------
# JOB MATCHING WITH RELEVANCE
# -------------------
def match_jobs(query, top_n=5):
    entities = extract_entities(query)
    intent = detect_intent(query)

    results = df.copy()
    results['score'] = 0

    # Skills match
    if entities['skills']:
        results['score'] += results['required_skills'].apply(
            lambda x: max([fuzz.partial_ratio(skill.lower(), x.lower()) for skill in entities['skills']]) if pd.notna(x) else 0
        )

    # Education match
    if entities['education']:
        results['score'] += results['education_required'].apply(
            lambda x: 100 if x in entities['education'] else 0
        )

    # Experience match
    if entities['experience']:
        results['score'] += results['experience_level'].apply(
            lambda x: 100 if x in entities['experience'] else 0
        )

    # Industry match
    if entities['industry']:
        results['score'] += results['industry'].apply(
            lambda x: 100 if x in entities['industry'] else 0
        )
    else:
        # fallback to intent
        results['score'] += results['industry'].apply(lambda x: 100 if x == intent else 0)

    # Location match
    if entities['location']:
        results['score'] += results['company_location'].apply(
            lambda x: 100 if x in entities['location'] else 0
        )

    # Sort by combined score and salary
    results = results.sort_values(by=['score', 'salary_usd'], ascending=False)
    return results.head(top_n)

# -------------------
# EXAMPLE QUERY
# -------------------
query = "Looking for an AI Software Engineer role requiring Python and Machine Learning in USA for entry level"
matched_jobs = match_jobs(query)

print("Extracted Entities & Intent:\n", extract_entities(query), "\nIntent:", detect_intent(query))
print("\nTop Matching Jobs:\n", matched_jobs[['job_title','company_name','salary_usd','required_skills','education_required','experience_level','industry','company_location','score']])
