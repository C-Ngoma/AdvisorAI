import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------
# Load dataset
# -------------------
df = pd.read_csv("datacollection/ai_job_dataset_merged.csv")

# -------------------
# Initialize spaCy
# -------------------
nlp = spacy.load("en_core_web_sm")

# -------------------
# Build skill, education, industry, experience lists
# -------------------
_skill_set = set()
for skills_str in df['required_skills'].dropna():
    for s in skills_str.split(','):
        s_clean = s.strip()
        if s_clean:
            _skill_set.add(s_clean)

_extra_skills = [
    "Machine Learning", "Deep Learning", "Natural Language Processing",
    "Data Analysis", "Computer Vision", "DevOps", "Cloud", "TensorFlow", "PyTorch"
]
_skill_set.update(_extra_skills)
skill_list = sorted(_skill_set)

education_list = df['education_required'].dropna().unique().tolist()
industry_list = df['industry'].dropna().unique().tolist()

experience_map = {
    "entry": "EN", "entry-level": "EN", "junior": "EN", "intern": "EN",
    "mid": "EX", "mid-level": "EX", "experienced": "EX",
    "senior": "SE", "lead": "SE", "manager": "SE"
}

# -------------------
# PhraseMatcher for skills
# -------------------
skill_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
_skill_patterns = [nlp.make_doc(s) for s in skill_list]
skill_matcher.add("SKILL", _skill_patterns)

# -------------------
# Entity extraction
# -------------------
def extract_entities(query):
    doc = nlp(query)
    entities = {"skills": [], "education": [], "experience": [], "industry": [], "location": []}

    # Skills
    matches = skill_matcher(doc)
    for match_id, start, end in matches:
        span_text = doc[start:end].text
        if span_text not in entities["skills"]:
            entities["skills"].append(span_text)

    # Education
    for edu in education_list:
        if edu and edu.lower() in query.lower():
            entities["education"].append(edu)

    # Experience
    for token in doc:
        t = token.text.lower()
        if t in experience_map:
            code = experience_map[t]
            if code not in entities["experience"]:
                entities["experience"].append(code)

    # Industry
    for ind in industry_list:
        if ind and ind.lower() in query.lower():
            entities["industry"].append(ind)

    # Location
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            if ent.text not in entities["location"]:
                entities["location"].append(ent.text)

    return entities

# -------------------
# Intent detection
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
# Job matching
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
        results = results[results['industry'] == intent]
    
    if entities['location']:
        results = results[results['company_location'].isin(entities['location'])]
    
    # Fallback: if no matches, return top N by salary
    if results.empty:
        results = df.sort_values(by='salary_usd', ascending=False)
    
    return results.head(top_n)

# -------------------
# Example usage
# -------------------
if __name__ == "__main__":
    queries = [
        "High school teacher position in South Africa requiring a Bachelor’s degree in Education.",
        "Doctor job in USA for a cardiologist with 10 years experience.",
        "Police officer job in South Africa with entry level training."
    ]
    
    for q in queries:
        print("\nQuery:", q)
        entities = extract_entities(q)
        intent = detect_intent(q)
        top_jobs = match_jobs(q)
        
        print("Entities:", entities)
        print("Intent:", intent)
        print("Top Jobs:\n", top_jobs[['job_title','company_name','salary_usd','industry','company_location']])
