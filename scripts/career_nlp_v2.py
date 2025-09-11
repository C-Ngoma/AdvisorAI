import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

# Load dataset
df = pd.read_csv(r"C:\Desktop\PROJECTT\datacollection\ai_job_dataset_merged.csv")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Build skill list
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

skill_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
_skill_patterns = [nlp.make_doc(s) for s in skill_list]
skill_matcher.add("SKILL", _skill_patterns)

def extract_entities(query):
    doc = nlp(query)
    entities = {"skills": [], "education": [], "experience": [], "industry": [], "location": []}

    # Skills
    matches = skill_matcher(doc)
    for match_id, start, end in matches:
        span_text = doc[start:end].text.strip()
        # Normalize capitalization (use skill_list for reference)
        for s in skill_list:
            if span_text.lower() == s.lower():
                span_text = s
                break
        if span_text not in entities["skills"]:
            entities["skills"].append(span_text)

    # Education
    for edu in education_list:
        if edu and edu.lower() in query.lower():
            if edu not in entities["education"]:
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
            if ind not in entities["industry"]:
                entities["industry"].append(ind)

    # Location
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            loc = ent.text.strip()
            if loc not in entities["location"]:
                entities["location"].append(loc)

    return entities

