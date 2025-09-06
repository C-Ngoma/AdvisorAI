import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Load prepared dataset ----------
DATA_PATH = "datacollection/resume_matching_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Keeping only what we need and make a clean skills text column
df = df[['job_id', 'job_title', 'required_skills', 'years_experience', 'salary_usd']].copy()
df['required_skills'] = df['required_skills'].fillna("")
# normalize punctuation -> spaces so TF-IDF sees separate tokens
df['skills_text'] = (
    df['required_skills']
    .str.lower()
    .str.replace(r'[;,/|]+', ' ', regex=True)
    .str.replace(r'\s+', ' ', regex=True)
    .str.strip()
)

# ---------- Build TF-IDF over job skill requirements ----------
# ngram_range=(1,2) lets us capture phrases like "deep learning"
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
job_matrix = vectorizer.fit_transform(df['skills_text'])

def recommend_jobs(user_skills: str,
                   top_n: int = 10,
                   max_years_experience: int | None = None,
                   min_salary_usd: int | None = None):
    """
    user_skills: a comma/space separated string e.g. "python, sql, machine learning"
    Filters are optional. Returns a DataFrame of top matches with similarity scores.
    """
    # Clean user input to match training preprocessing
    q = (user_skills or "").lower()
    q = (
        pd.Series(q)
        .str.replace(r'[;,/|]+', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .iloc[0]
    )

    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, job_matrix).ravel()

    results = df.copy()
    results['similarity'] = sims

    # Optional filters
    if max_years_experience is not None:
        results = results[results['years_experience'] <= max_years_experience]
    if min_salary_usd is not None:
        results = results[results['salary_usd'] >= min_salary_usd]

    # Rank and select top N
    results = (results
               .sort_values('similarity', ascending=False)
               .head(top_n)
               .loc[:, ['job_id','job_title','years_experience','salary_usd','similarity','required_skills']]
               .reset_index(drop=True))

    # nicer similarity (0–100%)
    results['match_%'] = (results['similarity'] * 100).round(1)
    return results.drop(columns=['similarity'])

if __name__ == "__main__":
    print("✅ Skill matcher ready. Example queries:")
    examples = [
        "python, sql, machine learning, statistics",
        "deep learning, nlp, transformers",
        "java, cloud, kubernetes, microservices",
        "computer vision, pytorch, opencv",
    ]
    for ex in examples:
        out = recommend_jobs(ex, top_n=5, max_years_experience=3)
        print("\n--------------------------------------------------")
        print("Your skills:", ex)
        print(out.to_string(index=False))
