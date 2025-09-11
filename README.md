# 🎓 Career Path Advisory AI

*Your AI-powered career counselor — matching students to real-world opportunities.*

---

## 📌 Overview

**Career Path Advisory AI** is an intelligent recommendation system that helps students discover career paths based on their academic background, skills, and job market trends.

It combines **NLP, time-series forecasting, and deep learning** to provide interactive career guidance.
Developed by a team of 10 students driven by the rising unemployment rate, the system’s mission is to **bridge the gap between opportunity and talent**.

---

## 🚀 Key Features

### 🗣️ NLP Career Queries

* Answers questions like: *“What can I do with a degree in Economics?”*

### 📈 Salary & Demand Forecasting

* Predicts job demand and salary trends using time-series models

### 🤖 Deep Resume & Skill Matching

* Matches student resumes to relevant job opportunities
* Provides **confidence scores** and **match explanations**

### 🎯 Smart Filtering

* Filter by **experience level**, **salary range**, or **industry**
* Extensible for custom filters

---

## ⚙️ Quick Start

### 1️⃣ Install Dependencies

```bash
pip install pandas scikit-learn flask sentence-transformers torch
```

### 2️⃣ Basic Usage

```python
from src.deep_resume_matcher import DeepResumeMatcherAPI

api = DeepResumeMatcherAPI()

result = api.match_resume_text("""
John Doe - Software Engineer
Experience: 3 years
Skills: Python, machine learning, AWS, Docker
Education: Master's degree
""", top_k=5)

print(result)
```

### 3️⃣ Run Web API

```bash
python src/web/deep_matcher_api.py
# Visit http://localhost:5000 for interactive demo
```

### 4️⃣ Run Demo & Tests

```bash
python demo_deep_matcher.py
python tests/test_deep_resume_matcher.py
```

---

## 🌐 API Endpoints

### `POST /match/resume` → Match resume text to jobs

**Request**

```json
{
  "resume_text": "John Doe - Software Engineer...",
  "top_k": 5,
  "filters": {
    "max_years_experience": 5,
    "min_salary_usd": 60000
  }
}
```

**Response**

```json
{
  "success": true,
  "candidate_info": {
    "skills": ["python", "machine learning", "aws"],
    "experience_years": 3,
    "education_level": "Master"
  },
  "matches": [
    {
      "job_id": "AI00123",
      "job_title": "Senior Python Developer",
      "confidence_score": 89.5,
      "salary_usd": 85000,
      "required_skills": "Python, Flask, AWS, Docker",
      "match_reasons": ["Matching skills: python, aws", "Experience requirement met"]
    }
  ],
  "total_matches": 5
}
```

### Other Endpoints

* `POST /match/candidate` → Match structured candidate data
* `GET /health` → Health check
* `GET /stats` → System statistics

---

## 📂 Project Structure

```
datacollection/
  └── ai_job_dataset_cleaned.csv   # Job dataset (15k+ entries)

src/
  ├── deep_resume_matcher.py       # Core deep learning matcher
  ├── skill_matcher.py             # TF-IDF based matcher
  └── web/
      └── deep_matcher_api.py      # Flask API

tests/
  └── test_deep_resume_matcher.py  # Test suite

scripts/
  └── test_datasets.sh             # Dataset validation

demo_deep_matcher.py               # Demo runner
requirements.txt                    # Dependencies
README.md                           # Documentation
.gitignore                          # Ignore files
```

---

## 📊 Dataset

* **15,000+ AI/Tech jobs** across industries
* Includes **skills, salary, experience, company, and location**
* Supports **remote and onsite roles**

---

## 🔧 Integration

```python
from src.deep_resume_matcher import DeepResumeMatcherAPI

matcher = DeepResumeMatcherAPI()

def process_candidate(resume_text, filters=None):
    matches = matcher.match_resume_text(resume_text, filters=filters)
    return matches["matches"]
```

---

## 📈 Performance Notes

* Transformer models (\~500MB memory) with TF-IDF fallback (\~50MB)
* Speed: \~100ms per query (transformers), \~10ms (TF-IDF)
* Works offline with fallback models

---

## 📜 License

MIT License — see `LICENSE` for details.

---


