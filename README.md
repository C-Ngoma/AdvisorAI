#🎓 Career Path Advisory AI
----
*“Your AI-powered career counselor — matching students to real-world opportunities.”*
----
##📌 Project Overview

Career Path Advisory AI is a system that recommends career paths to students based on their academic background, skills, and job market trends.
It uses NLP, time series forecasting, and deep learning to provide interactive career guidance.
Developed by a group of 10 dedicated students who witnessed a significant rise in unemployment, AdvisorAI aims to empower individuals by leveraging data collection and AI-driven insights. Our mission is to bridge the gap between opportunity and talent, offering actionable advice and guidance to help users achieve their goals. 
-----
##🚀 Features

🗣️ NLP Career Queries → Answer questions like “What can I do with a degree in Economics?”
📈 Salary & Demand Forecasting → Predict job demand and salary trends using time series analysis
🤖 Deep Resume & Skill Matching → Match a student’s skills with job requirements and recommend roles 
-----
## Quick Start

### 1. Install Dependencies
```bash
pip install pandas scikit-learn flask sentence-transformers torch
```

### 2. Basic Usage
```python
from src.deep_resume_matcher import DeepResumeMatcherAPI

# Initialize the API
api = DeepResumeMatcherAPI()

# Match resume text to jobs
result = api.match_resume_text("""
John Doe - Software Engineer
Experience: 3 years
Skills: Python, machine learning, AWS, Docker
Education: Master's degree
""", top_k=5)

print(f"Found {result['total_matches']} matches!")
```

### 3. Web API Demo
```bash
# Start the web server
python src/web/deep_matcher_api.py

# Visit http://localhost:5000 for interactive demo
```
### 4. Run Demo
```bash
# Comprehensive demo with examples
python demo_deep_matcher.py
```
### 5. Run Tests
```bash
python tests/test_deep_resume_matcher.py
```

## API Endpoints

### POST `/match/resume`
Match resume text to relevant jobs.

**Request:**
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

**Response:**
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

### POST `/match/candidate`
Match structured candidate data to jobs.

### GET `/health`
Health check endpoint.

### GET `/stats`
System statistics and dataset information.

##📂 Project Structure

```
datacollection/
└── ai_job_dataset_cleaned.csv    # AI job dataset (15,000 jobs)
scripts/
  |__test_datasets.sh
src/
├── deep_resume_matcher.py     # Main deep learning module
│   ├── ResumeParser           # Extract structured data from text
│   ├── EmbeddingModel         # Generate semantic embeddings
│   ├── JobMatcher             # Match candidates to jobs
│   └── DeepResumeMatcherAPI   # Complete API wrapper
├── skill_matcher.py           # Traditional TF-IDF matcher
└── web/
    └── deep_matcher_api.py    # Flask web API

tests/
└── test_deep_resume_matcher.py   # Comprehensive test suite
│──.gitignore/                    #directories to ignore
│── README.md                     # Project documentation
│──demo_deep_matcher.py           # project's matching functionality
│──requirements.txt               #lists all Python package dependencies needed to run your project


```
-------
## Key Features

### 🧠 Intelligent Resume Parsing
- **Skill Extraction**: Recognizes 60+ technical skills across categories
- **Experience Detection**: Extracts years of experience from various formats
- **Education Parsing**: Identifies education levels (Associate, Bachelor, Master, PhD)
- **Flexible Input**: Handles raw text or structured data

### 🔍 Advanced Job Matching
- **Semantic Similarity**: Uses transformer embeddings for deep understanding
- **Multiple Algorithms**: Transformer models with TF-IDF fallback
- **Confidence Scoring**: Provides match confidence percentages
- **Match Explanations**: Shows why jobs match candidates

### 🎯 Smart Filtering
- **Experience Level**: Filter by years of experience
- **Salary Range**: Set minimum salary requirements  
- **Industry Focus**: Filter by specific industries
- **Custom Filters**: Extensible filtering system

### 📈 Performance
- **15,000+ Jobs**: Large-scale job dataset
- **Fast Matching**: Optimized for real-time performance
- **Scalable**: Batch processing and caching support
- **Robust**: Handles errors gracefully with fallbacks
-------
## Examples

### Resume Text Matching
```python
# Example with filtering
result = api.match_resume_text(
    resume_text="""
    Data Scientist with 4 years experience.
    Skills: Python, TensorFlow, SQL, AWS, Tableau
    PhD in Statistics from MIT
    """,
    top_k=3,
    filters={
        'min_salary_usd': 80000,
        'max_years_experience': 6
    }
)
```
------
### Structured Candidate Matching
```python
candidate_data = {
    'skills': ['Java', 'Spring', 'Microservices', 'Kubernetes'],
    'experience_years': 5,
    'education_level': 'Master',
    'skills_text': 'java spring microservices kubernetes'
}

result = api.match_candidate_data(candidate_data, top_k=10)
```
------
## Dataset

The system uses `datacollection/ai_job_dataset_cleaned.csv` containing:
- **15,000 AI/Tech jobs** with detailed information
- **Skills, salary, experience** requirements
- **Company and industry** information
- **Geographic and remote work** data

## Integration

### With Existing AdvisorAI Backend
```python
# Import and integrate
from src.deep_resume_matcher import DeepResumeMatcherAPI

# Initialize in your application
matcher = DeepResumeMatcherAPI()

# Use in your existing workflows
def process_candidate_application(resume_text, filters=None):
    matches = matcher.match_resume_text(resume_text, filters=filters)
    return matches['matches']
```
----
### Web Integration
The Flask API provides RESTful endpoints that can be integrated with any web frontend or mobile application.
-----
### Custom Embedding Models
```python
# Use different transformer model
api = DeepResumeMatcherAPI()
api.job_matcher.embedding_model = EmbeddingModel('your-model-name')
```
----
### Extending Filters
Add custom filters in `JobMatcher._apply_filters()` method.
----
## Performance Notes

- **Transformer Models**: Require internet connection for first-time download
- **Offline Mode**: Automatically falls back to TF-IDF when transformers unavailable  
- **Memory Usage**: ~500MB for transformer models, ~50MB for TF-IDF fallback
- **Speed**: ~100ms per query with transformers, ~10ms with TF-IDF
------
## License

MIT License - see LICENSE file for details.
-----
