#!/usr/bin/env python3
"""
Deep Resume Matcher Demo Script

This script demonstrates the capabilities of the deep learning resume matcher
with various example resumes and candidate profiles.

Usage:
    python demo_deep_matcher.py
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from deep_resume_matcher import DeepResumeMatcherAPI

def print_separator(title=""):
    """Print a nice separator."""
    width = 60
    if title:
        padding = (width - len(title) - 2) // 2
        print("=" * padding + f" {title} " + "=" * padding)
    else:
        print("=" * width)

def print_match_results(results, title="Results"):
    """Print matching results in a nice format."""
    if not results['success']:
        print(f"❌ {title} failed: {results['error']}")
        return
    
    print(f"✅ {title} - Found {results['total_matches']} matches")
    
    if 'candidate_info' in results:
        candidate = results['candidate_info']
        print(f"\n👤 Candidate Profile:")
        if 'skills' in candidate:
            print(f"   Skills: {', '.join(candidate['skills'][:10])}")  # Show first 10 skills
        if 'experience_years' in candidate:
            print(f"   Experience: {candidate['experience_years']} years")
        if 'education_level' in candidate:
            print(f"   Education: {candidate['education_level']}")
    
    print(f"\n🎯 Top Matches:")
    for i, match in enumerate(results['matches'][:5], 1):
        print(f"\n{i}. {match['job_title']}")
        print(f"   📊 Confidence: {match['confidence_score']}%")
        print(f"   💰 Salary: ${match['salary_usd']:,}")
        print(f"   ⏰ Experience: {match['years_experience']} years")
        print(f"   🛠️  Skills: {match['required_skills'][:100]}{'...' if len(match['required_skills']) > 100 else ''}")
        if match['match_reasons']:
            print(f"   💡 Why: {'; '.join(match['match_reasons'])}")

def demo_resume_examples():
    """Demo with various resume examples."""
    print_separator("DEMO: Resume Text Matching")
    
    # Example resumes
    resumes = {
        "Data Scientist": """
        Jane Smith - Senior Data Scientist
        
        Experience: 5 years in data science and machine learning
        
        Skills:
        • Programming: Python, R, SQL, Scala
        • ML/AI: TensorFlow, PyTorch, Scikit-learn, Keras, XGBoost
        • Big Data: Spark, Hadoop, Kafka, Elasticsearch
        • Cloud: AWS (S3, EC2, SageMaker), GCP, Docker, Kubernetes
        • Visualization: Tableau, Power BI, Matplotlib, Seaborn
        • Databases: PostgreSQL, MongoDB, Redis
        
        Education: PhD in Statistics, Stanford University
        
        Experience:
        • Senior Data Scientist at Meta (3 years)
        • ML Engineer at Google (2 years)
        
        Specialized in deep learning, NLP, computer vision, and recommendation systems.
        Led cross-functional teams and deployed ML models at scale.
        """,
        
        "Software Engineer": """
        Mike Johnson - Full Stack Developer
        
        Experience: 3 years of web development
        
        Technical Skills:
        • Frontend: React, Vue.js, JavaScript, TypeScript, HTML5, CSS3
        • Backend: Node.js, Express, Python, Django, Flask
        • Databases: MySQL, PostgreSQL, MongoDB
        • DevOps: Docker, Kubernetes, Jenkins, Git, AWS
        • Testing: Jest, Cypress, Pytest
        
        Education: Bachelor's in Computer Science
        
        Previous Roles:
        • Software Engineer at Startup Inc. (2 years)
        • Frontend Developer at WebCorp (1 year)
        
        Built scalable web applications, RESTful APIs, and microservices.
        """,
        
        "DevOps Engineer": """
        Sarah Wilson - DevOps Engineer
        
        Experience: 4 years in DevOps and cloud infrastructure
        
        Core Skills:
        • Cloud Platforms: AWS (EC2, S3, Lambda, RDS), Azure, GCP
        • Containers: Docker, Kubernetes, Helm
        • CI/CD: Jenkins, GitLab CI, GitHub Actions, Travis CI
        • Infrastructure as Code: Terraform, CloudFormation, Ansible
        • Monitoring: Prometheus, Grafana, ELK Stack, DataDog
        • Scripting: Python, Bash, PowerShell
        
        Education: Master's in Information Systems
        
        Experience:
        • Senior DevOps Engineer at TechCorp (2 years)
        • Cloud Engineer at StartupXYZ (2 years)
        
        Automated deployment pipelines, managed Kubernetes clusters, and optimized cloud costs.
        """
    }
    
    # Initialize API
    try:
        api = DeepResumeMatcherAPI()
        print("✅ Deep Resume Matcher API initialized")
    except Exception as e:
        print(f"❌ Failed to initialize API: {e}")
        return
    
    # Test each resume
    for role, resume_text in resumes.items():
        print(f"\n" + "─" * 60)
        print(f"🧪 Testing: {role}")
        print("─" * 60)
        
        result = api.match_resume_text(
            resume_text=resume_text,
            top_k=5,
            filters={'min_salary_usd': 60000}
        )
        
        print_match_results(result, f"{role} Matching")

def demo_structured_candidates():
    """Demo with structured candidate data."""
    print_separator("DEMO: Structured Candidate Data")
    
    candidates = [
        {
            "name": "AI Research Scientist",
            "data": {
                'skills': ['Python', 'TensorFlow', 'PyTorch', 'Deep Learning', 'NLP', 'Computer Vision', 'Research', 'Statistics'],
                'experience_years': 6,
                'education_level': 'PhD',
                'skills_text': 'python tensorflow pytorch deep learning nlp computer vision machine learning research statistics'
            },
            "filters": {'min_salary_usd': 80000, 'max_years_experience': 10}
        },
        {
            "name": "Cloud Architect", 
            "data": {
                'skills': ['AWS', 'Kubernetes', 'Docker', 'Terraform', 'Microservices', 'Java', 'Spring', 'Jenkins'],
                'experience_years': 8,
                'education_level': 'Master',
                'skills_text': 'aws kubernetes docker terraform microservices java spring jenkins cloud architecture'
            },
            "filters": {'min_salary_usd': 90000}
        },
        {
            "name": "Junior Developer",
            "data": {
                'skills': ['JavaScript', 'React', 'Node.js', 'Git', 'HTML', 'CSS', 'MongoDB'],
                'experience_years': 1,
                'education_level': 'Bachelor',
                'skills_text': 'javascript react nodejs git html css mongodb web development'
            },
            "filters": {'max_years_experience': 3}
        }
    ]
    
    # Initialize API
    try:
        api = DeepResumeMatcherAPI()
        print("✅ Deep Resume Matcher API initialized")
    except Exception as e:
        print(f"❌ Failed to initialize API: {e}")
        return
    
    # Test each candidate
    for candidate in candidates:
        print(f"\n" + "─" * 60)
        print(f"🧪 Testing: {candidate['name']}")
        print("─" * 60)
        
        result = api.match_candidate_data(
            candidate_data=candidate['data'],
            top_k=5,
            filters=candidate['filters']
        )
        
        print_match_results(result, f"{candidate['name']} Matching")

def demo_performance_stats():
    """Show performance and system statistics."""
    print_separator("SYSTEM STATISTICS")
    
    try:
        api = DeepResumeMatcherAPI()
        
        # Basic stats
        job_count = len(api.job_matcher.df)
        embedding_dim = api.job_matcher.job_embeddings.shape[1]
        
        print(f"📊 Dataset Statistics:")
        print(f"   Total Jobs: {job_count:,}")
        print(f"   Embedding Dimensions: {embedding_dim}")
        print(f"   Dataset Path: {api.job_matcher.dataset_path}")
        
        # Sample job skills
        sample_skills = api.job_matcher.df['required_skills'].head(5).tolist()
        print(f"\n🛠️  Sample Job Skills:")
        for i, skills in enumerate(sample_skills, 1):
            print(f"   {i}. {skills[:80]}{'...' if len(skills) > 80 else ''}")
        
        # Model info
        model_type = "Transformer" if api.job_matcher.embedding_model.model else "TF-IDF"
        print(f"\n🤖 Model Information:")
        print(f"   Embedding Model: {model_type}")
        if model_type == "Transformer":
            print(f"   Model Name: {api.job_matcher.embedding_model.model_name}")
        
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")

def main():
    """Main demo function."""
    print("🚀 Deep Resume Matcher - Comprehensive Demo")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()
    
    try:
        # Run all demos
        demo_resume_examples()
        print("\n" + "=" * 80)
        demo_structured_candidates()
        print("\n" + "=" * 80)
        demo_performance_stats()
        
        print_separator("DEMO COMPLETED")
        print("🎉 All demos completed successfully!")
        print("\n💡 Next steps:")
        print("   • Start the web API: python src/web/deep_matcher_api.py")
        print("   • Visit http://localhost:5000 for interactive demo")
        print("   • Integrate with your application using the API endpoints")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()