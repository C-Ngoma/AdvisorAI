"""
Deep Learning Resume Parser and Skill Matching Module

This module implements a deep learning-based approach for resume parsing and 
job matching using transformer embeddings. It provides:

- ResumeParser: Extract skills and attributes from resume text
- EmbeddingModel: Generate semantic embeddings using pre-trained transformers
- JobMatcher: Match candidates to jobs using deep learning similarity
- DeepResumeMatcherAPI: Complete pipeline for candidate-job matching

Requirements:
- sentence-transformers>=2.0.0
- torch>=1.11.0
- pandas>=1.5.0
- scikit-learn>=1.2.0
"""

import os
import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available. Using fallback TF-IDF approach.")
    TRANSFORMERS_AVAILABLE = False

# Always import TfidfVectorizer as fallback
from sklearn.feature_extraction.text import TfidfVectorizer


class ResumeParser:
    """
    Parses resume text and extracts structured information including skills,
    experience, education, and other relevant attributes.
    """
    
    def __init__(self):
        # Common skill patterns and keywords
        self.skill_patterns = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'r', 'sql', 'php', 'ruby', 'go', 'rust', 'scala', 'kotlin'],
            'ml_frameworks': ['tensorflow', 'pytorch', 'scikit-learn', 'keras', 'xgboost', 'lightgbm', 'opencv', 'transformers'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'git'],
            'data': ['pandas', 'numpy', 'spark', 'hadoop', 'kafka', 'elasticsearch', 'mongodb', 'postgresql'],
            'web': ['react', 'angular', 'vue', 'node.js', 'express', 'flask', 'django', 'spring'],
            'tools': ['tableau', 'power bi', 'excel', 'jira', 'confluence', 'slack', 'notion']
        }
        
        # Education patterns
        self.education_patterns = [
            r'(bachelor|bs|ba|b\.s\.|b\.a\.)', 
            r'(master|ms|ma|m\.s\.|m\.a\.|mba)',
            r'(phd|ph\.d\.|doctorate|doctoral)'
        ]
        
        # Experience patterns
        self.experience_patterns = [
            r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+)\+?\s*(?:years?|yrs?)',
            r'over\s*(\d+)\s*(?:years?|yrs?)'
        ]
    
    def parse_resume(self, resume_text: str) -> Dict:
        """
        Parse resume text and extract structured information.
        
        Args:
            resume_text (str): Raw resume text
            
        Returns:
            Dict: Parsed resume data with skills, experience, education
        """
        if not resume_text or not isinstance(resume_text, str):
            return self._get_empty_resume()
            
        text_lower = resume_text.lower()
        
        return {
            'skills': self._extract_skills(text_lower),
            'experience_years': self._extract_experience(text_lower),
            'education_level': self._extract_education(text_lower),
            'raw_text': resume_text,
            'skills_text': ', '.join(self._extract_skills(text_lower))
        }
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text."""
        skills = []
        
        for category, skill_list in self.skill_patterns.items():
            for skill in skill_list:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, text):
                    skills.append(skill)
        
        return list(set(skills))  # Remove duplicates
    
    def _extract_experience(self, text: str) -> int:
        """Extract years of experience from resume text."""
        max_years = 0
        
        for pattern in self.experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    years = int(match)
                    max_years = max(max_years, years)
                except ValueError:
                    continue
        
        return max_years
    
    def _extract_education(self, text: str) -> str:
        """Extract education level from resume text."""
        for i, pattern in enumerate(self.education_patterns):
            if re.search(pattern, text, re.IGNORECASE):
                return ['Bachelor', 'Master', 'PhD'][i]
        
        return 'Associate'  # Default
    
    def _get_empty_resume(self) -> Dict:
        """Return empty resume structure."""
        return {
            'skills': [],
            'experience_years': 0,
            'education_level': 'Associate',
            'raw_text': '',
            'skills_text': ''
        }


class EmbeddingModel:
    """
    Generates semantic embeddings for text using pre-trained transformer models
    or falls back to TF-IDF if transformers are not available.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding model.
        
        Args:
            model_name (str): Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        self.fallback_vectorizer = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading transformer model: {model_name}")
                # Try to use a local model or create simple embeddings
                import os
                # For offline mode, we can use a simpler approach or cached models
                if not os.environ.get('HF_HUB_OFFLINE', '').lower() == 'true':
                    self.model = SentenceTransformer(model_name, device='cpu')
                    logger.info("Transformer model loaded successfully")
                else:
                    raise Exception("Offline mode - using fallback")
            except Exception as e:
                logger.warning(f"Failed to load transformer model: {e}")
                logger.info("Falling back to TF-IDF approach")
                self.model = None
        
        if self.model is None:
            logger.info("Using TF-IDF fallback for embeddings")
            self.fallback_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), 
                min_df=1, 
                max_features=5000,
                stop_words='english'
            )
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to encode
            batch_size (int): Batch size for processing
            
        Returns:
            np.ndarray: Embeddings matrix
        """
        if not texts:
            return np.array([])
        
        # Clean texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        if self.model is not None:
            # Use transformer model
            try:
                embeddings = self.model.encode(
                    cleaned_texts, 
                    batch_size=batch_size,
                    show_progress_bar=False
                )
                return embeddings
            except Exception as e:
                logger.warning(f"Transformer encoding failed: {e}")
                # Fall back to TF-IDF
                return self._encode_tfidf(cleaned_texts)
        else:
            # Use TF-IDF fallback
            return self._encode_tfidf(cleaned_texts)
    
    def _encode_tfidf(self, texts: List[str]) -> np.ndarray:
        """Fallback TF-IDF encoding."""
        if hasattr(self.fallback_vectorizer, 'vocabulary_'):
            # Transform using fitted vectorizer
            return self.fallback_vectorizer.transform(texts).toarray()
        else:
            # Fit and transform
            return self.fallback_vectorizer.fit_transform(texts).toarray()
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Basic cleaning
        text = re.sub(r'[^\w\s,.-]', ' ', text)  # Remove special chars except common punctuation
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = text.strip().lower()
        
        return text


class JobMatcher:
    """
    Matches candidates to jobs using deep learning embeddings and similarity scoring.
    """
    
    def __init__(self, 
                 dataset_path: str = "datacollection/ai_job_dataset_cleaned.csv",
                 embedding_model: Optional[EmbeddingModel] = None):
        """
        Initialize job matcher.
        
        Args:
            dataset_path (str): Path to job dataset CSV
            embedding_model (EmbeddingModel): Pre-initialized embedding model
        """
        self.dataset_path = dataset_path
        self.embedding_model = embedding_model or EmbeddingModel()
        self.df = None
        self.job_embeddings = None
        self.scaler = StandardScaler()
        
        self._load_and_prepare_data()
    
    def _load_and_prepare_data(self):
        """Load and prepare job dataset."""
        try:
            logger.info(f"Loading job dataset from {self.dataset_path}")
            self.df = pd.read_csv(self.dataset_path)
            
            # Clean and prepare job data
            self.df['required_skills'] = self.df['required_skills'].fillna("")
            self.df['job_text'] = self.df.apply(self._create_job_text, axis=1)
            
            # Generate embeddings for all jobs
            logger.info("Generating job embeddings...")
            job_texts = self.df['job_text'].tolist()
            self.job_embeddings = self.embedding_model.encode(job_texts)
            
            logger.info(f"Loaded {len(self.df)} jobs with {self.job_embeddings.shape[1]} embedding dimensions")
            
        except Exception as e:
            logger.error(f"Failed to load job dataset: {e}")
            # Create empty dataset for testing
            self.df = pd.DataFrame({
                'job_id': ['TEST001', 'TEST002'],
                'job_title': ['Test Job 1', 'Test Job 2'],
                'required_skills': ['python, machine learning', 'java, spring'],
                'years_experience': [2, 5],
                'salary_usd': [70000, 90000],
                'job_text': ['python machine learning', 'java spring']
            })
            self.job_embeddings = self.embedding_model.encode(self.df['job_text'].tolist())
    
    def _create_job_text(self, row) -> str:
        """Create searchable text representation of a job."""
        parts = []
        
        if pd.notna(row.get('job_title')):
            parts.append(str(row['job_title']))
        
        if pd.notna(row.get('required_skills')):
            # Clean skills text
            skills = str(row['required_skills']).replace(',', ' ').replace(';', ' ')
            parts.append(skills)
        
        if pd.notna(row.get('industry')):
            parts.append(str(row['industry']))
        
        return ' '.join(parts).lower()
    
    def match_candidate(self, 
                       candidate_data: Union[str, Dict], 
                       top_k: int = 10,
                       min_similarity: float = 0.1,
                       filters: Optional[Dict] = None) -> List[Dict]:
        """
        Match a candidate to relevant jobs.
        
        Args:
            candidate_data: Resume text (str) or parsed candidate dict
            top_k (int): Number of top matches to return
            min_similarity (float): Minimum similarity threshold
            filters (Dict): Additional filters (max_experience, min_salary, etc.)
            
        Returns:
            List[Dict]: Ranked list of job matches with scores
        """
        # Parse candidate data if needed
        if isinstance(candidate_data, str):
            parser = ResumeParser()
            candidate_info = parser.parse_resume(candidate_data)
            candidate_text = candidate_data
        elif isinstance(candidate_data, dict):
            candidate_info = candidate_data
            candidate_text = candidate_info.get('skills_text', '') or candidate_info.get('raw_text', '')
        else:
            raise ValueError("candidate_data must be string or dict")
        
        # Generate candidate embedding
        candidate_embedding = self.embedding_model.encode([candidate_text])
        
        # Calculate similarities
        similarities = cosine_similarity(candidate_embedding, self.job_embeddings).ravel()
        
        # Create results dataframe
        results_df = self.df.copy()
        results_df['similarity'] = similarities
        results_df['confidence_score'] = similarities * 100  # Convert to percentage
        
        # Apply filters
        results_df = self._apply_filters(results_df, candidate_info, filters)
        
        # Filter by minimum similarity
        results_df = results_df[results_df['similarity'] >= min_similarity]
        
        # Sort and get top K
        results_df = results_df.sort_values('similarity', ascending=False).head(top_k)
        
        # Convert to list of dictionaries
        matches = []
        for _, row in results_df.iterrows():
            match = {
                'job_id': row.get('job_id', ''),
                'job_title': row.get('job_title', ''),
                'required_skills': row.get('required_skills', ''),
                'years_experience': row.get('years_experience', 0),
                'salary_usd': row.get('salary_usd', 0),
                'similarity_score': round(row['similarity'], 3),
                'confidence_score': round(row['confidence_score'], 1),
                'company_name': row.get('company_name', ''),
                'industry': row.get('industry', ''),
                'match_reasons': self._generate_match_reasons(candidate_info, row)
            }
            matches.append(match)
        
        return matches
    
    def _apply_filters(self, df: pd.DataFrame, candidate_info: Dict, filters: Optional[Dict]) -> pd.DataFrame:
        """Apply additional filters to job matches."""
        if not filters:
            return df
        
        # Experience filter
        if 'max_years_experience' in filters:
            max_exp = filters['max_years_experience']
            if max_exp is not None:
                df = df[df.get('years_experience', 0) <= max_exp]
        
        # Salary filter
        if 'min_salary_usd' in filters:
            min_salary = filters['min_salary_usd']
            if min_salary is not None:
                df = df[df.get('salary_usd', 0) >= min_salary]
        
        # Industry filter
        if 'industry' in filters and filters['industry']:
            industry = filters['industry'].lower()
            df = df[df.get('industry', '').str.lower().str.contains(industry, na=False)]
        
        return df
    
    def _generate_match_reasons(self, candidate_info: Dict, job_row) -> List[str]:
        """Generate explanations for why a job matches a candidate."""
        reasons = []
        
        candidate_skills = set([s.lower() for s in candidate_info.get('skills', [])])
        job_skills_str = str(job_row.get('required_skills', '')).lower()
        
        # Find skill matches
        skill_matches = [skill for skill in candidate_skills 
                        if skill in job_skills_str]
        
        if skill_matches:
            reasons.append(f"Matching skills: {', '.join(skill_matches[:3])}")
        
        # Experience match
        candidate_exp = candidate_info.get('experience_years', 0)
        job_exp = job_row.get('years_experience', 0)
        
        if candidate_exp >= job_exp:
            reasons.append("Experience requirement met")
        elif job_exp - candidate_exp <= 2:
            reasons.append("Close experience match")
        
        return reasons


class DeepResumeMatcherAPI:
    """
    Complete API wrapper for deep learning resume matching functionality.
    """
    
    def __init__(self, dataset_path: str = "datacollection/ai_job_dataset_cleaned.csv"):
        """Initialize the complete matching pipeline."""
        self.resume_parser = ResumeParser()
        self.job_matcher = JobMatcher(dataset_path)
        
        logger.info("Deep Resume Matcher API initialized successfully")
    
    def match_resume_text(self, 
                         resume_text: str, 
                         top_k: int = 10, 
                         filters: Optional[Dict] = None) -> Dict:
        """
        Match resume text to jobs and return results.
        
        Args:
            resume_text (str): Raw resume text
            top_k (int): Number of matches to return
            filters (Dict): Additional filters
            
        Returns:
            Dict: Complete matching results
        """
        try:
            # Parse resume
            parsed_resume = self.resume_parser.parse_resume(resume_text)
            
            # Get job matches
            matches = self.job_matcher.match_candidate(
                parsed_resume, 
                top_k=top_k, 
                filters=filters
            )
            
            return {
                'success': True,
                'candidate_info': parsed_resume,
                'matches': matches,
                'total_matches': len(matches),
                'message': f'Found {len(matches)} matching jobs'
            }
            
        except Exception as e:
            logger.error(f"Resume matching failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'matches': [],
                'total_matches': 0
            }
    
    def match_candidate_data(self, 
                            candidate_data: Dict, 
                            top_k: int = 10, 
                            filters: Optional[Dict] = None) -> Dict:
        """
        Match structured candidate data to jobs.
        
        Args:
            candidate_data (Dict): Structured candidate information
            top_k (int): Number of matches to return
            filters (Dict): Additional filters
            
        Returns:
            Dict: Complete matching results
        """
        try:
            matches = self.job_matcher.match_candidate(
                candidate_data, 
                top_k=top_k, 
                filters=filters
            )
            
            return {
                'success': True,
                'candidate_info': candidate_data,
                'matches': matches,
                'total_matches': len(matches),
                'message': f'Found {len(matches)} matching jobs'
            }
            
        except Exception as e:
            logger.error(f"Candidate matching failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'matches': [],
                'total_matches': 0
            }


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Deep Resume Matcher - Example Usage")
    print("=" * 50)
    
    # Initialize the API
    try:
        api = DeepResumeMatcherAPI()
        print("✅ Deep Resume Matcher initialized successfully")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        exit(1)
    
    # Example 1: Resume text matching
    print("\n📄 Example 1: Resume Text Matching")
    sample_resume = """
    John Doe - Software Engineer
    
    Experience: 3 years of software development
    Skills: Python, machine learning, TensorFlow, AWS, Docker, SQL
    Education: Master's degree in Computer Science
    
    Previous roles:
    - Machine Learning Engineer at TechCorp (2 years)
    - Software Developer at StartupXYZ (1 year)
    
    Expertise in deep learning, natural language processing, and cloud computing.
    """
    
    result1 = api.match_resume_text(sample_resume, top_k=5)
    
    if result1['success']:
        print(f"✅ Found {result1['total_matches']} matches")
        print("\nTop matches:")
        for i, match in enumerate(result1['matches'][:3], 1):
            print(f"{i}. {match['job_title']} - {match['confidence_score']:.1f}% match")
            print(f"   Skills: {match['required_skills']}")
            print(f"   Reasons: {'; '.join(match['match_reasons'])}")
    else:
        print(f"❌ Matching failed: {result1['error']}")
    
    # Example 2: Structured candidate data matching
    print("\n📊 Example 2: Structured Candidate Data Matching")
    candidate_data = {
        'skills': ['Java', 'Spring', 'Microservices', 'Kubernetes', 'AWS'],
        'experience_years': 5,
        'education_level': 'Bachelor',
        'skills_text': 'java spring microservices kubernetes aws'
    }
    
    filters = {
        'max_years_experience': 7,
        'min_salary_usd': 60000
    }
    
    result2 = api.match_candidate_data(candidate_data, top_k=5, filters=filters)
    
    if result2['success']:
        print(f"✅ Found {result2['total_matches']} matches")
        print("\nTop matches:")
        for i, match in enumerate(result2['matches'][:3], 1):
            print(f"{i}. {match['job_title']} - {match['confidence_score']:.1f}% match")
            print(f"   Salary: ${match['salary_usd']:,}")
            print(f"   Experience: {match['years_experience']} years")
    else:
        print(f"❌ Matching failed: {result2['error']}")
    
    print("\n🎉 Deep Resume Matcher demo completed!")