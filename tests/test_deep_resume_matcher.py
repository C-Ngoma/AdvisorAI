#!/usr/bin/env python3
"""
Test Suite for Deep Resume Matcher

Tests for the deep learning resume parsing and job matching functionality.
"""

import os
import sys
import unittest
import tempfile
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

try:
    from deep_resume_matcher import ResumeParser, EmbeddingModel, JobMatcher, DeepResumeMatcherAPI
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    MODULES_AVAILABLE = False


class TestResumeParser(unittest.TestCase):
    """Test the ResumeParser class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Modules not available")
        self.parser = ResumeParser()
    
    def test_parse_simple_resume(self):
        """Test parsing a simple resume."""
        resume_text = """
        John Doe - Software Engineer
        Experience: 3 years
        Skills: Python, Java, SQL, Docker, AWS
        Education: Bachelor's degree
        """
        
        result = self.parser.parse_resume(resume_text)
        
        self.assertIsInstance(result, dict)
        self.assertIn('skills', result)
        self.assertIn('experience_years', result)
        self.assertIn('education_level', result)
        
        # Check if skills are detected
        skills = result['skills']
        self.assertIn('python', [s.lower() for s in skills])
        self.assertIn('java', [s.lower() for s in skills])
        
        # Check experience extraction
        self.assertGreaterEqual(result['experience_years'], 3)
    
    def test_parse_empty_resume(self):
        """Test parsing empty or invalid resume."""
        result = self.parser.parse_resume("")
        
        self.assertEqual(result['skills'], [])
        self.assertEqual(result['experience_years'], 0)
        self.assertEqual(result['education_level'], 'Associate')
    
    def test_skill_extraction(self):
        """Test skill extraction functionality."""
        text = "I have experience with python, machine learning, aws, and docker containers"
        skills = self.parser._extract_skills(text.lower())
        
        self.assertIn('python', skills)
        self.assertIn('aws', skills)
        self.assertIn('docker', skills)
    
    def test_experience_extraction(self):
        """Test experience years extraction."""
        texts = [
            "I have 5 years of experience",
            "Over 3 years in software development",
            "10+ years experience",
            "No specific experience mentioned"
        ]
        
        expected = [5, 3, 10, 0]
        
        for text, exp_years in zip(texts, expected):
            with self.subTest(text=text):
                years = self.parser._extract_experience(text.lower())
                self.assertEqual(years, exp_years)


class TestEmbeddingModel(unittest.TestCase):
    """Test the EmbeddingModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Modules not available")
        self.model = EmbeddingModel()
    
    def test_encode_texts(self):
        """Test text encoding functionality."""
        texts = [
            "python machine learning engineer",
            "java spring boot developer", 
            "data scientist with sql experience"
        ]
        
        embeddings = self.model.encode(texts)
        
        self.assertEqual(embeddings.shape[0], len(texts))
        self.assertGreater(embeddings.shape[1], 0)  # Should have embedding dimensions
    
    def test_encode_empty_list(self):
        """Test encoding empty list."""
        embeddings = self.model.encode([])
        self.assertEqual(embeddings.size, 0)
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        text = "Python, Machine-Learning & AI!!!"
        cleaned = self.model._clean_text(text)
        
        self.assertNotIn("!", cleaned)
        self.assertNotIn("&", cleaned)
        self.assertEqual(cleaned.lower(), cleaned)


class TestJobMatcher(unittest.TestCase):
    """Test the JobMatcher class."""
    
    def setUp(self):
        """Set up test fixtures with sample data."""
        if not MODULES_AVAILABLE:
            self.skipTest("Modules not available")
        
        # Create a temporary CSV file with test data
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_file.write("""job_id,job_title,required_skills,years_experience,salary_usd,industry,company_name
TEST001,Python Developer,"Python, Django, PostgreSQL",2,70000,Technology,TechCorp
TEST002,Data Scientist,"Python, Machine Learning, SQL",3,85000,Technology,DataCorp
TEST003,Java Engineer,"Java, Spring, MySQL",4,75000,Technology,JavaCorp
""")
        self.temp_file.close()
        
        self.matcher = JobMatcher(dataset_path=self.temp_file.name)
    
    def tearDown(self):
        """Clean up temporary files."""
        if hasattr(self, 'temp_file'):
            os.unlink(self.temp_file.name)
    
    def test_match_candidate_text(self):
        """Test candidate matching with text."""
        candidate_text = "Experienced Python developer with Django and machine learning skills"
        
        matches = self.matcher.match_candidate(candidate_text, top_k=3)
        
        self.assertIsInstance(matches, list)
        self.assertLessEqual(len(matches), 3)
        
        if matches:
            # Check match structure
            match = matches[0]
            self.assertIn('job_id', match)
            self.assertIn('job_title', match)
            self.assertIn('confidence_score', match)
            self.assertIn('similarity_score', match)
    
    def test_match_candidate_dict(self):
        """Test candidate matching with structured data."""
        candidate_data = {
            'skills': ['Python', 'Machine Learning', 'SQL'],
            'experience_years': 3,
            'education_level': 'Master',
            'skills_text': 'python machine learning sql'
        }
        
        matches = self.matcher.match_candidate(candidate_data, top_k=2)
        
        self.assertIsInstance(matches, list)
        self.assertLessEqual(len(matches), 2)
    
    def test_apply_filters(self):
        """Test filtering functionality."""
        candidate_data = {
            'skills': ['Python'],
            'experience_years': 5,
            'skills_text': 'python'
        }
        
        filters = {
            'min_salary_usd': 80000,
            'max_years_experience': 3
        }
        
        matches = self.matcher.match_candidate(candidate_data, filters=filters)
        
        # Should only return jobs that meet filter criteria
        for match in matches:
            self.assertGreaterEqual(match['salary_usd'], 80000)
            self.assertLessEqual(match['years_experience'], 3)


class TestDeepResumeMatcherAPI(unittest.TestCase):
    """Test the complete API wrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Modules not available")
        
        # Create temporary test dataset
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_file.write("""job_id,job_title,required_skills,years_experience,salary_usd,industry,company_name
API001,Senior Python Developer,"Python, Flask, AWS",4,90000,Technology,WebCorp
API002,ML Engineer,"Python, TensorFlow, Kubernetes",3,95000,AI,MLCorp
""")
        self.temp_file.close()
        
        self.api = DeepResumeMatcherAPI(dataset_path=self.temp_file.name)
    
    def tearDown(self):
        """Clean up temporary files."""
        if hasattr(self, 'temp_file'):
            os.unlink(self.temp_file.name)
    
    def test_match_resume_text_success(self):
        """Test successful resume text matching."""
        resume_text = """
        Senior Software Engineer with 5 years experience.
        Skills: Python, Flask, AWS, Machine Learning
        Education: Master's in Computer Science
        """
        
        result = self.api.match_resume_text(resume_text, top_k=2)
        
        self.assertTrue(result['success'])
        self.assertIn('matches', result)
        self.assertIn('candidate_info', result)
        self.assertIn('total_matches', result)
        self.assertLessEqual(len(result['matches']), 2)
    
    def test_match_resume_text_empty(self):
        """Test resume matching with empty text."""
        result = self.api.match_resume_text("", top_k=5)
        
        # Should handle gracefully
        self.assertIn('success', result)
        self.assertIn('matches', result)
    
    def test_match_candidate_data_success(self):
        """Test structured candidate data matching."""
        candidate_data = {
            'skills': ['Python', 'Flask', 'AWS'],
            'experience_years': 4,
            'education_level': 'Master',
            'skills_text': 'python flask aws web development'
        }
        
        result = self.api.match_candidate_data(candidate_data, top_k=2)
        
        self.assertTrue(result['success'])
        self.assertIn('matches', result)
        self.assertEqual(result['candidate_info'], candidate_data)
    
    def test_api_error_handling(self):
        """Test API error handling."""
        # Test with invalid data type
        try:
            result = self.api.match_candidate_data(None, top_k=5)
            self.assertFalse(result['success'])
            self.assertIn('error', result)
        except Exception:
            # Should handle gracefully without crashing
            pass


def run_tests():
    """Run all tests."""
    print("🧪 Running Deep Resume Matcher Tests")
    print("=" * 50)
    
    if not MODULES_AVAILABLE:
        print("❌ Cannot run tests - modules not available")
        return False
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestResumeParser,
        TestEmbeddingModel, 
        TestJobMatcher,
        TestDeepResumeMatcherAPI
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return True
    else:
        print(f"❌ {len(result.failures)} tests failed, {len(result.errors)} errors")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)