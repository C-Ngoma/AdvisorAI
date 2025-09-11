"""
Flask API for Deep Learning Resume Matching

This module provides a web API for the deep learning resume matcher,
allowing web-based candidate-job matching through HTTP endpoints.

Endpoints:
- POST /match/resume - Match resume text to jobs
- POST /match/candidate - Match structured candidate data to jobs
- GET /health - Health check
- GET /stats - Get system statistics
"""

import os
import sys
from flask import Flask, request, jsonify, render_template_string
from flask import send_from_directory
import logging
import traceback
from datetime import datetime

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from deep_resume_matcher import DeepResumeMatcherAPI
    MATCHER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import DeepResumeMatcherAPI: {e}")
    MATCHER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global matcher instance
matcher_api = None

def initialize_matcher():
    """Initialize the deep learning matcher API."""
    global matcher_api
    if MATCHER_AVAILABLE and matcher_api is None:
        try:
            logger.info("Initializing Deep Resume Matcher API...")
            matcher_api = DeepResumeMatcherAPI()
            logger.info("✅ Deep Resume Matcher API initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize matcher API: {e}")
            return False
    return matcher_api is not None

# Simple HTML template for the API interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep Resume Matcher API</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .method { background: #007bff; color: white; padding: 2px 8px; border-radius: 3px; font-size: 12px; }
        .method.post { background: #28a745; }
        .method.get { background: #17a2b8; }
        pre { background: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto; }
        .demo-area { background: #e9ecef; padding: 15px; border-radius: 5px; margin-top: 20px; }
        textarea { width: 100%; height: 150px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 3px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { background: #d4edda; padding: 10px; border-radius: 3px; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>🤖 Deep Learning Resume Matcher API</h1>
    <p>AI-powered resume parsing and job matching using transformer embeddings.</p>
    
    <h2>API Endpoints</h2>
    
    <div class="endpoint">
        <h3><span class="method post">POST</span> /match/resume</h3>
        <p>Match resume text to relevant jobs</p>
        <strong>Request Body:</strong>
        <pre>{
  "resume_text": "John Doe - Software Engineer...",
  "top_k": 5,
  "filters": {
    "max_years_experience": 5,
    "min_salary_usd": 60000
  }
}</pre>
    </div>
    
    <div class="endpoint">
        <h3><span class="method post">POST</span> /match/candidate</h3>
        <p>Match structured candidate data to jobs</p>
        <strong>Request Body:</strong>
        <pre>{
  "candidate_data": {
    "skills": ["python", "machine learning"],
    "experience_years": 3,
    "education_level": "Master"
  },
  "top_k": 10
}</pre>
    </div>
    
    <div class="endpoint">
        <h3><span class="method get">GET</span> /health</h3>
        <p>Check API health status</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method get">GET</span> /stats</h3>
        <p>Get system statistics</p>
    </div>
    
    <div class="demo-area">
        <h3>🚀 Try it out!</h3>
        <textarea id="resume-text" placeholder="Paste your resume text here...">
John Doe - Software Engineer

Experience: 3 years of software development
Skills: Python, machine learning, TensorFlow, AWS, Docker, SQL
Education: Master's degree in Computer Science

Previous roles:
- Machine Learning Engineer at TechCorp (2 years)
- Software Developer at StartupXYZ (1 year)

Expertise in deep learning, natural language processing, and cloud computing.
        </textarea>
        <br><br>
        <button onclick="matchResume()">Match Resume to Jobs</button>
        <div id="result"></div>
    </div>
    
    <script>
        async function matchResume() {
            const resumeText = document.getElementById('resume-text').value;
            const resultDiv = document.getElementById('result');
            
            try {
                const response = await fetch('/match/resume', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        resume_text: resumeText,
                        top_k: 5
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    let html = `<div class="result">
                        <h4>✅ Found ${data.total_matches} matches!</h4>
                        <h5>Parsed Skills: ${data.candidate_info.skills.join(', ')}</h5>
                        <h5>Top Matches:</h5>
                        <ol>`;
                    
                    data.matches.forEach(match => {
                        html += `<li>
                            <strong>${match.job_title}</strong> (${match.confidence_score}% match)<br>
                            Skills: ${match.required_skills}<br>
                            Salary: $${match.salary_usd.toLocaleString()}<br>
                            Experience: ${match.years_experience} years<br>
                            <small>Reasons: ${match.match_reasons.join(', ')}</small>
                        </li><br>`;
                    });
                    
                    html += `</ol></div>`;
                    resultDiv.innerHTML = html;
                } else {
                    resultDiv.innerHTML = `<div style="background: #f8d7da; padding: 10px; border-radius: 3px;">
                        ❌ Error: ${data.error}
                    </div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div style="background: #f8d7da; padding: 10px; border-radius: 3px;">
                    ❌ Network error: ${error.message}
                </div>`;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Serve the API documentation and demo interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health')
def health_check():
    """Health check endpoint."""
    try:
        if matcher_api is None:
            init_success = initialize_matcher()
            if not init_success:
                return jsonify({
                    'status': 'unhealthy',
                    'message': 'Matcher API not initialized',
                    'timestamp': datetime.utcnow().isoformat()
                }), 503
        
        return jsonify({
            'status': 'healthy',
            'message': 'Deep Resume Matcher API is running',
            'matcher_initialized': matcher_api is not None,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/stats')
def get_stats():
    """Get system statistics."""
    try:
        stats = {
            'matcher_initialized': matcher_api is not None,
            'matcher_available': MATCHER_AVAILABLE,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if matcher_api:
            # Add job dataset stats
            job_count = len(matcher_api.job_matcher.df)
            embedding_dim = matcher_api.job_matcher.job_embeddings.shape[1]
            
            stats.update({
                'total_jobs_loaded': job_count,
                'embedding_dimensions': embedding_dim,
                'dataset_path': matcher_api.job_matcher.dataset_path
            })
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/match/resume', methods=['POST'])
def match_resume():
    """Match resume text to jobs."""
    try:
        # Initialize matcher if needed
        if matcher_api is None:
            init_success = initialize_matcher()
            if not init_success:
                return jsonify({
                    'success': False,
                    'error': 'Matcher API not available',
                    'matches': []
                }), 503
        
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'matches': []
            }), 400
        
        resume_text = data.get('resume_text', '')
        if not resume_text:
            return jsonify({
                'success': False,
                'error': 'resume_text field is required',
                'matches': []
            }), 400
        
        top_k = data.get('top_k', 10)
        filters = data.get('filters', {})
        
        # Validate parameters
        if not isinstance(top_k, int) or top_k <= 0 or top_k > 100:
            top_k = 10
        
        # Perform matching
        result = matcher_api.match_resume_text(
            resume_text=resume_text,
            top_k=top_k,
            filters=filters
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Resume matching error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'matches': []
        }), 500

@app.route('/match/candidate', methods=['POST'])
def match_candidate():
    """Match structured candidate data to jobs."""
    try:
        # Initialize matcher if needed
        if matcher_api is None:
            init_success = initialize_matcher()
            if not init_success:
                return jsonify({
                    'success': False,
                    'error': 'Matcher API not available',
                    'matches': []
                }), 503
        
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'matches': []
            }), 400
        
        candidate_data = data.get('candidate_data', {})
        if not candidate_data:
            return jsonify({
                'success': False,
                'error': 'candidate_data field is required',
                'matches': []
            }), 400
        
        top_k = data.get('top_k', 10)
        filters = data.get('filters', {})
        
        # Validate parameters
        if not isinstance(top_k, int) or top_k <= 0 or top_k > 100:
            top_k = 10
        
        # Perform matching
        result = matcher_api.match_candidate_data(
            candidate_data=candidate_data,
            top_k=top_k,
            filters=filters
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Candidate matching error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'matches': []
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /',
            'GET /health',
            'GET /stats',
            'POST /match/resume',
            'POST /match/candidate'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Deep Resume Matcher API Server...")
    
    # Initialize the matcher
    if initialize_matcher():
        print("✅ Matcher API initialized successfully")
    else:
        print("⚠️ Matcher API initialization failed - running with limited functionality")
    
    # Start the Flask server
    print("🌐 Starting Flask server on http://localhost:5000")
    print("📖 API documentation available at: http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Avoid double initialization
    )