"""
Enterprise Spam Detection Application
Modified to use simple sklearn model instead of BERT
Professional Flask application for large-scale spam detection
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime
import re
import warnings
warnings.filterwarnings("ignore")

# Configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'enterprise-spam-detection-2024'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'txt', 'csv'}
    MODEL_PATH = 'models/simple_spam_model.pkl'
    CONFIG_PATH = 'model_config.json'
    MAX_LEN = 64
    SEUIL_OPTIMAL = 0.65

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spam_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Simple text preprocessing
def clean_text(text):
    """Simple text cleaning"""
    text = re.sub("[^a-zA-Z]", " ", str(text))
    return text.lower().strip()

class SpamDetector:
    """Simplified Spam Detector using sklearn model"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.config = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and vectorizer"""
        try:
            # Load the sklearn model
            with open(Config.MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
                self.vectorizer = model_data['vectorizer']
                self.model = model_data['model']
                # Use the optimized threshold from the model
                if 'threshold' in model_data:
                    Config.SEUIL_OPTIMAL = model_data['threshold']
            
            # Load configuration
            if os.path.exists(Config.CONFIG_PATH):
                with open(Config.CONFIG_PATH, 'r') as f:
                    self.config = json.load(f)
                    Config.SEUIL_OPTIMAL = self.config.get('seuil_optimal', Config.SEUIL_OPTIMAL)
                    Config.MAX_LEN = self.config.get('max_len', 64)
            
            logger.info(f"Model loaded successfully with threshold: {Config.SEUIL_OPTIMAL}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_text(self, text):
        """Preprocess text for prediction"""
        if not text or not isinstance(text, str):
            return ""
        
        # Clean the text
        cleaned = clean_text(text)
        return cleaned
    
    def predict(self, text):
        """Predict if text is spam"""
        try:
            if not self.model or not self.vectorizer:
                raise Exception("Model not loaded")
            
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Vectorize
            text_vector = self.vectorizer.transform([processed_text])
            
            # Predict
            probability = self.model.predict_proba(text_vector)[0][1]
            prediction = int(probability > Config.SEUIL_OPTIMAL)
            
            return {
                'prediction': prediction,
                'probability': float(probability),
                'confidence': float(max(probability, 1 - probability)),
                'processed_text': processed_text,
                'threshold': Config.SEUIL_OPTIMAL
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'prediction': 0,
                'probability': 0.0,
                'confidence': 0.0,
                'processed_text': text,
                'threshold': Config.SEUIL_OPTIMAL,
                'error': str(e)
            }
    
    def batch_predict(self, texts):
        """Predict multiple texts"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results

# Initialize the detector
detector = SpamDetector()

# Helper functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def read_uploaded_file(file_path):
    """Read uploaded file and extract messages"""
    messages = []
    
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            # Try different column names
            for col in ['message', 'text', 'content', 'v2', 'Text']:
                if col in df.columns:
                    messages = df[col].dropna().tolist()
                    break
            if not messages:
                # Use first column
                messages = df.iloc[:, 0].dropna().tolist()
        else:  # .txt file
            with open(file_path, 'r', encoding='utf-8') as f:
                messages = [line.strip() for line in f if line.strip()]
    
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
    
    return messages

# Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Single message prediction"""
    try:
        text = request.form.get('message', '').strip()
        
        if not text:
            flash('Please enter a message to analyze', 'warning')
            return redirect(url_for('index'))
        
        # Get prediction
        result = detector.predict(text)
        
        # Prepare result data
        result_data = {
            'original_text': text,
            'is_spam': bool(result['prediction']),
            'confidence': result['confidence'],
            'raw_score': result['probability'],
            'processed_text': result['processed_text'],
            'threshold': result['threshold'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return render_template('result.html', result=result_data)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        flash('An error occurred during analysis', 'danger')
        return render_template('error.html', error=str(e))

@app.route('/upload')
def upload_page():
    """File upload page"""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and batch processing"""
    try:
        if 'file' not in request.files:
            flash('No file selected', 'warning')
            return redirect(url_for('upload_page'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'warning')
            return redirect(url_for('upload_page'))
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Create upload directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            file.save(file_path)
            
            # Read and process messages
            messages = read_uploaded_file(file_path)
            
            if not messages:
                flash('No valid messages found in the file', 'warning')
                return redirect(url_for('upload_page'))
            
            # Batch prediction
            results = detector.batch_predict(messages)
            
            # Prepare results
            processed_results = []
            for i, (msg, result) in enumerate(zip(messages, results)):
                processed_results.append({
                    'index': i + 1,
                    'original_text': msg,
                    'is_spam': bool(result['prediction']),
                    'confidence': result['confidence'],
                    'raw_score': result['probability'],
                    'processed_text': result['processed_text']
                })
            
            # Calculate summary
            spam_count = sum(1 for r in processed_results if r['is_spam'])
            ham_count = len(processed_results) - spam_count
            
            summary = {
                'total': len(processed_results),
                'spam': spam_count,
                'ham': ham_count,
                'spam_rate': round((spam_count / len(processed_results)) * 100, 2),
                'avg_confidence': round(np.mean([r['confidence'] for r in processed_results]) * 100, 2)
            }
            
            # Clean up uploaded file
            os.remove(file_path)
            
            return render_template('batch_results.html', 
                               results=processed_results,
                               summary=summary,
                               timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        else:
            flash('Invalid file type. Please upload .txt or .csv files only', 'danger')
            return redirect(url_for('upload_page'))
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        flash('An error occurred while processing the file', 'danger')
        return render_template('error.html', error=str(e))

@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    return render_template('analytics.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if detector.model else 'unhealthy',
        'model_loaded': detector.model is not None,
        'vectorizer_loaded': detector.vectorizer is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }
    
    return jsonify(status)

# API Endpoints
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for single prediction"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text field is required'}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        result = detector.predict(text)
        
        return jsonify({
            'prediction': int(result['prediction']),
            'probability': result['probability'],
            'confidence': result['confidence'],
            'is_spam': bool(result['prediction']),
            'threshold': result['threshold'],
            'processed_text': result['processed_text']
        })
    
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def api_batch_predict():
    """API endpoint for batch prediction"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Texts field is required'}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({'error': 'Texts must be an array'}), 400
        
        if not texts:
            return jsonify({'error': 'Texts array cannot be empty'}), 400
        
        results = detector.batch_predict(texts)
        
        formatted_results = []
        for i, (text, result) in enumerate(zip(texts, results)):
            formatted_results.append({
                'index': i,
                'text': text,
                'prediction': int(result['prediction']),
                'probability': result['probability'],
                'confidence': result['confidence'],
                'is_spam': bool(result['prediction']),
                'threshold': result['threshold']
            })
        
        return jsonify({
            'results': formatted_results,
            'total': len(formatted_results),
            'spam_count': sum(1 for r in formatted_results if r['is_spam']),
            'ham_count': sum(1 for r in formatted_results if not r['is_spam'])
        })
    
    except Exception as e:
        logger.error(f"API batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info')
def api_model_info():
    """API endpoint for model information"""
    try:
        info = {
            'model_type': 'sklearn_logistic_regression',
            'threshold': Config.SEUIL_OPTIMAL,
            'max_length': Config.MAX_LEN,
            'features': len(detector.vectorizer.vocabulary_) if detector.vectorizer else 0,
            'model_loaded': detector.model is not None,
            'version': '1.0.0'
        }
        
        return jsonify(info)
    
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error='Internal server error'), 500

if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    print("=" * 50)
    print("SPAMGUARD ENTERPRISE - Starting Application")
    print("=" * 50)
    print(f"Model loaded: {detector.model is not None}")
    print(f"Vectorizer loaded: {detector.vectorizer is not None}")
    print(f"Threshold: {Config.SEUIL_OPTIMAL}")
    print(f"Upload folder: {Config.UPLOAD_FOLDER}")
    print("=" * 50)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)
