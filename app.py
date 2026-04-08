"""
Enterprise Spam Detection Application
Based on BERT model from Spam_Bert.ipynb
Professional Flask application for large-scale spam detection
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from transformers import BertTokenizer, TFAutoModel
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings("ignore")

# Configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'enterprise-spam-detection-2024'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'txt', 'csv'}
    MODEL_PATH = 'models/spam_classifier_bert.keras'
    TOKENIZER_PATH = './bert_tokenizer'
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

# Global variables for model components
model = None
tokenizer = None
config = None

class SpamDetector:
    """Enterprise-grade Spam Detection Class"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.stop_words = set(stopwords.words('english'))
        self.porter = PorterStemmer()
        self.load_model()
    
    def load_model(self):
        """Load BERT model and tokenizer"""
        try:
            logger.info("Loading BERT model and tokenizer...")
            
            # Load tokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            # Load or create model
            if os.path.exists(Config.MODEL_PATH):
                logger.info(f"Loading existing model from {Config.MODEL_PATH}")
                self.model = tf.keras.models.load_model(Config.MODEL_PATH)
            else:
                logger.info("Creating new BERT model...")
                self.model = self._build_spam_classifier()
            
            # Load config
            if os.path.exists(Config.CONFIG_PATH):
                with open(Config.CONFIG_PATH, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = {
                    'seuil_optimal': Config.SEUIL_OPTIMAL,
                    'max_len': Config.MAX_LEN
                }
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _build_spam_classifier(self):
        """Build spam classifier using BERT"""
        bert_model = TFAutoModel.from_pretrained('bert-base-uncased')
        
        # Freeze most layers, fine-tune last few
        bert_model.trainable = True
        for layer in bert_model.layers[:-4]:
            layer.trainable = False
        
        # Build the model
        input_word_ids = tf.keras.Input(shape=(Config.MAX_LEN,), dtype='int32')
        attention_masks = tf.keras.Input(shape=(Config.MAX_LEN,), dtype='int32')
        
        sequence_output = bert_model([input_word_ids, attention_masks])
        output = sequence_output[1]  # pooler_output
        
        # Classification head
        output = tf.keras.layers.Dense(64, activation='relu')(output)
        output = tf.keras.layers.Dropout(0.3)(output)
        output = tf.keras.layers.Dense(32, activation='relu')(output)
        output = tf.keras.layers.Dropout(0.2)(output)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
        
        model = tf.keras.models.Model(
            inputs=[input_word_ids, attention_masks],
            outputs=output
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def preprocess_text(self, text):
        """Preprocess text for prediction"""
        # Clean text
        text = re.sub("[^a-zA-Z]", " ", str(text))
        text = text.lower().strip()
        
        # Remove stopwords and apply stemming
        words = [self.porter.stem(word) for word in text.split() 
                if word not in self.stop_words]
        
        return " ".join(words)
    
    def encode_text(self, text):
        """Encode text for BERT"""
        if isinstance(text, str):
            text = [text]
        
        input_ids = []
        attention_masks = []
        
        for t in text:
            encoded = self.tokenizer.encode_plus(
                t,
                add_special_tokens=True,
                max_length=self.config['max_len'],
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='np'
            )
            input_ids.append(encoded['input_ids'][0])
            attention_masks.append(encoded['attention_mask'][0])
        
        return np.array(input_ids), np.array(attention_masks)
    
    def predict(self, text):
        """Make prediction on text"""
        try:
            # Preprocess
            processed_text = self.preprocess_text(text)
            
            # Encode
            input_ids, attention_masks = self.encode_text(processed_text)
            
            # Predict
            prediction = self.model.predict(
                [input_ids, attention_masks], 
                verbose=0
            )[0][0]
            
            # Classify
            threshold = self.config.get('seuil_optimal', Config.SEUIL_OPTIMAL)
            is_spam = prediction > threshold
            confidence = prediction if is_spam else 1 - prediction
            
            return {
                'is_spam': bool(is_spam),
                'confidence': float(confidence),
                'raw_score': float(prediction),
                'threshold': threshold,
                'processed_text': processed_text
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'error': str(e),
                'is_spam': False,
                'confidence': 0.0
            }
    
    def batch_predict(self, texts):
        """Make predictions on multiple texts"""
        results = []
        for text in texts:
            result = self.predict(text)
            result['original_text'] = text
            results.append(result)
        return results

# Initialize spam detector
spam_detector = SpamDetector()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

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
            flash('Please enter a message to analyze', 'error')
            return redirect(url_for('index'))
        
        result = spam_detector.predict(text)
        
        return render_template('result.html', 
                             text=text, 
                             result=result,
                             timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for single prediction"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        result = spam_detector.predict(data['text'])
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def api_batch_predict():
    """API endpoint for batch prediction"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'No texts provided'}), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({'error': 'texts must be a list'}), 400
        
        results = spam_detector.batch_predict(texts)
        
        # Summary statistics
        total = len(results)
        spam_count = sum(1 for r in results if r.get('is_spam', False))
        ham_count = total - spam_count
        
        return jsonify({
            'results': results,
            'summary': {
                'total': total,
                'spam': spam_count,
                'ham': ham_count,
                'spam_percentage': round((spam_count / total) * 100, 2) if total > 0 else 0
            }
        })
    
    except Exception as e:
        logger.error(f"API batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """File upload for batch processing"""
    if request.method == 'GET':
        return render_template('upload.html')
    
    try:
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Create upload directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            
            # Read and process file
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
                if 'message' in df.columns:
                    texts = df['message'].tolist()
                elif 'text' in df.columns:
                    texts = df['text'].tolist()
                else:
                    # Use first column
                    texts = df.iloc[:, 0].tolist()
            else:
                # TXT file - one message per line
                with open(filepath, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f if line.strip()]
            
            # Process texts
            results = spam_detector.batch_predict(texts)
            
            # Add original texts to results
            for i, result in enumerate(results):
                result['original_text'] = texts[i]
            
            # Generate summary
            total = len(results)
            spam_count = sum(1 for r in results if r.get('is_spam', False))
            ham_count = total - spam_count
            
            summary = {
                'total': total,
                'spam': spam_count,
                'ham': ham_count,
                'spam_percentage': round((spam_count / total) * 100, 2) if total > 0 else 0
            }
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return render_template('batch_results.html',
                                 results=results,
                                 summary=summary,
                                 filename=filename,
                                 timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        else:
            flash('Invalid file type. Please upload .txt or .csv files only.', 'error')
            return redirect(request.url)
    
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('upload_file'))

@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    return render_template('analytics.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': spam_detector.model is not None,
        'tokenizer_loaded': spam_detector.tokenizer is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('upload_file'))

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return render_template('error.html', error=str(e)), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    logger.info("Starting Enterprise Spam Detection Application")
    app.run(debug=True, host='0.0.0.0', port=5000)
