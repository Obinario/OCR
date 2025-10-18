#!/usr/bin/env python3
"""
PSAU OCR Service - API-based OCR with ML Classification
A comprehensive document processing system that combines API-based OCR with intelligent ML classification.
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tempfile
from gradio_client import Client, handle_file
import json
import time
import concurrent.futures
from contextlib import contextmanager
from ml_classifier import MLClassifier
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey123")
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size
OCR_TIMEOUT = 120  # 2 minutes timeout for OCR processing

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the OCR API client using Hugging Face token from environment
try:
    HF_TOKEN = os.getenv("HF_API_TOKEN")
    if not HF_TOKEN:
        raise ValueError("Missing Hugging Face API token. Set HF_API_TOKEN in environment variables.")

    # Initialize Gradio client securely
    ocr_client = Client("markobinario/OCRapi", hf_token=HF_TOKEN)
    api_available = True
    logger.info("✅ OCR API client initialized successfully!")
except Exception as e:
    logger.warning(f"Could not initialize OCR API client: {e}")
    ocr_client = None
    api_available = False

# Initialize ML Classifier
try:
    ml_classifier = MLClassifier()
    ml_available = ml_classifier.is_model_available()
    if ml_available:
        logger.info("✅ ML Classifier loaded successfully!")
    else:
        logger.warning("⚠️ ML Classifier not available - run 'python auto_train.py' to train the model")
except Exception as e:
    logger.warning(f"Could not initialize ML Classifier: {e}")
    ml_classifier = None
    ml_available = False


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def run_with_timeout(func, timeout_seconds):
    """Run a function with a timeout using concurrent.futures."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")


def process_pdf_with_ocr(file_path):
    """Process PDF file using the OCR API with timeout handling and ML classification."""
    try:
        if not api_available or not ocr_client:
            return None, "OCR API is not available. Please check your configuration or internet connection."

        def ocr_process():
            # Use the Hugging Face OCR API
            return ocr_client.predict(
                pdf_file=handle_file(file_path),
                api_name="/predict_1"
            )

        result = run_with_timeout(ocr_process, OCR_TIMEOUT)

        # Extract results
        extracted_text = result[0] if len(result) > 0 else ""
        detailed_results = result[1] if len(result) > 1 else "{}"
        processing_stats = result[2] if len(result) > 2 else ""

        # Prepare result dictionary
        result_dict = {
            'extracted_text': extracted_text,
            'detailed_results': detailed_results,
            'processing_stats': processing_stats,
            'success': True
        }

        # Add ML classification if available
        if ml_available and ml_classifier and extracted_text:
            try:
                # Convert extracted text to the format expected by ML classifier
                texts = [{'text': extracted_text, 'confidence': 100.0}]
                
                # Classify the document
                prediction = ml_classifier.classify_text(texts)
                result_dict['prediction'] = prediction
                
                # If it's a report card, verify pass/fail status
                if prediction == "Report Card":
                    status_info = ml_classifier.verify_report_card_status(texts)
                    result_dict['status_info'] = status_info
                    result_dict['is_report_card'] = True
                    result_dict['has_failed_remarks'] = status_info.get('status') == 'failed' if status_info else False
                else:
                    result_dict['is_report_card'] = False
                    result_dict['has_failed_remarks'] = False
                    
            except Exception as e:
                logger.warning(f"ML classification failed: {e}")
                result_dict['prediction'] = "Classification Error"
                result_dict['is_report_card'] = False
                result_dict['has_failed_remarks'] = False
        else:
            result_dict['prediction'] = "ML Not Available"
            result_dict['is_report_card'] = False
            result_dict['has_failed_remarks'] = False

        return result_dict, None

    except TimeoutError:
        return None, f"OCR processing timed out after {OCR_TIMEOUT} seconds. The PDF might be too large or complex."
    except Exception as e:
        return None, f"Error processing PDF: {str(e)}"


@app.route('/')
def index():
    """Main page with file upload form."""
    return render_template('index.html', api_available=api_available)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and OCR processing."""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a PDF file.')
        return redirect(request.url)

    # Check file size
    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    file.seek(0)

    if file_length > MAX_FILE_SIZE:
        flash('File too large. Maximum size is 16MB.')
        return redirect(request.url)

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        result, error = process_pdf_with_ocr(file_path)

        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass

        if error:
            flash(f'Error: {error}')
            return redirect(url_for('index'))

        return render_template('index.html', result=result, api_available=api_available)

    except Exception as e:
        flash(f'Error processing file: {str(e)}')
        return redirect(url_for('index'))


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'api_available': api_available,
        'ml_available': ml_available,
        'timestamp': time.time()
    })


@app.route('/api/classify', methods=['POST'])
def api_classify():
    """API endpoint for document classification."""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"})
        
        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Invalid file type"})
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Process the document
        result, error = process_pdf_with_ocr(file_path)
        
        # Clean up
        try:
            os.remove(file_path)
        except:
            pass
        
        if error:
            return jsonify({"success": False, "error": error})
        
        # Return API response
        return jsonify({
            "success": True,
            "prediction": result.get('prediction', 'Unknown'),
            "is_report_card": result.get('is_report_card', False),
            "has_failed_remarks": result.get('has_failed_remarks', False),
            "status_info": result.get('status_info'),
            "extracted_text": result.get('extracted_text', ''),
            "processing_stats": result.get('processing_stats', '')
        })
        
    except Exception as e:
        logger.error(f"Error in API classify: {str(e)}")
        return jsonify({"success": False, "error": f"Error processing document: {str(e)}"})


if __name__ == '__main__':
    logger.info("Starting PSAU OCR Service...")
    logger.info(f"OCR API Available: {api_available}")
    logger.info(f"ML Classification Available: {ml_available}")
    logger.info("Service ready at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
