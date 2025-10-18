from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tempfile
from gradio_client import Client
import json
import time
import concurrent.futures
from contextlib import contextmanager

app = Flask(__name__)
app.secret_key = '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'  # Change this to a random secret key

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size
OCR_TIMEOUT = 120  # 2 minutes timeout for OCR processing

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the OCR API client
try:
    ocr_client = Client("markobinario/OCRapi")
    api_available = True
except Exception as e:
    print(f"Warning: Could not initialize OCR API client: {e}")
    api_available = False

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
    """Process PDF file using the OCR API with timeout handling."""
    try:
        if not api_available:
            return None, "OCR API is not available. Please check your internet connection."
        
        # Define the OCR processing function
        def ocr_process():
            return ocr_client.predict(
                pdf_file=file_path,
                api_name="/predict_1"  # Using the successful endpoint
            )
        
        # Run OCR processing with timeout
        result = run_with_timeout(ocr_process, OCR_TIMEOUT)
        
        # Extract results from the tuple
        extracted_text = result[0] if len(result) > 0 else ""
        detailed_results = result[1] if len(result) > 1 else "{}"
        processing_stats = result[2] if len(result) > 2 else ""
        
        return {
            'extracted_text': extracted_text,
            'detailed_results': detailed_results,
            'processing_stats': processing_stats,
            'success': True
        }, None
        
    except TimeoutError as e:
        return None, f"OCR processing timed out after {OCR_TIMEOUT} seconds. The PDF might be too large or complex. Please try with a smaller file."
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            return None, f"OCR processing timed out. The PDF might be too large or complex. Please try with a smaller file."
        elif "connection" in error_msg.lower():
            return None, "Connection to OCR API failed. Please check your internet connection and try again."
        else:
            return None, f"Error processing PDF: {error_msg}"

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
    
    if file.content_length and file.content_length > MAX_FILE_SIZE:
        flash('File too large. Maximum size is 16MB.')
        return redirect(request.url)
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Process with OCR
        result, error = process_pdf_with_ocr(file_path)
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        if error:
            flash(f'Error: {error}')
            return redirect(url_for('index'))
        
        # Return results as JSON for AJAX request
        if request.headers.get('Content-Type') == 'application/json':
            return jsonify(result)
        
        # For regular form submission, render with results
        return render_template('index.html', 
                             result=result, 
                             api_available=api_available)
        
    except Exception as e:
        flash(f'Error processing file: {str(e)}')
        return redirect(url_for('index'))

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'api_available': api_available,
        'timestamp': time.time()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
