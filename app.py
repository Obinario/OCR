from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tempfile
from gradio_client import Client, handle_file
import json
import time
import concurrent.futures
from contextlib import contextmanager

app = Flask(__name__)

# Use a random or secret key (not your HF token!)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey123")

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size
OCR_TIMEOUT = 120  # 2 minutes timeout for OCR processing

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the OCR API client using Hugging Face token from environment
try:
    HF_TOKEN = os.getenv("HF_API_TOKEN")  # ✅ Use Render environment variable
    if not HF_TOKEN:
        raise ValueError("Missing Hugging Face API token. Set HF_API_TOKEN in Render Environment Variables.")

    # Initialize Gradio client securely
    ocr_client = Client("markobinario/OCRapi", hf_token=HF_TOKEN)
    api_available = True
except Exception as e:
    print(f"Warning: Could not initialize OCR API client: {e}")
    ocr_client = None
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
        if not api_available or not ocr_client:
            return None, "OCR API is not available. Please check your configuration or internet connection."

        def ocr_process():
            # You may need to adjust api_name depending on your Hugging Face Space setup
            return ocr_client.predict(
                pdf_file=handle_file(file_path),
                api_name="/predict_1"
            )

        result = run_with_timeout(ocr_process, OCR_TIMEOUT)

        # Extract results
        extracted_text = result[0] if len(result) > 0 else ""
        detailed_results = result[1] if len(result) > 1 else "{}"
        processing_stats = result[2] if len(result) > 2 else ""

        return {
            'extracted_text': extracted_text,
            'detailed_results': detailed_results,
            'processing_stats': processing_stats,
            'success': True
        }, None

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

    # Render doesn't automatically provide file size, so we manually check
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

        # Clean up
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
        'timestamp': time.time()
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
