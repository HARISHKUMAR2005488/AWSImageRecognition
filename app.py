"""
Flask Web Application for AWS Rekognition Image Label Detection

This application allows users to upload images through a web interface,
automatically uploads them to S3, and uses AWS Rekognition to detect labels.
"""

import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# ============================================================================
# CONFIGURATION - Load from environment variables
# ============================================================================

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

# File upload configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit

# Rekognition configuration
MAX_LABELS = 10
MIN_CONFIDENCE = 75

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_aws_config():
    """
    Validate that all required AWS environment variables are set.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not AWS_ACCESS_KEY_ID:
        return False, "AWS_ACCESS_KEY_ID environment variable is not set"
    if not AWS_SECRET_ACCESS_KEY:
        return False, "AWS_SECRET_ACCESS_KEY environment variable is not set"
    if not S3_BUCKET_NAME:
        return False, "S3_BUCKET_NAME environment variable is not set"
    return True, None


def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    
    Args:
        filename: Name of the uploaded file
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def upload_to_s3(file, filename):
    """
    Upload a file to S3 bucket.
    
    Args:
        file: File object from Flask request
        filename: Secure filename to use in S3
        
    Returns:
        tuple: (success, error_message)
    """
    try:
        # Create S3 client with credentials
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        
        # Upload file to S3
        s3_client.upload_fileobj(
            file,
            S3_BUCKET_NAME,
            filename,
            ExtraArgs={'ContentType': file.content_type}
        )
        
        return True, None
        
    except NoCredentialsError:
        return False, "AWS credentials are invalid or not configured properly"
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            return False, f"S3 bucket '{S3_BUCKET_NAME}' does not exist"
        elif error_code == 'AccessDenied':
            return False, "Access denied. Check your AWS permissions"
        else:
            return False, f"AWS error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error uploading to S3: {str(e)}"


def detect_labels_rekognition(filename):
    """
    Detect labels in an image using AWS Rekognition.
    
    Args:
        filename: Name of the file in S3 bucket
        
    Returns:
        tuple: (labels_list, error_message)
        labels_list format: [{'Name': 'label', 'Confidence': 95.5}, ...]
    """
    try:
        # Create Rekognition client with credentials
        rekognition_client = boto3.client(
            'rekognition',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        
        # Call Rekognition detect_labels API
        response = rekognition_client.detect_labels(
            Image={
                'S3Object': {
                    'Bucket': S3_BUCKET_NAME,
                    'Name': filename
                }
            },
            MaxLabels=MAX_LABELS,
            MinConfidence=MIN_CONFIDENCE
        )
        
        # Extract labels from response
        labels = response.get('Labels', [])
        
        # Format labels for display
        formatted_labels = [
            {
                'Name': label['Name'],
                'Confidence': round(label['Confidence'], 2)
            }
            for label in labels
        ]
        
        return formatted_labels, None
        
    except NoCredentialsError:
        return None, "AWS credentials are invalid or not configured properly"
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'InvalidS3ObjectException':
            return None, f"Image '{filename}' not found in S3 bucket"
        elif error_code == 'InvalidImageFormatException':
            return None, "Invalid image format. Please upload a valid image file"
        elif error_code == 'ImageTooLargeException':
            return None, "Image is too large for Rekognition processing"
        elif error_code == 'AccessDeniedException':
            return None, "Access denied. Check your AWS Rekognition permissions"
        else:
            return None, f"AWS Rekognition error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error during label detection: {str(e)}"


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """
    Render the main upload form page.
    """
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload, S3 upload, and Rekognition label detection.
    
    Process:
    1. Validate AWS configuration
    2. Validate uploaded file
    3. Upload to S3
    4. Detect labels with Rekognition
    5. Display results
    """
    
    # Step 1: Validate AWS configuration
    is_valid, error_msg = validate_aws_config()
    if not is_valid:
        flash(f'Configuration Error: {error_msg}', 'error')
        return redirect(url_for('index'))
    
    # Step 2: Validate uploaded file
    if 'file' not in request.files:
        flash('No file selected. Please choose an image to upload.', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected. Please choose an image to upload.', 'error')
        return redirect(url_for('index'))
    
    if not allowed_file(file.filename):
        flash(f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}', 'error')
        return redirect(url_for('index'))
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        flash(f'File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB', 'error')
        return redirect(url_for('index'))
    
    if file_size == 0:
        flash('File is empty. Please upload a valid image.', 'error')
        return redirect(url_for('index'))
    
    # Step 3: Secure filename and upload to S3
    filename = secure_filename(file.filename)
    
    # Add timestamp to avoid filename conflicts
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{filename}"
    
    success, error_msg = upload_to_s3(file, filename)
    
    if not success:
        flash(f'Upload Error: {error_msg}', 'error')
        return redirect(url_for('index'))
    
    # Step 4: Detect labels with Rekognition
    labels, error_msg = detect_labels_rekognition(filename)
    
    if labels is None:
        flash(f'Detection Error: {error_msg}', 'error')
        return redirect(url_for('index'))
    
    # Step 5: Store results in session and redirect to results page
    session['filename'] = filename
    session['labels'] = labels
    session['bucket'] = S3_BUCKET_NAME
    
    return redirect(url_for('results'))


@app.route('/results')
def results():
    """
    Display the detection results page.
    """
    # Retrieve results from session
    filename = session.get('filename')
    labels = session.get('labels')
    bucket = session.get('bucket')
    
    if not filename or not labels:
        flash('No results available. Please upload an image first.', 'error')
        return redirect(url_for('index'))
    
    # Generate S3 URL for image preview
    s3_url = f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{filename}"
    
    return render_template('result.html', 
                         filename=filename, 
                         labels=labels, 
                         s3_url=s3_url,
                         label_count=len(labels))


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    flash('File is too large. Please upload a smaller image.', 'error')
    return redirect(url_for('index'))


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors."""
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Run Flask development server
    # WARNING: Do not use in production. Use a production WSGI server like Gunicorn
    app.run(debug=True, host='0.0.0.0', port=5000)
