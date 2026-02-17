"""
Flask Web Application for AWS Rekognition Image Label Detection

This application allows users to upload images through a web interface,
automatically uploads them to S3, and uses AWS Rekognition to detect labels.
"""

import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from werkzeug.utils import secure_filename
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# CONFIGURATION - Load from environment variables

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

# File upload configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit

# Rekognition configuration
MAX_LABELS = 100  # Maximum number of labels to detect (increased for better detection)
MIN_CONFIDENCE = 55  # Minimum confidence threshold in percentage (lowered for more detections)

# HELPER FUNCTIONS

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


def download_from_s3(filename):
    """
    Download an image from S3 bucket to memory.
    
    Args:
        filename: Name of the file in S3 bucket
        
    Returns:
        tuple: (image_bytes, content_type, error_message)
    """
    try:
        # Create S3 client with credentials
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        
        # Download file from S3
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=filename)
        image_bytes = response['Body'].read()
        content_type = response.get('ContentType', 'image/jpeg')
        
        return image_bytes, content_type, None
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            return None, None, f"Image '{filename}' not found in S3 bucket"
        else:
            return None, None, f"Error downloading from S3: {str(e)}"
    except Exception as e:
        return None, None, f"Unexpected error downloading image: {str(e)}"


def detect_labels_rekognition(filename):
    """
    Detect labels in an image using AWS Rekognition.
    
    Args:
        filename: Name of the file in S3 bucket
        
    Returns:
        tuple: (labels_list, full_response, error_message)
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
                'Confidence': round(label['Confidence'], 2),
                'Instances': label.get('Instances', [])
            }
            for label in labels
        ]
        
        return formatted_labels, response, None
        
    except NoCredentialsError:
        return None, None, "AWS credentials are invalid or not configured properly"
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'InvalidS3ObjectException':
            return None, None, f"Image '{filename}' not found in S3 bucket"
        elif error_code == 'InvalidImageFormatException':
            return None, None, "Invalid image format. Please upload a valid image file"
        elif error_code == 'ImageTooLargeException':
            return None, None, "Image is too large for Rekognition processing"
        elif error_code == 'AccessDeniedException':
            return None, None, "Access denied. Check your AWS Rekognition permissions"
        else:
            return None, None, f"AWS Rekognition error: {str(e)}"
    except Exception as e:
        return None, None, f"Unexpected error during label detection: {str(e)}"


def generate_labeled_image(filename, rekognition_response):
    """
    Generate an image with bounding boxes and labels drawn on it.
    Includes smart overlap prevention and displays all detected labels.
    
    Args:
        filename: Name of the file in S3 bucket
        rekognition_response: Full response from Rekognition detect_labels
        
    Returns:
        tuple: (labeled_image_bytes, error_message)
    """
    try:
        # Download original image from S3
        image_bytes, _, error_msg = download_from_s3(filename)
        if error_msg:
            return None, error_msg
        
        # Open image with PIL
        img = Image.open(BytesIO(image_bytes))
        draw = ImageDraw.Draw(img)
        
        # Get image dimensions
        img_width, img_height = img.size
        
        # Try to load a font, fall back to default if not available
        try:
            # Try to use a TrueType font
            font_size = max(14, int(img_height * 0.025))  # Slightly larger font
            font = ImageFont.truetype("arial.ttf", font_size)
            small_font_size = max(10, int(img_height * 0.018))
            small_font = ImageFont.truetype("arial.ttf", small_font_size)
        except:
            # Fall back to default font
            font = ImageFont.load_default()
            small_font = font
        
        # Define colors for bounding boxes
        colors = ['#00FF00', '#FF0000', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', 
                  '#FFA500', '#FF1493', '#00CED1', '#32CD32']
        
        # Get all labels
        labels = rekognition_response.get('Labels', [])
        
        # Separate labels with instances (bounding boxes) and without
        labels_with_boxes = []
        labels_without_boxes = []
        
        for label in labels:
            label_name = label['Name']
            confidence = label['Confidence']
            instances = label.get('Instances', [])
            
            if instances:
                for idx, instance in enumerate(instances):
                    bbox = instance.get('BoundingBox', {})
                    if bbox:
                        labels_with_boxes.append({
                            'name': label_name,
                            'confidence': confidence,
                            'bbox': bbox,
                            'instance_num': idx + 1 if len(instances) > 1 else None
                        })
            else:
                labels_without_boxes.append({
                    'name': label_name,
                    'confidence': confidence
                })
        
        # Track occupied label positions to prevent overlap
        occupied_regions = []
        
        def check_overlap(x, y, width, height, padding=5):
            """Check if a region overlaps with any occupied region"""
            for ox, oy, ow, oh in occupied_regions:
                if not (x + width + padding < ox or x > ox + ow + padding or
                        y + height + padding < oy or y > oy + oh + padding):
                    return True
            return False
        
        def find_non_overlapping_position(initial_x, initial_y, text_width, text_height):
            """Find a position that doesn't overlap with existing labels"""
            x, y = initial_x, initial_y
            max_attempts = 50  # Increased from 20 for better overlap prevention
            step_size = max(5, int(text_height * 0.3))  # Smaller steps for finer positioning
            
            for attempt in range(max_attempts):
                if not check_overlap(x, y, text_width, text_height):
                    occupied_regions.append((x, y, text_width, text_height))
                    return x, y
                
                # Try multiple strategies for better positioning
                if attempt < 15:
                    # Strategy 1: Move down in small steps
                    y += step_size
                elif attempt < 30:
                    # Strategy 2: Move right and try different y positions
                    if attempt == 15:
                        x = initial_x + text_width + 10
                        y = initial_y
                    y += step_size
                else:
                    # Strategy 3: Try diagonal movements
                    offset = (attempt - 30) * 10
                    x = initial_x + offset
                    y = initial_y + offset
                
                # Keep within image bounds
                if y > img_height - text_height - 10:
                    y = 10
                    x += text_width + 10
                
                if x > img_width - text_width - 10:
                    x = 10
            
            # Fallback to original position
            occupied_regions.append((initial_x, initial_y, text_width, text_height))
            return initial_x, initial_y
        
        # Draw bounding boxes and labels for instances
        color_index = 0
        for item in labels_with_boxes:
            label_name = item['name']
            confidence = item['confidence']
            bbox = item['bbox']
            instance_num = item['instance_num']
            
            # Calculate pixel coordinates
            left = int(bbox['Left'] * img_width)
            top = int(bbox['Top'] * img_height)
            width = int(bbox['Width'] * img_width)
            height = int(bbox['Height'] * img_height)
            
            # Select color
            color = colors[color_index % len(colors)]
            color_index += 1
            
            # Draw rectangle
            line_width = max(3, int(img_height * 0.004))
            draw.rectangle(
                [(left, top), (left + width, top + height)],
                outline=color,
                width=line_width
            )
            
            # Create label text
            if instance_num:
                label_text = f"{label_name} #{instance_num} ({confidence:.1f}%)"
            else:
                label_text = f"{label_name} ({confidence:.1f}%)"
            
            # Calculate text size
            try:
                text_bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = text_bbox[2] - text_bbox[0] + 8
                text_height = text_bbox[3] - text_bbox[1] + 8
            except:
                text_width, text_height = draw.textsize(label_text, font=font)
                text_width += 8
                text_height += 8
            
            # Initial position (above the box)
            initial_x = max(5, left)
            initial_y = max(5, top - text_height - 5)
            
            # Find non-overlapping position
            text_x, text_y = find_non_overlapping_position(initial_x, initial_y, text_width, text_height)
            
            # Draw text background
            draw.rectangle(
                [(text_x, text_y), (text_x + text_width, text_y + text_height)],
                fill=color
            )
            
            # Draw text
            draw.text((text_x + 4, text_y + 4), label_text, fill='white', font=font)
        
        # Save to bytes
        output = BytesIO()
        img_format = img.format if img.format else 'JPEG'
        img.save(output, format=img_format)
        output.seek(0)
        
        return output.getvalue(), None
        
    except Exception as e:
        return None, f"Error generating labeled image: {str(e)}"


# ROUTES

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
    labels, rekognition_response, error_msg = detect_labels_rekognition(filename)
    
    if labels is None:
        flash(f'Detection Error: {error_msg}', 'error')
        return redirect(url_for('index'))
    
    # Step 5: Store results in session and redirect to results page
    session['filename'] = filename
    session['labels'] = labels
    session['bucket'] = S3_BUCKET_NAME
    session['rekognition_response'] = rekognition_response  # Store full response for labeled image
    
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


@app.route('/download/original/<filename>')
def download_original(filename):
    """
    Download the original uploaded image from S3.
    """
    # Validate that this filename is in the current session
    session_filename = session.get('filename')
    if not session_filename or session_filename != filename:
        flash('Invalid download request.', 'error')
        return redirect(url_for('index'))
    
    # Download from S3
    image_bytes, content_type, error_msg = download_from_s3(filename)
    
    if error_msg:
        flash(f'Download Error: {error_msg}', 'error')
        return redirect(url_for('results'))
    
    # Send file to user
    return send_file(
        BytesIO(image_bytes),
        mimetype=content_type,
        as_attachment=True,
        download_name=filename
    )


@app.route('/download/labeled/<filename>')
def download_labeled(filename):
    """
    Download the labeled image with bounding boxes and labels.
    """
    # Validate that this filename is in the current session
    session_filename = session.get('filename')
    rekognition_response = session.get('rekognition_response')
    
    if not session_filename or session_filename != filename or not rekognition_response:
        flash('Invalid download request.', 'error')
        return redirect(url_for('index'))
    
    # Generate labeled image
    labeled_image_bytes, error_msg = generate_labeled_image(filename, rekognition_response)
    
    if error_msg:
        flash(f'Error generating labeled image: {error_msg}', 'error')
        return redirect(url_for('results'))
    
    # Create download filename
    base_name = filename.rsplit('.', 1)[0]
    extension = filename.rsplit('.', 1)[1] if '.' in filename else 'jpg'
    download_filename = f"{base_name}_labeled.{extension}"
    
    # Send file to user
    return send_file(
        BytesIO(labeled_image_bytes),
        mimetype='image/jpeg',
        as_attachment=True,
        download_name=download_filename
    )


# ERROR HANDLERS

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

# MAIN

if __name__ == '__main__':
    # Run Flask development server
    # WARNING: Do not use in production. Use a production WSGI server like Gunicorn
    app.run(debug=True, host='0.0.0.0', port=5000)
