import boto3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO
import argparse
import os
import sys

def upload_image_to_s3(local_path, bucket, s3_key=None):
    """
    Upload a local image file to S3 bucket.
    
    Args:
        local_path: Path to the local image file
        bucket: S3 bucket name
        s3_key: Optional S3 key (filename in bucket). If not provided, uses the local filename.
    
    Returns:
        The S3 key of the uploaded file
    """
    try:
        # If no S3 key provided, use the basename of the local file
        if s3_key is None:
            s3_key = os.path.basename(local_path)
        
        # Check if file exists
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Image file not found: {local_path}")
        
        # Upload to S3
        s3_client = boto3.client('s3')
        print(f"Uploading {local_path} to s3://{bucket}/{s3_key}...")
        
        s3_client.upload_file(local_path, bucket, s3_key)
        print(f"Successfully uploaded to S3")
        
        return s3_key
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        sys.exit(1)

def detect_labels(photo, bucket):
    """
    Detect labels in an image stored in S3 using AWS Rekognition.
    
    Args:
        photo: S3 key of the image
        bucket: S3 bucket name
    
    Returns:
        Number of labels detected
    """
    try:
        client = boto3.client('rekognition')

        response = client.detect_labels(
            Image={'S3Object': {'Bucket': bucket, 'Name': photo}},
            MaxLabels=50)

        print(f'\nDetected labels for {photo}') 
        print()   

        # Print label information
        for label in response['Labels']:
            print(f"Label: {label['Name']}")
            print(f"Confidence: {label['Confidence']:.2f}%")
            print()

        # Load the image from S3
        s3 = boto3.resource('s3')
        obj = s3.Object(bucket, photo)
        img_data = obj.get()['Body'].read()
        img = Image.open(BytesIO(img_data))

        # Display the image
        plt.figure(figsize=(16, 12))
        plt.imshow(img)
        ax = plt.gca()

        # Define colors for different instances
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'orange', 'lime', 'purple']
        
        # Track instance counts and label positions to avoid overlap
        instance_counts = {}
        label_positions = []  # Store (left, top, width, height) of labels
        
        # Collect all instances with bounding boxes first
        all_instances = []
        for label in response['Labels']:
            label_name = label['Name']
            instances = label.get('Instances', [])
            if instances:
                for instance in instances:
                    all_instances.append((label_name, instance))
        
        # Sort by top position to process from top to bottom
        all_instances.sort(key=lambda x: x[1]['BoundingBox']['Top'])
        
        # Plot bounding boxes with smart label positioning
        for label_name, instance in all_instances:
            # Initialize counter for this label type if not exists
            if label_name not in instance_counts:
                instance_counts[label_name] = 0
            
            bbox = instance['BoundingBox']
            left = bbox['Left'] * img.width
            top = bbox['Top'] * img.height
            width = bbox['Width'] * img.width
            height = bbox['Height'] * img.height
            
            # Use different color for each instance
            color = colors[instance_counts[label_name] % len(colors)]
            instance_counts[label_name] += 1
            
            # Draw rectangle
            rect = patches.Rectangle((left, top), width, height, 
                                    linewidth=3, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Create label text
            total_instances = sum(1 for ln, _ in all_instances if ln == label_name)
            if total_instances > 1:
                label_text = f"{label_name} #{instance_counts[label_name]}"
            else:
                label_text = f"{label_name}"
            
            # Smart label positioning to avoid overlap
            label_top = top - 15
            label_left = left
            
            # Check for overlap with existing labels and adjust position
            overlap_found = True
            offset = 0
            while overlap_found and offset < 200:
                overlap_found = False
                for prev_left, prev_top, prev_width, prev_height in label_positions:
                    # Check if labels would overlap
                    if (abs(label_left - prev_left) < 150 and 
                        abs(label_top - prev_top) < 25):
                        overlap_found = True
                        label_top -= 30  # Move label up
                        offset += 30
                        break
            
            # Store this label's position
            label_positions.append((label_left, label_top, 150, 25))
            
            # Draw the label
            plt.text(label_left, label_top, label_text, color='white', fontsize=10, 
                    fontweight='bold',
                    bbox=dict(facecolor=color, alpha=0.85, edgecolor='white', 
                            boxstyle='round,pad=0.4', linewidth=2))

        plt.axis('off')
        plt.title(f'Image Recognition Results: {photo}', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save the labeled image to local directory
        output_filename = f"labeled_{photo.replace('/', '_')}"
        output_path = os.path.join(os.getcwd(), output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n>>> Labeled image saved to: {output_path}")
        
        plt.show()

        return len(response['Labels'])
    
    except client.exceptions.InvalidS3ObjectException:
        print(f"Error: Image '{photo}' not found in bucket '{bucket}'")
        sys.exit(1)
    except client.exceptions.InvalidParameterException as e:
        print(f"Error: Invalid parameter - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during label detection: {e}")
        sys.exit(1)

def main():
    """
    Main function to handle command-line arguments and orchestrate the workflow.
    """
    parser = argparse.ArgumentParser(
        description='Upload an image to S3 and detect labels using AWS Rekognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aws-image-labler.py --image "family image.jpg" --bucket my-bucket
  python aws-image-labler.py -i photo.png -b my-rekognition-bucket
        """
    )
    
    parser.add_argument('-i', '--image', 
                        required=True,
                        help='Path to the local image file to upload and analyze')
    
    parser.add_argument('-b', '--bucket',
                        required=True,
                        help='S3 bucket name where the image will be stored')
    
    parser.add_argument('-k', '--key',
                        help='Optional S3 key (filename in bucket). If not provided, uses the local filename')
    
    args = parser.parse_args()
    
    # Upload image to S3
    s3_key = upload_image_to_s3(args.image, args.bucket, args.key)
    
    # Detect labels
    label_count = detect_labels(s3_key, args.bucket)
    print(f"\n{'='*50}")
    print(f"Total labels detected: {label_count}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
