import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import argparse
import json
import sys
import os

def visualize_labels_from_json(image_path, json_path, output_path=None):
    """
    Visualize image recognition results from a JSON file.
    
    Args:
        image_path: Path to the image file
        json_path: Path to the JSON file containing Rekognition results
        output_path: Optional path to save the output image
    """
    try:
        # Check if files exist
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        # Load the JSON data
        with open(json_path, 'r') as f:
            response = json.load(f)
        
        # Check if Labels key exists
        if 'Labels' not in response:
            raise ValueError("JSON file must contain 'Labels' key with Rekognition results")
        
        # Load the image
        img = Image.open(image_path)
        
        # Print label information
        print(f'\nDetected labels for {os.path.basename(image_path)}')
        print()
        
        for label in response['Labels']:
            print(f"Label: {label['Name']}")
            print(f"Confidence: {label['Confidence']:.2f}%")
            print()
        
        # Display the image
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        ax = plt.gca()
        
        # Plot bounding boxes
        for label in response['Labels']:
            for instance in label.get('Instances', []):
                bbox = instance['BoundingBox']
                left = bbox['Left'] * img.width
                top = bbox['Top'] * img.height
                width = bbox['Width'] * img.width
                height = bbox['Height'] * img.height
                
                rect = patches.Rectangle((left, top), width, height, 
                                        linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                
                label_text = label['Name'] + ' (' + str(round(label['Confidence'], 2)) + '%)'
                plt.text(left, top - 5, label_text, color='red', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', 
                                boxstyle='round,pad=0.3'))
        
        plt.axis('off')
        plt.title(f'Image Recognition Results: {os.path.basename(image_path)}', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved visualization to: {output_path}")
        
        plt.show()
        
        print(f"\n{'='*50}")
        print(f"Total labels detected: {len(response['Labels'])}")
        print(f"{'='*50}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    """
    Main function to handle command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Visualize image recognition results from a JSON file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize-from-json.py -i "family image.jpg" -j results.json
  python visualize-from-json.py -i photo.png -j rekognition-output.json -o labeled-output.png
        """
    )
    
    parser.add_argument('-i', '--image',
                       required=True,
                       help='Path to the image file')
    
    parser.add_argument('-j', '--json',
                       required=True,
                       help='Path to the JSON file containing Rekognition results')
    
    parser.add_argument('-o', '--output',
                       help='Optional path to save the labeled image')
    
    args = parser.parse_args()
    
    visualize_labels_from_json(args.image, args.json, args.output)

if __name__ == "__main__":
    main()
