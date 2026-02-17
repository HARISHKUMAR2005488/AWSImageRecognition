# AWS Image Recognition with Amazon Rekognition

A Python-based image labeling and object detection tool that leverages AWS Rekognition to automatically identify and label objects, scenes, and activities in images. The project includes intelligent bounding box visualization with smart label positioning to avoid overlaps.

## 🌟 Features

- **Automatic Image Upload**: Upload images directly to Amazon S3 from your local machine
- **AI-Powered Object Detection**: Detect up to 50 labels per image using AWS Rekognition
- **Smart Visualization**: 
  - Color-coded bounding boxes for detected objects
  - Intelligent label positioning to prevent overlap
  - Instance counting for multiple objects of the same type
  - High-resolution output (300 DPI)
- **Dual Workflow Support**:
  - **Live Detection**: Upload and analyze images in real-time
  - **Offline Visualization**: Visualize pre-saved Rekognition JSON results

## 📋 Prerequisites

- Python 3.7 or higher
- AWS Account with:
  - Amazon S3 access
  - Amazon Rekognition access
  - Configured AWS CLI credentials
- Required Python packages (see `requirements.txt`)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/HARISHKUMAR2005488/AWSImageRecognition.git
   cd AWSImageRecognition
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS credentials**
   ```bash
   aws configure
   ```
   Enter your:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region (e.g., `us-east-1`)
   - Default output format (e.g., `json`)

## 📦 Dependencies

- `boto3` - AWS SDK for Python
- `Pillow` - Image processing library
- `matplotlib` - Visualization and plotting

## 🎯 Usage

### Method 1: Live Image Analysis (Recommended)

Upload an image to S3 and analyze it in real-time:

```bash
python aws-image-labler.py --image "path/to/image.jpg" --bucket your-s3-bucket-name
```

**Example:**
```bash
python aws-image-labler.py --image "family image.jpg" --bucket my-rekognition-bucket
```

**Optional Parameters:**
- `-k, --key`: Specify a custom S3 key (filename in bucket)
  ```bash
  python aws-image-labler.py -i "photo.png" -b my-bucket -k "custom-name.png"
  ```

### Method 2: Visualize from JSON

If you already have Rekognition results saved as JSON:

```bash
python visualize-from-json.py --image "path/to/image.jpg" --json "results.json"
```

**Save the output:**
```bash
python visualize-from-json.py -i "family image.jpg" -j sample-rekognition-output.json -o "output.png"
```

## 🏗️ Project Structure

```
AWS-Image_recognition/
├── aws-image-labler.py              # Main script for live image analysis
├── visualize-from-json.py           # Offline visualization from JSON
├── requirements.txt                 # Python dependencies
├── sample-rekognition-output.json   # Example Rekognition API response
├── .gitignore                       # Git ignore rules
└── README.md                        # Project documentation
```

## 🔧 AWS Services Used

| Service | Purpose |
|---------|---------|
| **Amazon S3** | Store images for processing |
| **Amazon Rekognition** | AI-powered image analysis and label detection |
| **AWS CLI** | Configure credentials and interact with AWS services |

## 📊 Output

The scripts generate:
1. **Console Output**: List of detected labels with confidence scores
2. **Labeled Image**: High-resolution image with:
   - Color-coded bounding boxes around detected objects
   - Smart-positioned labels (prevents overlap)
   - Instance numbering for multiple objects of the same type
   - Saved as `labeled_<original-filename>` in the current directory

**Example Output:**
```
Detected labels for family image.jpg

Label: Person
Confidence: 99.87%

Label: Clothing
Confidence: 98.45%

...

>>> Labeled image saved to: D:\AWS-Image_recognition\labeled_family image.jpg

==================================================
Total labels detected: 23
==================================================
```

## 🎨 Visualization Features

- **Multi-color bounding boxes**: 8 distinct colors for easy differentiation
- **Instance tracking**: Automatically numbers multiple instances (e.g., "Person #1", "Person #2")
- **Smart label positioning**: Prevents label overlap using intelligent positioning algorithm
- **High-quality output**: 300 DPI resolution for professional results

## ⚙️ Configuration

### S3 Bucket Setup

1. Create an S3 bucket in your AWS account
2. Ensure your IAM user/role has permissions:
   - `s3:PutObject` - Upload images
   - `s3:GetObject` - Retrieve images
   - `rekognition:DetectLabels` - Analyze images

### IAM Policy Example

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::your-bucket-name/*"
    },
    {
      "Effect": "Allow",
      "Action": "rekognition:DetectLabels",
      "Resource": "*"
    }
  ]
}
```

## 🐛 Troubleshooting

### Common Issues

**1. AWS Credentials Not Found**
```
Error: Unable to locate credentials
```
**Solution**: Run `aws configure` and enter your credentials

**2. S3 Bucket Access Denied**
```
Error: Access Denied
```
**Solution**: Verify your IAM permissions and bucket policy

**3. Image Not Found**
```
Error: Image file not found: <path>
```
**Solution**: Check the file path and ensure the image exists

**4. Invalid Image Format**
```
Error: Invalid parameter
```
**Solution**: Rekognition supports JPEG and PNG formats only

## 💰 Cost Considerations

- **AWS Free Tier**: 
  - Amazon Rekognition: 5,000 images/month for the first 12 months
  - Amazon S3: 5 GB storage, 20,000 GET requests, 2,000 PUT requests
- **After Free Tier**: 
  - Rekognition: ~$1 per 1,000 images
  - S3: Minimal storage costs for temporary image storage

## 🔒 Security Best Practices

- Never commit AWS credentials to version control
- Use IAM roles with least privilege principle
- Enable S3 bucket encryption
- Regularly rotate access keys
- Use AWS Secrets Manager for production deployments

## 📝 License

This project is open source and available for educational and commercial use.

## 👤 Author

**Harish Kumar**
- GitHub: [@HARISHKUMAR2005488](https://github.com/HARISHKUMAR2005488)
- Repository: [AWSImageRecognition](https://github.com/HARISHKUMAR2005488/AWSImageRecognition)

## 🙏 Acknowledgments

- AWS Rekognition documentation and examples
- Python boto3 library maintainers
- Open source community

## 📚 Additional Resources

- [AWS Rekognition Documentation](https://docs.aws.amazon.com/rekognition/)
- [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [AWS Free Tier Details](https://aws.amazon.com/free/)

---

**Note**: Make sure to delete images from S3 after processing to minimize storage costs, or set up S3 lifecycle policies for automatic cleanup.
