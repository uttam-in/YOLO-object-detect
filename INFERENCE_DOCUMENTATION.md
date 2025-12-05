# YOLO Inference Documentation

Complete guide for running inference using the trained YOLO model for park surveillance (bicycle, car, human detection).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Model Setup](#model-setup)
4. [Inference Scripts Overview](#inference-scripts-overview)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Configuration Parameters](#configuration-parameters)
8. [Output Formats](#output-formats)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run simple inference on an image
python inference_simple.py

# Run full inference with all features
python inference_yolo.py

# Process batch of images with CSV output
python inference_batch.py

# Real-time video inference with FPS counter
python inference_realtime.py
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install ultralytics torch torchvision opencv-python numpy matplotlib pillow pyyaml
```

### Verify Installation

```bash
python -c "from ultralytics import YOLO; print('Installation successful!')"
```

---

## Model Setup

### Using Pretrained Model
```python
model = YOLO('yolo11n.pt')  # Downloads automatically if not present
```

### Using Your Trained Model
After training, use your custom model:
```python
model = YOLO('runs/train/yolo_park_surveillance/weights/best.pt')
```

### Model Variants
- `yolo11n.pt` - Nano (fastest, least accurate)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large
- `yolo11x.pt` - Extra Large (slowest, most accurate)

---

## Inference Scripts Overview

### 1. inference_simple.py
**Purpose:** Quick and easy inference for beginners

**Features:**
- Minimal code
- Single image inference
- Automatic result saving

**Use Case:** Testing model quickly on one image

---

### 2. inference_yolo.py
**Purpose:** Comprehensive inference with full control

**Features:**
- Single image prediction
- Video file processing
- Webcam inference
- Batch folder processing
- Object tracking
- Detailed results extraction
- Class-based API

**Use Case:** Production applications, custom workflows

---

### 3. inference_batch.py
**Purpose:** Process multiple images and export results

**Features:**
- Batch processing
- CSV export with detection counts
- Summary statistics
- Automated file discovery

**Use Case:** Dataset evaluation, bulk processing

---

### 4. inference_realtime.py
**Purpose:** Real-time inference with performance monitoring

**Features:**
- Live FPS counter
- Detection statistics overlay
- Screenshot capture (press 's')
- Webcam/video/RTSP support
- Performance metrics

**Use Case:** Live monitoring, video analysis, demos

---

## Usage Examples

### Example 1: Simple Image Inference

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolo11n.pt')

# Run inference
results = model.predict('test_image.jpg', conf=0.25, save=True)

# Print detections
for result in results:
    for box in result.boxes:
        print(f"{model.names[int(box.cls)]}: {float(box.conf):.2f}")
```

**Output:**
```
car: 0.87
human: 0.92
bicycle: 0.78
Results saved in: runs/predict/
```

---

### Example 2: Using the YOLOInference Class

```python
from inference_yolo import YOLOInference

# Initialize
inference = YOLOInference('yolo11n.pt')

# Predict on image
inference.predict_image('test_image.jpg', conf=0.25)

# Predict on video
inference.predict_video('video.mp4', conf=0.25)

# Predict on folder
inference.predict_folder('data/test/images', conf=0.25)

# Get detailed results
detections = inference.get_detailed_results('test_image.jpg')
for det in detections:
    print(f"{det['class_name']}: {det['confidence']:.2f}")
    print(f"BBox: {det['bbox']}")
```

---

### Example 3: Batch Processing with CSV Export

```python
from inference_batch import batch_inference

# Process all images in folder
batch_inference(
    model_path='yolo11n.pt',
    source_dir='data/test/images',
    output_csv='results.csv',
    conf=0.25
)
```

**CSV Output:**
```csv
Image,Object_Count,Detections
image1.jpg,3,car(0.87); human(0.92); bicycle(0.78)
image2.jpg,1,car(0.91)
image3.jpg,0,No detections
```

---

### Example 4: Real-time Video Inference

```python
from inference_realtime import realtime_inference

# Webcam inference
realtime_inference(
    model_path='yolo11n.pt',
    source=0,  # 0 for default webcam
    conf=0.25
)

# Video file inference
realtime_inference(
    model_path='yolo11n.pt',
    source='pranitha_data1.mp4',
    conf=0.25
)

# RTSP stream inference
realtime_inference(
    model_path='yolo11n.pt',
    source='rtsp://192.168.1.100:554/stream',
    conf=0.25
)
```

**Controls:**
- Press `q` to quit
- Press `s` to save screenshot

---

### Example 5: Object Tracking in Videos

```python
from inference_yolo import YOLOInference

inference = YOLOInference('yolo11n.pt')

# Track objects across frames
inference.predict_with_tracking(
    video_path='video.mp4',
    conf=0.25
)
```

**Output:**
```
Total unique objects tracked: 15
Results saved in: runs/inference/tracking/
```

---

### Example 6: Webcam Inference

```python
from inference_yolo import YOLOInference

inference = YOLOInference('yolo11n.pt')

# Start webcam inference
inference.predict_webcam(conf=0.25, camera_id=0)
```

---

### Example 7: Custom Processing

```python
from ultralytics import YOLO
import cv2

model = YOLO('yolo11n.pt')

# Load image
image = cv2.imread('test_image.jpg')

# Run inference
results = model.predict(image, conf=0.25, verbose=False)

# Custom processing
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # Get class and confidence
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls]
        
        # Custom drawing or processing
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name} {conf:.2f}", 
                   (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save result
cv2.imwrite('custom_output.jpg', image)
```

---

## API Reference

### YOLOInference Class

#### `__init__(model_path)`
Initialize the inference object.

**Parameters:**
- `model_path` (str): Path to YOLO model file

**Example:**
```python
inference = YOLOInference('yolo11n.pt')
```

---

#### `predict_image(image_path, conf=0.25, save=True, output_dir='runs/inference')`
Run inference on a single image.

**Parameters:**
- `image_path` (str): Path to image file
- `conf` (float): Confidence threshold (0.0-1.0)
- `save` (bool): Save annotated image
- `output_dir` (str): Output directory

**Returns:**
- `results`: YOLO results object

**Example:**
```python
results = inference.predict_image('test.jpg', conf=0.3)
```

---

#### `predict_video(video_path, conf=0.25, save=True, output_dir='runs/inference')`
Run inference on a video file.

**Parameters:**
- `video_path` (str): Path to video file
- `conf` (float): Confidence threshold
- `save` (bool): Save annotated video
- `output_dir` (str): Output directory

**Returns:**
- `results`: Generator of YOLO results

**Example:**
```python
results = inference.predict_video('video.mp4', conf=0.25)
```

---

#### `predict_webcam(conf=0.25, camera_id=0)`
Run inference on webcam feed.

**Parameters:**
- `conf` (float): Confidence threshold
- `camera_id` (int): Camera device ID (0 for default)

**Example:**
```python
inference.predict_webcam(conf=0.25, camera_id=0)
```

---

#### `predict_folder(folder_path, conf=0.25, save=True, output_dir='runs/inference')`
Run inference on all images in a folder.

**Parameters:**
- `folder_path` (str): Path to folder containing images
- `conf` (float): Confidence threshold
- `save` (bool): Save annotated images
- `output_dir` (str): Output directory

**Returns:**
- `results`: List of YOLO results

**Example:**
```python
results = inference.predict_folder('data/test/images')
```

---

#### `predict_with_tracking(video_path, conf=0.25, save=True, output_dir='runs/inference')`
Run inference with object tracking.

**Parameters:**
- `video_path` (str): Path to video file
- `conf` (float): Confidence threshold
- `save` (bool): Save annotated video
- `output_dir` (str): Output directory

**Returns:**
- `results`: Generator of YOLO results with tracking IDs

**Example:**
```python
results = inference.predict_with_tracking('video.mp4')
```

---

#### `get_detailed_results(image_path, conf=0.25)`
Get detailed detection results as dictionary.

**Parameters:**
- `image_path` (str): Path to image file
- `conf` (float): Confidence threshold

**Returns:**
- `detections` (list): List of detection dictionaries

**Detection Dictionary Format:**
```python
{
    'class_id': 0,
    'class_name': 'bicycle',
    'confidence': 0.87,
    'bbox': [x1, y1, x2, y2],  # Pixel coordinates
    'bbox_normalized': [x_center, y_center, width, height]  # Normalized 0-1
}
```

**Example:**
```python
detections = inference.get_detailed_results('test.jpg')
for det in detections:
    print(f"{det['class_name']}: {det['confidence']:.2f}")
```

---

## Configuration Parameters

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conf` | float | 0.25 | Confidence threshold (0.0-1.0) |
| `iou` | float | 0.7 | IoU threshold for NMS |
| `imgsz` | int | 640 | Input image size |
| `max_det` | int | 300 | Maximum detections per image |
| `save` | bool | True | Save annotated results |
| `show` | bool | False | Display results in window |
| `show_labels` | bool | True | Show class labels |
| `show_conf` | bool | True | Show confidence scores |
| `line_width` | int | 2 | Bounding box line width |
| `device` | str | 'auto' | Device ('cpu', 'cuda', '0', '1', etc.) |

---

### Confidence Threshold Guidelines

| Threshold | Use Case |
|-----------|----------|
| 0.1-0.2 | High recall, many false positives |
| 0.25 | Balanced (default) |
| 0.3-0.4 | Fewer false positives |
| 0.5+ | High precision, may miss objects |

---

### Video Sources

| Source Type | Example | Description |
|-------------|---------|-------------|
| Webcam | `0` or `1` | Device ID |
| Video File | `'video.mp4'` | Local video file |
| Image | `'image.jpg'` | Single image |
| Folder | `'images/'` | Folder of images |
| URL | `'http://...'` | Video URL |
| RTSP | `'rtsp://...'` | IP camera stream |
| YouTube | `'https://youtube.com/...'` | YouTube video |

---

## Output Formats

### 1. Annotated Images/Videos
Saved in `runs/predict/` or `runs/inference/` with bounding boxes and labels.

### 2. Detection Results Object

```python
results = model.predict('image.jpg')

# Access detections
boxes = results[0].boxes

# Iterate through detections
for box in boxes:
    cls = int(box.cls[0])           # Class ID
    conf = float(box.conf[0])       # Confidence
    xyxy = box.xyxy[0].tolist()     # [x1, y1, x2, y2]
    xywh = box.xywh[0].tolist()     # [x_center, y_center, width, height]
```

### 3. CSV Export (from inference_batch.py)

```csv
Image,Object_Count,Detections
image1.jpg,3,car(0.87); human(0.92); bicycle(0.78)
image2.jpg,1,car(0.91)
```

### 4. JSON Format (custom)

```python
import json

detections = inference.get_detailed_results('image.jpg')
with open('results.json', 'w') as f:
    json.dump(detections, f, indent=2)
```

**Output:**
```json
[
  {
    "class_id": 1,
    "class_name": "car",
    "confidence": 0.87,
    "bbox": [100, 200, 300, 400],
    "bbox_normalized": [0.5, 0.6, 0.2, 0.3]
  }
]
```

---

## Troubleshooting

### Issue: Model not found

**Error:**
```
FileNotFoundError: Model not found at yolo11n.pt
```

**Solution:**
```python
# Download model automatically
from ultralytics import YOLO
model = YOLO('yolo11n.pt')  # Auto-downloads if missing
```

---

### Issue: CUDA out of memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size
2. Reduce image size
3. Use CPU instead

```python
# Use CPU
model.predict('image.jpg', device='cpu')

# Reduce image size
model.predict('image.jpg', imgsz=320)
```

---

### Issue: Low FPS on video

**Solutions:**
1. Use smaller model (yolo11n.pt)
2. Reduce image size
3. Use GPU
4. Skip frames

```python
# Skip frames for faster processing
cap = cv2.VideoCapture('video.mp4')
frame_skip = 2
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue
    
    results = model.predict(frame)
```

---

### Issue: No detections

**Solutions:**
1. Lower confidence threshold
2. Check if model is trained on correct classes
3. Verify image quality

```python
# Lower confidence
results = model.predict('image.jpg', conf=0.1)

# Check model classes
print(model.names)  # Should show: {0: 'bicycle', 1: 'car', 2: 'human'}
```

---

### Issue: Webcam not opening

**Error:**
```
Error: Cannot open video source 0
```

**Solutions:**
```python
# Try different camera IDs
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at ID: {i}")
        cap.release()
        break

# Or specify camera explicitly
inference.predict_webcam(camera_id=1)
```

---

### Issue: Slow inference on CPU

**Solutions:**
1. Use GPU if available
2. Use smaller model
3. Reduce image size
4. Use ONNX export for faster CPU inference

```python
# Export to ONNX for faster CPU inference
model.export(format='onnx')

# Load ONNX model
model = YOLO('yolo11n.onnx')
```

---

## Performance Tips

### 1. GPU Acceleration
```python
# Ensure CUDA is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Use GPU explicitly
model.predict('image.jpg', device='0')  # Use GPU 0
```

### 2. Batch Processing
```python
# Process multiple images at once
image_list = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = model.predict(image_list, batch=8)
```

### 3. Half Precision (FP16)
```python
# Use half precision for faster inference
model.predict('image.jpg', half=True)
```

### 4. Image Size Optimization
```python
# Smaller size = faster inference
model.predict('image.jpg', imgsz=320)  # Fast
model.predict('image.jpg', imgsz=640)  # Balanced (default)
model.predict('image.jpg', imgsz=1280) # Accurate but slow
```

---

## Advanced Usage

### Custom Visualization

```python
from ultralytics import YOLO
import cv2

model = YOLO('yolo11n.pt')
results = model.predict('image.jpg')

# Get annotated image
annotated = results[0].plot(
    conf=True,           # Show confidence
    labels=True,         # Show labels
    boxes=True,          # Show boxes
    line_width=3,        # Box thickness
    font_size=12,        # Label font size
    pil=False            # Return as numpy array
)

cv2.imwrite('custom_output.jpg', annotated)
```

### Filter by Class

```python
results = model.predict('image.jpg')

# Filter only cars
for box in results[0].boxes:
    if int(box.cls[0]) == 1:  # 1 = car
        print(f"Car detected with confidence: {float(box.conf[0]):.2f}")
```

### Count Objects

```python
from collections import Counter

results = model.predict('image.jpg')
class_counts = Counter()

for box in results[0].boxes:
    cls = int(box.cls[0])
    class_name = model.names[cls]
    class_counts[class_name] += 1

print(f"Detected: {dict(class_counts)}")
# Output: {'car': 3, 'human': 2, 'bicycle': 1}
```

---

## Support

For issues or questions:
1. Check [Ultralytics Documentation](https://docs.ultralytics.com)
2. Review this documentation
3. Check model training logs
4. Verify data.yaml configuration

---

## Classes Detected

This model detects 3 classes:
- **bicycle** (class_id: 0)
- **car** (class_id: 1)
- **human** (class_id: 2)

---

## File Structure

```
project/
├── inference_yolo.py          # Full-featured inference class
├── inference_simple.py        # Quick start inference
├── inference_batch.py         # Batch processing with CSV
├── inference_realtime.py      # Real-time with FPS counter
├── train_yolo.py             # Training script
├── requirements.txt          # Dependencies
├── data/
│   ├── data.yaml            # Dataset configuration
│   ├── train/images/        # Training images
│   ├── test/images/         # Test images
│   └── valid/images/        # Validation images
└── runs/
    ├── train/               # Training outputs
    ├── predict/             # Inference outputs
    └── inference/           # Custom inference outputs
```

---

**Last Updated:** December 2025  
**YOLO Version:** YOLOv11 (Ultralytics)  
**Model Classes:** bicycle, car, human
