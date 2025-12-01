# YOLO Object Detection

This project demonstrates how to use YOLO (You Only Look Once) for object detection in images.

## Prerequisites

Install the required dependencies:

```bash
pip install ultralytics opencv-python
```

The YOLO model will be automatically downloaded on first run.

## Usage

### Basic Usage

Run the detection script on the test image:

```bash
python detect_objects.py
```

### Custom Image

To detect objects in your own image, modify the `image_path` variable in `detect_objects.py`:

```python
image_path = "path/to/your/image.jpg"
```

Or use it as a module:

```python
from detect_objects import detect_objects

# Detect objects in your image
results = detect_objects("your_image.jpg")
```

### Advanced Options

You can customize the detection parameters:

```python
results = detect_objects(
    image_path="your_image.jpg",
    model_path="yolo11n.pt",  # or yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    conf_threshold=0.25  # Adjust confidence threshold (0.0 to 1.0)
)
```

## Available YOLO Models

- `yolo11n.pt` - Nano (fastest, least accurate)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large
- `yolo11x.pt` - Extra Large (slowest, most accurate)

## Output

The script will:
1. Print detected objects with confidence scores and bounding boxes
2. Save an annotated image with bounding boxes and labels (e.g., `test_image_detected.jpg`)

## Example Output

```
Loading YOLO model: yolo11n.pt
Running object detection on: test_image.jpg

Detected 5 objects:
  1. person (confidence: 0.89)
     Bounding box: [120.5, 180.3, 245.8, 420.1]
  2. car (confidence: 0.85)
     Bounding box: [300.2, 250.6, 580.9, 450.3]
  3. dog (confidence: 0.78)
     Bounding box: [150.1, 350.2, 200.5, 410.8]

Annotated image saved to: test_image_detected.jpg

✓ Object detection completed successfully!
```

## Features

- ✅ Automatic model downloading
- ✅ Detailed detection output
- ✅ Annotated image generation
- ✅ Configurable confidence threshold
- ✅ Support for multiple YOLO models
- ✅ Error handling and validation

## Supported Object Classes

YOLO11 can detect 80 different object classes including:
- People, animals (dog, cat, bird, etc.)
- Vehicles (car, truck, bicycle, motorcycle, etc.)
- Everyday objects (chair, bottle, laptop, phone, etc.)
- And many more!
