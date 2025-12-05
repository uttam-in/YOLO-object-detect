"""
Simple YOLO Inference Script
Quick inference on images or videos
"""

from ultralytics import YOLO

# Load model (use your trained model or pretrained)
model = YOLO('yolo11n.pt')  # or 'runs/train/yolo_park_surveillance/weights/best.pt'

# Inference on image
results = model.predict(
    source='test_image.jpg',
    conf=0.25,
    save=True
)

# Print detections
for result in results:
    boxes = result.boxes
    print(f"Found {len(boxes)} objects:")
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"  - {model.names[cls]}: {conf:.2f}")

print("\nResults saved in: runs/predict/")
