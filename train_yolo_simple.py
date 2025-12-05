"""
Simple YOLO Training Script
Quick start training with minimal configuration
"""

from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolo11n.pt')

# Train the model
results = model.train(
    data='data/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='park_surveillance'
)

# Validate
metrics = model.val()

# Print results
print(f"\nmAP50-95: {metrics.box.map:.3f}")
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"\nBest model: runs/detect/park_surveillance/weights/best.pt")
