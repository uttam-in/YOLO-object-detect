"""
Batch Inference Script
Process multiple images/videos and save results to CSV
"""

from ultralytics import YOLO
import csv
import os
from pathlib import Path

def batch_inference(model_path, source_dir, output_csv='results.csv', conf=0.25):
    """Run batch inference and save results to CSV"""
    
    # Load model
    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")
    print(f"Classes: {model.names}")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(source_dir).glob(f'*{ext}'))
        image_files.extend(Path(source_dir).glob(f'*{ext.upper()}'))
    
    print(f"\nFound {len(image_files)} images in {source_dir}")
    
    # Prepare CSV
    csv_data = []
    csv_data.append(['Image', 'Object_Count', 'Detections'])
    
    # Run inference
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        
        results = model.predict(
            source=str(img_path),
            conf=conf,
            save=True,
            verbose=False
        )
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[cls]
                detections.append(f"{class_name}({confidence:.2f})")
        
        # Add to CSV
        csv_data.append([
            img_path.name,
            len(detections),
            '; '.join(detections) if detections else 'No detections'
        ])
    
    # Save CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    
    print(f"\nResults saved to: {output_csv}")
    print(f"Annotated images saved in: runs/predict/")
    
    # Print summary
    total_detections = sum(int(row[1]) for row in csv_data[1:])
    print(f"\nSummary:")
    print(f"  Images processed: {len(image_files)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average per image: {total_detections/len(image_files):.1f}")

if __name__ == '__main__':
    # Configuration
    MODEL_PATH = 'yolo11n.pt'  # Change after training
    SOURCE_DIR = 'data/test/images'
    OUTPUT_CSV = 'inference_results.csv'
    CONFIDENCE = 0.25
    
    # Run batch inference
    batch_inference(MODEL_PATH, SOURCE_DIR, OUTPUT_CSV, CONFIDENCE)
