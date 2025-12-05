"""
YOLO Inference Script
Run inference on images, videos, or webcam using trained YOLO model
"""

from ultralytics import YOLO
import cv2
import os
from pathlib import Path

class YOLOInference:
    def __init__(self, model_path='runs/train/yolo_park_surveillance/weights/best.pt'):
        """Initialize YOLO model for inference"""
        self.model = YOLO(model_path)
        print(f"Model loaded from: {model_path}")
        print(f"Classes: {self.model.names}")
    
    def predict_image(self, image_path, conf=0.25, save=True, output_dir='runs/inference'):
        """Run inference on a single image"""
        results = self.model.predict(
            source=image_path,
            conf=conf,
            save=save,
            project=output_dir,
            name='images',
            exist_ok=True,
            show_labels=True,
            show_conf=True,
            line_width=2
        )
        
        # Print detections
        for result in results:
            boxes = result.boxes
            print(f"\nDetections in {image_path}:")
            print(f"Found {len(boxes)} objects")
            
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls]
                print(f"  - {class_name}: {conf:.2f}")
        
        return results
    
    def predict_video(self, video_path, conf=0.25, save=True, output_dir='runs/inference'):
        """Run inference on a video file"""
        results = self.model.predict(
            source=video_path,
            conf=conf,
            save=save,
            project=output_dir,
            name='videos',
            exist_ok=True,
            show_labels=True,
            show_conf=True,
            line_width=2,
            stream=True  # Use streaming for videos
        )
        
        # Process results
        frame_count = 0
        for result in results:
            frame_count += 1
            boxes = result.boxes
            if len(boxes) > 0:
                print(f"Frame {frame_count}: {len(boxes)} objects detected")
        
        print(f"\nProcessed {frame_count} frames")
        return results
    
    def predict_webcam(self, conf=0.25, camera_id=0):
        """Run inference on webcam feed"""
        results = self.model.predict(
            source=camera_id,
            conf=conf,
            show=True,
            stream=True,
            show_labels=True,
            show_conf=True,
            line_width=2
        )
        
        print("Press 'q' to quit webcam inference")
        for result in results:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def predict_folder(self, folder_path, conf=0.25, save=True, output_dir='runs/inference'):
        """Run inference on all images in a folder"""
        results = self.model.predict(
            source=folder_path,
            conf=conf,
            save=save,
            project=output_dir,
            name='batch',
            exist_ok=True,
            show_labels=True,
            show_conf=True,
            line_width=2
        )
        
        total_detections = 0
        for result in results:
            total_detections += len(result.boxes)
        
        print(f"\nProcessed folder: {folder_path}")
        print(f"Total detections: {total_detections}")
        return results
    
    def predict_with_tracking(self, video_path, conf=0.25, save=True, output_dir='runs/inference'):
        """Run inference with object tracking"""
        results = self.model.track(
            source=video_path,
            conf=conf,
            save=save,
            project=output_dir,
            name='tracking',
            exist_ok=True,
            show_labels=True,
            show_conf=True,
            line_width=2,
            stream=True,
            tracker='bytetrack.yaml'  # or 'botsort.yaml'
        )
        
        tracked_ids = set()
        for result in results:
            if result.boxes.id is not None:
                tracked_ids.update(result.boxes.id.int().cpu().tolist())
        
        print(f"\nTotal unique objects tracked: {len(tracked_ids)}")
        return results
    
    def get_detailed_results(self, image_path, conf=0.25):
        """Get detailed detection results without saving"""
        results = self.model.predict(source=image_path, conf=conf, save=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'class_id': int(box.cls[0]),
                    'class_name': self.model.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    'bbox_normalized': box.xywhn[0].cpu().numpy().tolist()  # [x_center, y_center, width, height]
                }
                detections.append(detection)
        
        return detections

def main():
    """Main function with usage examples"""
    
    # Initialize inference
    # Use your trained model or the pretrained one
    model_path = 'yolo11n.pt'  # Change to 'runs/train/yolo_park_surveillance/weights/best.pt' after training
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Using default YOLO model...")
        model_path = 'yolo11n.pt'
    
    inference = YOLOInference(model_path)
    
    # Example 1: Predict on single image
    if os.path.exists('test_image.jpg'):
        print("\n" + "="*50)
        print("Running inference on test_image.jpg")
        print("="*50)
        inference.predict_image('test_image.jpg', conf=0.25)
    
    # Example 2: Predict on video
    if os.path.exists('pranitha_data1.mp4'):
        print("\n" + "="*50)
        print("Running inference on pranitha_data1.mp4")
        print("="*50)
        inference.predict_video('pranitha_data1.mp4', conf=0.25)
    
    # Example 3: Predict on test folder
    if os.path.exists('data/test/images'):
        print("\n" + "="*50)
        print("Running inference on test images")
        print("="*50)
        inference.predict_folder('data/test/images', conf=0.25)
    
    # Example 4: Get detailed results
    if os.path.exists('test_image.jpg'):
        print("\n" + "="*50)
        print("Getting detailed detection results")
        print("="*50)
        detections = inference.get_detailed_results('test_image.jpg', conf=0.25)
        for i, det in enumerate(detections, 1):
            print(f"\nDetection {i}:")
            print(f"  Class: {det['class_name']}")
            print(f"  Confidence: {det['confidence']:.2f}")
            print(f"  BBox: {det['bbox']}")
    
    print("\n" + "="*50)
    print("Inference complete!")
    print("Results saved in: runs/inference/")
    print("="*50)

if __name__ == '__main__':
    main()
