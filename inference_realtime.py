"""
Real-time Inference Script
Run YOLO inference on webcam or video with FPS counter and statistics
"""

from ultralytics import YOLO
import cv2
import time
from collections import defaultdict

def realtime_inference(model_path='yolo11n.pt', source=0, conf=0.25, show_fps=True):
    """
    Run real-time inference with statistics
    
    Args:
        model_path: Path to YOLO model
        source: Video source (0 for webcam, or video file path)
        conf: Confidence threshold
        show_fps: Show FPS counter
    """
    
    # Load model
    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")
    print(f"Classes: {model.names}")
    print("\nPress 'q' to quit, 's' to save screenshot")
    
    # Open video source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if source != 0 else 30
    
    print(f"\nVideo: {width}x{height} @ {fps}fps")
    
    # Statistics
    frame_count = 0
    start_time = time.time()
    detection_counts = defaultdict(int)
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_start = time.time()
        
        # Run inference
        results = model.predict(
            source=frame,
            conf=conf,
            verbose=False,
            stream=False
        )
        
        # Process results
        annotated_frame = results[0].plot()
        
        # Count detections
        boxes = results[0].boxes
        frame_detections = defaultdict(int)
        
        for box in boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]
            detection_counts[class_name] += 1
            frame_detections[class_name] += 1
        
        # Calculate FPS
        if show_fps:
            frame_time = time.time() - frame_start
            current_fps = 1 / frame_time if frame_time > 0 else 0
            
            # Add FPS and stats to frame
            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add detection counts
            y_offset = 70
            for class_name, count in frame_detections.items():
                text = f"{class_name}: {count}"
                cv2.putText(annotated_frame, text, 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_offset += 30
        
        # Display frame
        cv2.imshow('YOLO Real-time Inference', annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_count += 1
            filename = f'screenshot_{screenshot_count}.jpg'
            cv2.imwrite(filename, annotated_frame)
            print(f"Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print statistics
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    print("\n" + "="*50)
    print("Inference Statistics")
    print("="*50)
    print(f"Total frames: {frame_count}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"\nDetection counts:")
    for class_name, count in detection_counts.items():
        print(f"  {class_name}: {count}")
    print(f"\nScreenshots saved: {screenshot_count}")

if __name__ == '__main__':
    # Configuration
    MODEL_PATH = 'yolo11n.pt'  # Change to your trained model
    
    # Option 1: Webcam
    # realtime_inference(MODEL_PATH, source=0, conf=0.25)
    
    # Option 2: Video file
    VIDEO_PATH = 'pranitha_data1.mp4'
    realtime_inference(MODEL_PATH, source=VIDEO_PATH, conf=0.25)
    
    # Option 3: RTSP stream
    # realtime_inference(MODEL_PATH, source='rtsp://your_stream_url', conf=0.25)
