from ultralytics import YOLO
import cv2
import os

def track_objects_video(video_path, model_path="yolo11n.pt", conf_threshold=0.25):
    """
    Track objects in a video using YOLO model.
    
    Args:
        video_path (str): Path to the input video
        model_path (str): Path to the YOLO model weights
        conf_threshold (float): Confidence threshold for detections
    
    Returns:
        output_path: Path to the output video with tracked objects
    """
    # Check if video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Load the pretrained YOLO model
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # Define output video path
    output_path = video_path.replace('.mp4', '_tracked.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\nProcessing video with object tracking...")
    frame_count = 0
    
    # Process video with tracking
    results = model.track(video_path, conf=conf_threshold, stream=True, persist=True)
    
    for result in results:
        frame_count += 1
        
        # Get annotated frame
        annotated_frame = result.plot()
        
        # Write frame to output video
        out.write(annotated_frame)
        
        # Print progress
        if frame_count % 30 == 0 or frame_count == total_frames:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        # Print detection info for first frame
        if frame_count == 1 and result.boxes is not None:
            num_detections = len(result.boxes)
            print(f"\n  Frame 1: Detected {num_detections} objects")
            for box in result.boxes[:5]:  # Show first 5 objects
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                print(f"    - {class_name} (confidence: {confidence:.2f})")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"\n✓ Tracking completed!")
    print(f"Output video saved to: {output_path}")
    
    return output_path


def main():
    """Main function to run video object tracking."""
    # Video path
    video_path = "pranitha_data1.mp4"
    
    # Model and parameters
    model_path = "yolo11n.pt"
    conf_threshold = 0.25
    
    try:
        output_path = track_objects_video(
            video_path=video_path,
            model_path=model_path,
            conf_threshold=conf_threshold
        )
        
        print("\n✓ Video object tracking completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease provide a valid video path.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
