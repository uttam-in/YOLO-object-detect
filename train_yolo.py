"""
YOLO Training Script
Train YOLO model on custom dataset using data.yaml configuration
"""

from ultralytics import YOLO
import torch
import os

def train_yolo():
    """Train YOLO model on custom dataset"""
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load a pretrained YOLO model (you can use yolo11n.pt or download a new one)
    model = YOLO('yolo11n.pt')  # nano model for faster training
    # Alternatives: yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large)
    
    # Training parameters
    results = model.train(
        data='data/data.yaml',           # path to data config
        epochs=100,                       # number of training epochs
        imgsz=640,                        # image size
        batch=16,                         # batch size (adjust based on GPU memory)
        device=device,                    # device to train on
        workers=4,                        # number of worker threads
        patience=50,                      # early stopping patience
        save=True,                        # save checkpoints
        project='runs/train',             # project directory
        name='yolo_park_surveillance',    # experiment name
        exist_ok=True,                    # overwrite existing experiment
        pretrained=True,                  # use pretrained weights
        optimizer='auto',                 # optimizer (auto, SGD, Adam, AdamW)
        verbose=True,                     # verbose output
        seed=42,                          # random seed for reproducibility
        deterministic=True,               # deterministic mode
        single_cls=False,                 # train as single-class dataset
        rect=False,                       # rectangular training
        cos_lr=False,                     # cosine learning rate scheduler
        close_mosaic=10,                  # disable mosaic augmentation for final epochs
        resume=False,                     # resume training from last checkpoint
        amp=True,                         # automatic mixed precision training
        fraction=1.0,                     # dataset fraction to train on
        profile=False,                    # profile ONNX and TensorRT speeds
        freeze=None,                      # freeze layers (list or int)
        # Learning rate settings
        lr0=0.01,                         # initial learning rate
        lrf=0.01,                         # final learning rate (lr0 * lrf)
        momentum=0.937,                   # SGD momentum/Adam beta1
        weight_decay=0.0005,              # optimizer weight decay
        warmup_epochs=3.0,                # warmup epochs
        warmup_momentum=0.8,              # warmup initial momentum
        warmup_bias_lr=0.1,               # warmup initial bias lr
        # Augmentation settings
        hsv_h=0.015,                      # HSV-Hue augmentation
        hsv_s=0.7,                        # HSV-Saturation augmentation
        hsv_v=0.4,                        # HSV-Value augmentation
        degrees=0.0,                      # rotation (+/- deg)
        translate=0.1,                    # translation (+/- fraction)
        scale=0.5,                        # scale (+/- gain)
        shear=0.0,                        # shear (+/- deg)
        perspective=0.0,                  # perspective (+/- fraction)
        flipud=0.0,                       # flip up-down probability
        fliplr=0.5,                       # flip left-right probability
        mosaic=1.0,                       # mosaic augmentation probability
        mixup=0.0,                        # mixup augmentation probability
        copy_paste=0.0,                   # copy-paste augmentation probability
    )
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"Best model saved at: {model.trainer.best}")
    print(f"Results saved in: runs/train/yolo_park_surveillance")
    
    return results

def validate_model(model_path='runs/train/yolo_park_surveillance/weights/best.pt'):
    """Validate the trained model"""
    model = YOLO(model_path)
    
    # Validate on test set
    metrics = model.val(
        data='data/data.yaml',
        split='test',
        imgsz=640,
        batch=16,
        save_json=True,
        save_hybrid=True,
        conf=0.25,
        iou=0.6,
        max_det=300,
        plots=True
    )
    
    print("\n" + "="*50)
    print("Validation Metrics:")
    print("="*50)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    return metrics

if __name__ == '__main__':
    # Train the model
    print("Starting YOLO training...")
    results = train_yolo()
    
    # Validate the trained model
    print("\nValidating trained model...")
    metrics = validate_model()
    
    print("\nTraining and validation complete!")
    print("Check 'runs/train/yolo_park_surveillance' for results and weights")
