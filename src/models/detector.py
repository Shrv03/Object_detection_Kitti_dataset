import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
import os


class KITTIObjectDetector:
    """
    Object Detection model for KITTI dataset using YOLOv8
    """
    
    def __init__(self, model_name='yolov8n', num_classes=8, device='auto'):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        
        # KITTI class names
        self.class_names = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
        
        # Colors for visualization (BGR format for OpenCV)
        self.colors = [
            (255, 0, 0),    # Car - Red
            (0, 255, 0),    # Van - Green
            (0, 0, 255),    # Truck - Blue
            (255, 255, 0),  # Pedestrian - Cyan
            (255, 0, 255),  # Person_sitting - Magenta
            (0, 255, 255),  # Cyclist - Yellow
            (128, 0, 128),  # Tram - Purple
            (128, 128, 128) # Misc - Gray
        ]
        
        # Initialize model
        self.model = None
        self.load_model()
    
    def load_model(self, weights_path=None):
        """Load YOLOv8 model"""
        if weights_path and os.path.exists(weights_path):
            self.model = YOLO(weights_path)
            print(f"Loaded model from {weights_path}")
        else:
            self.model = YOLO(f"{self.model_name}.pt")
            print(f"Loaded pretrained {self.model_name} model")
    
    def train(self, data_yaml, epochs=100, batch_size=16, img_size=640, 
              save_dir='runs/train', **kwargs):
        """
        Train the model on KITTI dataset
        
        Args:
            data_yaml: Path to dataset YAML file
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size
            save_dir: Directory to save results
        """
        print(f"Starting training with {epochs} epochs...")
        
        # Training arguments
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'project': save_dir,
            'name': 'kitti_detection',
            'save': True,
            'verbose': True,
            'device': self.device,
            **kwargs
        }
        
        # Start training
        results = self.model.train(**train_args)
        
        print("Training completed!")
        return results
    
    def validate(self, data_yaml, img_size=640, save_dir='runs/val'):
        """Validate the model"""
        print("Starting validation...")
        
        results = self.model.val(
            data=data_yaml,
            imgsz=img_size,
            project=save_dir,
            name='kitti_val',
            save=True,
            verbose=True
        )
        
        print("Validation completed!")
        return results
    
    def predict(self, source, conf=0.25, iou=0.45, save=True, save_dir='runs/predict'):
        """
        Run inference on images or video
        
        Args:
            source: Path to image, directory, or video
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save: Whether to save results
            save_dir: Directory to save results
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            save=save,
            project=save_dir,
            name='kitti_predict',
            verbose=True
        )
        
        return results
    
    def visualize_predictions(self, image_path, conf=0.25, iou=0.45, save_path=None):
        """
        Visualize predictions on a single image with custom styling
        
        Args:
            image_path: Path to input image
            conf: Confidence threshold
            iou: IoU threshold
            save_path: Path to save visualization
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            return None
        
        # Get predictions
        results = self.model.predict(image_path, conf=conf, iou=iou, verbose=False)
        
        # Draw predictions
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf_score = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name and color
                    class_name = self.class_names[class_id]
                    color = self.colors[class_id]
                    
                    # Draw bounding box
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw label
                    label = f"{class_name}: {conf_score:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(image, (int(x1), int(y1) - label_size[1] - 10), 
                                (int(x1) + label_size[0], int(y1)), color, -1)
                    cv2.putText(image, label, (int(x1), int(y1) - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save or display
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"Visualization saved to {save_path}")
        
        return image
    
    def evaluate_on_kitti(self, test_images_dir, ground_truth_dir=None, 
                         conf=0.25, iou=0.45, save_dir='evaluation'):
        """
        Evaluate model on KITTI test set and compute metrics
        
        Args:
            test_images_dir: Directory containing test images
            ground_truth_dir: Directory containing ground truth labels
            conf: Confidence threshold
            iou: IoU threshold
            save_dir: Directory to save evaluation results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Get all test images
        image_files = [f for f in os.listdir(test_images_dir) if f.endswith('.png')]
        image_files.sort()
        
        results = []
        
        print(f"Evaluating on {len(image_files)} images...")
        
        for img_file in image_files:
            img_path = os.path.join(test_images_dir, img_file)
            
            # Run inference
            prediction_results = self.model.predict(img_path, conf=conf, iou=iou, verbose=False)
            
            # Process results
            detections = []
            for result in prediction_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf_score = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf_score),
                            'class': self.class_names[class_id],
                            'class_id': class_id
                        })
            
            results.append({
                'image': img_file,
                'detections': detections
            })
        
        # Save results
        import json
        with open(os.path.join(save_dir, 'predictions.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation completed. Results saved to {save_dir}")
        return results
    
    def export_model(self, format='onnx', save_dir='exported_models'):
        """Export model to different formats"""
        os.makedirs(save_dir, exist_ok=True)
        
        export_path = self.model.export(format=format, project=save_dir)
        print(f"Model exported to {export_path}")
        return export_path
    
    def get_model_info(self):
        """Get model information and statistics"""
        info = self.model.info()
        print("Model Information:")
        print(f"Model: {self.model_name}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
        return info


def create_detection_report(results, save_path='detection_report.txt'):
    """Create a detailed detection report"""
    with open(save_path, 'w') as f:
        f.write("KITTI Object Detection Report\n")
        f.write("=" * 50 + "\n\n")
        
        total_detections = 0
        class_counts = {}
        
        for result in results:
            f.write(f"Image: {result['image']}\n")
            f.write(f"Number of detections: {len(result['detections'])}\n")
            
            for detection in result['detections']:
                class_name = detection['class']
                conf = detection['confidence']
                f.write(f"  - {class_name}: {conf:.3f}\n")
                
                total_detections += 1
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            f.write("\n")
        
        f.write("Summary:\n")
        f.write(f"Total detections: {total_detections}\n")
        f.write("Detections per class:\n")
        for class_name, count in sorted(class_counts.items()):
            f.write(f"  {class_name}: {count}\n")
    
    print(f"Detection report saved to {save_path}")