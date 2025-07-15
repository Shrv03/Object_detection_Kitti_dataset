import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml


class KITTIDataset(Dataset):
    """
    KITTI Dataset loader for object detection
    Handles KITTI format data and converts to YOLO format
    """
    
    def __init__(self, root_dir, split='training', transform=None, img_size=(640, 640)):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform
        
        # KITTI class mapping
        self.classes = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Data paths
        self.image_dir = os.path.join(root_dir, split, 'image_2')
        self.label_dir = os.path.join(root_dir, split, 'label_2')
        self.calib_dir = os.path.join(root_dir, split, 'calib')
        
        # Get all image files
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        self.image_files.sort()
        
        print(f"Loaded {len(self.image_files)} images from {split} split")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_name = img_name.replace('.png', '.txt')
        label_path = os.path.join(self.label_dir, label_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                
                obj_type = parts[0]
                
                # Skip DontCare objects
                if obj_type == 'DontCare':
                    continue
                
                # Map KITTI classes to our simplified classes
                if obj_type not in self.class_to_idx:
                    obj_type = 'Misc'  # Map unknown classes to Misc
                
                # Extract bounding box (left, top, right, bottom)
                left = float(parts[4])
                top = float(parts[5])
                right = float(parts[6])
                bottom = float(parts[7])
                
                # Convert to center coordinates and normalize
                h, w = image.shape[:2]
                center_x = (left + right) / 2.0 / w
                center_y = (top + bottom) / 2.0 / h
                width = (right - left) / w
                height = (bottom - top) / h
                
                # YOLO format: [class_id, center_x, center_y, width, height]
                boxes.append([center_x, center_y, width, height])
                labels.append(self.class_to_idx[obj_type])
        
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transformations
        if self.transform:
            # Convert boxes to albumentations format (x_min, y_min, x_max, y_max)
            if len(boxes) > 0:
                h, w = image.shape[:2]
                boxes_albumentations = []
                for box in boxes:
                    center_x, center_y, width, height = box
                    x_min = max(0, (center_x - width/2) * w)
                    y_min = max(0, (center_y - height/2) * h)
                    x_max = min(w, (center_x + width/2) * w)
                    y_max = min(h, (center_y + height/2) * h)
                    boxes_albumentations.append([x_min, y_min, x_max, y_max])
                
                transformed = self.transform(
                    image=image,
                    bboxes=boxes_albumentations,
                    class_labels=labels
                )
                image = transformed['image']
                
                # Convert back to YOLO format
                if len(transformed['bboxes']) > 0:
                    h, w = self.img_size
                    boxes = []
                    labels = transformed['class_labels']
                    for bbox in transformed['bboxes']:
                        x_min, y_min, x_max, y_max = bbox
                        center_x = (x_min + x_max) / 2.0 / w
                        center_y = (y_min + y_max) / 2.0 / h
                        width = (x_max - x_min) / w
                        height = (y_max - y_min) / h
                        boxes.append([center_x, center_y, width, height])
                    boxes = np.array(boxes, dtype=np.float32)
                else:
                    boxes = np.array([]).reshape(0, 4)
                    labels = np.array([])
            else:
                # No boxes, just transform image
                transformed = self.transform(image=image)
                image = transformed['image']
        
        # Resize image if no transform was applied
        if not self.transform:
            image = cv2.resize(image, self.img_size)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image,
            'boxes': torch.from_numpy(boxes) if len(boxes) > 0 else torch.empty(0, 4),
            'labels': torch.from_numpy(labels) if len(labels) > 0 else torch.empty(0, dtype=torch.long),
            'image_id': idx,
            'filename': img_name
        }


def get_train_transforms(img_size=(640, 640)):
    """Get training transforms with data augmentation"""
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


def get_val_transforms(img_size=(640, 640)):
    """Get validation transforms without augmentation"""
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


def create_kitti_yaml(data_dir, output_path='data/kitti.yaml'):
    """Create YOLO-format dataset YAML file for KITTI"""
    yaml_content = {
        'path': data_dir,
        'train': 'training/image_2',
        'val': 'testing/image_2',
        'nc': 8,  # number of classes
        'names': ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created KITTI YAML config at {output_path}")
    return output_path