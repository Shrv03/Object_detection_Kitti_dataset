import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import json
import yaml
from pathlib import Path


def download_kitti_dataset(download_dir='data/kitti'):
    """
    Instructions for downloading KITTI dataset
    Note: KITTI dataset requires manual download due to license agreements
    """
    print("KITTI Dataset Download Instructions:")
    print("=" * 50)
    print("1. Visit: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d")
    print("2. Register and download the following files:")
    print("   - Left color images of object data set (12 GB)")
    print("   - Training labels of object data set (5 MB)")
    print("   - Object development kit (1 MB)")
    print("3. Extract files to the following structure:")
    print(f"   {download_dir}/")
    print("   ├── training/")
    print("   │   ├── image_2/")
    print("   │   ├── label_2/")
    print("   │   └── calib/")
    print("   └── testing/")
    print("       ├── image_2/")
    print("       └── calib/")
    print("\nAfter downloading, run setup_kitti_dataset() to prepare the data.")


def setup_kitti_dataset(kitti_dir='data/kitti', output_dir='data/kitti_yolo'):
    """
    Convert KITTI dataset to YOLO format
    
    Args:
        kitti_dir: Path to original KITTI dataset
        output_dir: Path for YOLO format dataset
    """
    print("Converting KITTI dataset to YOLO format...")
    
    # KITTI class mapping to simplified classes
    kitti_to_yolo_classes = {
        'Car': 0,
        'Van': 1, 
        'Truck': 2,
        'Pedestrian': 3,
        'Person_sitting': 4,
        'Cyclist': 5,
        'Tram': 6,
        'Misc': 7,
        'DontCare': -1  # Ignore this class
    }
    
    class_names = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
    
    # Create output directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Process training data
    train_img_dir = os.path.join(kitti_dir, 'training', 'image_2')
    train_label_dir = os.path.join(kitti_dir, 'training', 'label_2')
    
    if not os.path.exists(train_img_dir):
        print(f"Error: KITTI training images not found at {train_img_dir}")
        print("Please download KITTI dataset first using download_kitti_dataset()")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(train_img_dir) if f.endswith('.png')]
    image_files.sort()
    
    # Split into train/val (80/20)
    split_idx = int(0.8 * len(image_files))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Processing {len(train_files)} training images...")
    _process_split(train_files, train_img_dir, train_label_dir, 
                   os.path.join(output_dir, 'train'), kitti_to_yolo_classes)
    
    print(f"Processing {len(val_files)} validation images...")
    _process_split(val_files, train_img_dir, train_label_dir, 
                   os.path.join(output_dir, 'val'), kitti_to_yolo_classes)
    
    # Create dataset YAML file
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Dataset converted successfully!")
    print(f"YOLO dataset saved to: {output_dir}")
    print(f"Dataset YAML saved to: {yaml_path}")
    
    return yaml_path


def _process_split(image_files, img_dir, label_dir, output_dir, class_mapping):
    """Process a data split (train/val)"""
    img_output_dir = os.path.join(output_dir, 'images')
    label_output_dir = os.path.join(output_dir, 'labels')
    
    for img_file in tqdm(image_files):
        # Copy image
        src_img = os.path.join(img_dir, img_file)
        dst_img = os.path.join(img_output_dir, img_file)
        shutil.copy2(src_img, dst_img)
        
        # Convert label
        label_file = img_file.replace('.png', '.txt')
        src_label = os.path.join(label_dir, label_file)
        dst_label = os.path.join(label_output_dir, label_file)
        
        if os.path.exists(src_label):
            _convert_kitti_to_yolo(src_label, dst_label, src_img, class_mapping)
        else:
            # Create empty label file
            with open(dst_label, 'w') as f:
                pass


def _convert_kitti_to_yolo(kitti_label_path, yolo_label_path, image_path, class_mapping):
    """Convert KITTI label format to YOLO format"""
    # Get image dimensions
    image = cv2.imread(image_path)
    if image is None:
        return
    
    h, w = image.shape[:2]
    
    yolo_labels = []
    
    with open(kitti_label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 15:
            continue
        
        obj_type = parts[0]
        
        # Skip DontCare objects
        if obj_type == 'DontCare':
            continue
        
        # Map class to YOLO class ID
        if obj_type not in class_mapping:
            obj_type = 'Misc'  # Unknown classes become Misc
        
        class_id = class_mapping[obj_type]
        if class_id == -1:  # Skip ignored classes
            continue
        
        # Get bounding box coordinates (left, top, right, bottom)
        left = float(parts[4])
        top = float(parts[5])
        right = float(parts[6])
        bottom = float(parts[7])
        
        # Convert to YOLO format (normalized center coordinates)
        center_x = (left + right) / 2.0 / w
        center_y = (top + bottom) / 2.0 / h
        width = (right - left) / w
        height = (bottom - top) / h
        
        # Clamp values to [0, 1]
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    
    # Write YOLO labels
    with open(yolo_label_path, 'w') as f:
        f.write('\n'.join(yolo_labels))


def analyze_dataset(dataset_dir):
    """Analyze dataset statistics"""
    print("Analyzing dataset...")
    
    stats = {
        'total_images': 0,
        'total_objects': 0,
        'class_distribution': {},
        'avg_objects_per_image': 0,
        'image_sizes': []
    }
    
    class_names = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
    
    for split in ['train', 'val']:
        img_dir = os.path.join(dataset_dir, split, 'images')
        label_dir = os.path.join(dataset_dir, split, 'labels')
        
        if not os.path.exists(img_dir):
            continue
        
        print(f"\nAnalyzing {split} split...")
        
        image_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        stats['total_images'] += len(image_files)
        
        for img_file in tqdm(image_files):
            # Check image size
            img_path = os.path.join(img_dir, img_file)
            image = cv2.imread(img_path)
            if image is not None:
                h, w = image.shape[:2]
                stats['image_sizes'].append((w, h))
            
            # Check labels
            label_file = img_file.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
            label_path = os.path.join(label_dir, label_file)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if 0 <= class_id < len(class_names):
                            class_name = class_names[class_id]
                            stats['class_distribution'][class_name] = stats['class_distribution'].get(class_name, 0) + 1
                            stats['total_objects'] += 1
    
    # Calculate statistics
    if stats['total_images'] > 0:
        stats['avg_objects_per_image'] = stats['total_objects'] / stats['total_images']
    
    # Print results
    print("\nDataset Statistics:")
    print("=" * 50)
    print(f"Total images: {stats['total_images']}")
    print(f"Total objects: {stats['total_objects']}")
    print(f"Average objects per image: {stats['avg_objects_per_image']:.2f}")
    
    if stats['image_sizes']:
        widths = [size[0] for size in stats['image_sizes']]
        heights = [size[1] for size in stats['image_sizes']]
        print(f"Image size range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
    
    print("\nClass distribution:")
    for class_name, count in sorted(stats['class_distribution'].items()):
        percentage = (count / stats['total_objects']) * 100 if stats['total_objects'] > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    return stats


def create_sample_dataset(output_dir='data/sample_kitti', num_samples=100):
    """
    Create a small sample dataset for testing
    """
    print(f"Creating sample dataset with {num_samples} synthetic images...")
    
    os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'labels'), exist_ok=True)
    
    class_names = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
    
    # Create synthetic images and labels
    for i in tqdm(range(num_samples)):
        # Determine split
        split = 'train' if i < int(0.8 * num_samples) else 'val'
        
        # Create synthetic image (640x640)
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Add some simple shapes as "objects"
        labels = []
        num_objects = np.random.randint(1, 6)  # 1-5 objects per image
        
        for _ in range(num_objects):
            # Random bounding box
            x1 = np.random.randint(0, 500)
            y1 = np.random.randint(0, 500)
            w = np.random.randint(50, 140)
            h = np.random.randint(50, 140)
            x2 = min(640, x1 + w)
            y2 = min(640, y1 + h)
            
            # Draw rectangle
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            
            # Convert to YOLO format
            center_x = (x1 + x2) / 2.0 / 640
            center_y = (y1 + y2) / 2.0 / 640
            width = (x2 - x1) / 640
            height = (y2 - y1) / 640
            
            class_id = np.random.randint(0, len(class_names))
            labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        # Save image and labels
        img_path = os.path.join(output_dir, split, 'images', f'{i:06d}.png')
        label_path = os.path.join(output_dir, split, 'labels', f'{i:06d}.txt')
        
        cv2.imwrite(img_path, img)
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
    
    # Create dataset YAML
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Sample dataset created at: {output_dir}")
    print(f"Dataset YAML: {yaml_path}")
    
    return yaml_path


def validate_dataset(dataset_dir):
    """Validate dataset format and integrity"""
    print("Validating dataset...")
    
    issues = []
    
    # Check directory structure
    required_dirs = [
        'train/images', 'train/labels',
        'val/images', 'val/labels'
    ]
    
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_dir, dir_path)
        if not os.path.exists(full_path):
            issues.append(f"Missing directory: {dir_path}")
    
    # Check dataset.yaml
    yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
    if not os.path.exists(yaml_path):
        issues.append("Missing dataset.yaml file")
    
    # Validate image-label pairs
    for split in ['train', 'val']:
        img_dir = os.path.join(dataset_dir, split, 'images')
        label_dir = os.path.join(dataset_dir, split, 'labels')
        
        if os.path.exists(img_dir) and os.path.exists(label_dir):
            img_files = set([os.path.splitext(f)[0] for f in os.listdir(img_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))])
            label_files = set([os.path.splitext(f)[0] for f in os.listdir(label_dir) 
                             if f.endswith('.txt')])
            
            missing_labels = img_files - label_files
            missing_images = label_files - img_files
            
            if missing_labels:
                issues.append(f"{split}: {len(missing_labels)} images missing labels")
            if missing_images:
                issues.append(f"{split}: {len(missing_images)} labels missing images")
    
    # Print results
    if issues:
        print("Dataset validation FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Dataset validation PASSED!")
        return True