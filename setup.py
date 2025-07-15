#!/usr/bin/env python3
"""
KITTI Object Detection Setup Script
Automates installation and dataset preparation
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, shell=False):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=shell, check=True, 
                              capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available, using CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False

def install_requirements():
    """Install required packages"""
    print("\n🔧 Installing requirements...")
    
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found!")
        return False
    
    success, output = run_command([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    
    if success:
        print("✅ Requirements installed successfully")
        return True
    else:
        print(f"❌ Failed to install requirements: {output}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        'data',
        'data/kitti',
        'data/sample_kitti',
        'weights',
        'logs',
        'results',
        'runs',
        'runs/train',
        'runs/val',
        'runs/predict'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Directories created")

def test_installation():
    """Test if installation is working"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        import torch
        import torchvision
        import ultralytics
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        
        print("✅ All packages imported successfully")
        
        # Test YOLO model loading
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 model loaded successfully")
        
        # Test custom modules
        sys.path.append('src')
        from models.detector import KITTIObjectDetector
        from utils.data_utils import create_sample_dataset
        
        print("✅ Custom modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def create_sample_dataset():
    """Create a sample dataset for testing"""
    print("\n🎯 Creating sample dataset...")
    
    try:
        sys.path.append('src')
        from utils.data_utils import create_sample_dataset as create_sample
        
        yaml_path = create_sample('data/sample_kitti', num_samples=50)
        print(f"✅ Sample dataset created: {yaml_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample dataset: {e}")
        return False

def download_pretrained_models():
    """Download pretrained YOLOv8 models"""
    print("\n⬇️  Downloading pretrained models...")
    
    models = ['yolov8n.pt', 'yolov8s.pt']
    
    try:
        from ultralytics import YOLO
        
        for model_name in models:
            print(f"   Downloading {model_name}...")
            model = YOLO(model_name)  # This will download if not present
            print(f"   ✅ {model_name} ready")
        
        print("✅ Pretrained models downloaded")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download models: {e}")
        return False

def show_usage_examples():
    """Show usage examples"""
    print("\n📖 Usage Examples:")
    print("=" * 50)
    
    examples = [
        ("Train model on sample dataset", 
         "python train.py --data data/sample_kitti/dataset.yaml --epochs 10"),
        
        ("Train with custom parameters",
         "python train.py --model yolov8s --batch-size 32 --epochs 100"),
        
        ("Run inference on image",
         "python inference.py --source path/to/image.jpg --weights runs/train/exp/weights/best.pt"),
        
        ("Run inference on webcam",
         "python inference.py --source 0 --weights yolov8n.pt"),
        
        ("Evaluate model",
         "python evaluate.py --weights runs/train/exp/weights/best.pt --data data/sample_kitti/dataset.yaml"),
        
        ("Convert KITTI dataset",
         "python -c \"from src.utils.data_utils import setup_kitti_dataset; setup_kitti_dataset()\"")
    ]
    
    for i, (description, command) in enumerate(examples, 1):
        print(f"{i}. {description}:")
        print(f"   {command}\n")

def main():
    parser = argparse.ArgumentParser(description='Setup KITTI Object Detection Project')
    parser.add_argument('--skip-install', action='store_true',
                        help='Skip package installation')
    parser.add_argument('--skip-test', action='store_true',
                        help='Skip installation test')
    parser.add_argument('--skip-sample', action='store_true',
                        help='Skip sample dataset creation')
    parser.add_argument('--skip-models', action='store_true',
                        help='Skip pretrained model download')
    args = parser.parse_args()
    
    print("🚗 KITTI Object Detection Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not args.skip_install:
        if not install_requirements():
            print("❌ Setup failed during package installation")
            sys.exit(1)
    
    # Check CUDA
    check_cuda()
    
    # Test installation
    if not args.skip_test:
        if not test_installation():
            print("❌ Setup failed during testing")
            sys.exit(1)
    
    # Download pretrained models
    if not args.skip_models:
        download_pretrained_models()
    
    # Create sample dataset
    if not args.skip_sample:
        create_sample_dataset()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Download KITTI dataset (see data/README.md for instructions)")
    print("2. Convert KITTI format using utils/data_utils.py")
    print("3. Start training: python train.py")
    
    show_usage_examples()

if __name__ == '__main__':
    main()