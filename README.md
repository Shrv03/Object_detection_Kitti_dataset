# KITTI Object Detection for Self-Driving Cars

A comprehensive deep learning project for object detection using the KITTI dataset, specifically designed for self-driving car applications. This project implements YOLOv8-based object detection with support for training, inference, and evaluation on KITTI data.

## 🚗 Overview

This project provides a complete pipeline for:
- **Object Detection**: Cars, vans, trucks, pedestrians, cyclists, trams, and misc objects
- **KITTI Dataset Integration**: Seamless handling of KITTI format data
- **YOLOv8 Implementation**: State-of-the-art object detection model
- **Self-Driving Car Focus**: Optimized for autonomous vehicle scenarios
- **Comprehensive Evaluation**: Detailed metrics and visualization

## 🏁 Quick Start

### 1. Setup Environment

```bash
# Clone the repository (if needed)
git clone <your-repo-url>
cd kitti-object-detection

# Run setup script
python setup.py
```

### 2. Train on Sample Data

```bash
# Train with default settings (uses sample dataset)
python train.py --epochs 10

# Train with custom parameters
python train.py --model yolov8s --batch-size 32 --epochs 100
```

### 3. Run Inference

```bash
# Inference on single image
python inference.py --source path/to/image.jpg --weights yolov8n.pt

# Inference on webcam
python inference.py --source 0 --weights yolov8n.pt --show

# Inference on directory
python inference.py --source path/to/images/ --weights runs/train/exp/weights/best.pt
```

## 📁 Project Structure

```
kitti-object-detection/
├── src/                          # Source code
│   ├── data/
│   │   └── kitti_dataset.py     # KITTI dataset loader
│   ├── models/
│   │   └── detector.py          # YOLOv8 detector wrapper
│   └── utils/
│       └── data_utils.py        # Data processing utilities
├── config/
│   └── config.yaml              # Configuration file
├── data/                        # Dataset directory
│   ├── kitti/                   # Original KITTI data
│   └── sample_kitti/            # Sample dataset for testing
├── weights/                     # Model weights
├── runs/                        # Training/inference results
├── train.py                     # Training script
├── inference.py                 # Inference script
├── evaluate.py                  # Evaluation script
├── setup.py                     # Setup and installation
└── requirements.txt             # Python dependencies
```

## 🎯 Features

### Core Functionality
- **Multi-class Object Detection**: 8 KITTI object classes
- **Real-time Inference**: Optimized for speed and accuracy
- **Video Processing**: Support for video files and webcam
- **Batch Processing**: Efficient directory processing
- **Custom Visualization**: Enhanced bounding box visualization

### Training Features
- **Transfer Learning**: Pre-trained YOLOv8 weights
- **Data Augmentation**: Advanced augmentation pipeline
- **Mixed Precision**: Memory-efficient training
- **Early Stopping**: Prevent overfitting
- **Experiment Tracking**: Organized result logging

### Evaluation Features
- **Comprehensive Metrics**: mAP, precision, recall, F1-score
- **Per-class Analysis**: Detailed class-wise performance
- **Visualization**: Plots and charts for analysis
- **KITTI Evaluation**: Compatible with KITTI evaluation protocols

## 📊 Dataset

### KITTI Object Classes
1. **Car** - Standard passenger cars
2. **Van** - Delivery vans and larger vehicles
3. **Truck** - Heavy trucks and commercial vehicles
4. **Pedestrian** - Walking people
5. **Person_sitting** - Sitting people
6. **Cyclist** - People on bicycles
7. **Tram** - Public transport trams
8. **Misc** - Other objects and vehicles

### Download KITTI Dataset

```bash
# 1. Visit KITTI website
# http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d

# 2. Download required files:
# - Left color images of object data set (12 GB)
# - Training labels of object data set (5 MB)

# 3. Extract to data/kitti/ directory
# 4. Convert to YOLO format
python -c "from src.utils.data_utils import setup_kitti_dataset; setup_kitti_dataset()"
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
# Model Configuration
model:
  name: "yolov8n"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  num_classes: 8

# Training Configuration
training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.01
  
# Dataset Configuration
dataset:
  img_size: [640, 640]
  classes: ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
```

## 🚀 Usage Examples

### Training

```bash
# Basic training
python train.py

# Custom dataset
python train.py --data path/to/dataset.yaml

# Resume training
python train.py --resume runs/train/exp/weights/last.pt

# Multi-GPU training
python train.py --device 0,1,2,3

# Validation during training
python train.py --validate
```

### Inference

```bash
# Image inference
python inference.py --source image.jpg --weights best.pt --conf 0.25

# Video inference
python inference.py --source video.mp4 --weights best.pt --save

# Webcam inference
python inference.py --source 0 --show

# Batch inference
python inference.py --source images_folder/ --save-json
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py --weights runs/train/exp/weights/best.pt --data dataset.yaml

# Custom evaluation settings
python evaluate.py --weights best.pt --data dataset.yaml --conf 0.001 --iou 0.6
```

## 📈 Results

Expected performance on KITTI dataset:

| Model | mAP@0.5 | mAP@0.5:0.95 | Speed (ms) | Params (M) |
|-------|---------|--------------|------------|------------|
| YOLOv8n | 0.65+ | 0.45+ | ~10 | 3.2 |
| YOLOv8s | 0.70+ | 0.50+ | ~15 | 11.2 |
| YOLOv8m | 0.75+ | 0.55+ | ~25 | 25.9 |
| YOLOv8l | 0.78+ | 0.58+ | ~35 | 43.7 |
| YOLOv8x | 0.80+ | 0.60+ | ~50 | 68.2 |

*Results may vary based on dataset size and training configuration*

## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data weights logs results runs

# Download sample models
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Docker Setup (Optional)

```dockerfile
FROM ultralytics/ultralytics:latest
WORKDIR /workspace
COPY . .
RUN pip install -r requirements.txt
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```bash
   # Reduce batch size
   python train.py --batch-size 8
   ```

2. **Dataset not found**
   ```bash
   # Create sample dataset
   python -c "from src.utils.data_utils import create_sample_dataset; create_sample_dataset()"
   ```

3. **Import errors**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

## 📚 Documentation

- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- KITTI Dataset creators and maintainers
- Ultralytics team for YOLOv8
- PyTorch community
- Self-driving car research community

## 📧 Contact

For questions and support:
- Create an issue in this repository
- Check the documentation
- Review existing issues for solutions

---

**Happy Autonomous Driving! 🚗💨**