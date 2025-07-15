# KITTI Dataset Setup Guide

This directory contains the KITTI dataset for object detection training and evaluation.

## 📁 Directory Structure

After setup, your data directory should look like this:

```
data/
├── README.md                    # This file
├── kitti/                       # Original KITTI dataset
│   ├── training/
│   │   ├── image_2/            # Left color images (training)
│   │   ├── label_2/            # Object labels (training)
│   │   └── calib/              # Calibration files (training)
│   └── testing/
│       ├── image_2/            # Left color images (testing)
│       └── calib/              # Calibration files (testing)
├── kitti_yolo/                 # Converted YOLO format dataset
│   ├── train/
│   │   ├── images/             # Training images
│   │   └── labels/             # Training labels (YOLO format)
│   ├── val/
│   │   ├── images/             # Validation images
│   │   └── labels/             # Validation labels (YOLO format)
│   └── dataset.yaml            # Dataset configuration
└── sample_kitti/               # Sample dataset for testing
    ├── train/
    ├── val/
    └── dataset.yaml
```

## 🔗 Download KITTI Dataset

### Step 1: Visit KITTI Website

Go to the official KITTI object detection evaluation page:
```
http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
```

### Step 2: Register and Download

1. **Register** for an account (free)
2. **Download** the following files:
   - **Left color images of object data set** (12 GB)
     - `data_object_image_2.zip`
   - **Training labels of object data set** (5 MB) 
     - `data_object_label_2.zip`
   - **Camera calibration matrices** (16 MB)
     - `data_object_calib.zip`

### Step 3: Extract Files

Extract the downloaded files to the `data/kitti/` directory:

```bash
# Navigate to data directory
cd data/kitti

# Extract files (adjust paths as needed)
unzip path/to/data_object_image_2.zip
unzip path/to/data_object_label_2.zip  
unzip path/to/data_object_calib.zip
```

### Step 4: Verify Structure

Check that your directory structure matches:

```bash
ls data/kitti/training/
# Should show: image_2/ label_2/ calib/

ls data/kitti/testing/
# Should show: image_2/ calib/
```

## 🔄 Convert to YOLO Format

### Automatic Conversion

Use the provided utility script:

```bash
# Convert KITTI to YOLO format
python -c "
from src.utils.data_utils import setup_kitti_dataset
setup_kitti_dataset('data/kitti', 'data/kitti_yolo')
"
```

### Manual Conversion

```bash
# Use the conversion script directly
python src/utils/data_utils.py
```

### Verify Conversion

```bash
# Check converted dataset
python -c "
from src.utils.data_utils import validate_dataset, analyze_dataset
validate_dataset('data/kitti_yolo')
analyze_dataset('data/kitti_yolo')
"
```

## 📊 Dataset Statistics

### Original KITTI Dataset
- **Training images**: ~7,481 images
- **Test images**: ~7,518 images  
- **Image size**: Varies (typically 1242x375 or 1224x370)
- **Object classes**: 8 main classes
- **Total objects**: ~80,000+ labeled objects

### Class Distribution (Approximate)
- **Car**: ~65% of objects
- **Pedestrian**: ~20% of objects
- **Cyclist**: ~8% of objects
- **Van**: ~3% of objects
- **Truck**: ~2% of objects
- **Person_sitting**: ~1% of objects
- **Tram**: ~1% of objects
- **Misc**: <1% of objects

## 🎯 Sample Dataset

For quick testing without downloading the full KITTI dataset:

```bash
# Create sample dataset
python -c "
from src.utils.data_utils import create_sample_dataset
create_sample_dataset('data/sample_kitti', num_samples=100)
"
```

This creates a synthetic dataset with:
- 100 sample images (80 train, 20 val)
- Random object annotations
- Compatible YOLO format
- Same class structure as KITTI

## 🔍 Data Formats

### KITTI Label Format

Each line in a KITTI label file represents one object:

```
Car 0.00 0 -1.57 614.24 181.78 727.31 284.77 1.57 1.65 4.06 2.45 1.35 69.44 -1.59
```

Fields:
1. **Class**: Object type (Car, Pedestrian, etc.)
2. **Truncated**: Float 0-1 (0=non-truncated, 1=truncated)
3. **Occluded**: Integer 0-3 (0=fully visible, 3=unknown)
4. **Alpha**: Observation angle [-pi, pi]
5. **Bbox**: 2D bounding box [left, top, right, bottom]
6. **Dimensions**: 3D object dimensions [height, width, length]
7. **Location**: 3D object location [x, y, z]
8. **Rotation_y**: Rotation around Y-axis [-pi, pi]

### YOLO Label Format

Each line represents one object:

```
0 0.618 0.394 0.186 0.273
```

Fields:
1. **Class ID**: Integer class index (0-7)
2. **Center X**: Normalized center x-coordinate (0-1)
3. **Center Y**: Normalized center y-coordinate (0-1)
4. **Width**: Normalized width (0-1)
5. **Height**: Normalized height (0-1)

## 🛠️ Troubleshooting

### Download Issues

1. **Slow download**: KITTI servers can be slow. Use a download manager.
2. **File corruption**: Verify checksums if provided.
3. **Access denied**: Ensure you're registered and logged in.

### Extraction Issues

1. **Insufficient space**: Dataset requires ~15GB free space.
2. **Permission errors**: Ensure write permissions in data directory.
3. **Path issues**: Use absolute paths for extraction.

### Conversion Issues

1. **Import errors**: Ensure all dependencies are installed.
2. **Missing files**: Verify KITTI directory structure.
3. **Memory issues**: Conversion processes large amounts of data.

### Common Commands

```bash
# Check dataset size
du -sh data/kitti/

# Count images
find data/kitti/training/image_2/ -name "*.png" | wc -l

# Check label files
find data/kitti/training/label_2/ -name "*.txt" | wc -l

# Validate YOLO dataset
python -c "
from src.utils.data_utils import validate_dataset
validate_dataset('data/kitti_yolo')
"
```

## 📚 Additional Resources

- [KITTI Dataset Paper](http://www.cvlibs.net/publications/Geiger2012CVPR.pdf)
- [KITTI Object Detection Benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php)
- [YOLO Format Documentation](https://docs.ultralytics.com/datasets/detect/)

## 📝 Notes

- **License**: KITTI dataset has its own license terms. Please review before use.
- **Citations**: If using KITTI in research, please cite the original papers.
- **Updates**: KITTI dataset may have updates. Check the official website.
- **Variants**: This setup uses the 2D object detection variant of KITTI.

---

For questions about the dataset setup, please refer to the main project README or create an issue.