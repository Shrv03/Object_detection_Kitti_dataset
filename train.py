#!/usr/bin/env python3
"""
KITTI Object Detection Training Script
Train YOLOv8 model on KITTI dataset for self-driving car applications
"""

import argparse
import os
import yaml
import torch
from datetime import datetime
import sys

# Add src to path
sys.path.append('src')

from models.detector import KITTIObjectDetector
from utils.data_utils import setup_kitti_dataset, analyze_dataset, validate_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train KITTI Object Detection Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to dataset YAML file (overrides config)')
    parser.add_argument('--model', type=str, default='yolov8n',
                        help='YOLOv8 model size (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, 0, 1, etc.)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--save-dir', type=str, default='runs/train',
                        help='Directory to save training results')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--validate', action='store_true', default=False,
                        help='Validate dataset before training')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment(args, config):
    """Setup experiment directories and logging"""
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"kitti_{args.model}_{timestamp}"
    
    experiment_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    config_save_path = os.path.join(experiment_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save arguments
    args_save_path = os.path.join(experiment_dir, 'args.yaml')
    with open(args_save_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    print(f"Experiment directory: {experiment_dir}")
    return experiment_dir


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Creating default configuration...")
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        # Create default config if not exists
        default_config = {
            'dataset': {
                'name': 'kitti',
                'root_dir': 'data/kitti',
                'classes': ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc'],
                'img_size': [640, 640]
            },
            'model': {
                'name': 'yolov8n',
                'num_classes': 8,
                'pretrained': True
            },
            'training': {
                'epochs': 100,
                'batch_size': 16,
                'learning_rate': 0.01,
                'patience': 50
            }
        }
        with open(args.config, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.data is not None:
        data_yaml = args.data
    else:
        # Use dataset from config or default
        data_yaml = config.get('dataset', {}).get('yaml_path', 'data/sample_kitti/dataset.yaml')
    
    # Check if dataset exists
    if not os.path.exists(data_yaml):
        print(f"Dataset YAML not found: {data_yaml}")
        print("Creating sample dataset for testing...")
        from utils.data_utils import create_sample_dataset
        data_yaml = create_sample_dataset()
    
    # Validate dataset if requested
    if args.validate:
        dataset_dir = os.path.dirname(data_yaml)
        if not validate_dataset(dataset_dir):
            print("Dataset validation failed. Please fix the issues before training.")
            return
        
        # Analyze dataset
        print("\nAnalyzing dataset...")
        analyze_dataset(dataset_dir)
    
    # Setup experiment
    experiment_dir = setup_experiment(args, config)
    
    # Initialize model
    print(f"Initializing {args.model} model...")
    detector = KITTIObjectDetector(
        model_name=args.model,
        num_classes=config['model']['num_classes'],
        device=args.device
    )
    
    # Display model info
    detector.get_model_info()
    
    # Training parameters
    train_params = {
        'epochs': config['training']['epochs'],
        'batch': config['training']['batch_size'],
        'imgsz': args.img_size,
        'lr0': config['training'].get('learning_rate', 0.01),
        'patience': config['training'].get('patience', 50),
        'workers': args.workers,
        'seed': args.seed,
        'project': args.save_dir,
        'name': args.name,
        'save_period': config['training'].get('save_period', 10),
        'cache': True,  # Cache images for faster training
        'device': args.device
    }
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        detector.load_model(args.resume)
    
    print("Starting training...")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {train_params['epochs']}")
    print(f"Batch size: {train_params['batch']}")
    print(f"Image size: {args.img_size}")
    print(f"Device: {args.device}")
    print(f"Experiment: {args.name}")
    print("=" * 50)
    
    # Start training
    try:
        results = detector.train(data_yaml, **train_params)
        print("Training completed successfully!")
        
        # Save final results
        results_path = os.path.join(experiment_dir, 'training_results.txt')
        with open(results_path, 'w') as f:
            f.write("KITTI Object Detection Training Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Dataset: {data_yaml}\n")
            f.write(f"Final mAP50: {getattr(results, 'box', {}).get('map50', 'N/A')}\n")
            f.write(f"Final mAP50-95: {getattr(results, 'box', {}).get('map', 'N/A')}\n")
            f.write(f"Training time: {getattr(results, 'speed', {}).get('train', 'N/A')} ms/img\n")
            f.write(f"Validation time: {getattr(results, 'speed', {}).get('val', 'N/A')} ms/img\n")
        
        print(f"Results saved to: {results_path}")
        
        # Run validation
        print("\nRunning final validation...")
        val_results = detector.validate(data_yaml, img_size=args.img_size, 
                                      save_dir=os.path.join(args.save_dir, f"{args.name}_val"))
        
        print("Training pipeline completed!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()