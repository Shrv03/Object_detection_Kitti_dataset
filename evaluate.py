#!/usr/bin/env python3
"""
KITTI Object Detection Evaluation Script
Evaluate trained model on KITTI test set with comprehensive metrics
"""

import argparse
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append('src')

from models.detector import KITTIObjectDetector


def parse_args():
    parser = argparse.ArgumentParser(description='KITTI Object Detection Evaluation')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default='yolov8n',
                        help='Model architecture')
    parser.add_argument('--conf', type=float, default=0.001,
                        help='Confidence threshold (low for evaluation)')
    parser.add_argument('--iou', type=float, default=0.6,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use')
    parser.add_argument('--save-dir', type=str, default='runs/evaluate',
                        help='Directory to save results')
    parser.add_argument('--name', type=str, default='exp',
                        help='Experiment name')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size')
    parser.add_argument('--plots', action='store_true', default=True,
                        help='Generate evaluation plots')
    parser.add_argument('--save-json', action='store_true', default=True,
                        help='Save results to JSON')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    return parser.parse_args()


def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_detections(pred_results, gt_results, iou_threshold=0.5, class_names=None):
    """
    Evaluate detection results against ground truth
    
    Args:
        pred_results: List of prediction results
        gt_results: List of ground truth results  
        iou_threshold: IoU threshold for matching
        class_names: List of class names
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if class_names is None:
        class_names = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
    
    # Initialize metrics
    metrics = {
        'per_class': {},
        'overall': {},
        'confusion_matrix': np.zeros((len(class_names), len(class_names))),
        'ap_per_class': {},
        'map50': 0.0,
        'map75': 0.0,
        'map50_95': 0.0
    }
    
    # Initialize per-class metrics
    for class_name in class_names:
        metrics['per_class'][class_name] = {
            'tp': 0, 'fp': 0, 'fn': 0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'detections': [], 'ground_truths': []
        }
    
    # Match predictions with ground truth
    for pred_result in pred_results:
        image_name = pred_result['image']
        pred_detections = pred_result['detections']
        
        # Find corresponding ground truth
        gt_result = None
        for gt in gt_results:
            if gt['image'] == image_name:
                gt_result = gt
                break
        
        if gt_result is None:
            # No ground truth for this image, all predictions are false positives
            for detection in pred_detections:
                class_name = detection['class']
                metrics['per_class'][class_name]['fp'] += 1
            continue
        
        gt_detections = gt_result['detections']
        gt_matched = [False] * len(gt_detections)
        
        # Sort predictions by confidence (highest first)
        pred_detections = sorted(pred_detections, key=lambda x: x['confidence'], reverse=True)
        
        # Match predictions to ground truth
        for pred_det in pred_detections:
            pred_class = pred_det['class']
            pred_box = pred_det['bbox']
            pred_conf = pred_det['confidence']
            
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, gt_det in enumerate(gt_detections):
                if gt_matched[gt_idx]:
                    continue
                
                if gt_det['class'] != pred_class:
                    continue
                
                gt_box = gt_det['bbox']
                iou = compute_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Record detection result
            metrics['per_class'][pred_class]['detections'].append({
                'confidence': pred_conf,
                'iou': best_iou,
                'matched': best_iou >= iou_threshold and best_gt_idx >= 0
            })
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # True positive
                metrics['per_class'][pred_class]['tp'] += 1
                gt_matched[best_gt_idx] = True
            else:
                # False positive
                metrics['per_class'][pred_class]['fp'] += 1
        
        # Count false negatives (unmatched ground truth)
        for gt_idx, gt_det in enumerate(gt_detections):
            if not gt_matched[gt_idx]:
                gt_class = gt_det['class']
                metrics['per_class'][gt_class]['fn'] += 1
            
            # Record ground truth
            gt_class = gt_det['class']
            metrics['per_class'][gt_class]['ground_truths'].append(gt_det)
    
    # Compute precision, recall, F1 for each class
    total_tp = total_fp = total_fn = 0
    
    for class_name in class_names:
        tp = metrics['per_class'][class_name]['tp']
        fp = metrics['per_class'][class_name]['fp']
        fn = metrics['per_class'][class_name]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics['per_class'][class_name]['precision'] = precision
        metrics['per_class'][class_name]['recall'] = recall
        metrics['per_class'][class_name]['f1'] = f1
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    metrics['overall'] = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn
    }
    
    return metrics


def generate_evaluation_plots(metrics, save_dir, class_names):
    """Generate evaluation plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Per-class metrics bar plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    classes = list(class_names)
    precisions = [metrics['per_class'][cls]['precision'] for cls in classes]
    recalls = [metrics['per_class'][cls]['recall'] for cls in classes]
    f1s = [metrics['per_class'][cls]['f1'] for cls in classes]
    
    x_pos = np.arange(len(classes))
    
    axes[0].bar(x_pos, precisions, alpha=0.7)
    axes[0].set_title('Precision per Class')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Precision')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(classes, rotation=45)
    axes[0].set_ylim(0, 1)
    
    axes[1].bar(x_pos, recalls, alpha=0.7, color='orange')
    axes[1].set_title('Recall per Class')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Recall')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(classes, rotation=45)
    axes[1].set_ylim(0, 1)
    
    axes[2].bar(x_pos, f1s, alpha=0.7, color='green')
    axes[2].set_title('F1-Score per Class')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(classes, rotation=45)
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detection counts
    fig, ax = plt.subplots(figsize=(12, 8))
    
    tp_counts = [metrics['per_class'][cls]['tp'] for cls in classes]
    fp_counts = [metrics['per_class'][cls]['fp'] for cls in classes]
    fn_counts = [metrics['per_class'][cls]['fn'] for cls in classes]
    
    width = 0.25
    x_pos = np.arange(len(classes))
    
    ax.bar(x_pos - width, tp_counts, width, label='True Positives', alpha=0.7)
    ax.bar(x_pos, fp_counts, width, label='False Positives', alpha=0.7)
    ax.bar(x_pos + width, fn_counts, width, label='False Negatives', alpha=0.7)
    
    ax.set_title('Detection Counts per Class')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'detection_counts.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Overall metrics pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    
    overall = metrics['overall']
    sizes = [overall['total_tp'], overall['total_fp'], overall['total_fn']]
    labels = ['True Positives', 'False Positives', 'False Negatives']
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Overall Detection Distribution')
    
    plt.savefig(os.path.join(save_dir, 'overall_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation plots saved to {save_dir}")


def create_evaluation_report(metrics, args, save_path):
    """Create detailed evaluation report"""
    class_names = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
    
    with open(save_path, 'w') as f:
        f.write("KITTI Object Detection Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        
        # Configuration
        f.write("Configuration:\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Weights: {args.weights}\n")
        f.write(f"Confidence threshold: {args.conf}\n")
        f.write(f"IoU threshold: {args.iou}\n")
        f.write(f"Image size: {args.img_size}\n")
        f.write(f"Device: {args.device}\n\n")
        
        # Overall metrics
        overall = metrics['overall']
        f.write("Overall Performance:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Precision: {overall['precision']:.4f}\n")
        f.write(f"Recall: {overall['recall']:.4f}\n")
        f.write(f"F1-Score: {overall['f1']:.4f}\n")
        f.write(f"Total True Positives: {overall['total_tp']}\n")
        f.write(f"Total False Positives: {overall['total_fp']}\n")
        f.write(f"Total False Negatives: {overall['total_fn']}\n\n")
        
        # Per-class metrics
        f.write("Per-Class Performance:\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TP':<5} {'FP':<5} {'FN':<5}\n")
        f.write("-" * 70 + "\n")
        
        for class_name in class_names:
            cls_metrics = metrics['per_class'][class_name]
            f.write(f"{class_name:<15} {cls_metrics['precision']:<10.4f} "
                   f"{cls_metrics['recall']:<10.4f} {cls_metrics['f1']:<10.4f} "
                   f"{cls_metrics['tp']:<5} {cls_metrics['fp']:<5} {cls_metrics['fn']:<5}\n")
        
        f.write("\n")
        
        # Class statistics
        f.write("Class Statistics:\n")
        f.write("-" * 30 + "\n")
        for class_name in class_names:
            cls_metrics = metrics['per_class'][class_name]
            num_gt = len(cls_metrics['ground_truths'])
            num_det = len(cls_metrics['detections'])
            f.write(f"{class_name}: {num_gt} ground truth objects, {num_det} detections\n")
    
    print(f"Evaluation report saved to {save_path}")


def main():
    args = parse_args()
    
    # Create save directory
    save_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize detector
    print(f"Loading model: {args.weights}")
    detector = KITTIObjectDetector(
        model_name=args.model,
        num_classes=8,
        device=args.device
    )
    detector.load_model(args.weights)
    
    print("Running YOLO validation...")
    # Run built-in YOLO validation
    yolo_results = detector.validate(
        data_yaml=args.data,
        img_size=args.img_size,
        save_dir=save_dir
    )
    
    print(f"Built-in validation completed!")
    
    # Additional custom evaluation would require ground truth labels
    # For now, we'll create a summary report of the YOLO results
    
    # Create summary report
    report_path = os.path.join(save_dir, 'evaluation_summary.txt')
    with open(report_path, 'w') as f:
        f.write("KITTI Object Detection Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Weights: {args.weights}\n")
        f.write(f"Dataset: {args.data}\n")
        f.write(f"Image size: {args.img_size}\n")
        f.write(f"Confidence threshold: {args.conf}\n")
        f.write(f"IoU threshold: {args.iou}\n\n")
        
        if hasattr(yolo_results, 'box') and yolo_results.box is not None:
            f.write("YOLO Validation Results:\n")
            f.write("-" * 30 + "\n")
            box_metrics = yolo_results.box
            
            # Check if metrics exist and write them
            if hasattr(box_metrics, 'map'):
                f.write(f"mAP@0.5:0.95: {box_metrics.map:.4f}\n")
            if hasattr(box_metrics, 'map50'):
                f.write(f"mAP@0.5: {box_metrics.map50:.4f}\n")
            if hasattr(box_metrics, 'map75'):
                f.write(f"mAP@0.75: {box_metrics.map75:.4f}\n")
            if hasattr(box_metrics, 'mp'):
                f.write(f"Mean Precision: {box_metrics.mp:.4f}\n")
            if hasattr(box_metrics, 'mr'):
                f.write(f"Mean Recall: {box_metrics.mr:.4f}\n")
        
        f.write(f"\nEvaluation completed at: {save_dir}\n")
    
    print(f"Evaluation summary saved to: {report_path}")
    print("Evaluation completed!")
    
    # Print key metrics
    if hasattr(yolo_results, 'box') and yolo_results.box is not None:
        box_metrics = yolo_results.box
        print("\nKey Metrics:")
        print("-" * 20)
        if hasattr(box_metrics, 'map'):
            print(f"mAP@0.5:0.95: {box_metrics.map:.4f}")
        if hasattr(box_metrics, 'map50'):
            print(f"mAP@0.5: {box_metrics.map50:.4f}")
        if hasattr(box_metrics, 'map75'):
            print(f"mAP@0.75: {box_metrics.map75:.4f}")


if __name__ == '__main__':
    main()