#!/usr/bin/env python3
"""
KITTI Object Detection Inference Script
Run inference on images, videos, or directories using trained model
"""

import argparse
import os
import sys
import cv2
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

from models.detector import KITTIObjectDetector, create_detection_report


def parse_args():
    parser = argparse.ArgumentParser(description='KITTI Object Detection Inference')
    parser.add_argument('--source', type=str, required=True,
                        help='Source: image, video, directory, or webcam (0)')
    parser.add_argument('--weights', type=str, default='yolov8n.pt',
                        help='Path to model weights')
    parser.add_argument('--model', type=str, default='yolov8n',
                        help='Model architecture (if using pretrained)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Inference image size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, 0, 1, etc.)')
    parser.add_argument('--save-dir', type=str, default='runs/predict',
                        help='Directory to save results')
    parser.add_argument('--name', type=str, default='exp',
                        help='Experiment name')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save results')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show results in real-time')
    parser.add_argument('--save-txt', action='store_true', default=False,
                        help='Save results as text files')
    parser.add_argument('--save-json', action='store_true', default=False,
                        help='Save results as JSON file')
    parser.add_argument('--max-det', type=int, default=1000,
                        help='Maximum detections per image')
    parser.add_argument('--line-thickness', type=int, default=2,
                        help='Bounding box line thickness')
    parser.add_argument('--benchmark', action='store_true', default=False,
                        help='Run speed benchmark')
    return parser.parse_args()


def is_image(path):
    """Check if file is an image"""
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    return Path(path).suffix.lower() in img_extensions


def is_video(path):
    """Check if file is a video"""
    vid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    return Path(path).suffix.lower() in vid_extensions


def process_image(detector, image_path, args, save_dir):
    """Process single image"""
    print(f"Processing image: {image_path}")
    
    start_time = time.time()
    
    # Run inference
    results = detector.predict(
        source=image_path,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        save_dir=save_dir
    )
    
    inference_time = time.time() - start_time
    
    # Custom visualization
    if args.save:
        save_path = os.path.join(save_dir, args.name, Path(image_path).name)
        detector.visualize_predictions(image_path, args.conf, args.iou, save_path)
    
    # Show results if requested
    if args.show:
        img = detector.visualize_predictions(image_path, args.conf, args.iou)
        if img is not None:
            cv2.imshow('KITTI Detection', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    print(f"Inference time: {inference_time:.3f}s")
    
    return results, inference_time


def process_directory(detector, dir_path, args, save_dir):
    """Process directory of images"""
    print(f"Processing directory: {dir_path}")
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(dir_path).glob(ext))
        image_files.extend(Path(dir_path).glob(ext.upper()))
    
    if not image_files:
        print("No image files found in directory!")
        return
    
    print(f"Found {len(image_files)} images")
    
    all_results = []
    total_time = 0
    
    for i, img_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {img_path.name}")
        
        start_time = time.time()
        
        # Run inference
        results = detector.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            save_dir=save_dir
        )
        
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Store results for report
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf_score = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(conf_score),
                        'class': detector.class_names[class_id],
                        'class_id': class_id
                    })
        
        all_results.append({
            'image': img_path.name,
            'detections': detections
        })
    
    avg_time = total_time / len(image_files)
    fps = 1.0 / avg_time
    
    print(f"\nProcessing completed!")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average time per image: {avg_time:.3f}s")
    print(f"Average FPS: {fps:.1f}")
    
    # Create detection report
    if args.save_json:
        import json
        report_path = os.path.join(save_dir, args.name, 'detections.json')
        with open(report_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Detection results saved to: {report_path}")
    
    # Create text report
    report_path = os.path.join(save_dir, args.name, 'detection_report.txt')
    create_detection_report(all_results, report_path)
    
    return all_results


def process_video(detector, video_path, args, save_dir):
    """Process video file"""
    print(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer if saving
    if args.save:
        output_path = os.path.join(save_dir, args.name, f"output_{Path(video_path).name}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Save frame temporarily for processing
            temp_frame_path = f"temp_frame_{frame_count}.jpg"
            cv2.imwrite(temp_frame_path, frame)
            
            start_time = time.time()
            
            # Run inference
            processed_img = detector.visualize_predictions(
                temp_frame_path, args.conf, args.iou
            )
            
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Clean up temp file
            os.remove(temp_frame_path)
            
            if processed_img is not None:
                # Save frame
                if args.save:
                    out.write(processed_img)
                
                # Show frame
                if args.show:
                    cv2.imshow('KITTI Video Detection', processed_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                avg_fps = frame_count / total_time if total_time > 0 else 0
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}), "
                      f"Avg FPS: {avg_fps:.1f}")
    
    except KeyboardInterrupt:
        print("\nVideo processing interrupted by user")
    
    finally:
        cap.release()
        if args.save:
            out.release()
        cv2.destroyAllWindows()
    
    avg_time = total_time / frame_count if frame_count > 0 else 0
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\nVideo processing completed!")
    print(f"Processed {frame_count} frames")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average time per frame: {avg_time:.3f}s")
    print(f"Average processing FPS: {avg_fps:.1f}")
    
    if args.save:
        print(f"Output video saved to: {output_path}")


def process_webcam(detector, args, save_dir):
    """Process webcam feed"""
    print("Starting webcam detection (press 'q' to quit)")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    total_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Save frame temporarily for processing
            temp_frame_path = f"temp_webcam_frame.jpg"
            cv2.imwrite(temp_frame_path, frame)
            
            start_time = time.time()
            
            # Run inference
            processed_img = detector.visualize_predictions(
                temp_frame_path, args.conf, args.iou
            )
            
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Clean up temp file
            os.remove(temp_frame_path)
            
            if processed_img is not None:
                # Add FPS counter
                fps = 1.0 / inference_time if inference_time > 0 else 0
                cv2.putText(processed_img, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('KITTI Webcam Detection', processed_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nWebcam detection interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"Webcam session completed: {frame_count} frames, Avg FPS: {avg_fps:.1f}")


def main():
    args = parse_args()
    
    # Create save directory
    save_dir = args.save_dir
    os.makedirs(os.path.join(save_dir, args.name), exist_ok=True)
    
    # Initialize detector
    print(f"Loading model: {args.weights}")
    detector = KITTIObjectDetector(
        model_name=args.model,
        num_classes=8,
        device=args.device
    )
    
    # Load custom weights if provided
    if args.weights != f"{args.model}.pt":
        detector.load_model(args.weights)
    
    # Display model info
    detector.get_model_info()
    
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Device: {args.device}")
    print("=" * 50)
    
    # Determine source type and process
    source = args.source
    
    if source == '0' or source.isdigit():
        # Webcam
        process_webcam(detector, args, save_dir)
    elif os.path.isfile(source):
        if is_image(source):
            # Single image
            process_image(detector, source, args, save_dir)
        elif is_video(source):
            # Video file
            process_video(detector, source, args, save_dir)
        else:
            print(f"Unsupported file format: {source}")
    elif os.path.isdir(source):
        # Directory of images
        process_directory(detector, source, args, save_dir)
    else:
        print(f"Source not found: {source}")
    
    print("Inference completed!")


if __name__ == '__main__':
    main()