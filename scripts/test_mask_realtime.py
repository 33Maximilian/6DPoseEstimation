"""YOLO+SAM 掩码演示"""

import sys

sys.path.append('./FoundationPose')

import logging
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)


def load_yolo_model(model_path='yolov8n-seg.pt'):
    """加载 YOLO 模型"""
    return YOLO(model_path)


def load_sam_model(model_type="vit_b", checkpoint_path="sam_vit_b_01ec64.pth"):
    """加载 SAM 模型"""
    import torch
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    if torch.cuda.is_available():
        sam = sam.cuda()
    return SamPredictor(sam)


def visualize_apple_detection_with_mask(rgb, yolo_results, sam_masks):
    """绘制苹果目标的检测框与 SAM 掩码"""
    vis_image = rgb.copy()
    apple_count = 0

    if yolo_results and yolo_results[0].boxes:
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        confs = yolo_results[0].boxes.conf.cpu().numpy()
        classes = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
        class_names = yolo_results[0].names

        apple_idx = 0
        for i, cls in enumerate(classes):
            class_name = class_names.get(cls, '').lower()
            if class_name == 'apple':
                box = boxes[i]
                conf = confs[i]
                apple_count += 1
                
                cv2.rectangle(vis_image, 
                            (int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), 
                            (0, 255, 0), 3)
                
                if sam_masks and apple_idx < len(sam_masks):
                    mask = sam_masks[apple_idx]
                    mask_overlay = np.zeros_like(vis_image)
                    mask_overlay[mask] = [255, 0, 0]  # Red mask
                    vis_image = cv2.addWeighted(vis_image, 0.7, mask_overlay, 0.3, 0)
                
                label = f'Apple #{apple_count}: {conf:.2f}'
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(vis_image,
                            (int(box[0]), int(box[1]) - label_size[1] - 10),
                            (int(box[0]) + label_size[0], int(box[1])),
                            (0, 255, 0), -1)
                cv2.putText(vis_image, label, 
                          (int(box[0]), int(box[1]) - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                apple_idx += 1

    info_text = f'Apples detected: {apple_count}'
    cv2.putText(vis_image, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis_image, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

    return vis_image, apple_count

def get_apple_detections(yolo_results):
    """筛选苹果目标"""
    apple_detections = []
    
    if yolo_results and yolo_results[0].boxes:
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        confs = yolo_results[0].boxes.conf.cpu().numpy()
        classes = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
        class_names = yolo_results[0].names

        for i, cls in enumerate(classes):
            class_name = class_names.get(cls, '').lower()
            if 'apple' in class_name:
                apple_detections.append({
                    'bbox': boxes[i],
                    'confidence': confs[i],
                    'class_name': 'apple'
                })
    
    return apple_detections

def _stable_detection(history, window=10, threshold=3):
    """判定是否存在稳定检测"""
    if len(history) > window:
        history = history[-window:]
    return sum(history) >= threshold


def _update_mask_window(image_shape, sam_masks, last_mask_display):
    """统一处理 SAM 掩码窗口"""
    if sam_masks:
        combined = np.zeros(image_shape[:2], dtype=np.uint8)
        for mask in sam_masks:
            combined[mask] = 255
        cv2.imshow('SAM Masks', combined)
        return combined

    if last_mask_display is not None:
        cv2.imshow('SAM Masks', last_mask_display)
        return last_mask_display

    empty = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.imshow('SAM Masks', empty)
    return None


def _print_frame_report(frame_idx, yolo_time, sam_time, total_time, fps, apple_count, apple_detections):
    """输出统计信息"""
    print(
        f"帧 {frame_idx:4d}: YOLO {yolo_time:.3f}s | SAM {sam_time:.3f}s | "
        f"总耗时 {total_time:.3f}s | FPS {fps:.1f} | 苹果 {apple_count}"
    )
    if apple_detections:
        for idx, detection in enumerate(apple_detections, start=1):
            bbox = detection['bbox']
            conf = detection['confidence']
            print(
                f"  - 目标 {idx}: 置信度={conf:.3f}, "
                f"bbox=[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
            )


def main():
    print("Loading YOLO model...")
    yolo = load_yolo_model()
    print("Loading SAM model...")
    sam_predictor = load_sam_model()
    print("Models loaded successfully!")

    # 初始化 RealSense
    print("Initializing RealSense camera...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        print("RealSense camera initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize RealSense: {e}")
        return

    frame_count = 0
    last_masks = []
    sam_frame_skip = 3
    detection_history = []
    last_mask_display = None

    try:
        print("开始实时检测")
        
        while True:
            start_time = time.time()

            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            bgr = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            frame_count += 1

            yolo_start = time.time()
            yolo_results = yolo(rgb, conf=0.12, iou=0.3, verbose=False, agnostic_nms=True)
            yolo_time = time.time() - yolo_start

            apple_detections = get_apple_detections(yolo_results)
            
            detection_history.append(len(apple_detections) > 0)
            if len(detection_history) > 10:
                detection_history.pop(0)
            stable_detection = _stable_detection(detection_history)
            
            sam_masks = []
            sam_time = 0
            
            if (apple_detections and frame_count % sam_frame_skip == 0) or (stable_detection and not apple_detections and last_masks):
                if apple_detections:
                    sam_start = time.time()
                    sam_predictor.set_image(rgb)
                    
                    for detection in apple_detections:
                        bbox = detection['bbox']
                        input_box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
                        masks, _, _ = sam_predictor.predict(box=input_box[None, :], multimask_output=False)
                        sam_masks.append(masks[0])
                    
                    sam_time = time.time() - sam_start
                    last_masks = sam_masks
                else:
                    sam_masks = last_masks
            elif apple_detections and last_masks:
                sam_masks = last_masks
            
            vis_image, apple_count = visualize_apple_detection_with_mask(rgb, yolo_results, sam_masks)

            last_mask_display = _update_mask_window(rgb.shape, sam_masks, last_mask_display)

            total_time = time.time() - start_time
            fps = 1.0 / total_time if total_time > 0 else 0
            _print_frame_report(frame_count, yolo_time, sam_time, total_time, fps, apple_count, apple_detections)

            cv2.imshow('Apple Detection with SAM Masks', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

            cv2.waitKey(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped and windows closed")

if __name__ == '__main__':
    main()
