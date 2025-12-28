"""实时掩码生成工具（YOLO+SAM / OpenCV退回）"""

import logging
import time

import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO

# Global models
yolo_model = None
sam_predictor = None

# 实时缓存降低 SAM 频率并提升稳定性
_mask_cache = {
    'last_masks': [],
    'last_detections': [],
    'detection_history': [],
    'frame_count': 0,
    'last_mask_time': 0,
    'last_mask_display': None,
}


def load_yolo_model(model_path='yolov8n-seg.pt'):
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO(model_path)
        logging.info(f"Loaded YOLO model: {model_path}")
    return yolo_model


def load_sam_model(model_type='vit_b', checkpoint_path='sam_vit_b_01ec64.pth'):
    global sam_predictor
    if sam_predictor is None:
        import torch

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        if torch.cuda.is_available():
            sam = sam.cuda()
            logging.info(f"Loaded SAM on GPU: {model_type}")
        else:
            logging.info(f"Loaded SAM on CPU: {model_type}")
        sam_predictor = SamPredictor(sam)
    return sam_predictor


def get_opencv_color_mask(color_image, **kwargs):
    """基于 HSV 的颜色掩码"""
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array(kwargs.get('lower_hsv', [0, 50, 50]))
    upper_hsv = np.array(kwargs.get('upper_hsv', [10, 255, 255]))
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(bool)


def get_opencv_depth_mask(depth_image, **kwargs):
    """基于深度阈值的前景掩码"""
    min_depth = kwargs.get('min_depth', 0.2)
    max_depth = kwargs.get('max_depth', 1.5)
    mask = (depth_image > min_depth) & (depth_image < max_depth) & (depth_image > 0)
    mask = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(bool)


def _combine_masks(color_image, depth_image, **kwargs):
    """OpenCV 颜色 + 深度组合"""
    color_mask = get_opencv_color_mask(color_image, **kwargs)
    depth_mask = get_opencv_depth_mask(depth_image, **kwargs)
    combined = color_mask & depth_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    return combined.astype(bool)


def _extract_target_detections(results, target_class):
    """从 YOLO 结果中过滤目标类别"""
    detections = []
    if results and results[0].boxes:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        class_names = results[0].names

        for idx, cls in enumerate(classes):
            class_name = class_names.get(cls, '').lower()
            if target_class.lower() in class_name:
                detections.append({
                    'bbox': boxes[idx],
                    'confidence': confs[idx],
                    'class_name': class_name,
                })
    return detections


def _should_run_sam(frame_id, detections, stable_detection, use_cache, sam_frame_skip):
    """决定是否需要重新调用 SAM"""
    if detections and frame_id % sam_frame_skip == 0:
        return True
    if stable_detection and not detections and _mask_cache['last_masks'] and use_cache:
        return True
    return False


def _visualize(color_image, sam_masks, detections, target_class):
    """绘制检测框与 SAM 掩码"""
    vis_image = color_image.copy()
    if detections:
        for idx, detection in enumerate(detections):
            bbox = detection['bbox']
            conf = detection['confidence']
            cv2.rectangle(
                vis_image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                3,
            )
            if sam_masks and idx < len(sam_masks):
                mask_overlay = np.zeros_like(vis_image)
                mask_overlay[sam_masks[idx]] = [255, 0, 0]
                vis_image = cv2.addWeighted(vis_image, 0.7, mask_overlay, 0.3, 0)

            label = f"{target_class.capitalize()} #{idx + 1}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(
                vis_image,
                (int(bbox[0]), int(bbox[1]) - label_size[1] - 10),
                (int(bbox[0]) + label_size[0], int(bbox[1])),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                vis_image,
                label,
                (int(bbox[0]), int(bbox[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

        info_text = f"{target_class.capitalize()}s: {len(detections)}"
        cv2.putText(
            vis_image,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis_image,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imshow(f'{target_class.capitalize()} Detection', vis_image)

    if sam_masks:
        combined_mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
        for mask in sam_masks:
            combined_mask[mask] = 255
        cv2.imshow('SAM Masks', combined_mask)
        _mask_cache['last_mask_display'] = combined_mask
    elif detections and _mask_cache['last_mask_display'] is not None:
        cv2.imshow('SAM Masks', _mask_cache['last_mask_display'])
    else:
        empty_mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
        cv2.imshow('SAM Masks', empty_mask)

    cv2.waitKey(1)


def generate_mask(color_image, depth_image, method='yolo_sam', **kwargs):
    """生成掩码"""
    if method == 'opencv_combined':
        return _combine_masks(color_image, depth_image, **kwargs)

    if method != 'yolo_sam':
        logging.error(f"未知方法 {method}，自动退回 OpenCV 模式。")
        return _combine_masks(color_image, depth_image, **kwargs)

    try:
        sam_frame_skip = kwargs.get('sam_frame_skip', 3)
        use_cache = kwargs.get('use_cache', True)
        conf_threshold = kwargs.get('conf_threshold', 0.12)
        iou_threshold = kwargs.get('iou_threshold', 0.3)
        target_class = kwargs.get('target_class', 'apple')
        return_all_masks = kwargs.get('return_all_masks', False)
        visualize = kwargs.get('visualize', True)

        yolo = load_yolo_model()
        sam = load_sam_model()

        _mask_cache['frame_count'] += 1
        frame_id = _mask_cache['frame_count']

        yolo_start = time.time()
        results = yolo(
            color_image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            agnostic_nms=True,
        )
        logging.debug("YOLO 耗时 %.3fs", time.time() - yolo_start)

        detections = _extract_target_detections(results, target_class)
        _mask_cache['detection_history'].append(bool(detections))
        if len(_mask_cache['detection_history']) > 10:
            _mask_cache['detection_history'].pop(0)
        stable_detection = sum(_mask_cache['detection_history']) >= 3

        sam_masks = []
        if _should_run_sam(frame_id, detections, stable_detection, use_cache, sam_frame_skip):
            if detections:
                sam_start = time.time()
                sam.set_image(color_image)
                for det in detections:
                    bbox = det['bbox']
                    input_box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
                    masks, _, _ = sam.predict(box=input_box[None, :], multimask_output=False)
                    sam_masks.append(masks[0])
                _mask_cache['last_masks'] = sam_masks
                _mask_cache['last_detections'] = detections
                _mask_cache['last_mask_time'] = time.time() - sam_start
            else:
                sam_masks = _mask_cache['last_masks']
        elif detections and _mask_cache['last_masks'] and use_cache:
            sam_masks = _mask_cache['last_masks']

        if visualize and (sam_masks or detections):
            _visualize(color_image, sam_masks, detections, target_class)

        if return_all_masks:
            return sam_masks if sam_masks else None

        if sam_masks:
            if detections:
                best_idx = np.argmax([d['confidence'] for d in detections])
                best_idx = min(best_idx, len(sam_masks) - 1)
                return sam_masks[best_idx].astype(bool)
            return sam_masks[0].astype(bool)

        logging.warning(f"未检测到 {target_class}，退回 OpenCV 掩码。")
    except Exception as err:
        logging.error("YOLO+SAM 失败: %s，退回 OpenCV 掩码。", err)

    return _combine_masks(color_image, depth_image, **kwargs)


def reset_mask_cache():
    """重置缓存"""
    global _mask_cache
    _mask_cache = {
        'last_masks': [],
        'last_detections': [],
        'detection_history': [],
        'frame_count': 0,
        'last_mask_time': 0,
        'last_mask_display': None,
    }
    logging.info("Mask cache reset")


def get_detection_info():
    """返回当前检测统计信息"""
    history = _mask_cache['detection_history']
    stable = sum(history) >= 3 if history else False
    return {
        'frame_count': _mask_cache['frame_count'],
        'num_detections': len(_mask_cache['last_detections']),
        'stable': stable,
        'sam_time': _mask_cache['last_mask_time'],
    }
