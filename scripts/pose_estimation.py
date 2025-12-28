#!/usr/bin/env python3
"""姿态估计入口脚本"""

import argparse
import logging
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh

from datareader import *
from estimater import *
from realsense_utils import cleanup_camera, get_frames, initialize_camera
from scripts.mask_generator import generate_mask, get_detection_info, reset_mask_cache

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MESH = Path('models/apple/apple.obj')
DEFAULT_DEBUG_DIR = Path('debug_realtime')


def to_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def rel_display(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def prepare_debug_dir(debug_dir: Path):
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    (debug_dir / 'track_vis').mkdir(parents=True, exist_ok=True)
    (debug_dir / 'ob_in_cam').mkdir(parents=True, exist_ok=True)


def draw_detection_info(image, detection_info, fps, pose_valid, frame_idx):
    """画面左上角绘制实时状态信息"""
    img = image.copy()
    h, w = img.shape[:2]
    
    # 信息面板背景
    cv2.rectangle(img, (10, 10), (400, 150), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 10), (400, 150), (0, 255, 0), 2)
    
    # 信息条目
    info_lines = [
        f"Frame: {frame_idx}",
        f"FPS: {fps:.1f}",
        f"Detections: {detection_info.get('num_detections', 0)}",
        f"Stable: {'Yes' if detection_info.get('stable', False) else 'No'}",
        f"SAM Time: {detection_info.get('sam_time', 0):.3f}s",
        f"Pose: {'Valid' if pose_valid else 'Lost'}"
    ]
    
    y_offset = 30
    for line in info_lines:
        cv2.putText(img, line, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y_offset += 20
    
    return img


def main():
    parser = argparse.ArgumentParser(description='Real-time 6D Pose Estimation with Optimized Pipeline')
    
    # 模型与数据配置
    parser.add_argument('--mesh_file', type=str, default=str(DEFAULT_MESH),
                       help='Path to object mesh file (.obj)')
    parser.add_argument('--target_class', type=str, default='apple',
                       help='Target object class for YOLO detection')
    
    # 姿态估计参数
    parser.add_argument('--est_refine_iter', type=int, default=5,
                       help='Refinement iterations for initial pose registration')
    parser.add_argument('--track_refine_iter', type=int, default=2,
                       help='Refinement iterations for pose tracking')
    
    # 实时优化参数
    parser.add_argument('--sam_frame_skip', type=int, default=3,
                       help='Process SAM every N frames (higher = faster but less accurate)')
    parser.add_argument('--conf_threshold', type=float, default=0.12,
                       help='YOLO confidence threshold (lower = more detections)')
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                       help='YOLO IoU threshold for NMS')
    parser.add_argument('--use_cache', action='store_true', default=True,
                       help='Use mask caching for performance')
    
    # 掩码策略
    parser.add_argument('--detect_method', type=str, default='yolo_sam',
                       choices=['opencv_combined', 'yolo_sam'],
                       help='Mask generation method')
    
    # 调试与可视化
    parser.add_argument('--debug', type=int, default=1,
                       help='Debug level: 0=none, 1=display, 2=save')
    parser.add_argument('--debug_dir', type=str, default=str(DEFAULT_DEBUG_DIR),
                       help='Directory to save debug outputs')
    parser.add_argument('--save_video', action='store_true',
                       help='Save output video')
    
    args = parser.parse_args()

    # Setup logging
    set_logging_format()
    set_seed(0)
    logging.info("="*60)
    logging.info("Real-time 6D Pose Estimation with Optimized Pipeline")
    logging.info("="*60)

    # 加载网格并计算包围盒
    mesh_path = to_project_path(args.mesh_file)
    logging.info(f"Loading mesh: {rel_display(mesh_path)}")
    mesh = trimesh.load(mesh_path)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
    logging.info(f"Mesh loaded. Extents: {extents}")

    # 初始化 FoundationPose 推理器
    logging.info("Initializing FoundationPose estimator...")
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=args.debug_dir,
        debug=args.debug,
        glctx=glctx
    )
    logging.info("Estimator initialized successfully!")

    # 初始化 RealSense 相机
    logging.info("Initializing RealSense camera...")
    try:
        pipeline, align, K, depth_scale = initialize_camera()
        logging.info(f"Camera intrinsics K:\n{K}")
        logging.info(f"Depth scale: {depth_scale}")
    except Exception as e:
        logging.error(f"Failed to initialize camera: {e}")
        return

    # 配置调试目录
    debug_dir = to_project_path(args.debug_dir)
    logging.info(f"Debug outputs: {rel_display(debug_dir)}")
    prepare_debug_dir(debug_dir)

    # 清空掩码缓存
    reset_mask_cache()

    # 视频写入器
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str((debug_dir / 'output.mp4')), fourcc, 30.0, (640, 480)
        )

    # 主循环
    frame_idx = 0
    pose = None
    pose_initialized = False
    last_valid_pose = None
    fps_history = []
    
    try:
        logging.info("="*60)
        logging.info("Starting real-time tracking loop...")
        logging.info("Use Ctrl+C to stop the session when needed")
        logging.info("="*60)
        
        while True:
            loop_start = time.time()
            
            # 拉取相机帧
            color_bgr, depth = get_frames(pipeline, align, depth_scale)
            if color_bgr is None or depth is None:
                continue
            
            # 转换为 RGB
            color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            
            # 首帧或丢失时重新生成掩码
            if frame_idx == 0 or not pose_initialized:
                logging.info(f"Frame {frame_idx}: Generating initial mask...")
                mask_start = time.time()
                
                try:
                    mask = generate_mask(
                        color,
                        depth,
                        method=args.detect_method,
                        target_class=args.target_class,
                        sam_frame_skip=args.sam_frame_skip,
                        conf_threshold=args.conf_threshold,
                        iou_threshold=args.iou_threshold,
                        use_cache=args.use_cache,
                        visualize=False,
                    )
                except Exception as e:
                    logging.error(f"Mask generation failed: {e}")
                    frame_idx += 1
                    continue
                
                mask_time = time.time() - mask_start
                
                if mask is None or np.sum(mask) == 0:
                    logging.warning(f"Frame {frame_idx}: No valid mask, skipping...")
                    frame_idx += 1
                    continue
                
                logging.info(f"Mask generated in {mask_time:.3f}s, pixels: {np.sum(mask)}")
                
                # 首帧注册
                logging.info("Registering initial pose...")
                reg_start = time.time()
                try:
                    pose = est.register(
                        K=K,
                        rgb=color,
                        depth=depth,
                        ob_mask=mask,
                        iteration=args.est_refine_iter
                    )
                except Exception as e:
                    logging.error(f"Pose registration failed: {e}", exc_info=True)
                    frame_idx += 1
                    continue
                
                reg_time = time.time() - reg_start
                
                if pose is None:
                    logging.warning(f"Failed to register pose, retrying...")
                    frame_idx += 1
                    continue
                
                pose_initialized = True
                last_valid_pose = pose.copy()
                logging.info(f"Pose registered in {reg_time:.3f}s")
                logging.info(f"Initial pose:\\n{pose}")
            
            else:
                # 后续帧使用 track_one
                track_start = time.time()
                try:
                    pose = est.track_one(
                        rgb=color,
                        depth=depth,
                        K=K,
                        iteration=args.track_refine_iter
                    )
                except Exception as e:
                    logging.error(f"Tracking failed: {e}")
                    pose = None
                
                track_time = time.time() - track_start
                
                if pose is not None:
                    last_valid_pose = pose.copy()
                else:
                    # 追踪失败时回退到上一帧
                    pose = last_valid_pose
                    logging.warning(f"Frame {frame_idx}: Tracking lost, using last valid pose")
            
            # 保存姿态矩阵
            if pose is not None:
                ob_path = debug_dir / 'ob_in_cam' / f'{frame_idx:06d}.txt'
                np.savetxt(ob_path, pose.reshape(4, 4))
            
            # 可视化
            vis_image = color.copy()
            pose_valid = pose is not None
            
            if args.debug >= 1 and pose is not None:
                # 绘制 3D 盒与坐标轴
                center_pose = pose @ np.linalg.inv(to_origin)
                vis_image = draw_posed_3d_box(K, img=vis_image, ob_in_cam=center_pose, bbox=bbox)
                vis_image = draw_xyz_axis(
                    vis_image, ob_in_cam=center_pose, scale=0.1, K=K,
                    thickness=3, transparency=0, is_input_rgb=True
                )
            
            # 覆盖性能信息
            detection_info = get_detection_info()
            loop_time = time.time() - loop_start
            fps = 1.0 / loop_time if loop_time > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)
            
            vis_image = draw_detection_info(
                vis_image, detection_info, avg_fps, pose_valid, frame_idx
            )
            
            # 转回 BGR 再显示
            vis_bgr = vis_image[...,::-1]
            cv2.imshow('FoundationPose - Real-time 6D Pose Estimation', vis_bgr)
            
            # 若 debug>=2 则存图
            if args.debug >= 2:
                track_path = debug_dir / 'track_vis' / f'{frame_idx:06d}.png'
                cv2.imwrite(str(track_path), vis_bgr)
            
            # 写入视频
            if video_writer is not None:
                video_writer.write(vis_bgr)
            
            # 记录帧日志
            if frame_idx % 10 == 0:  # 每 10 帧打印一次
                logging.info(
                    f"Frame {frame_idx:4d}: FPS {avg_fps:.1f} | "
                    f"Detections: {detection_info.get('num_detections', 0)} | "
                    f"Stable: {detection_info.get('stable', False)} | "
                    f"Pose: {'Valid' if pose_valid else 'Lost'}"
                )
                
            cv2.waitKey(1)
            
            frame_idx += 1
    
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    
    except Exception as e:
        logging.error(f"Error in main loop: {e}", exc_info=True)
    
    finally:
        # 资源回收
        cleanup_camera(pipeline)
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
        
        logging.info("="*60)
        logging.info(f"Session summary:")
        logging.info(f"  Total frames processed: {frame_idx}")
        logging.info(f"  Average FPS: {np.mean(fps_history):.1f}" if fps_history else "  No FPS data")
        logging.info(f"  Output saved to: {rel_display(debug_dir)}")
        logging.info("="*60)
        logging.info("Camera stopped and session ended")


if __name__ == '__main__':
    main()
