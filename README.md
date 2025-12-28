# FoundationPose 实时位姿估计

基于 RealSense D435i + YOLOv8 + SAM + FoundationPose 的实时6D位姿估计链路，覆盖「检测 → 掩码 → 注册 → 跟踪」全流程。

---

## 快速开始

### 1. 模型与资源
1. 下载 YOLOv8 权重 (`yolov8n.pt` 或自训练模型)。
2. 下载 SAM ViT 权重 (`sam_vit_h.pth`等) 并在 `mask_generator.py` 中配置路径。
3. 将目标物体 mesh（OBJ/PLY）放入 `models/<class>/`，名称与 `--target_class` 对应。
4. 配置FoundationPose：https://github.com/NVlabs/FoundationPose

### 2. 验证检测+分割链路
```bash
python scripts/test_mask_realtime.py
```

### 3. 运行实时姿态估计
```bash
python scripts/pose_estimation.py \
    --mesh_file models/apple/apple.obj \
    --target_class apple
```
---

## 目录结构

```
scripts/
├── get_K.py              # RealSense 内参 & 深度比例校验
├── mask_generator.py     # YOLO+SAM 掩码生成（缓存/帧跳跃/可视化）
├── pose_estimation.py    # 实时姿态估计主程序
└── test_mask_realtime.py # 检测+分割链路测试
README.md
```
---

## Scripts 功能说明

### 1. get_K.py
**功能**: 打印 RealSense 相机内参矩阵 `K` 与深度比例 `depth_scale`，同时展示彩色/深度图以验证标定。

| 方法 | 说明 |
|------|------|
| `initialize_camera()` | 初始化管线、对齐器及分辨率 |
| `print_camera_info()` | 输出内参矩阵、深度比例 |
| `preview_stream()` | OpenCV 窗口实时预览（按 `Ctrl+C` 关闭脚本） |

### 2. mask_generator.py
**功能**: 封装 YOLO 检测 + SAM 分割，支持缓存、帧跳跃与可选可视化。

| 核心函数 | 说明 |
|----------|------|
| `generate_mask(color_rgb, depth, ...)` | 返回单目标二值掩码，内部自动切换 YOLO+SAM 或 OpenCV 组合策略 |
| `reset_mask_cache()` | 清空历史检测/掩码缓存（切换目标或环境需手动调用） |
| `get_detection_info()` | 获取最近一次检测的置信度/边框/延迟等信息，便于调试面板展示 |

### 3. pose_estimation.py
**功能**: 主流程脚本，负责相机初始化、首帧注册、连续跟踪与可视化输出。

| 模块 | 描述 |
|------|------|
| `initialize_pipeline()` | 配置 RealSense 分辨率、对齐器及深度裁剪 |
| `register_first_frame()` | 使用掩码进行 FoundationPose 注册，输出初始位姿 |
| `track_loop()` | 处理每帧数据，调用 FoundationPose `track_one`，并根据配置做 refine |
| `log_status_panel()` | 打印 FPS、置信度、缓存命中等调试信息 |

### 4. test_mask_realtime.py
**功能**: 轻量级检测/分割测试程序，聚焦 YOLO+SAM 性能评估。

| 特性 | 说明 |
|------|------|
| 帧跳跃控制 | `--sam_frame_skip` 决定 SAM 调用频率 |
| 掩码可视化 | `--visualize True` 时显示检测框、掩码、FPS |
| 历史状态 | 终端实时输出置信度、缓存状态、耗时 |

---

## 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--sam_frame_skip` | 3 | SAM 调用间隔；提升速度但可能降低掩码质量 |
| `--conf_threshold` | 0.12 | YOLO 置信度；降低可更敏感，误检率上升 |
| `--track_refine_iter` | 2 | 跟踪阶段 refine 迭代数；提升稳定性但降低 FPS |
| `--est_refine_iter` | 5 | 首帧注册 refine 迭代数；影响初始精度与耗时 |
| `--detect_method` | `yolo_sam` | 可选 `yolo_sam` 或 `opencv_combined` |
| `--use_cache` | True | 是否复用历史掩码减少 SAM 频次 |

---

## 数据流

```
RealSense D435i (RGB+D) @640x480/30FPS
        ↓ 对齐 + 深度裁剪
mask_generator.generate_mask()
        ↓ (掩码 + 检测信息)
pose_estimation.register_first_frame()
        ↓ FoundationPose register
pose_estimation.track_loop()
        ↓ FoundationPose track_one
日志/可视化 → 状态面板 / OpenCV 窗口
```