"""查看相机内参矩阵与深度比例，获取K矩阵"""

import cv2
import numpy as np
import pyrealsense2 as rs


pipeline = rs.pipeline()
config = rs.config()
for stream, fmt in [
    (rs.stream.color, rs.format.bgr8),
    (rs.stream.depth, rs.format.z16),
]:
    config.enable_stream(stream, 640, 480, fmt, 30)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array(
    [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
    dtype=float,
)
print("K=\n", K, " depth_scale=", depth_scale)

try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data()) * depth_scale  # 单位：米

        cv2.imshow("color", color)
        cv2.imshow("depth(m)", (depth / 5.0).astype("float32"))

        cv2.waitKey(1)
except KeyboardInterrupt:
    print("手动中断")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
