#!/usr/bin/env python3
"""
相机管理模块 - 支持RealSense D435深度相机
"""

import numpy as np
import cv2


class CameraManager:
    """
    RealSense D435相机管理器

    支持2个D435相机同时采集RGB图像
    """

    def __init__(self, num_cameras=2, width=640, height=480, fps=30):
        """
        初始化RealSense D435相机管理器

        Args:
            num_cameras: 相机数量 (默认2个: wrist + front)
            width: 图像宽度 (默认640)
            height: 图像高度 (默认480)
            fps: 帧率 (默认30fps, 需要USB 3.0接口)
        """
        self.num_cameras = num_cameras
        self.width = width
        self.height = height
        self.fps = fps

        # 相机名称映射
        # 如果只有1个相机，只用wrist（手腕视角）
        self.camera_names = ['wrist', 'front'][:num_cameras]

        # RealSense pipeline列表
        self.pipelines = []

        # 初始化RealSense相机
        self._initialize_realsense_cameras()

    def _initialize_realsense_cameras(self):
        """初始化RealSense D435相机"""
        try:
            import pyrealsense2 as rs

            # 检测所有连接的RealSense设备
            ctx = rs.context()
            devices = ctx.query_devices()

            if len(devices) == 0:
                raise RuntimeError(
                    "❌ 未检测到RealSense D435设备！\n"
                    "   请检查:\n"
                    "   1. D435是否通过USB 3.0连接 (蓝色接口)\n"
                    "   2. 运行 'lsusb | grep Intel' 确认设备识别\n"
                    "   3. 运行 'rs-enumerate-devices' 测试RealSense SDK\n"
                )

            print(f"\n{'='*60}")
            print(f"检测到 {len(devices)} 个RealSense设备")

            # 初始化每个相机
            num_to_init = min(self.num_cameras, len(devices))
            for i in range(num_to_init):
                device = devices[i]
                serial = device.get_info(rs.camera_info.serial_number)
                name = device.get_info(rs.camera_info.name)

                # 创建pipeline和config
                pipeline = rs.pipeline()
                config = rs.config()

                # 指定设备序列号
                config.enable_device(serial)

                # 只启用RGB流 (不用深度，节省带宽)
                config.enable_stream(
                    rs.stream.color,
                    self.width,
                    self.height,
                    rs.format.bgr8,
                    self.fps
                )

                # 启动pipeline
                try:
                    pipeline.start(config)
                    self.pipelines.append(pipeline)
                    print(f"✓ 相机{i} [{self.camera_names[i]}]: {name} (S/N: {serial})")
                    print(f"  分辨率: {self.width}x{self.height} @ {self.fps}fps")
                except Exception as e:
                    print(f"✗ 相机{i}启动失败: {e}")

            print(f"{'='*60}\n")

            if len(self.pipelines) == 0:
                raise RuntimeError(
                    "❌ 所有RealSense相机启动失败！\n"
                    "   请检查是否有其他程序占用相机\n"
                )

        except ImportError:
            raise ImportError(
                "❌ pyrealsense2库未安装！\n"
                "   安装方法:\n"
                "   pip install pyrealsense2\n"
            )

    def get_frames(self):
        """
        同步获取所有相机的当前帧

        Returns:
            dict: {
                'wrist': np.ndarray (H, W, 3) BGR格式,
                'front': np.ndarray (H, W, 3) BGR格式
            }
            如果某个相机读取失败，返回黑色图像
        """
        frames = {}

        # RealSense模式
        for i, pipeline in enumerate(self.pipelines):
            camera_name = self.camera_names[i]
            try:
                # 等待帧 (超时1秒)
                frameset = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frameset.get_color_frame()

                if color_frame:
                    # 转换为numpy数组 (BGR格式)
                    frame = np.asanyarray(color_frame.get_data())
                    frames[camera_name] = frame
                else:
                    # 读取失败，使用黑色图像
                    print(f"⚠ 相机{i} [{camera_name}] 未能获取帧")
                    frames[camera_name] = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            except Exception as e:
                print(f"⚠ 相机{i} [{camera_name}] 读取失败: {e}")
                frames[camera_name] = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        return frames

    def get_num_cameras(self):
        """获取实际可用的相机数量"""
        return len(self.pipelines)

    def release(self):
        """释放所有RealSense相机资源"""
        for pipeline in self.pipelines:
            try:
                pipeline.stop()
            except:
                pass
        self.pipelines.clear()
        print("✓ 相机资源已释放")

    def __del__(self):
        """析构函数，确保资源释放"""
        self.release()


# 测试代码
if __name__ == '__main__':
    import time

    print("测试CameraManager...")

    # 创建相机管理器
    cam_manager = CameraManager(num_cameras=2, fps=15)

    print(f"\n可用相机数: {cam_manager.get_num_cameras()}")

    # 测试采集10帧
    print("\n测试采集10帧...")
    for i in range(10):
        start = time.time()
        frames = cam_manager.get_frames()
        elapsed = time.time() - start

        print(f"帧{i}: ", end='')
        for name, frame in frames.items():
            print(f"{name}={frame.shape} ", end='')
        print(f"耗时={elapsed*1000:.1f}ms")

        time.sleep(0.067)  # 15fps

    # 释放资源
    cam_manager.release()
    print("\n测试完成！")
