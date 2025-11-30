#!/usr/bin/env python3
"""
VR手柄读取模块
用于读取Meta Quest 3右手柄的位姿和按钮状态
"""

import openvr
import numpy as np
from scipy.spatial.transform import Rotation
import json
from pathlib import Path


class VRController:
    """VR手柄控制器类"""

    def __init__(self):
        self.vr_system = None
        self.controller_idx = None
        self.is_initialized = False

        # 标定相关
        self.is_calibrated = False
        self.reference_pose = None  # 4x4变换矩阵

        # 按钮状态
        self.last_trigger_state = False
        self.last_grip_state = False

    def initialize(self):
        """初始化OpenVR并找到右手柄"""
        try:
            openvr.init(openvr.VRApplication_Scene)
            self.vr_system = openvr.VRSystem()
            print("✓ OpenVR初始化成功")
        except openvr.OpenVRError as e:
            print(f"✗ OpenVR初始化失败: {e}")
            return False

        # 查找右手柄
        self.controller_idx = self._find_right_controller()
        if self.controller_idx is None:
            print("✗ 未找到右手柄")
            return False

        print(f"✓ 找到右手柄 (设备索引: {self.controller_idx})")
        self.is_initialized = True
        return True

    def _find_right_controller(self):
        """查找右手柄设备索引"""
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            device_class = self.vr_system.getTrackedDeviceClass(i)
            if device_class == openvr.TrackedDeviceClass_Controller:
                role = self.vr_system.getControllerRoleForTrackedDeviceIndex(i)
                if role == openvr.TrackedControllerRole_RightHand:
                    return i
        return None

    def get_pose(self):
        """
        获取手柄当前位姿

        Returns:
            numpy.ndarray: 4x4齐次变换矩阵，如果追踪丢失返回None
        """
        if not self.is_initialized:
            return None

        poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding,
            0,
            openvr.k_unMaxTrackedDeviceCount
        )

        if not poses[self.controller_idx].bPoseIsValid:
            return None

        # 转换为4x4矩阵
        pose_3x4 = poses[self.controller_idx].mDeviceToAbsoluteTracking
        T = np.eye(4)
        T[0, :] = pose_3x4[0]
        T[1, :] = pose_3x4[1]
        T[2, :] = pose_3x4[2]

        return T

    def get_buttons(self):
        """
        获取按钮状态

        Returns:
            dict: {
                'trigger': bool,  # Trigger是否按下
                'trigger_value': float,  # Trigger模拟值 0-1
                'grip': bool,  # Grip是否按下
                'trigger_pressed': bool,  # Trigger刚按下（边沿检测）
                'trigger_released': bool,  # Trigger刚松开（边沿检测）
            }
        """
        if not self.is_initialized:
            return None

        result, state = self.vr_system.getControllerState(self.controller_idx)
        if not result:
            return None

        # 按钮状态
        trigger_button_id = openvr.k_EButton_SteamVR_Trigger
        grip_button_id = openvr.k_EButton_Grip

        trigger_pressed = (state.ulButtonPressed & (1 << trigger_button_id)) != 0
        grip_pressed = (state.ulButtonPressed & (1 << grip_button_id)) != 0

        # 边沿检测
        trigger_just_pressed = trigger_pressed and not self.last_trigger_state
        trigger_just_released = not trigger_pressed and self.last_trigger_state

        self.last_trigger_state = trigger_pressed
        self.last_grip_state = grip_pressed

        return {
            'trigger': trigger_pressed,
            'trigger_value': state.rAxis[1].x,
            'grip': grip_pressed,
            'trigger_pressed': trigger_just_pressed,
            'trigger_released': trigger_just_released,
        }

    def calibrate(self, current_pose=None):
        """
        标定VR手柄参考位姿

        Args:
            current_pose: 如果为None，则使用当前手柄位姿
        """
        if current_pose is None:
            current_pose = self.get_pose()

        if current_pose is None:
            print("✗ 无法获取手柄位姿，标定失败")
            return False

        self.reference_pose = current_pose.copy()
        self.is_calibrated = True

        position = current_pose[:3, 3]
        print(f"✓ VR手柄标定完成")
        print(f"  参考位置: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] m")
        return True

    def get_relative_pose(self):
        """
        获取相对于标定点的位姿变化

        Returns:
            tuple: (delta_position, delta_rotation_matrix) 或 (None, None)
                - delta_position: 3D位移向量 [x, y, z]
                - delta_rotation_matrix: 3x3旋转矩阵
        """
        if not self.is_calibrated:
            return None, None

        current_pose = self.get_pose()
        if current_pose is None:
            return None, None

        # 计算相对变换: T_rel = T_ref^-1 * T_current
        T_ref_inv = np.linalg.inv(self.reference_pose)
        T_relative = T_ref_inv @ current_pose

        # 提取位移和旋转
        delta_position = T_relative[:3, 3]
        delta_rotation = T_relative[:3, :3]

        return delta_position, delta_rotation

    def save_calibration(self, filename="calibration_data.json"):
        """保存标定数据"""
        if not self.is_calibrated:
            return False

        position = self.reference_pose[:3, 3]
        rotation_matrix = self.reference_pose[:3, :3]
        quaternion = Rotation.from_matrix(rotation_matrix).as_quat()

        data = {
            "vr_reference_position": position.tolist(),
            "vr_reference_quaternion": quaternion.tolist(),
            "vr_reference_matrix": self.reference_pose.tolist(),
            "timestamp": None,  # 由调用者设置
            "coordinate_system": {
                "vr": "OpenVR (X:右, Y:上, Z:后)",
                "maniskill": "需要转换到 (X:前, Y:左, Z:上)"
            }
        }

        filepath = Path(filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return True

    def load_calibration(self, filename="calibration_data.json"):
        """加载标定数据"""
        filepath = Path(filename)
        if not filepath.exists():
            return False

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.reference_pose = np.array(data["vr_reference_matrix"])
            self.is_calibrated = True
            print(f"✓ 加载标定数据: {filepath}")
            return True
        except Exception as e:
            print(f"✗ 加载标定数据失败: {e}")
            return False

    def shutdown(self):
        """关闭OpenVR"""
        if self.is_initialized:
            openvr.shutdown()
            self.is_initialized = False
            print("✓ OpenVR已关闭")
