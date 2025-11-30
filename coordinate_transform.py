#!/usr/bin/env python3
"""
VR到ManiSkill坐标系转换工具
处理从VR坐标系到ManiSkill/SAPIEN坐标系的转换
"""

import numpy as np
from scipy.spatial.transform import Rotation

# sapien是可选依赖，只在实际控制ManiSkill时需要
try:
    import sapien
    HAS_SAPIEN = True
except ImportError:
    HAS_SAPIEN = False


class CoordinateTransformer:
    """
    VR坐标系到ManiSkill坐标系的转换器

    坐标系定义:
    VR (OpenVR):       X右, Y上, Z后
    ManiSkill/SAPIEN:  X前, Y左, Z上
    """

    def __init__(self, scale_factor=1.0, invert_y=False):
        """
        Args:
            scale_factor: 缩放因子，VR空间距离到真实空间的比例
                         例如: 2.0 表示VR中移动1m对应真实空间0.5m
                              0.5 表示VR中移动1m对应真实空间2m
            invert_y: 是否反转Y轴方向（默认False，保持原有行为）
        """
        self.scale_factor = scale_factor
        self.invert_y = invert_y

        # VR到ManiSkill的位置转换矩阵
        # VR:  [X右, Y上, Z后]
        # MS:  [X前, Y左, Z上]
        y_sign = 1 if invert_y else -1
        # 映射: VR_X右 -> MS_Y, VR_Y上 -> MS_Z上, VR_Z后 -> MS_X前(负)
        self.position_transform_matrix = np.array([
            [0, 0, -1],        # MS_X = -VR_Z (VR后方 -> MS前方)
            [y_sign, 0, 0],    # MS_Y = ±VR_X (可配置方向)
            [0, 1, 0]          # MS_Z = VR_Y  (VR上方 -> MS上方)
        ])

        # 旋转转换也需要相应的轴变换
        # 用四元数表示坐标系旋转
        # 从VR坐标系旋转到ManiSkill坐标系
        # 这是一个90度绕多个轴的组合旋转
        self._rotation_transform = self._compute_rotation_transform()

    def _compute_rotation_transform(self):
        """
        计算VR坐标系到ManiSkill坐标系的旋转变换

        VR坐标系到MS坐标系需要:
        1. 绕Y轴旋转180度 (Z后->Z前)
        2. 绕Z轴旋转±90度 (X右->Y左/右，取决于invert_y)
        """
        # R = Rz(±90°) @ Ry(180°)
        Ry_180 = Rotation.from_euler('y', 180, degrees=True)
        # 根据invert_y决定旋转方向
        z_angle = 90 if self.invert_y else -90
        Rz = Rotation.from_euler('z', z_angle, degrees=True)
        R_transform = Rz * Ry_180

        return R_transform

    def transform_position(self, vr_position):
        """
        将VR位置转换到ManiSkill坐标系

        Args:
            vr_position: numpy array [x, y, z] 在VR坐标系中的位置

        Returns:
            numpy array [x, y, z] 在ManiSkill坐标系中的位置
        """
        ms_position = self.position_transform_matrix @ vr_position
        ms_position = ms_position / self.scale_factor
        return ms_position

    def transform_rotation(self, vr_rotation_matrix):
        """
        将VR旋转矩阵转换到ManiSkill坐标系

        Args:
            vr_rotation_matrix: 3x3旋转矩阵，在VR坐标系中

        Returns:
            3x3旋转矩阵，在ManiSkill坐标系中
        """
        # R_ms = R_transform @ R_vr @ R_transform^-1
        R_vr = Rotation.from_matrix(vr_rotation_matrix)
        R_ms = self._rotation_transform * R_vr * self._rotation_transform.inv()
        return R_ms.as_matrix()

    def transform_pose(self, vr_position, vr_rotation_matrix):
        """
        将VR位姿转换到ManiSkill坐标系

        Args:
            vr_position: numpy array [x, y, z]
            vr_rotation_matrix: 3x3旋转矩阵

        Returns:
            tuple: (ms_position, ms_rotation_matrix)
        """
        ms_position = self.transform_position(vr_position)
        ms_rotation_matrix = self.transform_rotation(vr_rotation_matrix)
        return ms_position, ms_rotation_matrix

    def transform_to_sapien_pose(self, vr_position, vr_rotation_matrix):
        """
        将VR位姿转换为SAPIEN Pose对象

        Args:
            vr_position: numpy array [x, y, z]
            vr_rotation_matrix: 3x3旋转矩阵

        Returns:
            sapien.Pose对象

        Raises:
            ImportError: 如果sapien未安装
        """
        if not HAS_SAPIEN:
            raise ImportError("此方法需要安装sapien。请运行: pip install mani_skill")

        ms_position, ms_rotation_matrix = self.transform_pose(vr_position, vr_rotation_matrix)

        # 转换为四元数 [w, x, y, z] (SAPIEN格式)
        rotation = Rotation.from_matrix(ms_rotation_matrix)
        quat_xyzw = rotation.as_quat()  # [x, y, z, w]
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # [w, x, y, z]

        return sapien.Pose(p=ms_position, q=quat_wxyz)

    def create_workspace_clipper(self, x_range, y_range, z_range):
        """
        创建工作空间限制器

        Args:
            x_range: [x_min, x_max] 在ManiSkill坐标系中
            y_range: [y_min, y_max]
            z_range: [z_min, z_max]

        Returns:
            函数，用于将位置限制在工作空间内
        """
        def clip_position(position):
            """将位置限制在工作空间内"""
            clipped = np.array(position)
            clipped[0] = np.clip(clipped[0], x_range[0], x_range[1])
            clipped[1] = np.clip(clipped[1], y_range[0], y_range[1])
            clipped[2] = np.clip(clipped[2], z_range[0], z_range[1])
            return clipped

        return clip_position

    def is_in_workspace(self, position, x_range, y_range, z_range):
        """
        检查位置是否在工作空间内

        Args:
            position: numpy array [x, y, z] 在ManiSkill坐标系
            x_range, y_range, z_range: 工作空间范围

        Returns:
            bool
        """
        return (x_range[0] <= position[0] <= x_range[1] and
                y_range[0] <= position[1] <= y_range[1] and
                z_range[0] <= position[2] <= z_range[1])


class PositionFilter:
    """位置平滑滤波器，减少手柄抖动"""

    def __init__(self, alpha=0.5):
        """
        Args:
            alpha: 平滑系数，0-1之间
                  0: 完全使用旧值（不更新）
                  1: 完全使用新值（不平滑）
                  推荐: 0.3-0.7
        """
        self.alpha = alpha
        self.last_position = None
        self.last_rotation = None

    def filter_position(self, position):
        """
        平滑位置

        Args:
            position: numpy array [x, y, z]

        Returns:
            平滑后的位置
        """
        if self.last_position is None:
            self.last_position = position
            return position

        # 指数移动平均
        filtered = self.alpha * position + (1 - self.alpha) * self.last_position
        self.last_position = filtered
        return filtered

    def filter_rotation(self, rotation_matrix):
        """
        平滑旋转（使用球面线性插值）

        Args:
            rotation_matrix: 3x3旋转矩阵

        Returns:
            平滑后的旋转矩阵
        """
        if self.last_rotation is None:
            self.last_rotation = rotation_matrix
            return rotation_matrix

        # 使用SLERP（球面线性插值）
        R_last = Rotation.from_matrix(self.last_rotation)
        R_current = Rotation.from_matrix(rotation_matrix)

        # 插值
        quat_interpolated = (1 - self.alpha) * R_last.as_quat() + self.alpha * R_current.as_quat()
        # 归一化四元数
        quat_normalized = quat_interpolated / np.linalg.norm(quat_interpolated)
        # 从归一化的四元数创建 Rotation 对象
        R_filtered = Rotation.from_quat(quat_normalized)

        filtered_matrix = R_filtered.as_matrix()
        self.last_rotation = filtered_matrix
        return filtered_matrix

    def reset(self):
        """重置滤波器"""
        self.last_position = None
        self.last_rotation = None


# 测试代码
if __name__ == "__main__":
    # 测试坐标转换
    transformer = CoordinateTransformer(scale_factor=1.0)

    # 测试案例1: VR向右移动 -> MS向Y负方向（默认invert_y=False）
    vr_pos = np.array([0.1, 0, 0])  # VR右方向
    ms_pos = transformer.transform_position(vr_pos)
    print(f"VR右移0.1m: {vr_pos} -> MS: {ms_pos}")
    print(f"预期: [0, -0.1, 0], 实际: {ms_pos}\n")

    # 测试案例2: VR向上移动 -> MS向上移动
    vr_pos = np.array([0, 0.1, 0])  # VR上方向
    ms_pos = transformer.transform_position(vr_pos)
    print(f"VR上移0.1m: {vr_pos} -> MS上移: {ms_pos}")
    print(f"预期: [0, 0, 0.1], 实际: {ms_pos}\n")

    # 测试案例3: VR向后移动 -> MS向前移动
    vr_pos = np.array([0, 0, 0.1])  # VR后方向
    ms_pos = transformer.transform_position(vr_pos)
    print(f"VR后移0.1m: {vr_pos} -> MS前移: {ms_pos}")
    print(f"预期: [-0.1, 0, 0], 实际: {ms_pos}\n")

    # 测试案例4: 缩放
    transformer_scaled = CoordinateTransformer(scale_factor=2.0)
    vr_pos = np.array([0, 0.2, 0])  # VR上移0.2m
    ms_pos = transformer_scaled.transform_position(vr_pos)
    print(f"VR上移0.2m (scale=2.0): {vr_pos} -> MS上移: {ms_pos}")
    print(f"预期: [0, 0, 0.1], 实际: {ms_pos}\n")

    print("✓ 坐标转换测试完成")
