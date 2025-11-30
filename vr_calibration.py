#!/usr/bin/env python3
"""
VR手柄标定脚本
用于标定VR手柄的参考位姿，后续用于控制ManiSkill机械臂

使用方法:
1. 运行脚本
2. 将右手柄放到舒适的初始位置（如胸前）
3. 按下Trigger键完成标定
4. 标定后可以移动手柄查看相对位移
5. 按Ctrl+C退出

标定数据会保存到 calibration_data.json
"""

import openvr
import time
import sys
import numpy as np
import json
from pathlib import Path
from scipy.spatial.transform import Rotation


class VRCalibration:
    def __init__(self):
        self.vr_system = None
        self.is_calibrated = False
        self.reference_pose = None  # 标定的参考位姿

        # 用于平滑显示的上一次位置
        self.last_display_time = 0
        self.display_interval = 0.1  # 100ms更新一次显示

    def initialize_vr(self):
        """初始化OpenVR"""
        try:
            openvr.init(openvr.VRApplication_Scene)
            print("✓ OpenVR 初始化成功!")
            self.vr_system = openvr.VRSystem()
            return True
        except openvr.OpenVRError as e:
            print(f"✗ OpenVR初始化失败: {e}")
            print("请确保:")
            print("  1. SteamVR正在运行")
            print("  2. ALVR已连接")
            print("  3. Meta Quest 3已配对")
            return False

    def get_right_controller_index(self):
        """获取右手柄的设备索引"""
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            device_class = self.vr_system.getTrackedDeviceClass(i)
            if device_class == openvr.TrackedDeviceClass_Controller:
                role = self.vr_system.getControllerRoleForTrackedDeviceIndex(i)
                if role == openvr.TrackedControllerRole_RightHand:
                    return i
        return None

    def get_controller_pose(self, controller_idx):
        """获取手柄的位姿（4x4齐次变换矩阵）"""
        poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding,
            0,
            openvr.k_unMaxTrackedDeviceCount
        )

        if not poses[controller_idx].bPoseIsValid:
            return None

        # 获取3x4变换矩阵并转换为4x4
        pose_3x4 = poses[controller_idx].mDeviceToAbsoluteTracking
        T = np.eye(4)
        T[0, :] = pose_3x4[0]
        T[1, :] = pose_3x4[1]
        T[2, :] = pose_3x4[2]

        return T

    def get_button_state(self, controller_idx):
        """获取手柄按钮状态"""
        # 获取控制器状态
        result, state = self.vr_system.getControllerState(controller_idx)
        if not result:
            return None

        # Trigger按钮是按钮1（索引33），axis值在ulButtonPressed中
        # OpenVR中trigger是Axis1
        trigger_button_id = openvr.k_EButton_SteamVR_Trigger
        trigger_pressed = (state.ulButtonPressed & (1 << trigger_button_id)) != 0

        return {
            'trigger': trigger_pressed,
            'trigger_value': state.rAxis[1].x  # Trigger的模拟值 0-1
        }

    def pose_to_position_quaternion(self, T):
        """将4x4变换矩阵转换为位置和四元数"""
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        # 使用scipy转换为四元数 [x, y, z, w]
        rotation = Rotation.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()  # [x, y, z, w]
        return position, quaternion

    def calculate_relative_pose(self, current_pose):
        """计算相对于参考位姿的位移和旋转"""
        if self.reference_pose is None:
            return None, None

        # 计算相对变换: T_rel = T_ref^-1 * T_current
        T_ref_inv = np.linalg.inv(self.reference_pose)
        T_relative = T_ref_inv @ current_pose

        # 提取位移和旋转
        relative_position = T_relative[:3, 3]
        relative_rotation = Rotation.from_matrix(T_relative[:3, :3])
        relative_euler = relative_rotation.as_euler('xyz', degrees=True)  # 转换为欧拉角(度)

        return relative_position, relative_euler

    def calibrate(self, current_pose):
        """执行标定"""
        self.reference_pose = current_pose.copy()
        self.is_calibrated = True

        position, quaternion = self.pose_to_position_quaternion(current_pose)

        print("\n" + "="*60)
        print("✓ 标定完成!")
        print("="*60)
        print(f"参考位置 (VR坐标系): [{position[0]:7.4f}, {position[1]:7.4f}, {position[2]:7.4f}] m")
        print(f"参考四元数 [x,y,z,w]: [{quaternion[0]:7.4f}, {quaternion[1]:7.4f}, {quaternion[2]:7.4f}, {quaternion[3]:7.4f}]")
        print("="*60)
        print("现在可以移动手柄查看相对位移")
        print("按 Ctrl+C 退出并保存标定数据")
        print("="*60 + "\n")

    def save_calibration(self, filename="calibration_data.json"):
        """保存标定数据到JSON文件"""
        if not self.is_calibrated:
            print("未进行标定，无数据保存")
            return

        position, quaternion = self.pose_to_position_quaternion(self.reference_pose)

        calibration_data = {
            "vr_reference_position": position.tolist(),
            "vr_reference_quaternion": quaternion.tolist(),
            "vr_reference_matrix": self.reference_pose.tolist(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "coordinate_system": {
                "vr": "OpenVR (X:右, Y:上, Z:后)",
                "maniskill": "需要转换到 (X:前, Y:左, Z:上)"
            }
        }

        filepath = Path(__file__).parent / filename
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        print(f"\n✓ 标定数据已保存到: {filepath}")

    def display_status(self, controller_idx, current_pose):
        """显示当前状态"""
        current_time = time.time()
        if current_time - self.last_display_time < self.display_interval:
            return
        self.last_display_time = current_time

        position, quaternion = self.pose_to_position_quaternion(current_pose)

        # 清屏（简单方式）
        print("\033[H\033[J", end="")

        print("="*60)
        print("VR手柄标定工具 - 右手柄")
        print("="*60)
        print(f"标定状态: {'✓ 已标定' if self.is_calibrated else '✗ 未标定'}")
        print("-"*60)

        # 显示当前位姿
        print(f"当前位置 (VR): [{position[0]:7.4f}, {position[1]:7.4f}, {position[2]:7.4f}] m")
        print(f"当前四元数:     [{quaternion[0]:7.4f}, {quaternion[1]:7.4f}, {quaternion[2]:7.4f}, {quaternion[3]:7.4f}]")

        # 如果已标定，显示相对位移
        if self.is_calibrated:
            rel_pos, rel_euler = self.calculate_relative_pose(current_pose)
            print("-"*60)
            print("相对于标定点的位移:")
            print(f"  ΔX: {rel_pos[0]:+7.4f} m  (VR右方向)")
            print(f"  ΔY: {rel_pos[1]:+7.4f} m  (VR上方向)")
            print(f"  ΔZ: {rel_pos[2]:+7.4f} m  (VR后方向)")
            print(f"相对旋转 (欧拉角):")
            print(f"  Roll : {rel_euler[0]:+7.2f}°")
            print(f"  Pitch: {rel_euler[1]:+7.2f}°")
            print(f"  Yaw  : {rel_euler[2]:+7.2f}°")

        print("="*60)
        if not self.is_calibrated:
            print(">>> 将右手柄放到胸前初始位置，按住 Trigger 键标定 <<<")
        print("按 Ctrl+C 退出")
        print("="*60)

    def run(self):
        """主运行循环"""
        if not self.initialize_vr():
            return

        print("\n正在寻找右手柄...")
        controller_idx = self.get_right_controller_index()

        if controller_idx is None:
            print("✗ 未找到右手柄！")
            print("请确保:")
            print("  1. Meta Quest 3手柄已开启")
            print("  2. 手柄正在被追踪")
            openvr.shutdown()
            return

        print(f"✓ 找到右手柄 (设备索引: {controller_idx})")
        print("\n开始标定流程...")
        print("="*60)
        print("操作说明:")
        print("  1. 将右手柄放到舒适的初始位置（建议胸前）")
        print("  2. 按住 Trigger 键（食指扳机）完成标定")
        print("  3. 标定后移动手柄可查看相对位移")
        print("="*60)

        try:
            last_trigger_state = False

            while True:
                # 获取手柄位姿
                current_pose = self.get_controller_pose(controller_idx)
                if current_pose is None:
                    print("\r⚠ 手柄追踪丢失...", end="")
                    time.sleep(0.05)
                    continue

                # 获取按钮状态
                button_state = self.get_button_state(controller_idx)
                if button_state is None:
                    time.sleep(0.05)
                    continue

                # 检测trigger按下（边沿触发）
                trigger_pressed = button_state['trigger']
                if trigger_pressed and not last_trigger_state and not self.is_calibrated:
                    self.calibrate(current_pose)

                last_trigger_state = trigger_pressed

                # 显示当前状态
                self.display_status(controller_idx, current_pose)

                time.sleep(0.02)  # 50Hz更新

        except KeyboardInterrupt:
            print("\n\n正在退出...")
        finally:
            # 保存标定数据
            self.save_calibration()
            openvr.shutdown()
            print("✓ OpenVR已关闭")


def main():
    calibration = VRCalibration()
    calibration.run()


if __name__ == "__main__":
    main()
