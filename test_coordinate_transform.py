#!/usr/bin/env python3
"""
坐标转换测试和可视化工具
帮助验证VR坐标系到ManiSkill坐标系的转换是否正确

使用方法:
    python test_coordinate_transform.py

会实时显示：
- VR手柄的当前位置
- 相对于标定点的位移（VR坐标系）
- 转换后的位移（ManiSkill坐标系）
- 对应的机械臂末端目标位置
"""

import sys
import time
import numpy as np
from pathlib import Path

# 导入VR模块
from vr_controller import VRController
from coordinate_transform import CoordinateTransformer


class CoordinateVisualizationTool:
    """坐标转换可视化工具"""

    def __init__(self):
        self.vr = VRController()
        self.transformer = CoordinateTransformer(scale_factor=2.0)

        # 假设的机械臂初始末端位置
        self.robot_reference = np.array([0.0, 0.0, 0.5])

        self.is_calibrated = False

    def run(self):
        """主运行循环"""
        print("="*70)
        print("VR到ManiSkill坐标转换测试工具")
        print("="*70)

        # 初始化VR
        print("\n正在初始化VR...")
        if not self.vr.initialize():
            print("✗ VR初始化失败")
            return

        print("\n" + "="*70)
        print("操作说明:")
        print("  1. 将右手柄放到胸前舒适位置")
        print("  2. 按住 Trigger 键标定")
        print("  3. 标定后移动手柄观察坐标转换")
        print("  4. 按 Ctrl+C 退出")
        print("="*70 + "\n")

        try:
            while True:
                # 获取按钮状态
                buttons = self.vr.get_buttons()
                if buttons is None:
                    time.sleep(0.05)
                    continue

                # 标定
                if buttons['trigger_pressed'] and not self.is_calibrated:
                    if self.vr.calibrate():
                        self.is_calibrated = True

                # 显示状态
                self.display_status()

                time.sleep(0.1)  # 10Hz更新

        except KeyboardInterrupt:
            print("\n\n退出程序")
        finally:
            self.vr.shutdown()

    def display_status(self):
        """显示当前状态"""
        # 清屏
        print("\033[H\033[J", end="")

        print("="*70)
        print("VR坐标转换实时监控")
        print("="*70)

        if not self.is_calibrated:
            print("\n⚠  未标定 - 请按住Trigger键标定\n")
            print("="*70)
            return

        # 获取VR相对位姿
        vr_delta_pos, vr_delta_rot = self.vr.get_relative_pose()

        if vr_delta_pos is None:
            print("\n⚠  VR追踪丢失\n")
            print("="*70)
            return

        # 坐标转换
        ms_delta_pos, ms_delta_rot = self.transformer.transform_pose(
            vr_delta_pos, vr_delta_rot
        )

        # 计算机械臂目标位置
        robot_target = self.robot_reference + ms_delta_pos

        # 显示详细信息
        print("\n【1. VR手柄相对位移】(相对于标定点)")
        print(f"  VR坐标系: X(右) Y(上) Z(后)")
        print(f"  位移: [{vr_delta_pos[0]:+7.4f}, {vr_delta_pos[1]:+7.4f}, {vr_delta_pos[2]:+7.4f}] m")
        self._print_direction_hint(vr_delta_pos, "VR")

        print("\n【2. 转换后的位移】(ManiSkill坐标系)")
        print(f"  MS坐标系: X(前) Y(左) Z(上)")
        print(f"  位移: [{ms_delta_pos[0]:+7.4f}, {ms_delta_pos[1]:+7.4f}, {ms_delta_pos[2]:+7.4f}] m")
        self._print_direction_hint(ms_delta_pos, "MS")

        print("\n【3. 机械臂目标位置】(相对于base)")
        print(f"  参考位置: [{self.robot_reference[0]:.3f}, {self.robot_reference[1]:.3f}, {self.robot_reference[2]:.3f}] m")
        print(f"  目标位置: [{robot_target[0]:7.4f}, {robot_target[1]:7.4f}, {robot_target[2]:7.4f}] m")

        # 显示转换规则
        print("\n【坐标转换规则】")
        print("  VR_X(右) → MS_Y(左) [取负]")
        print("  VR_Y(上) → MS_Z(上)")
        print("  VR_Z(后) → MS_X(前) [取负]")

        # 验证
        expected_ms_delta = np.array([
            -vr_delta_pos[2],  # VR_Z后 → MS_X前(负)
            -vr_delta_pos[0],  # VR_X右 → MS_Y左(负)
             vr_delta_pos[1]   # VR_Y上 → MS_Z上
        ]) / self.transformer.scale_factor

        error = np.linalg.norm(ms_delta_pos - expected_ms_delta)

        print("\n【验证】")
        print(f"  预期MS位移: [{expected_ms_delta[0]:+7.4f}, {expected_ms_delta[1]:+7.4f}, {expected_ms_delta[2]:+7.4f}] m")
        print(f"  实际MS位移: [{ms_delta_pos[0]:+7.4f}, {ms_delta_pos[1]:+7.4f}, {ms_delta_pos[2]:+7.4f}] m")
        print(f"  误差: {error:.6f} m {'✓' if error < 0.001 else '✗'}")

        print("\n" + "="*70)
        print("按 Ctrl+C 退出")
        print("="*70)

    def _print_direction_hint(self, delta, coord_sys):
        """打印位移方向提示"""
        threshold = 0.01  # 1cm

        if coord_sys == "VR":
            labels = ["右", "上", "后"]
            neg_labels = ["左", "下", "前"]
        else:  # MS
            labels = ["前", "左", "上"]
            neg_labels = ["后", "右", "下"]

        hints = []
        for i, (pos_label, neg_label) in enumerate(zip(labels, neg_labels)):
            if abs(delta[i]) > threshold:
                direction = pos_label if delta[i] > 0 else neg_label
                hints.append(f"{direction}{abs(delta[i])*100:.1f}cm")

        if hints:
            print(f"  → {' + '.join(hints)}")
        else:
            print(f"  → 无明显移动")


def main():
    tool = CoordinateVisualizationTool()
    tool.run()


if __name__ == "__main__":
    main()
