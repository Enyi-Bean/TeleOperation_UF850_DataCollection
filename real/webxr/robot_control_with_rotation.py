#!/usr/bin/env python3
"""
UF850 WebXR 遥操作控制（带旋转版本）
基于robot_control_correct.py，增加姿态控制功能
适用于需要精确姿态控制的任务（如拉抽屉、开门等）
"""

import asyncio
import websockets
import json
import numpy as np
import time
from collections import deque
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xarm.wrapper import XArmAPI
from real import config


class UF850WebXRControlWithRotation:
    """UF850 WebXR 遥操作控制器（带旋转版本）"""

    def __init__(self, robot_ip=None):
        self.robot_ip = robot_ip or config.ROBOT_IP
        self.arm = None

        # ========== 旋转控制配置 ==========
        self.ENABLE_ROTATION = True  # 启用旋转控制
        self.ROTATION_MODE = "incremental"  # "incremental" 或 "absolute"
        self.ROTATION_SCALE = 0.5  # 旋转缩放因子（降低灵敏度）
        self.ROTATION_DEADZONE = 0.05  # 旋转死区（约3度）

        print("\n" + "="*60)
        print("坐标系配置（带旋转控制）:")
        print("  Quest 3 WebXR: X=右, Y=上, Z=后(朝向用户)")
        print("  UF850 机械臂: X=前, Y=左, Z=上")
        print("  位置映射:")
        print("    VR Z(后) → Robot X(前) [取反]")
        print("    VR X(右) → Robot Y(左) [取反]")
        print("    VR Y(上) → Robot Z(上)")
        print("  旋转控制:")
        print(f"    模式: {self.ROTATION_MODE}")
        print(f"    缩放: {self.ROTATION_SCALE}")
        print(f"    死区: {np.rad2deg(self.ROTATION_DEADZONE):.1f}°")
        print("="*60 + "\n")

        # 控制模式
        self.POSITION_SCALE = config.SCALE_FACTOR  # 位置缩放因子

        # 标定数据
        self.is_calibrated = False
        self.calibration_robot_pose = None  # 标定时机械臂位姿
        self.calibration_vr_pos = None      # 标定时VR位置(已转换)
        self.calibration_vr_rot_matrix = None  # 标定时VR旋转矩阵
        self.calibration_robot_rot_matrix = None  # 标定时机械臂旋转矩阵

        # 控制状态
        self.last_sent_pose = None
        self.last_command_time = None
        self.gripper_open = False
        self.last_trigger_state = False

        # 滤波
        self.position_buffer = deque(maxlen=3)
        self.rotation_buffer = deque(maxlen=3)  # 旋转也需要滤波

        # 统计
        self.frame_count = 0
        self.start_time = None

    def vr_to_robot_position(self, x, y, z):
        """
        将VR位置转换到机械臂坐标系

        VR: X=右, Y=上, Z=后
        Robot: X=前, Y=左, Z=上
        """
        # 正确的坐标映射
        robot_x = -z  # VR的后(+Z) → Robot的前(+X)，需要取反
        robot_y = -x  # VR的右(+X) → Robot的左(+Y)，需要取反
        robot_z = y   # VR的上(+Y) → Robot的上(+Z)

        # 单位转换：米 → 毫米，并应用缩放
        robot_x *= (1000.0 / self.POSITION_SCALE)
        robot_y *= (1000.0 / self.POSITION_SCALE)
        robot_z *= (1000.0 / self.POSITION_SCALE)

        return np.array([robot_x, robot_y, robot_z])

    def quaternion_to_rotation_matrix(self, qx, qy, qz, qw):
        """
        四元数转旋转矩阵
        """
        # 归一化四元数
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm < 1e-6:
            return np.eye(3)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

        # 转换为旋转矩阵
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz

        R = np.array([
            [1 - 2*(yy + zz),     2*(xy - wz),      2*(xz + wy)],
            [    2*(xy + wz), 1 - 2*(xx + zz),      2*(yz - wx)],
            [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)]
        ])
        return R

    def rotation_matrix_to_rpy(self, R):
        """
        旋转矩阵转RPY角
        """
        epsilon = 1e-6
        if abs(R[2, 0]) > 1 - epsilon:  # 万向节锁
            pitch = np.arcsin(-R[2, 0])
            roll = np.arctan2(-R[0, 1], R[1, 1])
            yaw = 0
        else:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arcsin(-R[2, 0])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        return [roll, pitch, yaw]

    def rpy_to_rotation_matrix(self, roll, pitch, yaw):
        """
        RPY角转旋转矩阵
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cp*cy,  -cr*sy + sr*sp*cy,   sr*sy + cr*sp*cy],
            [cp*sy,   cr*cy + sr*sp*sy,  -sr*cy + cr*sp*sy],
            [ -sp,         sr*cp,              cr*cp]
        ])
        return R

    def vr_rotation_to_robot(self, qx, qy, qz, qw):
        """
        将VR手柄旋转转换到机械臂坐标系
        考虑坐标系差异进行相应的变换
        """
        # VR四元数转旋转矩阵
        R_vr = self.quaternion_to_rotation_matrix(qx, qy, qz, qw)

        # 坐标系变换矩阵（与位置变换对应）
        # VR到Robot的坐标轴映射
        T_vr_to_robot = np.array([
            [ 0,  0, -1],  # Robot X = -VR Z
            [-1,  0,  0],  # Robot Y = -VR X
            [ 0,  1,  0]   # Robot Z =  VR Y
        ])

        # 变换旋转矩阵到机械臂坐标系
        R_robot = T_vr_to_robot @ R_vr @ T_vr_to_robot.T

        return R_robot

    async def initialize_robot(self):
        """初始化机械臂"""
        print("初始化UF850机械臂（带旋转控制）...")

        try:
            self.arm = XArmAPI(self.robot_ip, is_radian=True)
            print(f"✓ 连接成功: {self.robot_ip}")
        except Exception as e:
            print(f"✗ 连接失败: {e}")
            return False

        # 配置
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)  # 伺服模式
        self.arm.set_state(0)

        # 移动到初始位置
        print("移动到初始位置...")
        initial = config.INITIAL_POSITION.copy()
        initial[3:] = np.deg2rad(initial[3:])

        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.set_position(*initial, wait=True)

        # 切回伺服模式
        self.arm.set_mode(1)
        self.arm.set_state(0)

        print("✓ 初始化完成\n")
        return True

    def calibrate(self, vr_x, vr_y, vr_z, vr_qx, vr_qy, vr_qz, vr_qw):
        """标定"""
        print("\n" + "="*50)
        print("执行标定（带旋转控制）...")

        # 获取机械臂当前位姿
        code, pose = self.arm.get_position(is_radian=True)
        if code != 0:
            print("✗ 获取机械臂位姿失败")
            return False

        # 保存标定数据 - 位置
        self.calibration_robot_pose = pose
        self.calibration_vr_pos = self.vr_to_robot_position(vr_x, vr_y, vr_z)

        # 保存标定数据 - 旋转
        self.calibration_vr_rot_matrix = self.vr_rotation_to_robot(vr_qx, vr_qy, vr_qz, vr_qw)
        self.calibration_robot_rot_matrix = self.rpy_to_rotation_matrix(pose[3], pose[4], pose[5])

        # 显示标定信息
        print(f"机械臂位置: X={pose[0]:.1f}, Y={pose[1]:.1f}, Z={pose[2]:.1f} mm")
        print(f"机械臂姿态: Roll={np.rad2deg(pose[3]):.1f}°, Pitch={np.rad2deg(pose[4]):.1f}°, Yaw={np.rad2deg(pose[5]):.1f}°")
        print(f"VR手柄位置(原始): X={vr_x:.3f}, Y={vr_y:.3f}, Z={vr_z:.3f} m")
        print(f"VR手柄位置(转换): X={self.calibration_vr_pos[0]:.1f}, Y={self.calibration_vr_pos[1]:.1f}, Z={self.calibration_vr_pos[2]:.1f} mm")

        self.is_calibrated = True
        self.last_sent_pose = pose
        self.position_buffer.clear()
        self.rotation_buffer.clear()

        print("✓ 标定完成！")
        print("操作说明:")
        print("  • 移动手柄控制位置")
        print("  • 旋转手柄控制末端姿态")
        print("  • 按Trigger控制夹爪")
        print("="*50 + "\n")
        return True

    def compute_target_pose(self, vr_x, vr_y, vr_z, vr_qx, vr_qy, vr_qz, vr_qw):
        """计算目标位姿（包含旋转）"""
        if not self.is_calibrated:
            return None

        # === 位置部分 ===
        current_vr_pos = self.vr_to_robot_position(vr_x, vr_y, vr_z)
        delta_pos = current_vr_pos - self.calibration_vr_pos

        # 死区过滤（2mm）
        if np.linalg.norm(delta_pos) < 2.0:
            delta_pos = np.zeros(3)

        # 计算目标位置
        target_pos = self.calibration_robot_pose[:3] + delta_pos

        # 工作空间限制
        target_pos[0] = np.clip(target_pos[0],
                                config.WORKSPACE_LIMITS['x'][0],
                                config.WORKSPACE_LIMITS['x'][1])
        target_pos[1] = np.clip(target_pos[1],
                                config.WORKSPACE_LIMITS['y'][0],
                                config.WORKSPACE_LIMITS['y'][1])
        target_pos[2] = np.clip(target_pos[2],
                                config.WORKSPACE_LIMITS['z'][0],
                                config.WORKSPACE_LIMITS['z'][1])

        # === 旋转部分 ===
        if self.ENABLE_ROTATION:
            # 获取当前VR旋转矩阵
            current_vr_rot = self.vr_rotation_to_robot(vr_qx, vr_qy, vr_qz, vr_qw)

            if self.ROTATION_MODE == "incremental":
                # 增量模式：计算相对旋转
                # R_relative = R_current @ inv(R_calibration)
                R_relative = current_vr_rot @ self.calibration_vr_rot_matrix.T

                # 应用缩放因子（通过转换到轴角表示）
                axis_angle = self.rotation_matrix_to_axis_angle(R_relative)
                axis_angle[3] *= self.ROTATION_SCALE  # 缩放旋转角度
                R_relative_scaled = self.axis_angle_to_rotation_matrix(axis_angle)

                # 应用到机械臂
                R_target = self.calibration_robot_rot_matrix @ R_relative_scaled

            else:  # absolute模式
                # 直接映射（需要更复杂的处理）
                R_target = current_vr_rot

            # 旋转死区
            if self.last_sent_pose is not None:
                last_R = self.rpy_to_rotation_matrix(
                    self.last_sent_pose[3],
                    self.last_sent_pose[4],
                    self.last_sent_pose[5]
                )
                # 计算旋转变化量
                R_diff = R_target @ last_R.T
                angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))

                if angle_diff < self.ROTATION_DEADZONE:
                    # 保持上次的旋转
                    target_rpy = self.last_sent_pose[3:6]
                else:
                    # 转换为RPY
                    target_rpy = self.rotation_matrix_to_rpy(R_target)
            else:
                target_rpy = self.rotation_matrix_to_rpy(R_target)

        else:
            # 保持标定时的姿态
            target_rpy = self.calibration_robot_pose[3:6]

        # 组合位姿
        target_pose = list(target_pos) + list(target_rpy)

        # 速度限制
        if self.last_sent_pose is not None:
            delta = np.array(target_pose[:3]) - np.array(self.last_sent_pose[:3])
            dist = np.linalg.norm(delta)
            if dist > config.MAX_DELTA_POSITION:
                scale = config.MAX_DELTA_POSITION / dist
                target_pose[:3] = self.last_sent_pose[:3] + delta * scale

        return target_pose

    def rotation_matrix_to_axis_angle(self, R):
        """旋转矩阵转轴角表示"""
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if angle < 1e-6:
            return np.array([0, 0, 1, 0])  # 无旋转

        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(angle))

        return np.array([axis[0], axis[1], axis[2], angle])

    def axis_angle_to_rotation_matrix(self, axis_angle):
        """轴角表示转旋转矩阵"""
        axis = axis_angle[:3]
        angle = axis_angle[3]

        if angle < 1e-6:
            return np.eye(3)

        axis = axis / np.linalg.norm(axis)
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c

        x, y, z = axis
        R = np.array([
            [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
            [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
            [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
        ])
        return R

    def apply_filter(self, pose):
        """应用位置和旋转滤波"""
        # 位置滤波
        self.position_buffer.append(pose[:3])
        if len(self.position_buffer) > 0:
            filtered_pos = np.mean(self.position_buffer, axis=0)
        else:
            filtered_pos = pose[:3]

        # 旋转滤波（简单平均RPY）
        self.rotation_buffer.append(pose[3:6])
        if len(self.rotation_buffer) > 0:
            filtered_rot = np.mean(self.rotation_buffer, axis=0)
        else:
            filtered_rot = pose[3:6]

        return list(filtered_pos) + list(filtered_rot)

    def control_gripper(self):
        """控制夹爪开关"""
        if config.GRIPPER_TYPE == 0:
            print("未配置夹爪")
            return

        try:
            if config.GRIPPER_TYPE == 1:
                # xArm Gripper
                pos = 850 if self.gripper_open else 0
                ret = self.arm.set_gripper_position(pos, wait=False)
                print(f"夹爪: {'开(850)' if self.gripper_open else '关(0)'}")
            elif config.GRIPPER_TYPE == 2:
                # xArm Gripper G2
                pos = 84 if self.gripper_open else 0
                ret = self.arm.set_gripper_g2_position(pos, speed=225, wait=False)
                print(f"夹爪G2: {'开(84)' if self.gripper_open else '关(0)'}")
            elif config.GRIPPER_TYPE == 3:
                # BIO Gripper G2
                pos = 150 if self.gripper_open else 71
                ret = self.arm.set_bio_gripper_g2_position(pos, speed=4500, wait=False)
                print(f"BIO夹爪: {'开(150)' if self.gripper_open else '关(71)'}")
            else:
                print(f"未知夹爪类型: {config.GRIPPER_TYPE}")
                return

            if ret == 0:
                print(f"✓ 夹爪命令发送成功")
            else:
                print(f"⚠ 夹爪命令返回码: {ret}")

        except Exception as e:
            print(f"✗ 夹爪控制失败: {e}")

    async def handle_controller_data(self, websocket):
        """处理手柄数据"""
        client_ip = websocket.remote_address[0]
        print(f"Quest 3 已连接: {client_ip} (带旋转控制)")

        if self.start_time is None:
            self.start_time = time.time()

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    self.frame_count += 1

                    # 右手柄数据
                    if 'right' not in data or data['right'] is None:
                        continue

                    right = data['right']
                    pos = right['position']
                    ori = right['orientation']

                    # 调试输出（每50帧）
                    if self.frame_count % 50 == 0:
                        robot_pos = self.vr_to_robot_position(pos['x'], pos['y'], pos['z'])
                        print(f"\n[Frame {self.frame_count}]")
                        print(f"  VR位置: X={pos['x']:+.3f} Y={pos['y']:+.3f} Z={pos['z']:+.3f} (m)")
                        print(f"  →Robot位置: X={robot_pos[0]:+.1f} Y={robot_pos[1]:+.1f} Z={robot_pos[2]:+.1f} (mm)")

                        if self.is_calibrated and self.last_sent_pose:
                            print(f"  目标位置: X={self.last_sent_pose[0]:+.1f} Y={self.last_sent_pose[1]:+.1f} Z={self.last_sent_pose[2]:+.1f} (mm)")
                            print(f"  目标姿态: R={np.rad2deg(self.last_sent_pose[3]):+.1f}° P={np.rad2deg(self.last_sent_pose[4]):+.1f}° Y={np.rad2deg(self.last_sent_pose[5]):+.1f}°")

                    # 标定检查
                    if not self.is_calibrated:
                        if 'buttons' in right and len(right['buttons']) > 0:
                            if right['buttons'][0].get('pressed', False):
                                self.calibrate(pos['x'], pos['y'], pos['z'],
                                             ori['x'], ori['y'], ori['z'], ori['w'])
                        continue

                    # 计算目标位姿
                    target_pose = self.compute_target_pose(
                        pos['x'], pos['y'], pos['z'],
                        ori['x'], ori['y'], ori['z'], ori['w']
                    )

                    if target_pose is None:
                        continue

                    # 滤波
                    target_pose = self.apply_filter(target_pose)

                    # 频率控制
                    now = time.time()
                    if self.last_command_time is None or \
                       now - self.last_command_time >= 1.0/config.CONTROL_FREQUENCY:

                        # 发送到机械臂
                        try:
                            ret = self.arm.set_servo_cartesian(target_pose, is_radian=True)
                            if ret != 0 and config.VERBOSE:
                                print(f"⚠ 指令返回: {ret}")
                        except Exception as e:
                            print(f"✗ 发送失败: {e}")

                        self.last_command_time = now
                        self.last_sent_pose = target_pose

                    # 夹爪控制
                    if 'buttons' in right and len(right['buttons']) > 0:
                        trigger = right['buttons'][0].get('pressed', False)
                        if trigger and not self.last_trigger_state:
                            self.gripper_open = not self.gripper_open
                            self.control_gripper()  # 实际控制夹爪
                        self.last_trigger_state = trigger

                except Exception as e:
                    print(f"处理错误: {e}")
                    if config.VERBOSE:
                        import traceback
                        traceback.print_exc()

        except websockets.exceptions.ConnectionClosed:
            print(f"Quest 3 断开连接")

    async def run_server(self):
        """运行WebSocket服务器"""
        port = 8765
        print(f"WebSocket服务器启动（带旋转控制）: 端口 {port}")

        if not self.is_calibrated:
            print("\n等待标定...")
            print("将右手柄移到舒适位置，按住Trigger键标定")
            print("提示: 旋转控制已启用，适合需要精确姿态的任务\n")

        async with websockets.serve(self.handle_controller_data, "0.0.0.0", port):
            await asyncio.Future()

    def shutdown(self):
        """关闭"""
        if self.arm:
            try:
                self.arm.set_state(4)
                print("机械臂已暂停")
            except:
                pass

        if self.start_time:
            duration = time.time() - self.start_time
            print(f"运行时长: {duration:.1f}s, 帧数: {self.frame_count}")


async def main():
    """主函数"""
    robot_ip = sys.argv[1] if len(sys.argv) > 1 else None

    controller = UF850WebXRControlWithRotation(robot_ip)

    if not await controller.initialize_robot():
        return 1

    try:
        await controller.run_server()
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        controller.shutdown()

    return 0


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n退出")