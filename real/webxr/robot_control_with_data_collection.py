#!/usr/bin/env python3
"""
UF850 WebXR 遥操作控制 + 数据收集（半镜像+旋转版本）

功能:
1. VR遥操作控制 (Quest 3)
2. 实时数据收集 (30Hz采样)
3. GR00T LeRobot格式保存

按键说明:
  - Trigger: 控制夹爪开关
  - B键: 开始/结束录制
  - Joystick上下: 切换任务 (可选)
  - A键: 标定 (首次按Trigger标定)
"""

import asyncio
import websockets
import json
import numpy as np
import time
from collections import deque
import sys
import threading
from pathlib import Path

# 添加父目录到路径，以便导入config
sys.path.insert(0, str(Path(__file__).parent.parent))

from xarm.wrapper import XArmAPI
import config  # config.py在real/目录下
from data_collection import CameraManager, DataCollector


class UF850WebXRControlWithDataCollection:
    """UF850 WebXR遥操作控制器 + 数据收集"""

    def __init__(self, robot_ip=None, dataset_path="./uf850_teleop_dataset"):
        self.robot_ip = robot_ip or config.ROBOT_IP
        self.arm = None

        # ========== 原有遥操作配置 ==========
        self.ENABLE_ROTATION = True
        self.ROTATION_MODE = "incremental"
        self.ROTATION_SCALE = 0.5
        self.ROTATION_DEADZONE = 0.05
        self.POSITION_SCALE = config.SCALE_FACTOR

        # 标定数据
        self.is_calibrated = False
        self.calibration_robot_pose = None
        self.calibration_vr_pos = None
        self.calibration_vr_rot_matrix = None
        self.calibration_robot_rot_matrix = None

        # 控制状态
        self.last_sent_pose = None
        self.last_command_time = None
        self.gripper_open = False
        self.last_trigger_state = False

        # 滤波
        self.position_buffer = deque(maxlen=3)
        self.rotation_buffer = deque(maxlen=3)

        # 统计
        self.frame_count = 0
        self.start_time = None

        # ========== 新增: 数据收集组件 ==========
        self.data_collector = DataCollector(
            dataset_path=dataset_path,
            record_freq=30,      # 30Hz (100/3≈33Hz实际)
            control_freq=100
        )

        # ========== 新增: 相机管理 ==========
        self.camera_manager = None
        self.camera_thread = None
        self.latest_frames = {}
        self.camera_lock = threading.Lock()

        # ========== 新增: 按键状态跟踪 ==========
        self.last_b_button_state = False

        print("\n" + "="*60)
        print("UF850 WebXR遥操作 + 数据收集系统")
        print("="*60)
        print("坐标系配置（半镜像模式 + 旋转控制）:")
        print("  Quest 3 WebXR: X=右, Y=上, Z=后(朝向用户)")
        print("  UF850 机械臂: X=前, Y=左, Z=上")
        print("  位置映射(半镜像):")
        print("    VR Z(后) → Robot X(前) [前后不镜像，取反]")
        print("    VR X(右) → Robot Y(左) [左右镜像，不取反]")
        print("    VR Y(上) → Robot Z(上)")
        print("="*60)
        print("按键说明:")
        print("  • Trigger: 控制夹爪开关")
        print("  • B键: 开始/结束录制Episode")
        print("  • Joystick上下: 切换预定义任务")
        print("="*60)
        print(f"数据集路径: {dataset_path}")
        print(f"采样频率: 30Hz (从100Hz控制循环下采样)")
        print("="*60 + "\n")

    # ========== 原有方法保持不变 ==========
    def vr_to_robot_position(self, x, y, z):
        """VR位置转换到机械臂坐标系（半镜像版本）"""
        robot_x = -z
        robot_y = x
        robot_z = y

        robot_x *= (1000.0 / self.POSITION_SCALE)
        robot_y *= (1000.0 / self.POSITION_SCALE)
        robot_z *= (1000.0 / self.POSITION_SCALE)

        return np.array([robot_x, robot_y, robot_z])

    def quaternion_to_rotation_matrix(self, qx, qy, qz, qw):
        """四元数转旋转矩阵"""
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm < 1e-6:
            return np.eye(3)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

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
        """旋转矩阵转RPY角"""
        epsilon = 1e-6
        if abs(R[2, 0]) > 1 - epsilon:
            pitch = np.arcsin(-R[2, 0])
            roll = np.arctan2(-R[0, 1], R[1, 1])
            yaw = 0
        else:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arcsin(-R[2, 0])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        return [roll, pitch, yaw]

    def rpy_to_rotation_matrix(self, roll, pitch, yaw):
        """RPY角转旋转矩阵"""
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
        """VR手柄旋转转换到机械臂坐标系（半镜像版本）"""
        R_vr = self.quaternion_to_rotation_matrix(qx, qy, qz, qw)

        T_vr_to_robot_half_mirror = np.array([
            [ 0,  0, -1],
            [ 1,  0,  0],
            [ 0,  1,  0]
        ])

        R_robot = T_vr_to_robot_half_mirror @ R_vr @ T_vr_to_robot_half_mirror.T

        rpy = self.rotation_matrix_to_rpy(R_robot)
        rpy[0] = -rpy[0]
        rpy[2] = -rpy[2]

        R_robot_half_mirrored = self.rpy_to_rotation_matrix(rpy[0], rpy[1], rpy[2])
        return R_robot_half_mirrored

    def rotation_matrix_to_axis_angle(self, R):
        """旋转矩阵转轴角表示"""
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if angle < 1e-6:
            return np.array([0, 0, 1, 0])

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

    async def initialize_robot(self):
        """初始化机械臂 + 相机"""
        print("初始化UF850机械臂...")

        try:
            self.arm = XArmAPI(self.robot_ip, is_radian=True)
            print(f"✓ 连接成功: {self.robot_ip}")
        except Exception as e:
            print(f"✗ 连接失败: {e}")
            return False

        # 配置机械臂
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)
        self.arm.set_state(0)

        # 移动到初始位置
        print("移动到初始位置...")
        initial = config.INITIAL_POSITION.copy()
        initial[3:] = np.deg2rad(initial[3:])

        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.set_position(*initial, wait=True)

        self.arm.set_mode(1)
        self.arm.set_state(0)

        print("✓ 机械臂初始化完成\n")

        # ========== 新增: 初始化相机 ==========
        print("初始化相机...")
        try:
            self.camera_manager = CameraManager(num_cameras=1, fps=30)  # 临时使用1个相机

            # 启动相机采集线程
            self.camera_thread = threading.Thread(
                target=self._camera_capture_loop,
                daemon=True
            )
            self.camera_thread.start()

            print("✓ 相机初始化完成\n")
        except Exception as e:
            print(f"⚠ 相机初始化失败: {e}")
            print("   将继续运行但不保存视频数据\n")

        return True

    def _camera_capture_loop(self):
        """相机采集线程 (30fps)"""
        print("相机采集线程已启动 (30fps)")
        while True:
            try:
                frames = self.camera_manager.get_frames()
                with self.camera_lock:
                    self.latest_frames = frames
                time.sleep(1.0 / 30)  # 30fps
            except Exception as e:
                print(f"相机采集错误: {e}")
                time.sleep(0.1)

    def _get_current_state(self):
        """
        获取机械臂当前状态 (8维)

        Returns:
            np.ndarray [8]: [j0, j1, j2, j3, j4, j5, j6, gripper] (弧度)
        """
        # 获取关节角度 (弧度)
        code, angles = self.arm.get_servo_angle(is_radian=True)
        if code != 0:
            return None

        # 获取夹爪位置并归一化
        code, gripper_pos = self.arm.get_gripper_position()
        if code != 0:
            gripper_normalized = 0.5
        else:
            gripper_normalized = gripper_pos / 850.0

        # 组合为8维state (7个关节 + 1个夹爪)
        state = np.append(angles, gripper_normalized)
        return state

    def _compute_current_action(self, target_pose):
        """
        从目标笛卡尔位姿计算action (8维关节空间)

        Args:
            target_pose: [x, y, z, roll, pitch, yaw] (笛卡尔空间, mm和弧度)

        Returns:
            np.ndarray [8]: 目标关节角度 + 夹爪 (弧度)
        """
        # 方案1: 使用IK计算目标关节角度
        code, joint_angles = self.arm.get_inverse_kinematics(target_pose)
        if code != 0 or joint_angles is None:
            # IK失败，使用当前角度
            code, joint_angles = self.arm.get_servo_angle(is_radian=True)
            if code != 0:
                return None

        # 夹爪目标
        gripper_target = 1.0 if self.gripper_open else 0.0

        action = np.append(joint_angles, gripper_target)
        return action

    def calibrate(self, vr_x, vr_y, vr_z, vr_qx, vr_qy, vr_qz, vr_qw):
        """标定"""
        print("\n" + "="*50)
        print("执行标定...")

        code, pose = self.arm.get_position(is_radian=True)
        if code != 0:
            print("✗ 获取机械臂位姿失败")
            return False

        self.calibration_robot_pose = pose
        self.calibration_vr_pos = self.vr_to_robot_position(vr_x, vr_y, vr_z)
        self.calibration_vr_rot_matrix = self.vr_rotation_to_robot(vr_qx, vr_qy, vr_qz, vr_qw)
        self.calibration_robot_rot_matrix = self.rpy_to_rotation_matrix(pose[3], pose[4], pose[5])

        print(f"机械臂位置: X={pose[0]:.1f}, Y={pose[1]:.1f}, Z={pose[2]:.1f} mm")
        print(f"机械臂姿态: Roll={np.rad2deg(pose[3]):.1f}°, Pitch={np.rad2deg(pose[4]):.1f}°, Yaw={np.rad2deg(pose[5]):.1f}°")

        self.is_calibrated = True
        self.last_sent_pose = pose
        self.position_buffer.clear()
        self.rotation_buffer.clear()

        print("✓ 标定完成！")
        print("="*50 + "\n")
        return True

    def compute_target_pose(self, vr_x, vr_y, vr_z, vr_qx, vr_qy, vr_qz, vr_qw):
        """计算目标位姿（包含旋转）"""
        if not self.is_calibrated:
            return None

        # 位置部分
        current_vr_pos = self.vr_to_robot_position(vr_x, vr_y, vr_z)
        delta_pos = current_vr_pos - self.calibration_vr_pos

        if np.linalg.norm(delta_pos) < 2.0:
            delta_pos = np.zeros(3)

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

        # 旋转部分
        if self.ENABLE_ROTATION:
            current_vr_rot = self.vr_rotation_to_robot(vr_qx, vr_qy, vr_qz, vr_qw)

            if self.ROTATION_MODE == "incremental":
                R_relative = current_vr_rot @ self.calibration_vr_rot_matrix.T
                axis_angle = self.rotation_matrix_to_axis_angle(R_relative)
                axis_angle[3] *= self.ROTATION_SCALE
                R_relative_scaled = self.axis_angle_to_rotation_matrix(axis_angle)
                R_target = self.calibration_robot_rot_matrix @ R_relative_scaled
            else:
                R_target = current_vr_rot

            if self.last_sent_pose is not None:
                last_R = self.rpy_to_rotation_matrix(
                    self.last_sent_pose[3],
                    self.last_sent_pose[4],
                    self.last_sent_pose[5]
                )
                R_diff = R_target @ last_R.T
                angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))

                if angle_diff < self.ROTATION_DEADZONE:
                    target_rpy = self.last_sent_pose[3:6]
                else:
                    target_rpy = self.rotation_matrix_to_rpy(R_target)
            else:
                target_rpy = self.rotation_matrix_to_rpy(R_target)
        else:
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

    def apply_filter(self, pose):
        """应用位置和旋转滤波"""
        self.position_buffer.append(pose[:3])
        if len(self.position_buffer) > 0:
            filtered_pos = np.mean(self.position_buffer, axis=0)
        else:
            filtered_pos = pose[:3]

        self.rotation_buffer.append(pose[3:6])
        if len(self.rotation_buffer) > 0:
            filtered_rot = np.mean(self.rotation_buffer, axis=0)
        else:
            filtered_rot = pose[3:6]

        return list(filtered_pos) + list(filtered_rot)

    def control_gripper(self):
        """控制夹爪开关"""
        if config.GRIPPER_TYPE == 0:
            return

        try:
            if config.GRIPPER_TYPE == 1:
                pos = 850 if self.gripper_open else 0
                self.arm.set_gripper_position(pos, wait=False)
            elif config.GRIPPER_TYPE == 2:
                pos = 84 if self.gripper_open else 0
                self.arm.set_gripper_g2_position(pos, speed=225, wait=False)
            elif config.GRIPPER_TYPE == 3:
                pos = 150 if self.gripper_open else 71
                self.arm.set_bio_gripper_g2_position(pos, speed=4500, wait=False)
        except Exception as e:
            print(f"✗ 夹爪控制失败: {e}")

    async def handle_controller_data(self, websocket):
        """处理手柄数据 + 数据收集"""
        client_ip = websocket.remote_address[0]
        print(f"Quest 3 已连接: {client_ip}")

        if self.start_time is None:
            self.start_time = time.time()

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    self.frame_count += 1
                    self.data_collector.step_count += 1

                    if 'right' not in data or data['right'] is None:
                        continue

                    right = data['right']
                    pos = right['position']
                    ori = right['orientation']

                    # 调试输出
                    if self.frame_count % 50 == 0 and config.VERBOSE:
                        robot_pos = self.vr_to_robot_position(pos['x'], pos['y'], pos['z'])
                        print(f"\n[Frame {self.frame_count}]")
                        print(f"  VR位置: X={pos['x']:+.3f} Y={pos['y']:+.3f} Z={pos['z']:+.3f}")
                        print(f"  →Robot: X={robot_pos[0]:+.1f} Y={robot_pos[1]:+.1f} Z={robot_pos[2]:+.1f}")

                    # 标定检查
                    if not self.is_calibrated:
                        if 'buttons' in right and len(right['buttons']) > 0:
                            if right['buttons'][0].get('pressed', False):
                                self.calibrate(pos['x'], pos['y'], pos['z'],
                                             ori['x'], ori['y'], ori['z'], ori['w'])
                        continue

                    # ========== B键控制录制 ==========
                    # Quest 3手柄: buttons[5] = B键
                    if 'buttons' in right and len(right['buttons']) > 5:
                        b_button = right['buttons'][5].get('pressed', False)

                        # 检测B键按下 (上升沿)
                        if b_button and not self.last_b_button_state:
                            if not self.data_collector.is_recording:
                                # 开始录制
                                self.data_collector.start_episode()
                            else:
                                # 结束录制
                                self.data_collector.stop_episode()

                        self.last_b_button_state = b_button

                    # 计算目标位姿
                    target_pose = self.compute_target_pose(
                        pos['x'], pos['y'], pos['z'],
                        ori['x'], ori['y'], ori['z'], ori['w']
                    )

                    if target_pose is None:
                        continue

                    # 滤波
                    target_pose = self.apply_filter(target_pose)

                    # ========== 新增: 数据收集 ==========
                    if self.data_collector.should_record_this_step():
                        # 1. 获取当前state
                        current_state = self._get_current_state()

                        # 2. 计算当前action
                        current_action = self._compute_current_action(target_pose)

                        # 3. 获取相机帧
                        with self.camera_lock:
                            frames = self.latest_frames.copy()

                        # 4. 记录
                        if current_state is not None and current_action is not None:
                            self.data_collector.record_step(
                                state=current_state,
                                action=current_action,
                                frames=frames,
                                timestamp=time.time()
                            )

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

                    # Trigger控制夹爪
                    if 'buttons' in right and len(right['buttons']) > 0:
                        trigger = right['buttons'][0].get('pressed', False)
                        if trigger and not self.last_trigger_state:
                            self.gripper_open = not self.gripper_open
                            self.control_gripper()
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
        print(f"\nWebSocket服务器启动: 端口 {port}")

        if not self.is_calibrated:
            print("\n等待标定...")
            print("将右手柄移到舒适位置，按住Trigger键标定\n")

        print("数据收集说明:")
        print("  • 标定后，按B键开始录制Episode")
        print("  • 执行任务操作")
        print("  • 再次按B键结束录制")
        print("  • 可用Joystick上下切换预定义任务\n")

        async with websockets.serve(self.handle_controller_data, "0.0.0.0", port):
            await asyncio.Future()

    def shutdown(self):
        """关闭"""
        # 停止录制
        if self.data_collector.is_recording:
            print("\n检测到正在录制，自动保存...")
            self.data_collector.stop_episode()

        # 打印统计
        self.data_collector.print_statistics()

        # 释放相机
        if self.camera_manager:
            self.camera_manager.release()

        # 停止机械臂
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
    import sys

    # 解析命令行参数
    robot_ip = sys.argv[1] if len(sys.argv) > 1 else None
    dataset_path = sys.argv[2] if len(sys.argv) > 2 else "./uf850_teleop_dataset"

    controller = UF850WebXRControlWithDataCollection(
        robot_ip=robot_ip,
        dataset_path=dataset_path
    )

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
