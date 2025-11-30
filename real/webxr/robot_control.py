#!/usr/bin/env python3
"""
UF850 WebXR 遥操作控制（最终版）
- 使用 xarm_webxr_control.py 的 WebSocket 连接架构（能正常连接）
- 使用 xarm_webxr_control_fixed.py 的坐标转换逻辑（正确的映射）
"""

import asyncio
import websockets
import json
import numpy as np
import time
from collections import deque
from datetime import datetime
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xarm.wrapper import XArmAPI
from real import config


class Transformations:
    """坐标转换工具类"""

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """将四元数转换为旋转矩阵（xyzw顺序）"""
        x, y, z, w = q
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        if norm < 1e-6:
            return np.eye(3)

        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        R = np.array([
            [1 - 2*(yy + zz),     2*(xy - wz),      2*(xz + wy)],
            [    2*(xy + wz), 1 - 2*(xx + zz),      2*(yz - wx)],
            [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)]
        ])
        return R

    @staticmethod
    def rpy_to_rotation_matrix(roll, pitch, yaw):
        """RPY角到旋转矩阵的转换"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cp*cy,  -cr*sy + sr*sp*cy,    sr*sy + cr*sp*cy],
            [cp*sy,   cr*cy + sr*sp*sy,   -sr*cy + cr*sp*sy],
            [ -sp,        sr*cp,               cr*cp],
        ])
        return R

    @staticmethod
    def rotation_matrix_to_rpy(R, yaw_zero=True):
        """旋转矩阵到RPY角的转换"""
        epsilon = 1e-6
        if abs(R[2, 0]) > 1 - epsilon:  # 万向节锁
            pitch = np.arcsin(-R[2, 0])
            roll_yaw = np.arctan2(-R[0, 1], R[1, 1])
            if yaw_zero:
                roll, yaw = roll_yaw, 0
            else:
                roll, yaw = 0, roll_yaw
        else:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arcsin(-R[2, 0])
            yaw = np.arctan2(R[1, 0], R[0, 0])

        return roll, pitch, yaw


class MovingAverageFilter:
    """移动平均滤波器 - 减少抖动"""

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffers = {}

    def filter(self, key, value):
        if key not in self.buffers:
            self.buffers[key] = deque(maxlen=self.window_size)

        self.buffers[key].append(value)
        return np.mean(self.buffers[key], axis=0)

    def reset(self):
        self.buffers.clear()


class DeadZoneFilter:
    """死区滤波器 - 过滤微小抖动"""

    def __init__(self, position_threshold=0.002, rotation_threshold=0.01):
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold
        self.last_values = {}

    def filter(self, key, position, rotation=None):
        if key not in self.last_values:
            self.last_values[key] = (np.array(position),
                                    np.array(rotation) if rotation is not None else None)
            return (position, rotation) if rotation is not None else position

        last_pos, last_rot = self.last_values[key]

        # 位置死区
        pos_diff = np.linalg.norm(np.array(position) - last_pos)
        if pos_diff < self.position_threshold:
            filtered_pos = last_pos
        else:
            filtered_pos = np.array(position)
            last_pos = filtered_pos

        # 旋转死区
        if rotation is not None and last_rot is not None:
            rot_diff = np.linalg.norm(np.array(rotation) - last_rot)
            if rot_diff < self.rotation_threshold:
                filtered_rot = last_rot
            else:
                filtered_rot = np.array(rotation)
                last_rot = filtered_rot
        else:
            filtered_rot = rotation

        self.last_values[key] = (filtered_pos, filtered_rot)

        return (filtered_pos.tolist(), filtered_rot.tolist() if filtered_rot is not None else None) \
               if rotation is not None else filtered_pos.tolist()


class UF850WebXRControl:
    """UF850 WebXR 遥操作控制器（最终版）"""

    def __init__(self, robot_ip=None):
        self.robot_ip = robot_ip or config.ROBOT_IP
        self.arm = None

        # VR 坐标系到机械臂坐标系的固定变换
        # 面对面操作：绕X轴旋转180度
        face_to_face = False  # 可以改为 False 切换到同向模式

        if face_to_face:
            self.vr_to_robot_transform = self.xyzrpy_to_matrix(
                0, 0, 0,  # 位置偏移
                np.pi, 0, 0  # 面对面镜像：绕X轴180度
            )
        else:
            self.vr_to_robot_transform = self.xyzrpy_to_matrix(
                0, 0, 0,  # 位置偏移
                0, 0, 0  # 同向：不镜像
            )

        # 滤波器 - 减少抖动
        self.moving_avg_filter = MovingAverageFilter(window_size=3)
        self.deadzone_filter = DeadZoneFilter(
            position_threshold=0.001,  # 1mm 死区
            rotation_threshold=0.005   # 约 0.3 度死区
        )

        # 标定状态
        self.is_calibrated = False
        self.robot_base_matrix = None      # 机械臂标定时的初始位姿矩阵
        self.vr_begin_robot_matrix = None  # VR 标定时在机械臂坐标系中的位姿矩阵

        # 控制状态
        self.last_target_pose = None
        self.last_command_time = None
        self.gripper_open = False
        self.last_trigger_state = False

        # 统计
        self.frame_count = 0
        self.start_time = None
        self.connected_clients = set()

    @staticmethod
    def xyzrpy_to_matrix(x, y, z, roll, pitch, yaw):
        """构造4x4齐次变换矩阵"""
        T = np.eye(4)
        T[:3, :3] = Transformations.rpy_to_rotation_matrix(roll, pitch, yaw)
        T[:3, 3] = [x, y, z]
        return T

    @staticmethod
    def xyzq_to_matrix(x, y, z, qx, qy, qz, qw):
        """从位置和四元数构造4x4矩阵"""
        T = np.eye(4)
        T[:3, :3] = Transformations.quaternion_to_rotation_matrix([qx, qy, qz, qw])
        T[:3, 3] = [x, y, z]
        return T

    @staticmethod
    def matrix_to_xyzrpy(T):
        """从4x4矩阵提取xyzrpy"""
        x, y, z = T[0, 3], T[1, 3], T[2, 3]
        roll, pitch, yaw = Transformations.rotation_matrix_to_rpy(T[:3, :3])
        return [x, y, z, roll, pitch, yaw]

    def vr_pose_to_robot_matrix(self, x, y, z, qx, qy, qz, qw):
        """将 VR 手柄位姿转换到机械臂坐标系"""
        # VR 手柄的4x4矩阵（米）
        vr_matrix = self.xyzq_to_matrix(x, y, z, qx, qy, qz, qw)

        # 应用缩放（米 → 毫米）
        vr_matrix[:3, 3] *= (1000.0 / config.SCALE_FACTOR)

        # 转换到机械臂坐标系
        robot_matrix = vr_matrix @ self.vr_to_robot_transform

        return robot_matrix

    def compute_robot_target_pose(self, vr_end_robot_matrix):
        """计算机械臂目标位姿"""
        # robot_target = robot_base @ inv(vr_begin_robot) @ vr_end_robot
        inv_vr_begin = np.linalg.inv(self.vr_begin_robot_matrix)
        relative_transform = inv_vr_begin @ vr_end_robot_matrix
        robot_target_matrix = self.robot_base_matrix @ relative_transform

        target_pose = self.matrix_to_xyzrpy(robot_target_matrix)

        # 应用安全限制
        target_pose = self.apply_safety_limits(target_pose)

        return target_pose

    def apply_safety_limits(self, target_pose):
        """应用安全限制"""
        # 1. 工作空间限制
        target_pose[0] = np.clip(target_pose[0],
                                config.WORKSPACE_LIMITS['x'][0],
                                config.WORKSPACE_LIMITS['x'][1])
        target_pose[1] = np.clip(target_pose[1],
                                config.WORKSPACE_LIMITS['y'][0],
                                config.WORKSPACE_LIMITS['y'][1])
        target_pose[2] = np.clip(target_pose[2],
                                config.WORKSPACE_LIMITS['z'][0],
                                config.WORKSPACE_LIMITS['z'][1])

        # 2. 单步最大位移限制
        if self.last_target_pose is not None:
            delta_pos = np.array(target_pose[:3]) - np.array(self.last_target_pose[:3])
            delta_distance = np.linalg.norm(delta_pos)

            if delta_distance > config.MAX_DELTA_POSITION:
                # 限制移动幅度
                scale = config.MAX_DELTA_POSITION / delta_distance
                target_pose[:3] = self.last_target_pose[:3] + delta_pos * scale

        # 3. 平滑处理
        if self.last_target_pose is not None and config.POSITION_SMOOTHING > 0:
            alpha = 1.0 - config.POSITION_SMOOTHING
            for i in range(3):  # 只平滑位置
                target_pose[i] = alpha * target_pose[i] + (1 - alpha) * self.last_target_pose[i]

        return target_pose

    async def initialize_robot(self):
        """初始化机械臂"""
        print("=" * 60)
        print("UF850 WebXR 遥操作系统初始化")
        print("=" * 60)

        print(f"连接机械臂: {self.robot_ip}")
        try:
            self.arm = XArmAPI(self.robot_ip, is_radian=True)
            print("✓ 机械臂连接成功")
        except Exception as e:
            print(f"✗ 机械臂连接失败: {e}")
            return False

        # 配置机械臂
        print("配置机械臂...")
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(config.ROBOT_MODE)
        self.arm.set_state(0)

        # 设置速度限制因子
        self.arm.set_linear_spd_limit_factor(2.0)

        mode_name = {1: "伺服模式", 7: "在线轨迹规划模式"}.get(config.ROBOT_MODE, "未知模式")
        print(f"模式: {mode_name} (Mode {config.ROBOT_MODE})")
        print(f"缩放因子: {config.SCALE_FACTOR}")

        # 移动到初始位置（不等待完成，避免阻塞）
        print("移动到初始安全位置...")
        initial_pos_rad = config.INITIAL_POSITION.copy()
        initial_pos_rad[3:] = np.deg2rad(initial_pos_rad[3:])

        ret = self.arm.set_position(*initial_pos_rad, wait=False, radius=0)
        if ret != 0:
            print(f"⚠ 移动命令返回码: {ret}")
        else:
            print("✓ 移动指令已发送（机械臂将在后台移动到初始位置）")

        # 等待几秒让机械臂开始移动
        await asyncio.sleep(3)

        print("✓ 所有组件初始化完成")
        print("=" * 60)
        return True

    def calibrate(self, vr_robot_matrix):
        """执行标定"""
        print("\n" + "=" * 60)
        print("执行标定...")
        print("=" * 60)

        # 获取机械臂当前位姿
        code, current_pose = self.arm.get_position(is_radian=True)
        if code != 0:
            print(f"✗ 无法获取机械臂位姿，错误代码: {code}")
            return False

        print(f"机械臂位姿: [{current_pose[0]:.1f}, {current_pose[1]:.1f}, {current_pose[2]:.1f}] mm")

        # 保存标定数据
        self.robot_base_matrix = self.xyzrpy_to_matrix(*current_pose)
        self.vr_begin_robot_matrix = vr_robot_matrix

        # 重置滤波器
        self.moving_avg_filter.reset()

        # 标定完成
        self.is_calibrated = True
        self.last_target_pose = current_pose
        self.last_command_time = time.time()

        print("✓ 标定完成！")
        print("  - 现在可以移动手柄控制机械臂")
        print("=" * 60 + "\n")

        return True

    def send_to_robot(self, target_pose):
        """发送目标位姿到机械臂"""
        try:
            if config.ROBOT_MODE == 7:
                # Mode 7: 在线轨迹规划模式
                ret = self.arm.set_position(
                    *target_pose,
                    radius=0,
                    speed=config.TRAJECTORY_SPEED,
                    mvacc=config.TRAJECTORY_ACCELERATION,
                    wait=False
                )
            else:
                # Mode 1: 伺服模式
                ret = self.arm.set_servo_cartesian(
                    target_pose,
                    speed=config.SERVO_SPEED,
                    mvacc=config.SERVO_ACCELERATION
                )

            if ret != 0 and config.VERBOSE:
                print(f"⚠ 运动指令返回码: {ret}")

        except Exception as e:
            print(f"✗ 发送运动指令失败: {e}")

    def handle_gripper(self, trigger_pressed):
        """处理夹爪控制"""
        if config.GRIPPER_TYPE == 0:
            return

        if trigger_pressed and not self.last_trigger_state:
            self.gripper_open = not self.gripper_open

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

                status = "打开" if self.gripper_open else "关闭"
                print(f"夹爪{status}")
            except Exception as e:
                print(f"✗ 夹爪控制失败: {e}")

        self.last_trigger_state = trigger_pressed

    async def handle_controller_data(self, websocket):
        """处理来自 Quest 3 的手柄数据"""
        self.connected_clients.add(websocket)
        client_ip = websocket.remote_address[0]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Quest 3 已连接! IP: {client_ip}")

        if self.start_time is None:
            self.start_time = time.time()

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    self.frame_count += 1

                    # 使用右手柄控制
                    if 'right' not in data or data['right'] is None:
                        continue

                    right_hand = data['right']

                    # 提取位置和旋转（VR 原始坐标，米）
                    pos = right_hand['position']
                    ori = right_hand['orientation']

                    # 转换到机械臂坐标系
                    vr_robot_matrix = self.vr_pose_to_robot_matrix(
                        pos['x'], pos['y'], pos['z'],
                        ori['x'], ori['y'], ori['z'], ori['w']
                    )

                    # 检查标定
                    if not self.is_calibrated:
                        if 'buttons' in right_hand and len(right_hand['buttons']) > 0:
                            if right_hand['buttons'][0].get('pressed', False):
                                self.calibrate(vr_robot_matrix)
                        continue

                    # 计算目标位姿
                    target_pose = self.compute_robot_target_pose(vr_robot_matrix)

                    # 应用滤波
                    target_pose = self.moving_avg_filter.filter('target_pose', target_pose)

                    # 死区滤波
                    position = target_pose[:3]
                    rotation = target_pose[3:]
                    position, rotation = self.deadzone_filter.filter('pose', position, rotation)
                    target_pose = list(position) + list(rotation)

                    # 限制发送频率
                    now = time.time()
                    if self.last_command_time is None or \
                       now - self.last_command_time >= 1.0/config.CONTROL_FREQUENCY:

                        # 发送到机械臂
                        self.send_to_robot(target_pose)
                        self.last_command_time = now
                        self.last_target_pose = target_pose

                    # 处理夹爪
                    if 'buttons' in right_hand and len(right_hand['buttons']) > 0:
                        trigger_pressed = right_hand['buttons'][0].get('pressed', False)
                        self.handle_gripper(trigger_pressed)

                    # 显示信息
                    if config.SHOW_REALTIME_POSE and self.frame_count % 100 == 0:
                        print(f"位姿: [{target_pose[0]:.1f}, {target_pose[1]:.1f}, {target_pose[2]:.1f}] mm")

                except json.JSONDecodeError as e:
                    print(f"JSON 解析错误: {e}")
                except Exception as e:
                    print(f"处理数据错误: {e}")
                    import traceback
                    traceback.print_exc()

        except websockets.exceptions.ConnectionClosed:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Quest 3 断开连接")
        finally:
            self.connected_clients.remove(websocket)

    async def run_server(self):
        """运行 WebSocket 服务器"""
        port = 8765
        print("=" * 60)
        print("WebSocket 服务器启动中...")
        print(f"监听端口: {port}")

        if not self.is_calibrated:
            print("\n等待标定...")
            print("在 Quest 3 VR 中,将右手柄移动到舒适位置")
            print("然后按住 Trigger 键进行标定")

        print("=" * 60)

        async with websockets.serve(self.handle_controller_data, "0.0.0.0", port):
            await asyncio.Future()

    def shutdown(self):
        """关闭系统"""
        print("\n" + "=" * 60)
        print("关闭系统...")
        print("=" * 60)

        if self.start_time is not None:
            duration = time.time() - self.start_time
            avg_freq = self.frame_count / duration if duration > 0 else 0
            print(f"运行时长: {duration:.1f} 秒")
            print(f"接收帧数: {self.frame_count}")
            print(f"平均频率: {avg_freq:.1f} Hz")

        if self.arm is not None:
            try:
                self.arm.set_state(4)
                print("✓ 机械臂已暂停")
            except:
                pass

        print("✓ 系统已关闭")
        print("=" * 60)


async def main():
    """主函数"""
    robot_ip = None
    if len(sys.argv) >= 2:
        robot_ip = sys.argv[1]

    controller = UF850WebXRControl(robot_ip=robot_ip)

    if not await controller.initialize_robot():
        print("初始化失败，退出")
        return 1

    try:
        await controller.run_server()
    except KeyboardInterrupt:
        print("\n用户中断程序")
    finally:
        controller.shutdown()

    return 0


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已退出")