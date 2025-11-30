import openvr
import time
import sys
import numpy as np

def main():
    # 初始化 OpenVR
    try:
        openvr.init(openvr.VRApplication_Scene)
        print("OpenVR 初始化成功!")
    except openvr.OpenVRError as e:
        print(f"初始化失败: {e}")
        sys.exit(1)

    vr_system = openvr.VRSystem()

    print("开始读取手柄数据，按 Ctrl+C 退出...\n")

    try:
        while True:
            # 获取设备姿态
            poses = vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding,
                0,
                openvr.k_unMaxTrackedDeviceCount
            )

            # 遍历所有设备
            for i in range(openvr.k_unMaxTrackedDeviceCount):
                if poses[i].bPoseIsValid:
                    device_class = vr_system.getTrackedDeviceClass(i)

                    # 只显示手柄（Controller）
                    if device_class == openvr.TrackedDeviceClass_Controller:
                        # 获取设备角色（左手或右手）
                        role = vr_system.getControllerRoleForTrackedDeviceIndex(i)

                        if role == openvr.TrackedControllerRole_LeftHand:
                            hand = "左手"
                        elif role == openvr.TrackedControllerRole_RightHand:
                            hand = "右手"
                        else:
                            hand = "未知"

                        # 获取 3x4 变换矩阵
                        pose_3x4 = poses[i].mDeviceToAbsoluteTracking

                        # 转换为 4x4 齐次变换矩阵
                        T = np.eye(4)
                        T[0, :] = pose_3x4[0]
                        T[1, :] = pose_3x4[1]
                        T[2, :] = pose_3x4[2]

                        # 提取位置
                        position = T[:3, 3]

                        # 提取旋转矩阵
                        rotation = T[:3, :3]

                        print(f"\n{hand} 手柄 [设备 {i}]:")
                        print(f"位置 (x, y, z): [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
                        print(f"齐次变换矩阵 T:")
                        print(T)
                        print("-" * 60)

            time.sleep(0.5)  # 500ms 更新一次

    except KeyboardInterrupt:
        print("\n停止读取")
    finally:
        openvr.shutdown()
        print("OpenVR 已关闭")

if __name__ == "__main__":
    main()
