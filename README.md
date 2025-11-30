# TeleOperation_UF850_DataCollection

基于Quest 3的UF850 机械臂VR遥操作系统，用于采集LeRobot格式的专家演示轨迹数据，可用于VLA模型训练。

## 功能特性

- **VR 遥操作**: 使用 Quest 3 手柄实时控制 UF850 机械臂
- **半镜像模式**: 前后不镜像，左右镜像，适合面对面操作
- **旋转控制**: 支持末端执行器姿态控制
- **数据采集**: 采集符合 LeRobot/GR00T 格式的演示轨迹数据
- **多相机支持**: 支持单/双 RealSense 相机同时录制

## 环境配置

### 1. 创建虚拟环境

```bash
cd teleVR
python3 -m venv teleReal
source teleReal/bin/activate
```

### 2. 安装依赖

```bash
pip install numpy>=1.20.0
pip install pandas>=1.3.0
pip install pyarrow>=6.0.0
pip install opencv-python>=4.5.0
pip install websockets>=10.0
pip install pyrealsense2>=2.54.0  # 如需使用 RealSense 相机
```

### 3. 安装 xArm SDK

```bash
pip install xArm-Python-SDK
```

或从源码安装:
```bash
git clone https://github.com/xArm-Developer/xArm-Python-SDK.git
cd xArm-Python-SDK
pip install .
```

### 4. 安装 ADB (Android Debug Bridge)

```bash
sudo apt install android-tools-adb
```

## 硬件准备

1. **UF850 机械臂**: 确保机械臂已上电并连接到局域网
2. **Quest 3**: 开启开发者模式，通过 USB 连接到电脑
3. **RealSense 相机** (可选): 用于录制演示数据

## 配置

编辑 `real/config.py` 配置机械臂 IP 和其他参数:

```python
ROBOT_IP = "192.168.1.117"  # 修改为你的机械臂 IP
GRIPPER_TYPE = 1            # 夹爪类型: 1=标准夹爪, 2=G2夹爪, 3=仿生夹爪
SCALE_FACTOR = 2            # VR 到机械臂的缩放因子
```

---

## 功能一: 纯遥操作模式 (不采集数据)

适用于调试、测试或演示，不需要保存轨迹数据。

### 使用步骤

**终端 1** - 启动机械臂控制服务:
```bash
cd teleVR/real/webxr
source ../teleReal/bin/activate  # 或你的虚拟环境
python3 robot_control_half_mirror_with_rotation.py
```

**终端 2** - 启动 USB 连接服务:
```bash
cd teleVR/real/webxr
./start_usb.sh
```

**Quest 3** - 打开浏览器访问:
```
http://localhost:8080/index.html
```

### 操作说明

| 按键 | 功能 |
|------|------|
| Trigger | 首次按下进行标定，之后控制夹爪开关 |
| 移动手柄 | 控制机械臂末端位置 |
| 旋转手柄 | 控制机械臂末端姿态 |

---

## 功能二: 遥操作 + 数据采集模式

在遥操作的基础上，同时采集观测数据和轨迹数据，保存为 LeRobot 格式。

### 相机配置

编辑 `real/webxr/robot_control_with_data_collection.py` 第 252 行:

```python
# 单相机模式
self.camera_manager = CameraManager(num_cameras=1, fps=30)

# 双相机模式
self.camera_manager = CameraManager(num_cameras=2, fps=30)
```

### 使用步骤

**终端 1** - 启动带数据采集的控制服务:
```bash
cd teleVR/real/webxr
source ../teleReal/bin/activate
python3 robot_control_with_data_collection.py
```

**终端 2** - 启动 USB 连接服务:
```bash
cd teleVR/real/webxr
./start_usb.sh
```

**Quest 3** - 打开浏览器访问:
```
http://localhost:8080/index.html
```

### 操作说明

| 按键 | 功能 |
|------|------|
| Trigger | 首次按下进行标定，之后控制夹爪开关 |
| B 键 | 开始/结束录制当前 Episode |
| Joystick 上下 | 切换预定义任务 |
| 移动/旋转手柄 | 控制机械臂 |

### 数据保存位置

采集的数据保存在:
```
teleVR/real/webxr/uf850_teleop_dataset/
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
├── videos/
│   └── chunk-000/
│       └── observation.images.cam_0/
│           └── episode_000000.mp4
└── meta/
    ├── info.json
    ├── stats.json
    ├── episodes.jsonl
    └── tasks.jsonl
```
## License

MIT License
