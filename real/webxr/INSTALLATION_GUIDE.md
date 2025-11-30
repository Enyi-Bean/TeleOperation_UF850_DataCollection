# teleVR环境安装指南

## 📦 依赖库安装

### 前提条件

- 已有teleVR虚拟环境（包含xArm-Python-SDK）
- Python版本: 3.8+
- Ubuntu 20.04/22.04

### 安装步骤

#### 1. 激活teleVR虚拟环境

```bash
cd /home/enyi/Code/UF850/xArm-Python-SDK
source uf850/bin/activate  # 实际路径可能不同，根据你的环境调整
```

#### 2. 安装数据收集核心依赖

```bash
cd /home/enyi/Code/UF850/teleVR/real/webxr

# 方法1: 使用requirements.txt (推荐)
pip install -r data_collection/requirements.txt

# 方法2: 手动逐个安装
pip install numpy>=1.20.0
pip install pandas>=1.3.0
pip install pyarrow>=6.0.0
pip install opencv-python>=4.5.0
pip install websockets>=10.0
pip install pyrealsense2>=2.54.0
```

### 📋 依赖说明

| 库名 | 版本要求 | 用途 | 是否必需 |
|------|----------|------|---------|
| **numpy** | >=1.20.0 | 数组操作、关节角度处理 | ✅ 必需 |
| **pandas** | >=1.3.0 | DataFrame构建、Parquet保存 | ✅ 必需 |
| **pyarrow** | >=6.0.0 | Parquet文件格式支持 | ✅ 必需 |
| **opencv-python** | >=4.5.0 | 图像处理、MP4视频编码 | ✅ 必需 |
| **websockets** | >=10.0 | VR手柄数据通信（已有） | ✅ 必需 |
| **pyrealsense2** | >=2.54.0 | RealSense D435相机驱动 | ✅ 必需 |
| **xArm-Python-SDK** | - | UF850机械臂控制（已安装） | ✅ 必需 |

### 🔍 验证安装

#### 方法1: 快速检查

```bash
python3 -c "
import numpy
import pandas
import pyarrow
import cv2
import websockets
import pyrealsense2 as rs
print('✅ 所有依赖安装成功！')
"
```

**成功输出**：
```
✅ 所有依赖安装成功！
```

**如果失败**，会显示缺少哪个库，例如：
```
ModuleNotFoundError: No module named 'pandas'
```

#### 方法2: 检查各个库版本

```bash
python3 << 'EOF'
import numpy as np
import pandas as pd
import pyarrow as pa
import cv2
import websockets
import pyrealsense2 as rs

print("✅ 依赖库版本:")
print(f"  numpy:        {np.__version__}")
print(f"  pandas:       {pd.__version__}")
print(f"  pyarrow:      {pa.__version__}")
print(f"  opencv:       {cv2.__version__}")
print(f"  websockets:   {websockets.__version__}")
print(f"  pyrealsense2: {rs.__version__}")
EOF
```

**预期输出示例**：
```
✅ 依赖库版本:
  numpy:        1.24.3
  pandas:       2.0.2
  pyarrow:      12.0.0
  opencv:       4.7.0
  websockets:   11.0.3
  pyrealsense2: 2.54.1
```

#### 方法3: 测试RealSense相机

```bash
# 检测D435设备
rs-enumerate-devices

# 测试相机管理器
cd /home/enyi/Code/UF850/teleVR/real/webxr/data_collection
python3 camera_manager.py
```

**成功输出**：
```
============================================================
检测到 2 个RealSense设备
✓ 相机0 [wrist]: Intel RealSense D435 (S/N: xxxxxxx)
  分辨率: 640x480 @ 25fps
✓ 相机1 [front]: Intel RealSense D435 (S/N: yyyyyyy)
  分辨率: 640x480 @ 25fps
============================================================
```

### ⚠️ 常见问题

#### 问题1: pyrealsense2安装失败

**症状**：
```bash
pip install pyrealsense2
# 输出: ERROR: Could not find a version that satisfies the requirement pyrealsense2
```

**解决方案**：
```bash
# 方案1: 使用官方预编译包
pip install pyrealsense2

# 方案2: 从源码编译 (如果上面失败)
sudo apt-get install librealsense2-dkms librealsense2-utils
pip install pyrealsense2

# 方案3: 检查Python版本兼容性
python3 --version  # 确保是3.8-3.11
```

#### 问题2: pyarrow安装慢或失败

**症状**：下载速度慢或超时

**解决方案**：
```bash
# 使用清华镜像加速
pip install pyarrow -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 问题3: opencv-python与系统OpenCV冲突

**症状**：
```
ImportError: libGL.so.1: cannot open shared object file
```

**解决方案**：
```bash
# 安装系统依赖
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# 或使用headless版本
pip uninstall opencv-python
pip install opencv-python-headless
```

#### 问题4: RealSense设备权限问题

**症状**：
```
RuntimeError: No device connected
```

**解决方案**：
```bash
# 添加udev规则
sudo apt-get install librealsense2-udev-rules

# 重新插拔USB线，或重启电脑
```

### 🚀 完整测试流程

安装完成后，运行完整测试：

```bash
cd /home/enyi/Code/UF850/teleVR/real/webxr

# 测试1: 相机初始化
python3 -c "from data_collection import CameraManager; cm = CameraManager(num_cameras=2)"

# 测试2: 数据收集器初始化
python3 -c "from data_collection import DataCollector; dc = DataCollector('./test_dataset')"

# 测试3: 启动完整系统 (不连接机械臂，仅测试依赖)
# python3 robot_control_with_data_collection.py  # 会尝试连接机械臂
```

### 📝 依赖摘要（复制粘贴版）

```bash
# 激活环境
cd /home/enyi/Code/UF850/xArm-Python-SDK
source uf850/bin/activate

# 一键安装所有依赖
cd /home/enyi/Code/UF850/teleVR/real/webxr
pip install numpy>=1.20.0 pandas>=1.3.0 pyarrow>=6.0.0 opencv-python>=4.5.0 websockets>=10.0 pyrealsense2>=2.54.0

# 验证
python3 -c "import numpy, pandas, pyarrow, cv2, websockets, pyrealsense2; print('✅ 安装成功')"
```

### ✅ 安装完成标志

如果以下命令都成功，说明环境已就绪：

```bash
✅ python3 -c "import numpy, pandas, pyarrow, cv2, websockets, pyrealsense2"
✅ rs-enumerate-devices  # 显示2个D435设备
✅ python3 data_collection/camera_manager.py  # 成功初始化相机
```

**现在可以开始收集数据了！** 🎉

参考 `DATA_COLLECTION_GUIDE.md` 了解完整使用流程。
