# 🤖 UF850数据收集完整指南

## 📋 目录
1. [系统概述](#系统概述)
2. [环境准备](#环境准备)
3. [使用流程](#使用流程)
4. [数据格式说明](#数据格式说明)
5. [故障排除](#故障排除)

---

## 系统概述

本系统用于通过Quest 3 VR头显遥操作UF850机械臂，并自动收集符合GR00T LeRobot V2.0格式的演示轨迹数据。

### 核心特性
- ✅ **实时遥操作**: 100Hz控制频率，流畅自然
- ✅ **智能采样**: 30Hz数据记录，与GR00T官方示例对齐
- ✅ **双相机支持**: 手腕+正面视角，支持RealSense D435
- ✅ **一键录制**: B键开始/结束，简单直观
- ✅ **标准格式**: 完全符合LeRobot V2.0 + GR00T modality规范

---

## 环境准备

### 1. 检查硬件连接

```bash
# 检查机械臂
ping 192.168.1.117

# 检查相机 (RealSense)
rs-enumerate-devices

# 检查相机 (USB)
ls /dev/video*
```

### 2. 安装依赖

```bash
cd /home/enyi/Code/UF850/xArm-Python-SDK
source uf850/bin/activate

# 安装必需依赖
pip install numpy pandas pyarrow opencv-python websockets

# 安装RealSense支持 (如果使用D435)
pip install pyrealsense2
```

### 3. 验证安装

```bash
# 测试相机
cd /home/enyi/Code/UF850/teleVR/real/webxr
python3 -c "from data_collection import CameraManager; CameraManager()"

# 应该看到: ✓ 相机X初始化成功
```

---

## 使用流程

### 步骤1: 启动系统

**终端1 - 遥操作+数据收集**
```bash
cd /home/enyi/Code/UF850/xArm-Python-SDK
source uf850/bin/activate
cd /home/enyi/Code/UF850/teleVR/real/webxr

# 启动 (使用默认数据集路径 ./uf850_teleop_dataset)
python3 robot_control_with_data_collection.py

# 或指定自定义路径
python3 robot_control_with_data_collection.py 192.168.1.117 /path/to/your/dataset
```

**终端2 - WebXR服务**
```bash
cd /home/enyi/Code/UF850/teleVR/real/webxr
./start_usb.sh
```

应该看到输出：
```
============================================================
UF850 WebXR遥操作 + 数据收集系统
============================================================
...
✓ 机械臂初始化完成
✓ 相机初始化完成
WebSocket服务器启动: 端口 8765
```

### 步骤2: 连接Quest 3

1. 戴上Quest 3头显
2. 打开浏览器
3. 输入网址: `http://localhost:8080/index.html`
4. 点击"Start VR"

### 步骤3: 标定

- 将右手柄移动到舒适的操作位置
- **按住Trigger键** (食指扳机)
- 终端显示:
  ```
  ==================================================
  执行标定...
  机械臂位置: X=400.0, Y=0.0, Z=450.0 mm
  ✓ 标定完成！
  ==================================================
  ```

### 步骤4: 收集数据

#### 4.1 切换任务 (可选)

用右手柄**Joystick上下拨动**切换预定义任务:
- 向上: 下一个任务
- 向下: 上一个任务

终端显示:
```
切换任务: [1] pick the bottle and place it in the box
```

#### 4.2 开始录制

- 按**B键** (右手柄上方按钮)
- 终端显示:
  ```
  ============================================================
  🔴 开始录制 Episode 0
     任务: [0] pick the cup and place it on the plate
     采样频率: 30 Hz
  ============================================================
  ```

#### 4.3 执行任务

- 移动手柄控制机械臂
- **Trigger键**控制夹爪开关
- 完成任务操作 (例如: 抓取→移动→放置)

系统会每100帧显示进度:
```
  录制中... 已记录100帧 (3.3s, 实际频率=30.1Hz)
  录制中... 已记录200帧 (6.7s, 实际频率=30.0Hz)
```

#### 4.4 结束录制

- 再次按**B键**
- 终端显示:
  ```
  ============================================================
  ⏹ 停止录制 Episode 0
     帧数: 450
     时长: 15.00s
     实际频率: 30.0 Hz
  ============================================================
  💾 保存数据中...
    ✓ Parquet: episode_000000.parquet (450行)
    ✓ 视频: wrist 450帧 (1.12MB)
    ✓ 视频: front 450帧 (1.08MB)
  ✅ Episode 0 保存完成!
  ```

#### 4.5 继续收集

- 重置场景 (手动复位物体)
- 重复步骤4.2-4.4
- Episode索引自动递增 (0, 1, 2, ...)

### 步骤5: 查看统计

按 `Ctrl+C` 退出时，系统自动显示:
```
============================================================
数据收集统计
============================================================
  已收集Episodes: 10
  下一个Episode索引: 10
  录制状态: ⚪ 未录制
  当前任务: [0] pick the cup and place it on the plate
============================================================
```

---

## 数据格式说明

### 数据集目录结构

```
uf850_teleop_dataset/
├── meta/
│   ├── modality.json          # GR00T特有配置
│   ├── info.json              # 数据集元信息
│   ├── episodes.jsonl         # Episode元数据
│   └── tasks.jsonl            # 任务列表
├── data/
│   └── chunk-000/
│       └── episode_XXXXXX.parquet  # 低维数据
└── videos/
    └── chunk-000/
        ├── observation.images.wrist/
        │   └── episode_XXXXXX.mp4
        └── observation.images.front/
            └── episode_XXXXXX.mp4
```

### Parquet文件内容

| 列名 | 类型 | 说明 |
|------|------|------|
| `observation.state` | float64[7] | 6关节角度+夹爪 (弧度) |
| `action` | float64[7] | 目标关节角度+夹爪 |
| `timestamp` | float64 | 时间戳 (秒) |
| `episode_index` | int64 | Episode索引 |
| `frame_index` | int64 | 帧索引 |
| `task_index` | int64 | 任务索引 |
| `next.done` | bool | 是否结束 |

### 视频格式

- **编码**: H.264 (MP4)
- **分辨率**: 640x480
- **帧率**: 30 fps
- **格式**: BGR (OpenCV标准)

---

## 故障排除

### 问题1: 相机初始化失败

**症状**:
```
⚠ 未检测到RealSense设备，回退到USB相机模式
```

**解决方案**:
1. 检查D435连接: `lsusb | grep Intel`
2. 重新安装SDK: `pip install --upgrade pyrealsense2`
3. 使用USB 3.0接口 (蓝色)
4. 如果仍失败，系统会自动使用USB相机

### 问题2: 机械臂连接超时

**症状**:
```
✗ 连接失败: timeout
```

**解决方案**:
1. 检查网络: `ping 192.168.1.117`
2. 检查IP配置: `cat ../config.py | grep ROBOT_IP`
3. 重启机械臂控制器

### 问题3: Quest 3无法连接

**症状**: 浏览器显示"无法访问此网站"

**解决方案**:
1. 确保终端2的`start_usb.sh`正在运行
2. 检查Quest 3与电脑在同一网络
3. 尝试使用电脑IP: `http://192.168.x.x:8080/index.html`

### 问题4: 数据保存失败

**症状**:
```
❌ 保存失败: No module named 'pandas'
```

**解决方案**:
```bash
pip install pandas pyarrow opencv-python
```

### 问题5: Episode太短被拒绝

**症状**:
```
⚠ 警告: Episode太短 (<10帧)，可能无效
是否仍要保存? (y/n):
```

**原因**: 录制时间 < 0.3秒 (少于10帧)

**解决方案**: 执行完整的任务操作后再结束录制

---

## 高级配置

### 自定义任务列表

编辑 `data_collection/data_collector.py`:
```python
PREDEFINED_TASKS = [
    "pick the cup and place it on the plate",
    "pick the bottle and place it in the box",
    "open the drawer and take out the pen",  # 添加新任务
]
```

### 调整采样频率

编辑 `robot_control_with_data_collection.py`:
```python
self.data_collector = DataCollector(
    dataset_path=dataset_path,
    record_freq=20,      # 改为20Hz
    control_freq=100
)
```

### 禁用相机 (仅测试用)

编辑 `robot_control_with_data_collection.py`:
```python
# 在initialize_robot()中注释掉相机初始化部分
# self.camera_manager = CameraManager(...)
```

---

## 数据质量检查

### 检查Episode数量
```bash
ls -l uf850_teleop_dataset/data/chunk-000/ | wc -l
```

### 检查视频大小
```bash
du -sh uf850_teleop_dataset/videos/
```

### 读取Parquet文件
```python
import pandas as pd
df = pd.read_parquet('uf850_teleop_dataset/data/chunk-000/episode_000000.parquet')
print(df.info())
print(df.head())
```

### 播放视频
```bash
vlc uf850_teleop_dataset/videos/chunk-000/observation.images.wrist/episode_000000.mp4
```

---

## 下一步: 训练GR00T

数据收集完成后，按照GR00T官方教程进行训练：

```bash
# 1. 准备数据集
python scripts/load_dataset.py --data_path ./uf850_teleop_dataset/

# 2. 训练
python scripts/gr00t_finetune.py \
    --dataset-path ./uf850_teleop_dataset/ \
    --num-gpus 1 \
    --batch-size 32 \
    --max-steps 10000
```

---

## 联系与支持

- 问题反馈: [GitHub Issues](https://github.com/your-repo/issues)
- GR00T官方: https://github.com/NVlabs/GR00T
- LeRobot文档: https://huggingface.co/docs/lerobot
