# UF850 VR遥操作数据收集系统

为GR00T训练收集符合LeRobot V2.0格式的演示轨迹数据。

## 🎯 功能特性

- ✅ Quest 3 VR遥操作控制
- ✅ 实时数据收集 (30Hz采样)
- ✅ 双相机支持 (RealSense D435 / USB相机)
- ✅ GR00T LeRobot V2.0格式保存
- ✅ 按键控制录制开始/结束
- ✅ 预定义任务管理

## 📦 系统要求

### 硬件
- UF850机械臂
- Meta Quest 3 VR头显
- 2个RealSense D435相机 (可选: 普通USB相机)
- 主机: Ubuntu 20.04/22.04

### 软件依赖
```bash
# 基础依赖
pip install numpy pandas pyarrow opencv-python websockets

# 相机支持 (可选)
pip install pyrealsense2

# xArm SDK (已安装在虚拟环境中)
```

## 🚀 快速开始

### 1. 启动系统

**终端1: 启动遥操作+数据收集**
```bash
cd /home/enyi/Code/UF850/xArm-Python-SDK
source uf850/bin/activate
cd /home/enyi/Code/UF850/teleVR/real/webxr

# 使用默认数据集路径
python3 robot_control_with_data_collection.py

# 或指定自定义路径
python3 robot_control_with_data_collection.py 192.168.1.117 ./my_dataset
```

**终端2: 启动WebXR服务**
```bash
cd /home/enyi/Code/UF850/teleVR/real/webxr
./start_usb.sh
```

### 2. Quest 3连接

1. 打开Quest 3浏览器
2. 输入网址: `http://localhost:8080/index.html`
3. 点击"Start VR"进入VR模式

### 3. 标定

- 将右手柄移动到舒适位置
- 按住**Trigger键**进行标定
- 看到"✓ 标定完成！"提示

### 4. 数据收集

1. **切换任务** (可选):
   - 摇杆向上: 下一个任务
   - 摇杆向下: 上一个任务

2. **开始录制**:
   - 按**B键**开始录制Episode
   - 终端显示: "🔴 开始录制 Episode X"

3. **执行任务**:
   - 使用VR手柄遥操作机械臂
   - Trigger键控制夹爪开关
   - 系统自动30Hz采样记录数据

4. **结束录制**:
   - 再次按**B键**停止录制
   - 终端显示: "⏹ 停止录制 Episode X"
   - 数据自动保存

5. **重复**:
   - 重复步骤2-4收集更多episodes

## 🎮 按键说明

| 按键 | 功能 |
|------|------|
| **Trigger** | 标定 (首次) / 控制夹爪开关 |
| **B键** | 开始/结束录制Episode |
| **Joystick上下** | 切换预定义任务 |

## 📂 数据集结构

```
uf850_teleop_dataset/
├── meta/
│   ├── modality.json          # GR00T配置文件
│   ├── info.json              # 数据集元信息
│   ├── episodes.jsonl         # Episode列表
│   └── tasks.jsonl            # 任务描述
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.wrist/
        │   ├── episode_000000.mp4
        │   └── ...
        └── observation.images.front/
            ├── episode_000000.mp4
            └── ...
```

## 🔧 配置说明

### 预定义任务

编辑 `data_collection/data_collector.py`:
```python
PREDEFINED_TASKS = [
    "pick the cup and place it on the plate",
    "pick the bottle and place it in the box",
    # 添加你的任务...
]
```

### 采样频率

在 `robot_control_with_data_collection.py`:
```python
self.data_collector = DataCollector(
    dataset_path=dataset_path,
    record_freq=30,      # 数据记录频率 (Hz)
    control_freq=100     # 控制循环频率 (Hz)
)
```

### 相机配置

在 `data_collection/camera_manager.py`:
```python
cam_manager = CameraManager(
    num_cameras=2,        # 相机数量
    width=640,            # 图像宽度
    height=480,           # 图像高度
    fps=30,               # 帧率
    use_realsense=True    # True=RealSense, False=USB相机
)
```

## 📊 数据格式

### State (observation.state)
- 维度: 7
- 内容: [joint_0, joint_1, ..., joint_5, gripper]
- 单位: 弧度 (关节), 归一化0-1 (夹爪)

### Action
- 维度: 7
- 内容: [target_joint_0, ..., target_joint_5, target_gripper]
- 单位: 弧度 (关节), 归一化0-1 (夹爪)
- 类型: 绝对位置

### Video
- 格式: MP4 (H.264编码)
- 分辨率: 640x480
- 帧率: 30fps
- 相机: wrist (手腕) + front (正面)

## 🧪 测试

### 测试相机
```bash
cd data_collection
python3 camera_manager.py
```

### 查看数据集统计
```python
from data_collection import DataCollector

collector = DataCollector("./uf850_teleop_dataset")
collector.print_statistics()
```

## 🐛 常见问题

### 1. 相机检测失败
```
⚠ 未检测到RealSense设备，回退到USB相机模式
```
**解决**:
- 检查D435是否正确连接
- 安装pyrealsense2: `pip install pyrealsense2`
- 使用USB 3.0接口

### 2. Episode保存失败
```
❌ 保存失败: No module named 'pandas'
```
**解决**:
```bash
pip install pandas pyarrow
```

### 3. 视频编码失败
```
✗ 无法创建视频文件
```
**解决**:
- 检查opencv安装: `pip install opencv-python`
- 确保有写入权限

## 📈 数据收集建议

### 初期验证 (1-2小时)
- 单任务: "pick cup"
- 收集: 20条成功轨迹
- 目的: 验证pipeline

### 小规模实验 (半天)
- 单任务: "pick cup"
- 收集: 100条轨迹 (不同位置/角度)
- 目的: 验证模型学习

### 正式数据集 (1-2天)
- 3-5个任务
- 每任务: 100-200条
- 总计: 500-1000条

## 📚 相关链接

- [GR00T官方文档](https://github.com/NVlabs/GR00T)
- [LeRobot V2.0格式](https://huggingface.co/docs/lerobot)
- [xArm Python SDK](https://github.com/xArm-Developer/xArm-Python-SDK)

## 📝 许可证

MIT License
