# VR遥操作ManiSkill机械臂 - 使用说明

使用Meta Quest 3手柄通过OpenVR/SteamVR/ALVR遥操作ManiSkill仿真中的Panda机械臂。

## 📋 目录

- [系统要求](#系统要求)
- [安装依赖](#安装依赖)
- [文件结构](#文件结构)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [参数配置](#参数配置)
- [故障排除](#故障排除)
- [坐标系说明](#坐标系说明)

---

## 系统要求

- **操作系统**: Ubuntu 22.04
- **硬件**: Meta Quest 3 + 手柄
- **软件**:
  - SteamVR
  - ALVR (Air Light VR)
  - ManiSkill3
  - Python 3.9+

---

## 安装依赖

```bash
# 1. 安装Python依赖
pip install openvr scipy gymnasium

# 2. 确保ManiSkill已安装
cd /home/enyi/Code/Maniskill/ManiSkill
pip install -e .

# 3. 确保SteamVR和ALVR已配置
# 参考ALVR官方文档: https://github.com/alvr-org/ALVR
```

---

## 文件结构

```
/home/enyi/Code/UF850/teleVR/
├── test_controllers.py          # VR手柄测试脚本（您原有的）
├── vr_calibration.py            # VR标定脚本
├── vr_controller.py             # VR控制器模块
├── coordinate_transform.py      # 坐标转换模块
└── README_VR_TELEOPERATION.md   # 本文档

/home/enyi/Code/Maniskill/ManiSkill/mani_skill/examples/teleoperation/
└── vr_teleoperation_panda.py    # VR遥操作主脚本
```

---

## 快速开始

### 第1步：测试VR连接

```bash
cd /home/enyi/Code/UF850/teleVR

# 测试VR手柄是否正常工作
python test_controllers.py

# 应该看到右手柄的位置和变换矩阵实时显示
```

### 第2步：运行标定（可选，用于验证）

```bash
# 运行标定工具
python vr_calibration.py

# 操作步骤：
# 1. 将右手柄放到胸前
# 2. 按住Trigger键（食指扳机）
# 3. 看到"✓ 标定完成"
# 4. 移动手柄查看相对位移
# 5. 按Ctrl+C退出，数据保存到 calibration_data.json
```

### 第3步：运行VR遥操作

```bash
cd /home/enyi/Code/Maniskill/ManiSkill

# 基础用法（默认参数）
python -m mani_skill.examples.teleoperation.vr_teleoperation_panda

# 带自定义参数
python -m mani_skill.examples.teleoperation.vr_teleoperation_panda \
    --scale-factor 2.0 \
    --position-smooth 0.5 \
    --control-freq 30

# 录制轨迹
python -m mani_skill.examples.teleoperation.vr_teleoperation_panda \
    --record-dir ./demos/vr_demos
```

### 第4步：开始遥操作

1. **等待环境加载** - 看到ManiSkill的GUI窗口
2. **将右手柄放到胸前** - 选择一个舒适的起始位置
3. **按住Trigger键标定** - 看到"✓ 标定完成"提示
4. **开始控制**:
   - 移动手柄 → 机械臂末端跟随移动
   - 按Trigger键 → 切换夹爪开合
5. **完成任务** - 抓取红色方块，移动到绿色目标位置
6. **退出** - 在GUI窗口按'Q'键

---

## 详细使用说明

### 控制映射

| VR手柄操作 | 机械臂响应 |
|-----------|----------|
| 向前移动手柄 | 末端向后移动 |
| 向后移动手柄 | 末端向前移动 |
| 向左移动手柄 | 末端向右移动 |
| 向右移动手柄 | 末端向左移动 |
| 向上移动手柄 | 末端向上移动 |
| 向下移动手柄 | 末端向下移动 |
| 旋转手柄 | 末端旋转（保持相对关系） |
| 按Trigger | 切换夹爪开/关 |

### 工作流程详解

#### 1. 标定的意义

标定建立了VR手柄位置和机械臂末端位置的"虚拟链接"：

```
标定时：
VR手柄在: [1.2, 1.5, -0.8] m (胸前)
机械臂在: [0.0, 0.0, 0.5] m (初始位置)

运行时：
VR手柄移动到: [1.3, 1.5, -0.8] m (向右0.1m)
→ VR相对位移: [+0.1, 0, 0] m
→ 坐标转换: [-0, -0.1, 0] m (MS坐标系)
→ 机械臂目标: [0.0, -0.1, 0.5] m
```

#### 2. 坐标系转换

**VR坐标系 (OpenVR)**:
- X轴: 右
- Y轴: 上
- Z轴: 后（向您）

**ManiSkill坐标系**:
- X轴: 前（远离机械臂base）
- Y轴: 左
- Z轴: 上

**映射关系**:
```
VR_X (右)  → MS_Y (左) 取负号
VR_Y (上)  → MS_Z (上)
VR_Z (后)  → MS_X (前) 取负号
```

#### 3. 工作空间限制

默认工作空间（相对于机械臂base）：
- X: [0.0, 0.8] m (前后)
- Y: [-0.5, 0.5] m (左右)
- Z: [0.0, 0.6] m (上下)

超出范围的目标位置会被自动限制在边界内。

---

## 参数配置

### 命令行参数

```bash
python -m mani_skill.examples.teleoperation.vr_teleoperation_panda --help
```

主要参数：

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--env-id` | `PickCube-v1` | 环境ID |
| `--control-mode` | `pd_ee_delta_pose` | 控制模式（IK） |
| `--scale-factor` | `2.0` | VR空间缩放（越大越精细） |
| `--position-smooth` | `0.5` | 位置平滑（0=完全平滑，1=无平滑） |
| `--rotation-smooth` | `0.3` | 旋转平滑 |
| `--control-freq` | `30` | 控制频率（Hz） |
| `--max-delta-pos` | `0.05` | 每步最大位移（m） |
| `--workspace-x` | `(0.0, 0.8)` | X轴工作空间 |
| `--workspace-y` | `(-0.5, 0.5)` | Y轴工作空间 |
| `--workspace-z` | `(0.0, 0.6)` | Z轴工作空间 |
| `--record-dir` | `None` | 录制目录 |

### 调整建议

**如果机械臂移动太快/太大**:
```bash
--scale-factor 3.0  # 增大缩放因子
```

**如果机械臂抖动**:
```bash
--position-smooth 0.3  # 增加平滑
--rotation-smooth 0.2
```

**如果响应延迟**:
```bash
--position-smooth 0.8  # 减少平滑
--control-freq 60      # 提高频率
```

**如果想要更大的工作空间**:
```bash
--workspace-x "(0.0, 1.0)" \
--workspace-y "(-0.6, 0.6)" \
--workspace-z "(0.0, 0.8)"
```

---

## 故障排除

### 问题1: "OpenVR初始化失败"

**原因**: SteamVR未运行或ALVR未连接

**解决**:
1. 启动SteamVR
2. 启动ALVR服务器
3. 在Quest 3上启动ALVR客户端
4. 确认连接成功后再运行脚本

### 问题2: "未找到右手柄"

**原因**: 手柄未开启或未配对

**解决**:
1. 按下手柄上的任意按钮唤醒
2. 在SteamVR中查看手柄是否显示为绿色（已追踪）
3. 确保手柄在Quest的追踪范围内

### 问题3: "VR追踪丢失"

**原因**: 手柄超出追踪范围或遮挡

**解决**:
1. 将手柄移回摄像头可见范围
2. 确保房间光线充足
3. 避免快速移动手柄

### 问题4: 机械臂不动或移动方向错误

**原因**: 坐标系映射问题或未标定

**解决**:
1. 确保已标定（看到"✓ 标定完成"）
2. 重新标定（重启程序）
3. 检查坐标转换矩阵（见coordinate_transform.py）

**调试方法**:
```bash
# 测试坐标转换
cd /home/enyi/Code/UF850/teleVR
python coordinate_transform.py

# 应该看到：
# VR右移0.1m: [0.1 0. 0.] -> MS左移: [0. -0.1 0.]
# VR上移0.1m: [0. 0.1 0.] -> MS上移: [0. 0. 0.1]
# VR后移0.1m: [0. 0. 0.1] -> MS前移: [-0.1 0. 0.]
```

### 问题5: 机械臂移动太慢/太快

**原因**: 缩放因子不合适

**解决**:
```bash
# 移动太慢（VR中移动很多，机械臂移动很少）
--scale-factor 1.0  # 减小缩放因子

# 移动太快（VR中轻微移动，机械臂移动很大）
--scale-factor 5.0  # 增大缩放因子
```

### 问题6: "超出工作空间"频繁提示

**原因**: 工作空间设置太小或标定位置不佳

**解决**:
1. 重新标定，选择更靠近机械臂中心的位置
2. 调整工作空间参数（见上文）
3. 避免大幅度移动手柄

---

## 坐标系说明

### VR世界坐标系 (OpenVR/SteamVR)

- **原点**: 房间地板中心（play space中心）
- **X轴**: 向右（+X）
- **Y轴**: 向上（+Y）
- **Z轴**: 向后，朝向用户（+Z）
- **单位**: 米

### ManiSkill仿真世界坐标系

- **原点**: 仿真世界中心
- **X轴**: 向前（+X）
- **Y轴**: 向左（+Y）
- **Z轴**: 向上（+Z）
- **单位**: 米

### Panda机械臂Base坐标系

在PickCube环境中：
- **Base位置**: [-0.615, 0, 0] (世界坐标系)
- **初始末端位置**: 约 [0.0, 0.0, 0.5] (相对base)

### 转换示例

假设标定时：
- VR手柄: [1.0, 1.5, -1.0] (胸前，世界坐标)
- 机械臂末端: [0.0, 0.0, 0.5] (相对base)

当VR手柄移动到 [1.1, 1.5, -1.0] (向右0.1m)：

```python
# 1. VR相对位移
vr_delta = [1.1, 1.5, -1.0] - [1.0, 1.5, -1.0] = [0.1, 0, 0]

# 2. 坐标系转换 (VR → MS)
#    VR_X右 → MS_Y左(负)
ms_delta = [0, -0.1, 0]

# 3. 缩放（如果scale_factor=2.0）
ms_delta_scaled = [0, -0.05, 0]

# 4. 机械臂目标位置
target = [0.0, 0.0, 0.5] + [0, -0.05, 0] = [0.0, -0.05, 0.5]
```

---

## 高级使用

### 1. 录制演示轨迹

```bash
python -m mani_skill.examples.teleoperation.vr_teleoperation_panda \
    --record-dir ./demos/vr_pickcube

# 完成后会生成：
# - trajectory.h5 (轨迹数据)
# - trajectory.json (元数据)
# - videos/ (视频文件)
```

### 2. 修改控制模式

当前默认使用 `pd_ee_delta_pose`（实时IK控制），您也可以尝试：

```bash
# 使用motion planning（更安全，但可能较慢）
# 需要修改代码集成motion planner
```

### 3. 自定义坐标转换

如果默认的坐标映射不合适，可以修改 `coordinate_transform.py`:

```python
# 在 CoordinateTransformer.__init__ 中修改
self.position_transform_matrix = np.array([
    [0, 0, -1],   # 修改这里
    [-1, 0, 0],   # 修改这里
    [0, 1, 0]     # 修改这里
])
```

### 4. 添加更多按钮控制

在 `vr_teleoperation_panda.py` 的 `step()` 函数中添加：

```python
# 例如：使用Grip按钮重置环境
if buttons['grip']:
    self.env.reset()
    print(">>> 环境已重置")
```

---

## 开发者信息

### 项目结构

```
VR遥操作系统
├── VR输入层 (vr_controller.py)
│   ├── OpenVR接口
│   ├── 手柄追踪
│   └── 按钮读取
│
├── 坐标转换层 (coordinate_transform.py)
│   ├── VR→MS坐标映射
│   ├── 位置平滑滤波
│   └── 工作空间限制
│
├── 控制层 (vr_teleoperation_panda.py)
│   ├── 标定管理
│   ├── 目标位姿计算
│   ├── 动作生成
│   └── 环境交互
│
└── 仿真环境 (ManiSkill)
    └── Panda机械臂
```

### 扩展到其他机器人

修改 `vr_teleoperation_panda.py`:

```python
args.robot_uid = "fetch"  # 或 "xarm6_robotiq", "so100" 等
```

确保机器人支持 `pd_ee_delta_pose` 控制模式。

---

## 常见问题 (FAQ)

**Q: 每次都需要标定吗？**
A: 是的，建议每次运行时重新标定。因为您的站立位置可能不同，标定可以确保舒适的控制体验。

**Q: 可以用左手柄吗？**
A: 可以，修改 `vr_controller.py` 中的 `TrackedControllerRole_RightHand` 为 `TrackedControllerRole_LeftHand`。

**Q: 如何提高控制精度？**
A: 增大 `scale_factor`（如3.0或4.0），这样VR中需要更大的移动才对应真实空间的位移。

**Q: 为什么有时候机械臂不跟随手柄？**
A: 可能是IK求解失败或超出关节限制。尝试更小幅度的移动，或者调整工作空间范围。

**Q: 可以控制关节空间吗？**
A: 可以，但需要自定义映射。当前实现是末端空间控制，更直观易用。

---

## 致谢

- ManiSkill团队提供的优秀仿真框架
- OpenVR/SteamVR提供的VR接口
- ALVR团队提供的无线串流方案

---

## 联系与反馈

如有问题或建议，请在ManiSkill GitHub仓库提issue。

**祝您遥操作愉快！🎮🤖**
