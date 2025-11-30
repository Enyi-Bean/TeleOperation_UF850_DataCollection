# Action与State的时序关系说明

## 📊 当前实现

### 数据记录时序（robot_control_with_data_collection.py 第537-555行）

```python
if self.data_collector.should_record_this_step():  # 每4个控制周期(25Hz)
    # t 时刻
    current_state = self._get_current_state()       # state(t)
    current_action = self._compute_current_action(target_pose)  # action(t)

    with self.camera_lock:
        frames = self.latest_frames.copy()          # frames(t)

    self.data_collector.record_step(
        state=current_state,    # 当前观测到的状态
        action=current_action,  # 当前计算的目标动作
        frames=frames,
        timestamp=time.time()
    )
```

## ❓ Action的语义

### 当前实现的Action定义

```python
def _compute_current_action(self, target_pose):
    """从VR手柄位姿计算目标关节角度"""
    # 使用逆运动学(IK)将笛卡尔空间目标转换为关节空间
    code, joint_angles = self.arm.get_inverse_kinematics(target_pose)
    gripper_target = 1.0 if self.gripper_open else 0.0
    action = np.append(joint_angles, gripper_target)
    return action
```

**语义**：`action(t)` = "机械臂应该移动到的目标位置"

- **不是**增量/delta（相对于当前位置的变化）
- **是**绝对目标位置（关节空间的目标角度）

## 🤔 是否需要action比state晚一个timestep？

### 标准机器人学习数据格式

在标准的行为克隆(Behavioral Cloning)和模仿学习中：

```
时刻 t:
  observation(t) → state(t) + image(t)
  action(t) → "在观测到state(t)后，expert执行的动作"
```

### 两种常见的action定义

#### 方案A：action = state(t+1) - state(t) (Delta)
```
记录: [state(t), action=delta, image(t)]
训练: 学习 policy(state(t)) → delta
```
**问题**：我们的action是**绝对位置**，不是delta

#### 方案B：action = target_state(t) (Absolute)
```
记录: [state(t), action=target, image(t)]
训练: 学习 policy(state(t)) → target
```
**这就是我们当前的实现！**

### GR00T/LeRobot的action定义

查看LeRobot文档和GR00T示例：

```python
# LeRobot_compatible_data_schema.md 第124行
"action": {
    "<action_key>": {
        "absolute": <bool>,  # true for absolute values, false for relative/delta
    }
}
```

**GR00T支持两种模式**：
- `absolute=true`: 绝对目标位置（我们当前实现）
- `absolute=false`: 相对增量

### 我们的action是否正确？

**✅ 当前实现是正确的！**

原因：
1. **action语义清晰**：action(t) = "看到state(t)后，VR操作员指示的目标位置"
2. **时间同步**：state(t)和action(t)在同一个控制周期采集，**时间戳一致**
3. **因果关系正确**：
   ```
   VR手柄位置(t) → 计算目标(t) → action(t)
   机械臂传感器(t) → 读取关节(t) → state(t)
   ```
4. **符合GR00T格式**：设置`modality.json`中`absolute=true`即可

## 🔍 是否需要改成state(t), action(t+1)？

### 不需要！理由如下：

#### 1. **我们的action已经是"未来目标"**

```python
# 当前时刻 t
current_state = [j0, j1, j2, j3, j4, j5, gripper]  # 当前位置
current_action = IK(target_pose)                    # 目标位置（未来）

# 机械臂会在 t → t+Δt 期间移动：
#   从 current_state 移动到 current_action
```

#### 2. **LeRobot的delta_timestamps机制处理时序**

GR00T训练时使用`delta_timestamps`来定义时序关系：

```python
# gr00t/experiment/data_config.py
delta_timestamps = {
    "observation.images.wrist": [0],        # 当前帧
    "observation.state": [0],               # 当前状态
    "action": [0, 0.04, 0.08, ..., 0.64],  # 当前+未来16步
}
```

**这里的`[0, 0.04, ...]`表示**：
- 0: 当前时刻的action(t)
- 0.04: 未来0.04秒的action(t+1)（如果25Hz，就是下一帧）
- ...

**GR00T会学习预测未来的action序列！**

#### 3. **官方SO-100示例也是同时采集**

查看`Isaac-GR00T/examples/SO-100/eval_lerobot.py`：

```python
# 推理时
observation = get_observation()  # 当前观测
action = policy(observation)     # 预测动作
robot.execute(action)            # 执行
```

没有时间偏移，都是同一时刻。

## 📝 总结

### ✅ 当前实现 (正确)

```python
时刻 t (25Hz采样):
  state(t)  = 机械臂当前实际位置 (传感器读数)
  action(t) = VR操作员指示的目标位置 (通过IK计算)
  frames(t) = 相机当前图像
  timestamp = t
```

### ❌ 不需要改成 state(t), action(t+1)

原因：
1. action已经是"目标/未来位置"，不需要时间偏移
2. GR00T通过`delta_timestamps`机制学习时序
3. 我们的action语义是绝对目标（`absolute=true`）
4. 官方示例也是同时采集state和action

### 📋 需要确保的配置

在`episode_recorder.py`的`modality.json`中（第345-353行）：

```json
"action": {
    "single_arm": {
        "start": 0,
        "end": 6,
        "absolute": true  // ← 关键：标记为绝对值
    },
    "gripper": {
        "start": 6,
        "end": 7,
        "absolute": true
    }
}
```

**已正确配置！** （但原代码没有写`absolute`字段，建议添加）

## 🎯 推荐的修改

在`episode_recorder.py`第345行附近添加`absolute: true`：

```python
modality = {
    "state": {
        "single_arm": {"start": 0, "end": 6},
        "gripper": {"start": 6, "end": 7}
    },
    "action": {
        "single_arm": {
            "start": 0,
            "end": 6,
            "absolute": True  # ← 添加这行
        },
        "gripper": {
            "start": 6,
            "end": 7,
            "absolute": True  # ← 添加这行
        }
    },
    # ...
}
```

## 🔗 参考

- [LeRobot Dataset Format](https://docs.phospho.ai/learn/lerobot-dataset)
- [Robot Learning Tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial)
- GR00T官方示例: `Isaac-GR00T/examples/SO-100/`
