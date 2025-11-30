# 数据验证报告

## 数据集位置
`/home/enyi/Code/UF850/teleVR/real/webxr/uf850_teleop_dataset`

## 总体统计

- **总Episodes数**: 2
- **总帧数**: 389帧 (Episode 0: 331帧, Episode 1: 58帧)
- **数据格式**: LeRobot V2.0
- **机器人**: UF850
- **采样频率**: 30.0 Hz (实际测量)
- **任务**: pick the cup and place it on the plate

---

## Episode详情

### Episode 0
- **帧数**: 331帧
- **时长**: 11.0秒
- **实际频率**: 30.0 Hz
- **视频大小**: 1.42 MB
- **Parquet大小**: 47.9 KB

### Episode 1
- **帧数**: 58帧
- **时长**: 1.9秒
- **实际频率**: 30.0 Hz
- **视频大小**: 0.24 MB
- **Parquet大小**: 13.8 KB

---

## 数据结构验证

### ✅ 目录结构
```
uf850_teleop_dataset/
├── meta/
│   ├── info.json          ✓ 有效
│   ├── episodes.jsonl     ✓ 有效 (2条记录)
│   ├── tasks.jsonl        ✓ 有效 (2条记录)
│   └── modality.json      ✓ 有效
├── data/chunk-000/
│   ├── episode_000000.parquet  ✓ 有效
│   └── episode_000001.parquet  ✓ 有效
└── videos/chunk-000/
    └── observation.images.wrist/
        ├── episode_000000.mp4  ✓ 有效
        └── episode_000001.mp4  ✓ 有效
```

### ✅ Parquet数据列
```
1. observation.state         (object)   - 机械臂状态
2. action                    (object)   - 动作指令
3. timestamp                 (float64)  - 时间戳
4. task_index                (int64)    - 任务索引
5. annotation.human.action.task_description (int64)
6. annotation.human.validity (int64)    - 有效性标记
7. episode_index             (int64)    - Episode索引
8. frame_index               (int64)    - 帧索引
9. index                     (int64)    - 全局索引
10. next.reward              (float64)  - 奖励
11. next.done                (bool)     - 结束标志
```

### ✅ 数据维度检查

**observation.state**: 8维数组
```
示例: [0.029985, -0.078293, -0.402606, -3.167252, 0.36518, 3.150346, 0.0, 0.98588235]
       [joint0,  joint1,   joint2,   joint3,   joint4, joint5,  joint6, gripper]
```

**action**: 8维数组
```
示例: [0.0298905, -0.07841398, -0.4027299, 3.11592507, 0.36518219, -3.13292599, 0.0, 1.0]
       [joint0,   joint1,    joint2,   joint3,    joint4,     joint5,      joint6, gripper]
```

**注意**: 实际数据是8维 (7个关节 + 1个夹爪)，这与UF850是7自由度机械臂一致。第7个关节(joint6, index=6)始终为0.0，可能是冗余自由度未启用。

### ⚠️ 发现的问题

**modality.json配置不匹配**

当前modality.json定义为7维:
```json
"state": {
  "single_arm": {
    "start": 0,
    "end": 6     ← 表示索引0-5 (6个关节)
  },
  "gripper": {
    "start": 6,
    "end": 7     ← 表示索引6 (1个夹爪)
  }
}
```

但实际数据是8维:
- joint0-5: 索引0-5
- joint6: 索引6 (始终为0.0)
- gripper: 索引7

**修正建议**: 将modality.json更新为8维配置。

---

## 数据完整性验证

### ✅ 时间戳
- **连续性**: ✓ 单调递增
- **采样间隔**: 33.3ms (标准差极小)
- **实际频率**: 30.0 Hz (符合预期)

### ✅ 帧索引
- **frame_index**: ✓ 从0连续递增到N-1
- **episode_index**: ✓ 正确标记
- **index**: ✓ 正确标记

### ✅ 终止标志
- **next.done**: ✓ 仅最后一帧为True
- **其他帧**: ✓ 全部为False

### ✅ 任务标注
- **task_index**: ✓ 全部为0 (对应第一个任务)
- **annotation.human.validity**: ✓ 全部为1 (有效)

### ✅ 视频文件
- **Episode 0**: 1.42 MB (331帧，约11秒)
- **Episode 1**: 0.24 MB (58帧，约2秒)
- **格式**: MP4
- **分辨率**: 预期640x480 (未验证，ffprobe未安装)

---

## 数据质量评估

### ✅ 优点
1. **采样频率稳定**: 30.0 Hz，间隔一致
2. **数据完整**: 所有必需字段齐全
3. **格式正确**: 符合LeRobot V2.0规范
4. **时间同步**: 时间戳单调递增，无跳变
5. **元数据完整**: info.json, episodes.jsonl, tasks.jsonl都存在且有效

### ⚠️ 建议改进
1. **Episode时长偏短**:
   - Episode 0: 11秒 (尚可)
   - Episode 1: 1.9秒 (太短)
   - **建议**: 录制5-10秒的完整任务轨迹

2. **modality.json需要更新**:
   - 当前配置为7维，实际数据8维
   - 需要明确joint6的作用（冗余自由度？）

3. **相机数量**:
   - 当前仅有wrist相机数据
   - front相机数据缺失（因USB线损坏）
   - **建议**: 修复后添加第二个相机

---

## GR00T训练兼容性

### ✅ 兼容项
- ✓ LeRobot V2.0 Parquet格式
- ✓ MP4视频编码
- ✓ 绝对action (absolute: true)
- ✓ 时间戳对齐
- ✓ 任务标注

### ⚠️ 需要确认
- modality.json维度定义 (7维 vs 8维)
- joint6 (index=6) 的物理含义
- 是否需要双目相机（当前仅单目）

---

## 结论

**数据质量**: ✅ **良好**

数据结构完整，格式正确，采样稳定，符合LeRobot V2.0规范。唯一需要修正的是modality.json的维度配置，以及建议增加episode时长和添加第二个相机。

**可用于GR00T训练**: ✅ **是** (修正modality.json后)

---

## 下一步操作建议

1. **立即修正**: 更新modality.json以匹配8维数据
2. **短期**: 录制更多5-10秒的完整轨迹
3. **中期**: 修复USB线，添加front相机数据
4. **长期**: 收集多任务数据集 (当前仅1个任务)
