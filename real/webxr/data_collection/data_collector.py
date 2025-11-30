#!/usr/bin/env python3
"""
数据收集核心模块
"""

import time
import numpy as np
from pathlib import Path
from .episode_recorder import EpisodeRecorder


class DataCollector:
    """
    GR00T LeRobot格式数据收集器

    核心职责:
    1. 管理episode录制状态
    2. 按30Hz下采样收集数据 (从100Hz控制循环)
    3. 内存缓存episode数据
    4. 调用EpisodeRecorder保存
    """

    # 预定义任务列表
    PREDEFINED_TASKS = [
        "pick the cup and place it on the plate",
    ]

    def __init__(self, dataset_path, record_freq=30, control_freq=100):
        """
        初始化数据收集器

        Args:
            dataset_path: 数据集保存路径
            record_freq: 数据记录频率 (Hz, 默认30, 接近对齐: 100/3≈33Hz)
            control_freq: 控制循环频率 (Hz, 默认100)
        """
        self.dataset_path = Path(dataset_path)
        self.record_freq = record_freq
        self.control_freq = control_freq
        self.record_interval = control_freq // record_freq  # 每3个控制周期记录1次 (100//30=3)

        # Episode状态
        self.is_recording = False
        self.current_episode_data = None
        self.episode_index = 0
        self.step_count = 0  # 全局step计数 (用于下采样)
        self.episode_start_time = None
        self.episode_step_count = 0  # 当前episode内的step数

        # 任务管理
        self.current_task_index = 0
        self.tasks = self.PREDEFINED_TASKS.copy()

        # 初始化数据集目录结构
        self._init_dataset_structure()

        # 加载已有episode数量
        self._load_existing_episodes()

        print(f"\n{'='*60}")
        print("数据收集器初始化完成")
        print(f"  数据集路径: {self.dataset_path}")
        print(f"  控制频率: {self.control_freq} Hz")
        print(f"  记录频率: {self.record_freq} Hz (下采样比例: 1/{self.record_interval})")
        print(f"  当前Episode索引: {self.episode_index}")
        print(f"  预定义任务数: {len(self.tasks)}")
        print(f"{'='*60}\n")

    def _init_dataset_structure(self):
        """创建LeRobot数据集目录结构"""
        # meta目录
        (self.dataset_path / 'meta').mkdir(parents=True, exist_ok=True)

        # data目录
        (self.dataset_path / 'data' / 'chunk-000').mkdir(parents=True, exist_ok=True)

        # videos目录
        video_base = self.dataset_path / 'videos' / 'chunk-000'
        (video_base / 'observation.images.wrist').mkdir(parents=True, exist_ok=True)
        (video_base / 'observation.images.front').mkdir(parents=True, exist_ok=True)

        print(f"✓ 数据集目录结构已创建: {self.dataset_path}")

    def _load_existing_episodes(self):
        """加载已有的episode数量，从现有基础上继续"""
        data_dir = self.dataset_path / 'data' / 'chunk-000'
        if data_dir.exists():
            existing_episodes = list(data_dir.glob('episode_*.parquet'))
            if existing_episodes:
                # 找到最大的episode索引
                max_idx = max([int(p.stem.split('_')[1]) for p in existing_episodes])
                self.episode_index = max_idx + 1
                print(f"✓ 检测到已有{len(existing_episodes)}个episodes，从Episode {self.episode_index}开始")

    def should_record_this_step(self):
        """
        判断当前control循环是否需要记录数据 (下采样)

        Returns:
            bool: True表示本次需要记录
        """
        return self.is_recording and (self.step_count % self.record_interval == 0)

    def get_current_task(self):
        """获取当前任务描述"""
        return self.tasks[self.current_task_index]

    def set_task_by_index(self, task_index):
        """
        手动设置任务索引

        Args:
            task_index: 任务索引 (0到len(tasks)-1)
        """
        if 0 <= task_index < len(self.tasks):
            self.current_task_index = task_index
            print(f"\n✓ 已设置任务: [{task_index}] {self.get_current_task()}\n")
        else:
            print(f"⚠ 无效的任务索引 {task_index}，有效范围: 0-{len(self.tasks)-1}")

    def start_episode(self, task_description=None):
        """
        开始录制新episode

        Args:
            task_description: 任务描述，如果为None则使用当前任务
        """
        if self.is_recording:
            print("⚠ 已在录制中，请先结束当前episode")
            return False

        # 确定任务描述
        if task_description is None:
            task_description = self.get_current_task()

        # 初始化episode数据buffer
        self.current_episode_data = {
            'states': [],           # observation.state (关节角度 + 夹爪)
            'actions': [],          # action (目标关节角度 + 夹爪)
            'timestamps': [],       # timestamp (秒)
            'frames_wrist': [],     # 手腕相机帧
            'frames_front': [],     # 正面相机帧
            'task_description': task_description,
            'task_index': self.current_task_index
        }

        self.is_recording = True
        self.episode_start_time = time.time()
        self.episode_step_count = 0

        print(f"\n{'='*60}")
        print(f"🔴 开始录制 Episode {self.episode_index}")
        print(f"   任务: [{self.current_task_index}] {task_description}")
        print(f"   采样频率: {self.record_freq} Hz")
        print(f"{'='*60}\n")

        return True

    def record_step(self, state, action, frames, timestamp):
        """
        记录单步数据

        Args:
            state: np.ndarray [8] (7关节角度 + 1夹爪, 弧度)
            action: np.ndarray [8] (目标关节角度 + 夹爪, 弧度)
            frames: dict {'wrist': np.ndarray, 'front': np.ndarray}
            timestamp: float (秒)
        """
        if not self.is_recording:
            return

        # 保存数据
        self.current_episode_data['states'].append(state.copy())
        self.current_episode_data['actions'].append(action.copy())
        self.current_episode_data['timestamps'].append(timestamp)

        if 'wrist' in frames:
            self.current_episode_data['frames_wrist'].append(frames['wrist'].copy())
        if 'front' in frames:
            self.current_episode_data['frames_front'].append(frames['front'].copy())

        self.episode_step_count += 1

        # 每100帧打印一次进度
        if self.episode_step_count % 100 == 0:
            elapsed = time.time() - self.episode_start_time
            fps = self.episode_step_count / elapsed
            print(f"  录制中... 已记录{self.episode_step_count}帧 "
                  f"({elapsed:.1f}s, 实际频率={fps:.1f}Hz)")

    def stop_episode(self):
        """结束当前episode并保存"""
        if not self.is_recording:
            print("⚠ 当前未在录制")
            return

        self.is_recording = False
        duration = time.time() - self.episode_start_time
        num_frames = len(self.current_episode_data['states'])

        print(f"\n{'='*60}")
        print(f"⏹ 停止录制 Episode {self.episode_index}")
        print(f"   帧数: {num_frames}")
        print(f"   时长: {duration:.2f}s")
        if duration > 0:
            print(f"   实际频率: {num_frames/duration:.1f} Hz")
        print(f"{'='*60}\n")

        # 检查数据有效性
        if num_frames < 10:
            print("⚠ 警告: Episode太短 (<10帧)，可能无效")
            response = input("是否仍要保存? (y/n): ").strip().lower()
            if response != 'y':
                print("❌ Episode已丢弃\n")
                self.current_episode_data = None
                return

        # 保存数据
        print("💾 保存数据中...")
        try:
            recorder = EpisodeRecorder(
                dataset_path=self.dataset_path,
                episode_index=self.episode_index,
                fps=self.record_freq
            )
            recorder.save(self.current_episode_data)

            print(f"✅ Episode {self.episode_index} 保存完成!\n")

            # 递增episode索引
            self.episode_index += 1

        except Exception as e:
            print(f"❌ 保存失败: {e}")
            import traceback
            traceback.print_exc()

        # 清空buffer
        self.current_episode_data = None
        self.episode_step_count = 0

    def get_statistics(self):
        """获取数据收集统计信息"""
        data_dir = self.dataset_path / 'data' / 'chunk-000'
        episodes = list(data_dir.glob('episode_*.parquet')) if data_dir.exists() else []

        stats = {
            'total_episodes': len(episodes),
            'next_episode_index': self.episode_index,
            'is_recording': self.is_recording,
            'current_task': self.get_current_task(),
            'current_task_index': self.current_task_index
        }

        return stats

    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()

        print(f"\n{'='*60}")
        print("数据收集统计")
        print(f"{'='*60}")
        print(f"  已收集Episodes: {stats['total_episodes']}")
        print(f"  下一个Episode索引: {stats['next_episode_index']}")
        print(f"  录制状态: {'🔴 录制中' if stats['is_recording'] else '⚪ 未录制'}")
        print(f"  当前任务: [{stats['current_task_index']}] {stats['current_task']}")
        print(f"{'='*60}\n")
