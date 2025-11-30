#!/usr/bin/env python3
"""
详细检查Episode数据内容
"""

import sys
import json
from pathlib import Path

def inspect_episode(dataset_path, episode_idx=0):
    """详细检查Episode数据内容"""
    dataset_path = Path(dataset_path)

    print(f"\n{'='*60}")
    print(f"检查 Episode {episode_idx}")
    print(f"{'='*60}\n")

    # 1. Parquet文件检查
    parquet_path = dataset_path / 'data' / 'chunk-000' / f'episode_{episode_idx:06d}.parquet'

    if not parquet_path.exists():
        print(f"✗ Parquet文件不存在: {parquet_path}")
        return

    try:
        import pandas as pd
        import numpy as np

        df = pd.read_parquet(parquet_path)

        print("1. Parquet数据概览:")
        print(f"  总帧数: {len(df)}")
        print(f"  列名: {list(df.columns)}")
        print(f"\n2. 列数据类型:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")

        print(f"\n3. 关键列数据检查:")

        # 检查observation.state
        if 'observation.state' in df.columns:
            states = df['observation.state']
            print(f"\n  observation.state:")
            print(f"    类型: {type(states.iloc[0])}")
            if isinstance(states.iloc[0], (list, np.ndarray)):
                print(f"    维度: {len(states.iloc[0])}")
                print(f"    前3帧:")
                for i in range(min(3, len(states))):
                    state = states.iloc[i]
                    if isinstance(state, list):
                        state = np.array(state)
                    print(f"      [{i}] {state} (shape={state.shape})")
            else:
                print(f"    ⚠ 非数组格式: {states.iloc[0]}")

        # 检查action
        if 'action' in df.columns:
            actions = df['action']
            print(f"\n  action:")
            print(f"    类型: {type(actions.iloc[0])}")
            if isinstance(actions.iloc[0], (list, np.ndarray)):
                print(f"    维度: {len(actions.iloc[0])}")
                print(f"    前3帧:")
                for i in range(min(3, len(actions))):
                    action = actions.iloc[i]
                    if isinstance(action, list):
                        action = np.array(action)
                    print(f"      [{i}] {action} (shape={action.shape})")
            else:
                print(f"    ⚠ 非数组格式: {actions.iloc[0]}")

        # 检查timestamp
        if 'timestamp' in df.columns:
            timestamps = df['timestamp']
            print(f"\n  timestamp:")
            print(f"    类型: {timestamps.dtype}")
            print(f"    范围: {timestamps.min():.3f} ~ {timestamps.max():.3f}")
            print(f"    时长: {timestamps.max() - timestamps.min():.3f}s")
            if len(timestamps) > 1:
                diffs = timestamps.diff().dropna()
                print(f"    平均间隔: {diffs.mean()*1000:.1f}ms")
                print(f"    实际频率: {1.0/diffs.mean():.1f}Hz")

        # 检查task_index
        if 'task_index' in df.columns:
            task_indices = df['task_index'].unique()
            print(f"\n  task_index: {task_indices}")

        # 检查next.done
        if 'next.done' in df.columns:
            done_flags = df['next.done']
            num_done = done_flags.sum()
            last_done = done_flags.iloc[-1]
            print(f"\n  next.done:")
            print(f"    True数量: {num_done}")
            print(f"    最后一帧: {last_done}")
            if num_done == 1 and last_done:
                print(f"    ✓ 正确 (仅最后一帧为True)")
            else:
                print(f"    ⚠ 异常")

        # 检查episode_index和frame_index
        if 'episode_index' in df.columns:
            ep_indices = df['episode_index'].unique()
            print(f"\n  episode_index: {ep_indices}")

        if 'frame_index' in df.columns:
            frame_indices = df['frame_index']
            print(f"\n  frame_index: {frame_indices.min()} ~ {frame_indices.max()}")
            if (frame_indices == range(len(frame_indices))).all():
                print(f"    ✓ 连续递增")
            else:
                print(f"    ⚠ 不连续")

        print(f"\n4. 数据样本 (前2帧):")
        print(df.head(2).to_string())

    except ImportError:
        print("✗ 需要pandas: pip install pandas pyarrow")
        return
    except Exception as e:
        print(f"✗ 读取Parquet失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 视频文件检查
    print(f"\n{'='*60}")
    print("5. 视频文件:")
    video_path = dataset_path / 'videos' / 'chunk-000' / 'observation.images.wrist' / f'episode_{episode_idx:06d}.mp4'

    if video_path.exists():
        size_mb = video_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {video_path.name} ({size_mb:.2f} MB)")

        # 用ffprobe检查视频属性
        import subprocess
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration',
                 '-of', 'json', str(video_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                import json
                video_info = json.loads(result.stdout)
                if 'streams' in video_info and len(video_info['streams']) > 0:
                    stream = video_info['streams'][0]
                    print(f"    分辨率: {stream.get('width')}x{stream.get('height')}")
                    print(f"    帧率: {stream.get('r_frame_rate')}")
                    if 'nb_frames' in stream:
                        print(f"    总帧数: {stream.get('nb_frames')}")
                    if 'duration' in stream:
                        print(f"    时长: {float(stream.get('duration')):.2f}s")
        except FileNotFoundError:
            print("    (ffprobe未安装，无法检查视频详情)")
        except Exception as e:
            print(f"    (视频检查失败: {e})")
    else:
        print(f"  ✗ 视频文件不存在")

    print(f"{'='*60}\n")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = '/home/enyi/Code/UF850/teleVR/real/webxr/uf850_teleop_dataset'

    episode_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    inspect_episode(dataset_path, episode_idx)
