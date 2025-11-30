#!/usr/bin/env python3
"""
简单验证脚本 - 检查LeRobot数据集完整性
"""

import json
from pathlib import Path

def verify_dataset(dataset_path):
    """验证数据集完整性"""
    dataset_path = Path(dataset_path)

    print(f"\n{'='*60}")
    print(f"验证数据集: {dataset_path}")
    print(f"{'='*60}\n")

    # 1. 检查目录结构
    print("1. 目录结构检查:")
    required_dirs = [
        'meta',
        'data/chunk-000',
        'videos/chunk-000/observation.images.wrist',
    ]

    for dir_path in required_dirs:
        full_path = dataset_path / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (不存在)")

    # 2. 检查Parquet文件
    print("\n2. Parquet数据文件:")
    data_dir = dataset_path / 'data' / 'chunk-000'
    parquet_files = sorted(data_dir.glob('episode_*.parquet'))

    if parquet_files:
        print(f"  找到 {len(parquet_files)} 个episode文件:")
        for pf in parquet_files:
            size_kb = pf.stat().st_size / 1024
            print(f"    • {pf.name} ({size_kb:.1f} KB)")
    else:
        print("  ✗ 未找到任何episode文件")

    # 3. 检查视频文件
    print("\n3. 视频文件:")
    video_dir = dataset_path / 'videos' / 'chunk-000' / 'observation.images.wrist'
    video_files = sorted(video_dir.glob('episode_*.mp4'))

    if video_files:
        print(f"  找到 {len(video_files)} 个视频文件:")
        for vf in video_files:
            size_mb = vf.stat().st_size / (1024 * 1024)
            print(f"    • {vf.name} ({size_mb:.2f} MB)")
    else:
        print("  ✗ 未找到任何视频文件")

    # 4. 检查meta文件
    print("\n4. Meta元数据文件:")
    meta_files = {
        'info.json': 'JSON',
        'episodes.jsonl': 'JSONL',
        'tasks.jsonl': 'JSONL',
        'modality.json': 'JSON'
    }

    for filename, file_type in meta_files.items():
        meta_path = dataset_path / 'meta' / filename
        if meta_path.exists():
            try:
                if file_type == 'JSON':
                    with open(meta_path, 'r') as f:
                        data = json.load(f)
                    print(f"  ✓ {filename} (有效JSON)")
                else:  # JSONL
                    with open(meta_path, 'r') as f:
                        lines = f.readlines()
                    for line in lines:
                        json.loads(line)  # 验证每行都是有效JSON
                    print(f"  ✓ {filename} ({len(lines)}条记录)")
            except Exception as e:
                print(f"  ✗ {filename} (格式错误: {e})")
        else:
            print(f"  ⚠ {filename} (不存在)")

    # 5. 读取关键信息
    print("\n5. 数据集统计信息:")

    # 读取info.json
    info_path = dataset_path / 'meta' / 'info.json'
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
        print(f"  总Episodes数: {info.get('total_episodes', 'N/A')}")
        print(f"  总帧数: {info.get('total_frames', 'N/A')}")
        print(f"  总任务数: {info.get('total_tasks', 'N/A')}")
        print(f"  FPS: {info.get('fps', 'N/A')}")
        print(f"  机器人类型: {info.get('robot_type', 'N/A')}")
        print(f"  代码版本: {info.get('codebase_version', 'N/A')}")

    # 读取episodes.jsonl
    episodes_path = dataset_path / 'meta' / 'episodes.jsonl'
    if episodes_path.exists():
        print("\n  Episodes详情:")
        with open(episodes_path, 'r') as f:
            for line in f:
                ep = json.loads(line)
                print(f"    Episode {ep['episode_index']}: {ep['length']}帧, 任务={ep['tasks']}")

    # 读取tasks.jsonl
    tasks_path = dataset_path / 'meta' / 'tasks.jsonl'
    if tasks_path.exists():
        print("\n  任务列表:")
        with open(tasks_path, 'r') as f:
            for line in f:
                task = json.loads(line)
                print(f"    [{task['task_index']}] {task['task']}")

    # 6. 数据一致性检查
    print("\n6. 数据一致性:")

    num_parquet = len(parquet_files)
    num_videos = len(video_files)

    if num_parquet == num_videos:
        print(f"  ✓ Parquet文件数 ({num_parquet}) = 视频文件数 ({num_videos})")
    else:
        print(f"  ✗ 数量不匹配: Parquet={num_parquet}, 视频={num_videos}")

    if info_path.exists():
        expected_episodes = info.get('total_episodes', 0)
        if num_parquet == expected_episodes:
            print(f"  ✓ 文件数量与info.json一致 ({expected_episodes})")
        else:
            print(f"  ⚠ 文件数量({num_parquet}) ≠ info.json中的total_episodes({expected_episodes})")

    print(f"\n{'='*60}")

    if num_parquet > 0 and num_videos > 0:
        print("✅ 数据集结构完整！")
    else:
        print("⚠️ 数据集不完整")

    print(f"{'='*60}\n")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = '/home/enyi/Code/UF850/teleVR/real/webxr/lerobot_dataset'

    verify_dataset(dataset_path)
