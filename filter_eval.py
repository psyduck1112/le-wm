"""
从 scene_expert.h5 过滤出指定任务的 episode，生成专门用于 eval 的 HDF5。

与 filter_dataset.py 的差别：
  - 训练只需要：pixels, action, observation, qpos, qvel
  - eval 多保留：button_states, privileged_target/drawer_pos（用于复现 + 成功判定）
  - 默认数量小（200 而非 10000）
  - 支持 --offset 从源 episode 序列的指定位置开始挑，避免和训练集重叠

运行：
    python filter_eval.py --task drawer --n-episodes 200
"""

import argparse
import os
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm


KEYS_TO_KEEP = [
    'pixels', 'qpos', 'qvel', 'button_states',
    'privileged_target_drawer_pos',
    'privileged_drawer_pos',
    'privileged_target_task',
    'action', 'observation',
]
# ep_idx / step_idx 在 write_filtered 末尾合成（重新编号到 0..N-1），
# 不能直接从源 copy（源里的 episode 编号在新 dataset 里会越界）


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',         type=str, default='ogbench/scene_expert')
    parser.add_argument('--dst',         type=str, default='ogbench/scene_drawer_eval')
    parser.add_argument('--task',        type=str, default='drawer',
                        choices=['cube', 'button', 'drawer', 'window'])
    parser.add_argument('--n-episodes',  type=int, default=200)
    parser.add_argument('--offset',      type=int, default=0,
                        help='Skip the first N matching episodes (avoid overlap with train set)')
    return parser.parse_args()


def find_task_episodes(f, task, n_needed, offset=0):
    """扫描所有 episode 首步 task 标签，从第 offset 个匹配项开始取 n_needed 个。"""
    ep_offsets = f['ep_offset'][:]
    task_at_start = f['privileged_target_task'][ep_offsets]

    skipped = 0
    indices = []
    for i, t in enumerate(task_at_start):
        t = t.decode() if isinstance(t, bytes) else t
        if t != task:
            continue
        if skipped < offset:
            skipped += 1
            continue
        indices.append(i)
        if len(indices) >= n_needed:
            break
    return indices


def write_filtered(src_path, dst_path, ep_indices, keys):
    """把选中 episode 的指定 key 顺序写入新 HDF5，重建 ep_len/ep_offset。"""
    with h5py.File(src_path, 'r') as src:
        src_ep_offsets = src['ep_offset'][:]
        src_ep_lens = src['ep_len'][:]

        new_ep_lens = src_ep_lens[ep_indices].astype(np.int32)
        new_ep_offsets = np.concatenate([[0], np.cumsum(new_ep_lens[:-1])]).astype(np.int64)
        total_steps = int(new_ep_lens.sum())

        with h5py.File(dst_path, 'w', libver='latest',
                       fs_strategy='page', fs_page_size=4 * 1024 * 1024) as dst:
            dst.swmr_mode = True

            dst.create_dataset('ep_len', data=new_ep_lens)
            dst.create_dataset('ep_offset', data=new_ep_offsets)

            for key in keys:
                if key not in src:
                    print(f'  skip: {key} not in source')
                    continue
                sample = src[key][0:1]
                shape = (total_steps,) + sample.shape[1:]
                dtype = sample.dtype

                if sample.ndim >= 2:
                    chunks = (min(100, total_steps),) + sample.shape[1:]
                    comp = hdf5plugin.Blosc(cname='lz4', clevel=5,
                                            shuffle=hdf5plugin.Blosc.SHUFFLE)
                    dst.create_dataset(key, shape=shape, dtype=dtype,
                                       chunks=chunks, **comp)
                else:
                    dst.create_dataset(key, shape=shape, dtype=dtype)

            ptr = 0
            for ep_idx in tqdm(ep_indices, desc='Copying episodes'):
                s = int(src_ep_offsets[ep_idx])
                l = int(src_ep_lens[ep_idx])
                for key in keys:
                    if key not in src:
                        continue
                    dst[key][ptr:ptr + l] = src[key][s:s + l]
                ptr += l

            # 合成 ep_idx / step_idx（按新 episode 顺序 0..N-1 重新编号）
            new_ep_idx = np.repeat(np.arange(len(ep_indices), dtype=np.int32), new_ep_lens)
            new_step_idx = np.concatenate([np.arange(l, dtype=np.int64) for l in new_ep_lens])
            dst.create_dataset('ep_idx', data=new_ep_idx)
            dst.create_dataset('step_idx', data=new_step_idx)

            dst.flush()

        return total_steps


def main():
    args = parse_args()
    cache_dir = os.environ.get('STABLEWM_HOME', os.path.expanduser('~/.stable_worldmodel'))
    src_path = Path(cache_dir) / f'{args.src}.h5'
    dst_path = Path(cache_dir) / f'{args.dst}.h5'
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Source: {src_path}')
    print(f'Dest:   {dst_path}')
    print(f'Task:   {args.task}, target episodes: {args.n_episodes}, offset: {args.offset}')

    with h5py.File(src_path, 'r') as f:
        indices = find_task_episodes(f, args.task, args.n_episodes, args.offset)

    print(f'Found {len(indices)} {args.task} episodes in source (after offset={args.offset}).')
    if len(indices) < args.n_episodes:
        print(f'  WARNING: only {len(indices)} available, requested {args.n_episodes}.')

    total_steps = write_filtered(src_path, dst_path, indices, KEYS_TO_KEEP)
    print(f'Done. Wrote {len(indices)} episodes / {total_steps} steps to {dst_path}')


if __name__ == '__main__':
    main()
