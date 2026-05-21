"""
从 scene_expert.h5 过滤出指定任务的 episode，输出训练就绪的 HDF5。

只保留训练 + eval 必要的 key：
    pixels, action, observation (=proprio), qpos, qvel
    + ep_len / ep_offset 索引

运行：
    python filter_dataset.py --task drawer --n-episodes 10000
"""

import argparse
import os
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm


KEYS_TO_KEEP = ['pixels', 'action', 'observation', 'qpos', 'qvel']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',         type=str, default='ogbench/scene_expert')
    parser.add_argument('--dst',         type=str, default='ogbench/scene_drawer')
    parser.add_argument('--task',        type=str, default='drawer',
                        choices=['cube', 'button', 'drawer', 'window'])
    parser.add_argument('--n-episodes',  type=int, default=10000)
    return parser.parse_args()


def find_task_episodes(f, task, n_needed):
    """扫描所有 episode 的首步 task 标签，返回前 n_needed 个匹配的 episode 索引。"""
    ep_offsets = f['ep_offset'][:]
    task_at_start = f['privileged_target_task'][ep_offsets]

    indices = []
    for i, t in enumerate(task_at_start):
        t = t.decode() if isinstance(t, bytes) else t
        if t == task:
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
    print(f'Task:   {args.task}, target episodes: {args.n_episodes}')

    with h5py.File(src_path, 'r') as f:
        indices = find_task_episodes(f, args.task, args.n_episodes)

    print(f'Found {len(indices)} {args.task} episodes in source.')
    if len(indices) < args.n_episodes:
        print(f'  WARNING: only {len(indices)} available, requested {args.n_episodes}.')
        print(f'  Run OGB_collect.py to gather more, then re-run this script.')

    total_steps = write_filtered(src_path, dst_path, indices, KEYS_TO_KEEP)
    print(f'Done. Wrote {len(indices)} episodes / {total_steps} steps to {dst_path}')


if __name__ == '__main__':
    main()
