"""
从新版 OGB_collect 采集的数据中抽出 N 个 episode 作为 eval 数据集。

新版 OGB_collect.py 特性：
  - set_new_target 被 patch 成 drawer-only，所有 episode 都是 drawer 任务
  - terminate_at_goal=False + max_episode_steps=500 → 每个 episode 500 步
  - 一个 episode 内 drawer 在 [-0.16, 0.0] 之间反复开关 (~10 次)

filter_eval 的作用：
  - 抽 N 个 episode 作为 eval 子集（避免与训练集重叠：用 --offset 跳过前面的）
  - 只保留 eval 必需的列（减小文件体积）：
      pixels, qpos, qvel, button_states,
      privileged_target_drawer_pos, privileged_drawer_pos,
      action, observation, success
  - 重建 ep_idx / step_idx（0..N-1 重新编号）
  - 不再做 task 过滤（collect 端已强制 drawer-only）

运行：
    python filter_eval.py --src ogbench/scene_drawer_train \
                          --dst ogbench/scene_drawer_eval \
                          --n-episodes 50 --offset 0
"""

import argparse
import os
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm


# 新版 collect 产出的列里，eval 真正用得到的子集：
#   - pixels / action / observation: policy 输入 + normalizer 拟合
#   - qpos / qvel / button_states:   set_state 还原初始状态
#   - privileged_target_drawer_pos:  set_target_drawer_pos 注入目标
#   - privileged_drawer_pos:         （可选）可视化/调试用
#   - success:                       找每个ep第一次 success 帧作为 goal pixels
KEYS_TO_KEEP = [
    'pixels', 'qpos', 'qvel', 'button_states',
    'privileged_target_drawer_pos',
    'privileged_drawer_pos',
    'action', 'observation',
    'success',
]
# ep_idx / step_idx 在末尾重新合成（不能直接 copy 源里的，因为 episode 被重新编号）


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',        type=str, default='ogbench/scene_drawer_train',
                        help='源数据集名（不带 .h5）')
    parser.add_argument('--dst',        type=str, default='ogbench/scene_drawer_eval',
                        help='输出数据集名（不带 .h5）')
    parser.add_argument('--n-episodes', type=int, default=50,
                        help='要抽多少个 episode 作为 eval')
    parser.add_argument('--offset',     type=int, default=0,
                        help='跳过源里前 N 个 episode（避免和训练集重叠）')
    return parser.parse_args()


def write_filtered(src_path, dst_path, ep_indices, keys):
    """把选中 episode 的指定 key 顺序写入新 HDF5，重建 ep_len / ep_offset / ep_idx / step_idx。"""
    with h5py.File(src_path, 'r') as src:
        src_ep_offsets = src['ep_offset'][:]
        src_ep_lens = src['ep_len'][:]

        new_ep_lens = src_ep_lens[ep_indices].astype(np.int32)
        new_ep_offsets = np.concatenate(
            [[0], np.cumsum(new_ep_lens[:-1])]
        ).astype(np.int64)
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
            new_ep_idx = np.repeat(
                np.arange(len(ep_indices), dtype=np.int32), new_ep_lens
            )
            new_step_idx = np.concatenate(
                [np.arange(l, dtype=np.int64) for l in new_ep_lens]
            )
            dst.create_dataset('ep_idx', data=new_ep_idx)
            dst.create_dataset('step_idx', data=new_step_idx)

            dst.flush()

        return total_steps


def main():
    args = parse_args()
    cache_dir = os.environ.get(
        'STABLEWM_HOME', os.path.expanduser('~/.stable_worldmodel')
    )
    src_path = Path(cache_dir) / f'{args.src}.h5'
    dst_path = Path(cache_dir) / f'{args.dst}.h5'
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Source: {src_path}')
    print(f'Dest:   {dst_path}')
    print(f'Pick {args.n_episodes} episodes starting from index {args.offset}')

    with h5py.File(src_path, 'r') as f:
        n_total = len(f['ep_len'][:])

    if args.offset + args.n_episodes > n_total:
        raise ValueError(
            f'Source 只有 {n_total} 个 episode，请求 offset={args.offset} + '
            f'n_episodes={args.n_episodes} 越界'
        )

    indices = list(range(args.offset, args.offset + args.n_episodes))
    total_steps = write_filtered(src_path, dst_path, indices, KEYS_TO_KEEP)
    print(f'Done. Wrote {len(indices)} episodes / {total_steps} steps to {dst_path}')


if __name__ == '__main__':
    main()
