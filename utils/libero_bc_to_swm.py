"""
Convert LIBERO/robomimic-format BC rollout HDF5 → SWM HDF5 format.

Source layout:
  data/demo_X/obs/agentview_rgb  (T, 224, 224, 3) uint8
  data/demo_X/obs/ee_pos         (T, 3)
  data/demo_X/obs/ee_ori         (T, 3)  axis-angle
  data/demo_X/obs/gripper_states (T, 2)
  data/demo_X/actions            (T, 7)

Target layout (SWM HDF5Dataset format):
  ep_len      (N,)        int64
  ep_offset   (N,)        int64
  pixels      (total, 224, 224, 3)  uint8   agentview
  eye_in_hand (total, 224, 224, 3)  uint8   wrist camera (M1: 进视觉 emb)
  action      (total, 7)  float32
  proprio     (total, 8)  float32   [ee_pos(3) | ee_ori(3) | gripper(2)]
  drawer_qpos (total, 1)  float32   抽屉关节真值 (探测3 回归靶子, 不进模型)

Usage:
  python utils/libero_bc_to_swm.py
"""

import argparse
import numpy as np
import h5py
from pathlib import Path


SRC = "/home/yikang/stable-wm/libero_bc_rollouts/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_bc.hdf5"
DST = "/home/yikang/git/le-wm/data/libero_bc_drawer_v2.h5"

# states = sim.get_state().flatten() = [time(1), qpos..., qvel...]
# 抽屉关节 model qpos 地址=37 (find_drawer_joint), states 下标 = 37 + 1(time) = 38。
# 经验校验: 关抽屉 demo 中该列起≈-0.15 -> 终≈0.001 (open=-0.16, closed=+0.01)。
DRAWER_COL = 38


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=SRC)
    p.add_argument("--dst", default=DST)
    return p.parse_args()


def main():
    args = parse_args()
    src_path = Path(args.src)
    dst_path = Path(args.dst)

    print(f"Source: {src_path}")
    print(f"Dest:   {dst_path}")

    with h5py.File(src_path, "r") as src:
        data_grp = src["data"]
        demo_keys = sorted(
            [k for k in data_grp.keys() if k.startswith("demo_")],
            key=lambda k: int(k.split("_")[1]),
        )
        n_demos = len(demo_keys)
        print(f"Demos: {n_demos}")

        # --- pass 1: collect ep_lens from attrs ---
        ep_lens = np.array(
            [int(data_grp[k].attrs["num_samples"]) for k in demo_keys],
            dtype=np.int64,
        )
        ep_offsets = np.concatenate([[0], np.cumsum(ep_lens)[:-1]]).astype(np.int64)
        total = int(ep_lens.sum())
        print(f"Total steps: {total}")

        # --- create output file ---
        with h5py.File(dst_path, "w") as dst:
            dst.create_dataset("ep_len",    data=ep_lens)
            dst.create_dataset("ep_offset", data=ep_offsets)

            # pre-allocate (chunked so HDF5 doesn't barf on 56GB)
            px_ds = dst.create_dataset(
                "pixels", shape=(total, 224, 224, 3), dtype=np.uint8,
                chunks=(64, 224, 224, 3), compression=None,
            )
            eih_ds = dst.create_dataset(
                "eye_in_hand", shape=(total, 224, 224, 3), dtype=np.uint8,
                chunks=(64, 224, 224, 3), compression=None,
            )
            act_ds = dst.create_dataset(
                "action", shape=(total, 7), dtype=np.float32,
            )
            prop_ds = dst.create_dataset(
                "proprio", shape=(total, 8), dtype=np.float32,
            )
            dq_ds = dst.create_dataset(
                "drawer_qpos", shape=(total, 1), dtype=np.float32,
            )

            # --- pass 2: fill ---
            cursor = 0
            for i, key in enumerate(demo_keys):
                demo = data_grp[key]
                obs  = demo["obs"]
                T    = ep_lens[i]

                px_ds[cursor:cursor + T] = obs["agentview_rgb"][:]      # (T,224,224,3)
                eih_ds[cursor:cursor + T] = obs["eye_in_hand_rgb"][:]   # (T,224,224,3) 原样, 不翻转(与 agentview 一致)
                act_ds[cursor:cursor + T] = demo["actions"][:].astype(np.float32)

                proprio = np.concatenate([
                    obs["ee_pos"][:].astype(np.float32),          # (T,3)
                    obs["ee_ori"][:].astype(np.float32),           # (T,3)
                    obs["gripper_states"][:].astype(np.float32),   # (T,2)
                ], axis=1)
                prop_ds[cursor:cursor + T] = proprio

                dq_ds[cursor:cursor + T] = demo["states"][:, DRAWER_COL:DRAWER_COL + 1].astype(np.float32)

                cursor += T

                if (i + 1) % 100 == 0 or i == n_demos - 1:
                    print(f"  [{i+1:5d}/{n_demos}]  cursor={cursor}  ({100*(i+1)/n_demos:.1f}%)")

    print(f"\nSaved → {dst_path}")


if __name__ == "__main__":
    main()
