"""
Random-exploration re-sampling: fill the (observation, action) space.

Goal: the expert dataset is a narrow tube. We want to widen the JOINT (s, a)
distribution the world model sees, so p(s'|s,a) is learned broadly (not just on the
expert path). We reuse each demo's EXACT initial full MuJoCo state (-> inherits the
per-demo randomized object placements & arm pose), then drive the arm with RANDOM
actions. The simulator gives REAL dynamics, so every recorded
(frame_t, action_t, frame_{t+1}) triple is physically consistent — only now both the
visited states AND the actions taken in them sweep far off the expert tube.

Why action-repeat (hold) instead of per-step i.i.d. noise:
  OSC is incremental control (action = delta command). Per-step zero-mean noise
  cancels -> the arm jitters in place -> actions are diverse but STATES don't move,
  so (s,a) pairs stay clustered on the expert states. Holding a random action for
  `hold` steps makes net displacement grow ~linearly -> the arm sweeps through new
  states, and since the state keeps changing each step every state is still paired
  with a random action -> the JOINT (s,a) space gets filled. `hold` trades off:
  smaller -> more action diversity per state; larger -> wider state spread.

Output schema == libero_bc_drawer_v2.h5 (drop-in for HDF5Dataset / ConcatDataset):
  ep_len, ep_offset, pixels, eye_in_hand, action, proprio(8), drawer_qpos(1)

Usage:
  MUJOCO_GL=egl EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
    python utils/perturb_resim.py --n-episodes 5000 \
        --dst /mnt/hdd1/data/yikang/libero_bc_drawer_perturb.h5
"""
import argparse
import os
import time
import numpy as np
import h5py

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle

SRC = "/home/yikang/stable-wm/libero_bc_rollouts/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_bc.hdf5"
DST = "/mnt/hdd1/data/yikang/libero_bc_drawer_perturb.h5"
DRAWER_ADDR = 37  # wooden_cabinet_1_top_level qpos addr (verified; = v2 states col 38 - 1 time)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=SRC)
    p.add_argument("--dst", default=DST)
    p.add_argument("--n-episodes", type=int, default=5000, help="how many source-demo scenes to roll")
    p.add_argument("--hold", type=int, default=4, help="action-repeat length (steps a random action is held)")
    p.add_argument("--scale", type=float, default=1.0, help="action magnitude (1.0 = full [-1,1] range)")
    p.add_argument("--drift", type=float, default=0.0,
                   help="per-episode constant bias magnitude added to every action. 0 = pure zero-mean "
                        "random walk (episodes all look alike). >0 gives each episode its own fixed "
                        "random direction so net displacement ~drift*T*dir -> episodes fan out to "
                        "different regions instead of jittering around the start.")
    p.add_argument("--drift-spectrum", action="store_true",
                   help="draw EACH episode's drift magnitude ~ U(0, drift) instead of using a fixed "
                        "drift. Gives a continuum: small-mag episodes stay near the expert tube (dense "
                        "near-distribution coverage), large-mag episodes rail out to the workspace "
                        "extremes (boundary / OOD-recovery states). --drift is then the MAX magnitude.")
    p.add_argument("--start", choices=["random", "zero"], default="random",
                   help="random: start from a random step along the expert demo (covers all drawer openings); "
                        "zero: always start from states[0] (drawer always open)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def clear_done(env):
    base = env.env if hasattr(env, "env") and hasattr(env.env, "done") else env
    base.done = False
    base.ignore_done = True
    base.timestep = 0


def get_obs(env):
    """Current observation WITHOUT stepping (robosuite env is at env.env)."""
    tgt = env if hasattr(env, "_get_observations") else env.env
    return tgt._get_observations()


def proprio_of(obs):
    return np.concatenate([
        obs["robot0_eef_pos"].astype(np.float32),                   # (3,)
        quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),  # (3,)
        obs["robot0_gripper_qpos"].astype(np.float32),              # (2,)
    ])


def build_env(img_size=224):
    bench = get_benchmark("LIBERO_90")(0)
    task = bench.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=img_size, camera_widths=img_size)


def main():
    args = parse_args()

    src = h5py.File(args.src, "r")
    dgrp = src["data"]
    demo_keys = sorted([k for k in dgrp.keys() if k.startswith("demo_")],
                       key=lambda k: int(k.split("_")[1]))[:args.n_episodes]
    n = len(demo_keys)
    ep_lens = np.array([int(dgrp[k].attrs["num_samples"]) for k in demo_keys], dtype=np.int64)
    ep_offsets = np.concatenate([[0], np.cumsum(ep_lens)[:-1]]).astype(np.int64)
    total = int(ep_lens.sum())
    print(f"src={args.src}\ndst={args.dst}\nepisodes={n}  total_frames={total}  "
          f"hold={args.hold}  scale={args.scale}", flush=True)

    env = build_env()

    # --- orientation self-check vs the source-stored frame (must be RAW, no flip) ---
    s0 = dgrp[demo_keys[0]]["states"][0]
    src_img0 = dgrp[demo_keys[0]]["obs"]["agentview_rgb"][0].astype(np.float32)
    env.reset(); env.sim.set_state_from_flattened(s0); env.sim.forward()
    fresh = get_obs(env)["agentview_image"].astype(np.float32)
    mse_raw = ((fresh - src_img0) ** 2).mean()
    mse_flip = ((fresh[::-1] - src_img0) ** 2).mean()
    assert mse_raw < mse_flip, f"orientation mismatch: raw={mse_raw:.1f} flip={mse_flip:.1f}"
    print(f"orientation check OK (raw={mse_raw:.1f} < flip={mse_flip:.1f}) -> RAW", flush=True)

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    with h5py.File(args.dst, "w") as dst:
        dst.create_dataset("ep_len", data=ep_lens)
        dst.create_dataset("ep_offset", data=ep_offsets)
        px = dst.create_dataset("pixels", (total, 224, 224, 3), np.uint8, chunks=(64, 224, 224, 3))
        eih = dst.create_dataset("eye_in_hand", (total, 224, 224, 3), np.uint8, chunks=(64, 224, 224, 3))
        act = dst.create_dataset("action", (total, 7), np.float32)
        prop = dst.create_dataset("proprio", (total, 8), np.float32)
        dq = dst.create_dataset("drawer_qpos", (total, 1), np.float32)

        cursor = 0
        t_start = time.time()
        for i, key in enumerate(demo_keys):
            demo = dgrp[key]
            states = demo["states"][:]
            T = int(ep_lens[i])
            rng = np.random.default_rng(args.seed + 1 + int(key.split("_")[1]))  # per-episode reproducible

            start = int(rng.integers(0, T)) if args.start == "random" else 0  # drawer opening varies along demo
            env.reset()
            env.sim.set_state_from_flattened(states[start]); env.sim.forward()
            clear_done(env)
            obs = get_obs(env)

            # per-episode constant drift bias (per-dim Gaussian) -> each episode has its own
            # sustained push, so episodes fan out to different regions. With --drift-spectrum the
            # per-episode magnitude is drawn ~U(0,drift): a continuum from tube to workspace extremes.
            if args.drift > 0:
                mag = rng.uniform(0, args.drift) if args.drift_spectrum else args.drift
                drift_vec = (mag * rng.standard_normal(7)).astype(np.float32)
            else:
                drift_vec = np.zeros(7, dtype=np.float32)

            a = np.zeros(7, dtype=np.float32)  # current held action
            for t in range(T):
                # resample a fresh space-filling action every `hold` steps;
                # the per-episode drift biases the walk so it travels, not just jitters
                if t % args.hold == 0:
                    a = np.clip(drift_vec + args.scale * rng.uniform(-1.0, 1.0, 7),
                                -1.0, 1.0).astype(np.float32)

                # record frame t (obs + sim drawer_q are at the SAME instant)
                j = cursor + t
                px[j] = obs["agentview_image"]
                eih[j] = obs["robot0_eye_in_hand_image"]
                prop[j] = proprio_of(obs)
                dq[j] = env.sim.data.qpos[DRAWER_ADDR]
                act[j] = a

                obs, _, _, _ = env.step(a)

            cursor += T
            if (i + 1) % 50 == 0 or i == n - 1:
                el = time.time() - t_start
                eta = el / (i + 1) * (n - i - 1)
                print(f"  [{i+1:5d}/{n}] cursor={cursor} "
                      f"({100*(i+1)/n:.1f}%)  elapsed={el/60:.1f}m  eta={eta/60:.1f}m", flush=True)

    src.close(); env.close()
    print(f"\nSaved -> {args.dst}", flush=True)


if __name__ == "__main__":
    main()
