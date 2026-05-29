"""
LeWM eval on LIBERO KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet.

Uses swm.World + WorldModelPolicy (CEM) for clean success-rate tracking
and automatic video recording.

Usage:
    EGL_DEVICE_ID=0 python libero_eval/bc_drawer_eval.py \
        --ckpt /home/yikang/stable-wm/outputs/lewm_libero_bc_drawer_epoch_75 \
        --n-eval 20 --eval-budget 150
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import h5py
from sklearn import preprocessing
from torchvision.transforms import v2 as T
import stable_pretraining as spt
import stable_worldmodel as swm

import envs.libero_env  # registers libero-bc-drawer-lewm-v0


H5_PATH    = "/home/yikang/stable-wm/libero_bc_drawer.h5"
CACHE_DIR  = "/home/yikang/stable-wm"
HISTORY    = 3


def build_process_and_transform(h5_path: str):
    """Compute action/proprio normalizers from training data."""
    process = {}
    with h5py.File(h5_path, "r") as f:
        for col in ("action", "proprio"):
            data = f[col][:]
            scaler = preprocessing.StandardScaler()
            scaler.fit(data)
            process[col] = scaler

    transform = {
        "pixels": T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(**spt.data.dataset_stats.ImageNet),
            T.Resize(224),
        ]),
        "goal": T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(**spt.data.dataset_stats.ImageNet),
            T.Resize(224),
        ]),
    }
    return process, transform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",
                   default="/home/yikang/stable-wm/outputs/lewm_libero_bc_drawer_epoch_75",
                   help="Path prefix for _object.ckpt (omit suffix)")
    p.add_argument("--h5",          default=H5_PATH)
    p.add_argument("--n-eval",      type=int, default=1)
    p.add_argument("--eval-budget", type=int, default=150,
                   help="Max env steps per episode")
    p.add_argument("--horizon",           type=int, default=7)
    p.add_argument("--receding-horizon",  type=int, default=2,
                   help="Steps to execute before re-planning")
    p.add_argument("--num-samples",       type=int, default=300)
    p.add_argument("--n-steps",           type=int, default=5,
                   help="CEM iterations per planning call")
    p.add_argument("--topk",              type=int, default=50)
    p.add_argument("--video-dir",   default=None,
                   help="Directory to save rollout videos (None = skip)")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # ── model ────────────────────────────────────────────────────────────────
    ckpt_path = args.ckpt if args.ckpt.endswith(".ckpt") else args.ckpt + "_object.ckpt"
    model = swm.policy.AutoCostModel(ckpt_path.replace("_object.ckpt", ""))
    model = model.cuda().eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    # ── normalizers + image transform ────────────────────────────────────────
    process, transform = build_process_and_transform(args.h5)

    # ── CEM solver + policy ──────────────────────────────────────────────────
    solver = swm.solver.CEMSolver(
        model=model,
        num_samples=args.num_samples,
        n_steps=args.n_steps,
        topk=args.topk,
        device="cuda",
    )
    config = swm.PlanConfig(
        horizon=args.horizon,
        receding_horizon=args.receding_horizon,
        action_block=1,   # frameskip=1
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )

    # ── world ────────────────────────────────────────────────────────────────
    world = swm.World(
        env_name="libero-bc-drawer-lewm-v0",
        num_envs=1,
        image_shape=(224, 224),
        history_size=HISTORY,
        frame_skip=1,
        max_episode_steps=args.eval_budget,
        goal_conditioned=True,
        demo_hdf5_path=args.h5,
    )
    world.set_policy(policy)

    # ── eval loop ────────────────────────────────────────────────────────────
    import imageio
    from pathlib import Path

    rng = np.random.default_rng(args.seed)
    vdir = Path(args.video_dir) if args.video_dir else None
    if vdir:
        vdir.mkdir(parents=True, exist_ok=True)

    successes = 0

    for ep in range(args.n_eval):
        seed = int(rng.integers(0, 10000))
        world.reset(seed=[seed])

        ep_frames = []
        success = False

        for step in range(args.eval_budget):
            # collect frame before step (world.infos["pixels"][0, -1] = latest HWC frame)
            if vdir is not None:
                ep_frames.append(world.infos["pixels"][0, -1].copy())

            world.step()

            if world.terminateds[0]:
                success = True
                break

        # collect final frame
        if vdir is not None:
            ep_frames.append(world.infos["pixels"][0, -1].copy())

        successes += int(success)
        tag = "SUCCESS" if success else "fail   "
        print(f"ep {ep+1:3d}/{args.n_eval}  {tag}  steps={step+1:3d}  "
              f"rate={successes/(ep+1):.2f}")

        if vdir is not None and ep_frames:
            vpath = vdir / f"ep{ep:02d}_{tag.strip()}.mp4"
            writer = imageio.get_writer(str(vpath), fps=20)
            for frame in ep_frames:
                writer.append_data(frame[::-1, ::-1])  # OpenGL → display
            writer.close()

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Success rate: {successes}/{args.n_eval} = {successes/args.n_eval:.1%}")
    print(f"{'='*50}")
    if vdir:
        print(f"Videos saved to {vdir}")

    world.envs.close()


if __name__ == "__main__":
    main()
