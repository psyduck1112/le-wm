import os
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import envs  # triggers gymnasium.register("libero-goal-lewm-v0")
import h5py
import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig
from PIL import Image
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


def img_transform(cfg):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def get_normalization(cfg):
    dataset_path = Path(cfg.world.demo_hdf5_path).parent
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    process = {}
    for col in cfg.dataset.keys_to_cache:
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
    return process


def load_short_goals(hdf5_path: str, K: int) -> list[np.ndarray]:
    """Load frame[K] of each demo as a short-horizon goal."""
    goals = []
    with h5py.File(hdf5_path, "r") as f:
        offsets = f["ep_offset"][:]
        lengths = f["ep_len"][:]
        pixels = f["pixels"]
        for o, l in zip(offsets, lengths):
            idx = min(K, l - 1)
            goals.append(pixels[o + idx])
    return goals


@hydra.main(version_base=None, config_path="./config/eval", config_name="libero_short")
def run(cfg: DictConfig):
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }
    process = get_normalization(cfg)

    # build world
    cfg.world.max_episode_steps = cfg.eval.max_steps
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # load policy (same as eval_libero.py)
    model = swm.policy.AutoCostModel(cfg.policy)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    plan_cfg = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=plan_cfg, process=process, transform=transform
    )
    world.set_policy(policy)

    # load short-horizon goal pool from HDF5
    goals = load_short_goals(cfg.world.demo_hdf5_path, cfg.eval.goal_offset)
    print(f"Loaded {len(goals)} goal candidates (frame {cfg.eval.goal_offset})")

    # set up output dir
    out_dir = Path(__file__).parent / "videos" / f"short_off{cfg.eval.goal_offset}_task{cfg.world.task_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # pick N goals deterministically from the pool
    rng = np.random.RandomState(cfg.seed)
    goal_indices = rng.choice(len(goals), size=cfg.eval.num_eval, replace=False)

    # loop: one episode per iteration, each with a fixed goal
    for i, gi in enumerate(goal_indices):
        gi = int(gi)
        # monkey-patch: pool of 1 => env.reset() always picks this goal
        world.envs[0].unwrapped._goal_images = [goals[gi]]
        # save goal frame for visual inspection
        Image.fromarray(goals[gi]).save(out_dir / f"goal_{i:03d}_demo{gi:03d}.png")
        # record video
        world.record_video(
            video_path=out_dir / f"ep_{i:03d}",
            max_steps=cfg.eval.max_steps,
            fps=30,
            viewname="pixels",
            seed=cfg.seed + i,
        )
        print(f"Episode {i}: goal=demo{gi} -> ep_{i:03d}")

    print(f"\nAll videos & goals saved to: {out_dir}")


if __name__ == "__main__":
    run()
