import os
os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import envs  # triggers gymnasium.register("libero-goal-lewm-v0")
import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def img_transform(cfg):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def get_normalization(cfg):
    """Compute action/proprio normalization stats from training HDF5."""
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


@hydra.main(version_base=None, config_path="./config/eval", config_name="libero_goal")
def run(cfg: DictConfig):
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    process = get_normalization(cfg)

    # create world environment
    cfg.world.max_episode_steps = 600
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # load policy
    policy_path = cfg.get("policy", "random")
    if policy_path != "random":
        model = swm.policy.AutoCostModel(policy_path)
        model = model.to("cuda").eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )
    else:
        policy = swm.policy.RandomPolicy()

    world.set_policy(policy)

    # run evaluation
    start = time.time()
    metrics = world.evaluate(episodes=cfg.eval.num_eval, seed=cfg.seed)
    elapsed = time.time() - start

    print(metrics)

    # video recording
    video_dir = Path(__file__).parent / "videos" / f"task_{cfg.world.task_id}_{policy_path}"
    video_dir.mkdir(parents=True, exist_ok=True)
    world.record_video(
        video_path=video_dir,
        max_steps=600,
        fps=30,
        viewname="pixels",
        seed=cfg.seed,
    )
    print(f"Videos saved to {video_dir}")

    results_path = Path(__file__).parent / cfg.output.filename
    with results_path.open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {elapsed:.1f}s\n")


if __name__ == "__main__":
    run()

