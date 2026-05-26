"""
OGBScene 系列任务的 eval 入口（独立于通用 eval.py）。

为什么单独一个脚本：
  - SceneEnv.set_state 把 button_states 拆成两个 kwargs (button_state_0/_1)，
    但 Hydra callable 机制只能整列传，无法在 YAML 里切片 button_states (2,)。
  - 这里通过 monkey-patch 让 set_state 同时接受 button_states= 数组形式。
  - 不污染 eval.py，避免影响其他环境的 eval。

主流程与 eval.py 完全一致：
  HDF5Dataset → 选起始帧 + goal_offset → world.evaluate_from_dataset
  policy = WorldModelPolicy(CEMSolver(JEPA ckpt))

用法：
  MUJOCO_GL=egl EGL_DEVICE_ID=0 python ogb/ogb_eval.py \
      --config-name=ogbscene_drawer policy=<ckpt 路径>
"""

import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


def _patch_scene_env_set_state():
    """让 SceneEnv.set_state 多接受 button_states= 数组参数。

    dataset 里 button_states 是 (2,) int64。Hydra callable 传整列时只能整体传，
    包一层让它在 set_state 内部拆成 button_state_0 / button_state_1。
    """
    try:
        from stable_worldmodel.envs.ogbench.scene_env import SceneEnv
    except ImportError:
        return
    if getattr(SceneEnv.set_state, "_lewm_patched", False):
        return
    orig_set_state = SceneEnv.set_state

    def set_state(self, qpos, qvel, button_states=None, **kwargs):
        if button_states is not None:
            arr = np.asarray(button_states).reshape(-1)
            kwargs["button_state_0"] = int(arr[0])
            kwargs["button_state_1"] = int(arr[1])
        orig_set_state(self, qpos, qvel, **kwargs)

    set_state._lewm_patched = True
    SceneEnv.set_state = set_state


_patch_scene_env_set_state()


def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


@hydra.main(version_base=None, config_path="../config/eval", config_name="ogbscene_drawer")
def run(cfg: DictConfig):
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    cfg.world.max_episode_steps = 100
    # image_shape 必须和 dataset pixels 分辨率一致 (scene_drawer 是 64×64)。
    # 否则 world.evaluate_from_dataset 里 video_frames 会被 dataset pixels 的 shape 误分配，
    # env 后续渲染 224×224 时塞不进去。policy.transform 之后还是会 resize 到 224 给 encoder，
    # 所以这里渲染分辨率改小不影响模型，只影响导出视频的清晰度。
    world = swm.World(**cfg.world, image_shape=(64, 64))

    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    policy_cfg = cfg.get("policy", "random")
    if policy_cfg != "random":
        model = swm.policy.AutoCostModel(policy_cfg)
        model = model.to("cuda").eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        plan_config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=plan_config, process=process, transform=transform
        )
    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # pick valid (episode, start) pairs: start + goal_offset_steps must be in-episode
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=results_path / "results",
    )
    end_time = time.time()

    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
