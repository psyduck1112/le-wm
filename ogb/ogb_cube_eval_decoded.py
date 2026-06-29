"""OGBench cube eval with DECODED physics cost (改进版, 不改 jepa.py).

与 eval.py 完全相同的 World/solver/policy 机制, 唯一区别:
加载 co-train 模型后, 在【实例上】把 criterion 换成 decoded_criterion ——
用模型自带的 task_head 把 pred/goal 的最后一帧 emb 解码成方块 3D 位置,
在【米空间】求平方距离 (反归一化乘 σ), 与成功判据 ‖cube−target‖≤0.04m 同度量.

jepa.py 一行不动: 替换发生在本脚本里 (types.MethodType 绑到 model 实例).

跑:
  MUJOCO_GL=egl EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
    python ogb/ogb_cube_eval_decoded.py --config-name=cube \
      policy=/mnt/hdd1/data/yikang/outputs_cube_phys/lewm_cube_phys_epoch_25 \
      norm_path=/home/yikang/stable-wm/cube/cube_pos_norm.npz

对照 (emb_l2 baseline) 仍用原 eval.py:
  python eval.py --config-name=cube policy=<同一 ckpt>
"""

import os
import sys

os.environ["MUJOCO_GL"] = "egl"
# 脚本在 ogb/ 下, sys.path[0]=ogb/; 把项目根插进 path 让 torch.load 能 import jepa
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from pathlib import Path
from types import MethodType

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


# ----------------------------------------------------------------------------
# DECODED COST: 本脚本内定义, 实例级替换 model.criterion (不进 jepa.py)
# ----------------------------------------------------------------------------
def decoded_criterion(self, info_dict: dict):
    """用 task_head 解码方块位置, 在米空间求距离. 返回 (B, S) per-candidate cost.

    pred_emb: (B, S, T, D)   goal_emb: (B, T, D) 或 (B, 1, T, D)
    COST 只用 cube_pos = head 输出的前 3 维 (label_keys 把 privileged_block_0_pos 排第一).
    head 可能还输出 eef/gripper 等 proprio (训练塑形用), 但规划 cost 不读它们 -> 切 [:3].
    cost = ‖(cube_pred − cube_goal) ⊙ σ_cube‖²   (μ 在差里抵消, 只需 σ 反归一化到米)
    """
    pred_emb = info_dict["predicted_emb"]      # (B, S, T, D)
    goal_emb = info_dict["goal_emb"]
    if goal_emb.ndim == 3:
        goal_emb = goal_emb.unsqueeze(1)       # (B, 1, T, D)

    sig = self._decoded_sigma.to(pred_emb.device)   # (3,)  cube σ
    k = sig.shape[0]                                 # cube_pos 维数 = 3
    p = self.task_head(pred_emb[..., -1, :])[..., :k]   # (B, S, 3)  只取 cube
    g = self.task_head(goal_emb[..., -1, :])[..., :k]   # (B, 1, 3)
    diff = (p - g) * sig                                # 反归一化到米
    return diff.pow(2).sum(-1)                          # (B, S)


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


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
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


@hydra.main(version_base=None, config_path="../config/eval", config_name="cube")
def run(cfg: DictConfig):
    """Eval with decoded physics cost (改进版)."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- load model. cost_mode=decoded 时实例级替换 criterion; emb_l2 时用模型原 criterion.
    # 一个脚本跑两种条件, 视频/结果都落到 le-wm (video_dir), 便于对照.
    policy = cfg.get("policy", "random")
    assert policy != "random", "需要一个 ckpt"
    cost_mode = cfg.get("cost_mode", "decoded")

    model = swm.policy.AutoCostModel(cfg.policy)

    std = None
    if cost_mode == "decoded":
        assert getattr(model, "task_head", None) is not None, (
            "ckpt 没有 task_head, 不能用 decoded cost. 用 co-train 的 lewm_cube_phys ckpt."
        )
        # σ 反归一化向量: 优先 norm_path npz, 否则现场从 h5 算 (确定性一致)
        norm_path = cfg.get("norm_path", None)
        if norm_path and Path(norm_path).exists():
            std = np.load(norm_path)["std"].reshape(-1)
            print(f"[decoded] σ from {norm_path}: {std.tolist()}")
        else:
            col = stats_dataset.get_col_data("privileged_block_0_pos")
            col = col[~np.isnan(col).any(axis=1)]
            std = torch.from_numpy(np.asarray(col)).std(0).numpy()
            print(f"[decoded] σ recomputed from h5: {std.tolist()}")
        model._decoded_sigma = torch.tensor(std, dtype=torch.float32)
        model.criterion = MethodType(decoded_criterion, model)  # ← 实例级替换, jepa.py 不变
    else:
        print(f"[emb_l2] 用模型原 criterion (192 维 emb-L2), 不替换")

    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )

    # 视频/结果落地目录: 默认写进 le-wm/cube_videos/<cost_mode> (不留在 hdd1 的 results)
    repo_root = Path(__file__).resolve().parent.parent
    video_dir = Path(cfg.get("video_dir") or (repo_root / "cube_videos" / cost_mode))
    video_dir.mkdir(parents=True, exist_ok=True)
    results_path = video_dir

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
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
        video_path=video_dir,                       # 视频直接落到 le-wm/cube_videos/<mode>
    )
    end_time = time.time()

    print(f"[{cost_mode}] cost", metrics)

    out_path = video_dir / f"results_{cost_mode}.txt"
    with out_path.open("a") as f:
        f.write(f"\n==== CONFIG ({cost_mode} cost) ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        if std is not None:
            f.write(f"norm_sigma: {std.tolist()}\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
