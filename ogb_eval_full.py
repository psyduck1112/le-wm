"""
OGBScene drawer 全任务 eval（不用 offset，跑完整 task 到 terminated）。

与 ogb_eval.py 的区别：
  ogb_eval.py:  随机起点 + 固定 goal_offset（替代 cube/pusht 的子序列eval）
  ogb_eval_full.py:
    - 每个 episode 从 step 0 开始（drawer 在初始位置）
    - target_drawer_pos 由 dataset 提供（call set_target_drawer_pos）
    - goal pixels = 同 episode 内 success=True 的第一帧
    - eval env mode='data_collection' + terminate_at_goal=True
      → _success 只看 drawer_success（非全场景成功）
    - success rate = terminated within eval_budget 的 episode 比例

用法:
  MUJOCO_GL=egl EGL_DEVICE_ID=0 python ogb_eval_full.py \
      --config-name=ogbscene_drawer_full policy=<ckpt 路径>
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
    """让 SceneEnv.set_state 多接受 button_states= 数组参数（同 ogb_eval.py）。"""
    from stable_worldmodel.envs.ogbench.scene_env import SceneEnv
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
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


def find_first_success_steps(dataset, ep_indices, col_name):
    """对每个 episode 找第一个 success=True 的 step_idx；找不到的 episode 标记 -1。"""
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    success = dataset.get_col_data("success").flatten().astype(bool)

    result = {}
    for ep_id in ep_indices:
        mask = (episode_idx == ep_id) & success
        if not np.any(mask):
            result[int(ep_id)] = -1
            continue
        first_idx = np.nonzero(mask)[0][0]
        result[int(ep_id)] = int(step_idx[first_idx])
    return result


@hydra.main(version_base=None, config_path="./config/eval",
            config_name="ogbscene_drawer_full")
def run(cfg: DictConfig):
    # Eval env: data_collection mode + terminate_at_goal=True
    # → _success 只看 _target_task (= 'drawer')，drawer 到位即 terminated
    cfg.world.max_episode_steps = max(cfg.eval.eval_budget * 2, 200)
    world = swm.World(**cfg.world, image_shape=(64, 64))

    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices = np.unique(dataset.get_col_data(col_name))

    # 对每个 episode 找第一次 success 的 step（作为 goal step）
    first_success = find_first_success_steps(dataset, ep_indices, col_name)
    valid_eps = np.array(
        [ep for ep in ep_indices if first_success[int(ep)] > 0],
        dtype=ep_indices.dtype,
    )
    print(f"{len(valid_eps)}/{len(ep_indices)} episodes 有 success 帧可用作 goal")

    # 抽 num_eval 个 episode（按 episode 而非 row 抽样）
    g = np.random.default_rng(cfg.seed)
    n_eval = min(cfg.eval.num_eval, len(valid_eps))
    chosen_eps = g.choice(valid_eps, size=n_eval, replace=False)
    chosen_eps = np.sort(chosen_eps)
    print(f"Eval episodes: {chosen_eps.tolist()}")

    # 起点 = 每个 ep 的 step_idx=0；goal_offset_steps = first_success（每个ep独立）
    # evaluate_from_dataset 只接受 scalar offset，所以这里手动准备 start_steps（=0）
    # 并用最大 first_success 作为 offset（保证所有ep都已成功）。
    # 但每个ep的 first_success 不同 → 改用循环逐个评测。
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
        model = swm.policy.AutoCostModel(policy_cfg).to("cuda").eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        plan_config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=plan_config, process=process, transform=transform
        )
    else:
        policy = swm.policy.RandomPolicy()

    world.set_policy(policy)

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # 逐个 episode 评测（因为每个ep的 goal step 不一样）
    # 这里 num_envs=1，每次循环跑一个ep
    assert cfg.world.num_envs == 1, \
        "ogb_eval_full 暂只支持 num_envs=1（每个ep的 goal step 不同）"

    successes = []
    eval_metrics = []
    start_time = time.time()
    video_dir = results_path / "results_full"
    video_dir.mkdir(parents=True, exist_ok=True)

    for i, ep in enumerate(chosen_eps):
        ep = int(ep)
        goal_step = first_success[ep]
        print(f"\n=== Episode {i+1}/{n_eval} (ep_idx={ep}, goal_step={goal_step}) ===")

        metrics = world.evaluate_from_dataset(
            dataset,
            start_steps=[0],
            goal_offset_steps=goal_step,    # per-ep offset
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=[ep],
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video_path=video_dir / f"ep{ep}",
        )
        eval_metrics.append(metrics)
        # metrics 里通常有 'success_rate' / 'episode_successes'
        ep_success = metrics.get("episode_successes", metrics.get("success_rate"))
        if hasattr(ep_success, "__len__"):
            successes.extend(list(np.asarray(ep_success).flatten()))
        else:
            successes.append(float(ep_success))
        print(f"  success={successes[-1]}")

    end_time = time.time()
    success_rate = float(np.mean(successes))

    print(f"\n==== RESULTS ====")
    print(f"Success rate: {success_rate:.3f} ({sum(successes):.0f}/{len(successes)})")
    print(f"Eval time:    {end_time - start_time:.1f}s")

    out_path = results_path / cfg.output.filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"success_rate: {success_rate}\n")
        f.write(f"per_episode: {successes}\n")
        f.write(f"eval_time: {end_time - start_time}\n")


if __name__ == "__main__":
    run()
