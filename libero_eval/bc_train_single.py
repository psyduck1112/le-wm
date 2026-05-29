"""
Train BC on a single LIBERO task without running the full sequential loop.

cd /home/yikang/git/le-wm
MUJOCO_GL=egl EGL_DEVICE_ID=0 python libero_eval/bc_train_single.py \
    --task-id 3 \
    --n-epochs 50
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import yaml
import pprint

os.chdir("/home/yikang/LIBERO/libero")  # hydra writes outputs relative to cwd

from easydict import EasyDict
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.algos import get_algo_class
from libero.lifelong.datasets import get_dataset, SequenceVLDataset
from libero.lifelong.main import get_task_embs
from libero.lifelong.utils import (
    control_seed, safe_device, torch_load_model,
    create_experiment_dir, NpEncoder, compute_flops,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=3,
        help="Task index within the benchmark (0-based)")
    parser.add_argument("--benchmark", type=str, default="LIBERO_90")
    parser.add_argument("--task-order-index", type=int, default=0)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config", type=str,
        default="/home/yikang/LIBERO/libero/experiments/LIBERO_90/Sequential"
                "/BCTransformerPolicy_seed10000/run_008/config.json",
        help="Path to an existing config.json to reuse architecture settings")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── load config from existing run (reuse architecture / hyperparams) ────
    with open(args.config) as f:
        cfg = EasyDict(json.load(f))

    # override training-specific fields
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.train.n_epochs = args.n_epochs
    cfg.benchmark_name = args.benchmark
    cfg.data.task_order_index = args.task_order_index
    cfg.use_wandb = False
    cfg.eval.eval = False
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")

    control_seed(cfg.seed)

    task_id = args.task_id

    # ── benchmark & task info ────────────────────────────────────────────────
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_manip_tasks = benchmark.n_tasks

    assert task_id < n_manip_tasks, \
        f"task_id={task_id} out of range (benchmark has {n_manip_tasks} tasks)"

    # need ALL descriptions to build task_embs (language conditioning)
    descriptions = [benchmark.get_task(i).language for i in range(n_manip_tasks)]
    print(f"Training on task {task_id}: {descriptions[task_id]}")

    # ── load ONLY task_id's dataset ──────────────────────────────────────────
    dataset_path = os.path.join(cfg.folder, benchmark.get_task_demonstration(task_id))
    print(f"Dataset: {dataset_path}")

    task_dataset, shape_meta = get_dataset(
        dataset_path=dataset_path,
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True,
        seq_len=cfg.data.seq_len,
    )

    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    vl_dataset = SequenceVLDataset(task_dataset, task_embs[task_id])
    print(f"  demos: {vl_dataset.n_demos}  sequences: {vl_dataset.total_num_sequences}")

    # ── build algo ───────────────────────────────────────────────────────────
    cfg.shape_meta = shape_meta
    create_experiment_dir(cfg)  # sets cfg.experiment_dir, creates run_XXX/

    algo = safe_device(
        get_algo_class(cfg.lifelong.algo)(n_manip_tasks, cfg),
        cfg.device,
    )

    with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)

    GFLOPs, MParams = compute_flops(algo, vl_dataset, cfg)
    print(f"Policy: {GFLOPs:.1f} GFLOPs, {MParams:.1f} MParams")

    # ── train ────────────────────────────────────────────────────────────────
    result_summary = {
        "L_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),
        "S_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),
        "L_fwd": np.zeros((n_manip_tasks,)),
        "S_fwd": np.zeros((n_manip_tasks,)),
    }

    algo.train()
    algo.learn_one_task(vl_dataset, task_id, benchmark, result_summary)

    ckpt = os.path.join(cfg.experiment_dir, f"task{task_id}_model.pth")
    print(f"\nDone. Checkpoint saved to:\n  {ckpt}")


if __name__ == "__main__":
    main()