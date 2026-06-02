"""Probe 4 (LIBERO_DIAGNOSTICS.md 探测4): BC baseline control.

Eval a standard LIBERO BCTransformerPolicy (the same policy that generated le-wm's
training rollouts) on the SAME drawer-close task, and compare its success rate to
le-wm-MPC (baseline 10%). Separates "data insufficient" from "le-wm WM+planning bad".

  BC succeeds, MPC fails  => data is enough; bug is in le-wm world-model + planning.
  BC also fails           => data coverage / task difficulty (not le-wm-specific).

Reuses bc_rollout.py's rollout core, stripped of all HDF5 writing.

Usage:
  CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl EGL_DEVICE_ID=0 \
    python diagnostics/p4_bc_eval.py --n-eval 20
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import numpy as np
import torch
import cv2
import robomimic.utils.obs_utils as ObsUtils
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.lifelong.algos import Sequential
from libero.lifelong.metric import raw_obs_to_tensor_obs
from libero.lifelong.utils import torch_load_model, safe_device
from libero.lifelong.main import get_task_embs

CKPT = ("/home/yikang/LIBERO/libero/experiments/LIBERO_90/Sequential"
        "/BCTransformerPolicy_seed10000/run_008/task0_model.pth")
WARMUP = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sd, cfg, previous_mask = torch_load_model(args.ckpt, map_location=args.device)
    cfg.device = args.device
    algo = safe_device(Sequential(10, cfg), cfg.device)
    algo.policy.previous_mask = previous_mask
    algo.policy.load_state_dict(sd)
    algo.eval()
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})

    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    descriptions = [benchmark.get_task(i).language for i in range(10)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)
    task = benchmark.get_task(args.task_id)
    task_emb = benchmark.get_task_emb(args.task_id)
    print(f"task: {benchmark.get_task_names()[args.task_id]}")

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states = torch.load(os.path.join(get_libero_path("init_states"),
                             task.problem_folder, task.init_states_file), weights_only=False)
    BC_SIZE = cfg.data.img_h
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=224, camera_widths=224)

    successes = 0
    with torch.no_grad():
        for ep in range(args.n_eval):
            algo.reset()
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(WARMUP):
                obs, _, _, _ = env.step(np.zeros(7))
            success, steps = False, 0
            for t in range(args.max_steps):
                obs_bc = {**obs}
                for cam in (cfg.data.obs_key_mapping["agentview_rgb"],
                            cfg.data.obs_key_mapping["eye_in_hand_rgb"]):
                    obs_bc[cam] = cv2.resize(obs[cam], (BC_SIZE, BC_SIZE),
                                             interpolation=cv2.INTER_AREA)
                data = raw_obs_to_tensor_obs([obs_bc], task_emb, cfg)
                action = algo.policy.get_action(data)[0]
                obs, _, done, _ = env.step(action)
                steps = t + 1
                if done:
                    success = True
                    break
            successes += int(success)
            print(f"ep {ep+1:3d}/{args.n_eval}  {'SUCCESS' if success else 'fail   '}  "
                  f"steps={steps:3d}  rate={successes/(ep+1):.2f}", flush=True)

    print(f"\n==== BC success rate: {successes}/{args.n_eval} = "
          f"{successes/args.n_eval:.1%}  (le-wm MPC baseline = 10%) ====")
    env.close()


if __name__ == "__main__":
    main()
