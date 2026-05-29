'''
MUJOCO_GL=egl EGL_DEVICE_ID=0 python libero_eval/bc_rollout.py \
    --ckpt .../run_009/task3_model.pth \
    --task-id 3 \
    --n-episodes 200 --n-videos 3
'''
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import json
import sys
import numpy as np
import torch
import h5py
import robosuite.utils.transform_utils as T
import robosuite.macros as macros
from pathlib import Path
import imageio
import cv2

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.lifelong.algos import Sequential
from libero.lifelong.metric import raw_obs_to_tensor_obs
from libero.lifelong.utils import torch_load_model, safe_device
from libero.lifelong.main import get_task_embs
import robomimic.utils.obs_utils as ObsUtils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
        default="/home/yikang/LIBERO/libero/experiments/LIBERO_90/Sequential"
                "/BCTransformerPolicy_seed10000/run_008/task0_model.pth")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--n-videos", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--out-dir", type=str,
        default="/home/yikang/stable-wm/libero_bc_rollouts")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── load checkpoint ──────────────────────────────────────────────────────
    sd, cfg, previous_mask = torch_load_model(args.ckpt, map_location=args.device) # Libero pth包含权重 训练完整配置 PackNet专用掩码
    cfg.device = args.device

    algo = safe_device(Sequential(10, cfg), cfg.device)
    algo.policy.previous_mask = previous_mask
    algo.policy.load_state_dict(sd)
    algo.eval()

    # initialise robomimic obs modality registry (normally done inside get_dataset)
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})

    # ── task setup ───────────────────────────────────────────────────────────
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    descriptions = [benchmark.get_task(i).language for i in range(10)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    task = benchmark.get_task(args.task_id)
    task_emb = benchmark.get_task_emb(args.task_id)
    task_name = benchmark.get_task_names()[args.task_id]

    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    init_states_path = os.path.join(
        get_libero_path("init_states"), task.problem_folder, task.init_states_file
    )
    init_states = torch.load(init_states_path, weights_only=False)

    # render at 224×224 for LeWM training; BC inference downscales to cfg.data.img_h
    RENDER_SIZE = 224
    BC_SIZE = cfg.data.img_h  # 128, what BC was trained on

    # ── environment ──────────────────────────────────────────────────────────
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=RENDER_SIZE,
        camera_widths=RENDER_SIZE,
    )

    # ── output paths ─────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = out_dir / f"{task_name}_bc.hdf5"
    video_dir = out_dir / f"{task_name}videos"
    video_dir.mkdir(exist_ok=True)

    env_args_meta = {
        "type": 1,
        "env_name": task_name,
        "problem_name": task.problem_folder,
        "bddl_file": bddl_file,
        "env_kwargs": {
            "bddl_file_name": bddl_file,
            "camera_heights": RENDER_SIZE,
            "camera_widths": RENDER_SIZE,
        },
    }
    problem_info = {
        "problem_name": task.problem_folder,
        "domain_name": "robosuite",
        "language_instruction": task.language,
    }

    # ── HDF5 file ────────────────────────────────────────────────────────────
    f = h5py.File(hdf5_path, "w")
    grp = f.create_group("data")
    grp.attrs["env_name"] = task_name
    grp.attrs["env_args"] = json.dumps(env_args_meta)
    grp.attrs["problem_info"] = json.dumps(problem_info)
    grp.attrs["bddl_file_name"] = bddl_file
    grp.attrs["bddl_file_content"] = open(bddl_file).read()
    grp.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION

    total_len = 0
    success_count = 0

    with torch.no_grad():
        for ep in range(args.n_episodes):
            algo.reset()

            # reset env and set init state
            env.reset()
            init_idx = ep % init_states.shape[0]
            obs = env.set_init_state(init_states[init_idx])

            # grab model xml right after reset (before warmup modifies physics)
            model_xml = env.sim.model.get_xml()

            # 5 zero-action warmup steps (stabilise physics, same as evaluate.py)
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7))

            # state before first BC action (used as init_state attr)
            init_state_flat = env.sim.get_state().flatten()

            # ── rollout ──────────────────────────────────────────────────────
            actions_buf      = []
            states_buf       = []
            robot_states_buf = []
            agentview_buf    = []
            eyeinhand_buf    = []
            gripper_buf      = []
            joint_buf        = []
            ee_buf           = []

            success = False

            for _ in range(args.max_steps):
                # BC was trained on BC_SIZE images; downscale before inference
                obs_bc = {**obs}
                for cam_key in (cfg.data.obs_key_mapping["agentview_rgb"],
                                cfg.data.obs_key_mapping["eye_in_hand_rgb"]):
                    obs_bc[cam_key] = cv2.resize(
                        obs[cam_key], (BC_SIZE, BC_SIZE), interpolation=cv2.INTER_AREA
                    )
                data = raw_obs_to_tensor_obs([obs_bc], task_emb, cfg)
                action = algo.policy.get_action(data)[0]  # (1,7) → (7,)

                next_obs, reward, done, info = env.step(action)

                # record current-step obs + resulting state (same convention as create_dataset.py)
                actions_buf.append(action)
                states_buf.append(env.sim.get_state().flatten())
                robot_states_buf.append(env.env.get_robot_state_vector(obs))
                agentview_buf.append(obs["agentview_image"])
                eyeinhand_buf.append(obs["robot0_eye_in_hand_image"])
                gripper_buf.append(obs["robot0_gripper_qpos"])
                joint_buf.append(obs["robot0_joint_pos"])
                ee_buf.append(np.concatenate([
                    obs["robot0_eef_pos"],
                    T.quat2axisangle(obs["robot0_eef_quat"]),
                ]))

                obs = next_obs
                if done:
                    success = True
                    success_count += 1
                    break

            T_ep = len(actions_buf)
            rewards_arr = np.zeros(T_ep, dtype=np.uint8)
            dones_arr   = np.zeros(T_ep, dtype=np.uint8)
            dones_arr[-1] = 1
            if success:
                rewards_arr[-1] = 1

            # ── write episode ─────────────────────────────────────────────────
            ep_grp = grp.create_group(f"demo_{ep}")
            ep_grp.attrs["init_state"]  = init_state_flat
            ep_grp.attrs["model_file"]  = model_xml
            ep_grp.attrs["num_samples"] = T_ep

            ep_grp.create_dataset("actions",      data=np.stack(actions_buf))
            ep_grp.create_dataset("states",       data=np.stack(states_buf))
            ep_grp.create_dataset("robot_states", data=np.stack(robot_states_buf))
            ep_grp.create_dataset("rewards",      data=rewards_arr)
            ep_grp.create_dataset("dones",        data=dones_arr)

            obs_grp = ep_grp.create_group("obs")
            obs_grp.create_dataset("agentview_rgb",    data=np.stack(agentview_buf))
            obs_grp.create_dataset("eye_in_hand_rgb",  data=np.stack(eyeinhand_buf))
            obs_grp.create_dataset("gripper_states",   data=np.stack(gripper_buf))
            obs_grp.create_dataset("joint_states",     data=np.stack(joint_buf))
            ee_arr = np.stack(ee_buf)
            obs_grp.create_dataset("ee_states", data=ee_arr)
            obs_grp.create_dataset("ee_pos",    data=ee_arr[:, :3])
            obs_grp.create_dataset("ee_ori",    data=ee_arr[:, 3:])

            total_len += T_ep

            # ── optional video ────────────────────────────────────────────────
            if ep < args.n_videos:
                tag = "success" if success else "fail"
                vpath = video_dir / f"ep{ep:04d}_{tag}.mp4"
                writer = imageio.get_writer(str(vpath), fps=20)
                for img in agentview_buf:
                    writer.append_data(img[::-1, ::-1])  # OpenGL origin bottom-left → display
                writer.close()

            print(f"ep {ep+1:4d}/{args.n_episodes} | steps={T_ep:3d} "
                  f"| {'SUCCESS' if success else 'fail   '} "
                  f"| success_rate={success_count/(ep+1):.2f}")

    grp.attrs["num_demos"] = args.n_episodes
    grp.attrs["total"]     = total_len
    f.close()
    env.close()

    print(f"\nSaved → {hdf5_path}")
    print(f"Success rate: {success_count/args.n_episodes:.2%}")


if __name__ == "__main__":
    main()
