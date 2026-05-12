import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import h5py
import numpy as np
import torch
from PIL import Image

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.lifelong.algos import get_algo_class
from libero.lifelong.main import get_task_embs
from libero.lifelong.metric import raw_obs_to_tensor_obs
from libero.lifelong.utils import control_seed, safe_device


BENCHMARK_ALIASES = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
    "libero_90": "LIBERO_90",
}


def load_policy(checkpoint_path, device):
    """Load a LIBERO lifelong policy checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    cfg.device = device
    cfg.folder = cfg.get("folder") or get_libero_path("datasets")
    cfg.bddl_folder = cfg.get("bddl_folder") or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.get("init_states_folder") or get_libero_path(
        "init_states"
    )
    if not hasattr(cfg.data, "task_order_index"):
        cfg.data.task_order_index = 0

    algo = safe_device(get_algo_class(cfg.lifelong.algo)(10, cfg), cfg.device)
    algo.policy.load_state_dict(ckpt["state_dict"])
    algo.eval()
    algo.reset()
    return algo, cfg


def resolve_task_id(benchmark, task_id, task_name):
    if task_name is None:
        return task_id

    matches = [
        i
        for i, task in enumerate(benchmark.tasks)
        if task.name == task_name
        or task.bddl_file == task_name
        or task.bddl_file == f"{task_name}.bddl"
    ]
    if not matches:
        names = "\n".join(task.name for task in benchmark.tasks)
        raise ValueError(
            f"task name {task_name!r} not found in {benchmark.name}. "
            f"Available tasks:\n{names}"
        )
    if len(matches) > 1:
        raise ValueError(f"task name {task_name!r} matched multiple tasks: {matches}")
    return matches[0]


def prepare_task(cfg, benchmark_name, task_id, task_name=None):
    """Create benchmark metadata, task embedding, and the offscreen env."""
    benchmark_name = BENCHMARK_ALIASES.get(benchmark_name, benchmark_name)
    benchmark = get_benchmark(benchmark_name)(cfg.data.task_order_index)
    task_id = resolve_task_id(benchmark, task_id, task_name)

    descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    task = benchmark.get_task(task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
        camera_heights=cfg.data.img_h,
        camera_widths=cfg.data.img_w,
    )

    init_states_path = os.path.join(
        cfg.init_states_folder, task.problem_folder, task.init_states_file
    )
    init_states = torch.load(init_states_path, weights_only=False)
    task_info = {
        "benchmark": benchmark.name,
        "task_id": task_id,
        "task_language": task.language,
        "bddl_file": os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
        "init_states_file": init_states_path,
        "policy_camera_height": int(cfg.data.img_h),
        "policy_camera_width": int(cfg.data.img_w),
    }
    return benchmark, task, task_id, task_embs[task_id], env, init_states, task_info


def image_to_hwc_uint8(image):
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"expected image with 3 dims, got shape {image.shape}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = image.transpose(1, 2, 0)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def resize_hwc_uint8(image, size):
    if size is None:
        return image
    height, width = image.shape[:2]
    if height == size and width == size:
        return image
    return np.asarray(Image.fromarray(image).resize((size, size), Image.BILINEAR))


def proprio_from_obs(obs):
    parts = []
    for key in ("robot0_gripper_qpos", "robot0_joint_pos"):
        if key in obs:
            parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def rollout_episode(algo, cfg, env, init_state, task_emb, max_steps, save_img_size):
    """Collect one rollout. Failed and successful episodes are both kept."""
    env.reset()
    obs = env.set_init_state(init_state)
    algo.reset()

    zero_action = np.zeros(7, dtype=np.float32)
    for _ in range(5):
        obs, _, _, _ = env.step(zero_action)

    pixels = []
    actions = []
    proprios = []
    rewards = []
    dones = []
    success = False

    with torch.no_grad():
        for _ in range(max_steps):
            image = image_to_hwc_uint8(obs["agentview_image"])
            pixels.append(resize_hwc_uint8(image, save_img_size))
            proprios.append(proprio_from_obs(obs))

            data = raw_obs_to_tensor_obs([obs], task_emb, cfg)
            action = algo.policy.get_action(data)[0]
            action = np.asarray(action, dtype=np.float32)

            obs, reward, done, info = env.step(action)
            actions.append(action)
            rewards.append(float(reward))
            dones.append(bool(done))
            success = success or bool(done)

            if done:
                image = image_to_hwc_uint8(obs["agentview_image"])
                pixels.append(resize_hwc_uint8(image, save_img_size))
                proprios.append(proprio_from_obs(obs))
                break

    if len(pixels) == len(actions) + 1:
        actions.append(np.zeros_like(actions[-1]))
        rewards.append(0.0)
        dones.append(True)

    return {
        "pixels": np.stack(pixels).astype(np.uint8),
        "action": np.stack(actions).astype(np.float32),
        "proprio": np.stack(proprios).astype(np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
        "done": np.asarray(dones, dtype=np.bool_),
        "success": success,
    }


def create_or_append_dataset(group, name, array, maxshape):
    if name not in group:
        group.create_dataset(
            name,
            data=array,
            maxshape=maxshape,
            chunks=True,
            compression="gzip" if name == "pixels" else None,
        )
        return

    dataset = group[name]
    old_len = dataset.shape[0]
    dataset.resize((old_len + array.shape[0], *dataset.shape[1:]))
    dataset[old_len:] = array


def write_attrs_once(f, attrs):
    for key, value in attrs.items():
        if key not in f.attrs:
            f.attrs[key] = value


def append_episode(h5_path, episode, task_id, init_state_id, init_state, seed, attrs):
    """Append an episode using the flat LeWM HDF5 layout."""
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    length = episode["pixels"].shape[0]
    for key in ("action", "proprio", "reward", "done"):
        if episode[key].shape[0] != length:
            raise ValueError(
                f"{key} length {episode[key].shape[0]} does not match pixels {length}"
            )

    with h5py.File(h5_path, "a") as f:
        write_attrs_once(f, attrs)
        offset = int(f["ep_len"][:].sum()) if "ep_len" in f else 0
        create_or_append_dataset(
            f,
            "pixels",
            episode["pixels"],
            (None, *episode["pixels"].shape[1:]),
        )
        create_or_append_dataset(
            f,
            "action",
            episode["action"],
            (None, episode["action"].shape[1]),
        )
        create_or_append_dataset(
            f,
            "proprio",
            episode["proprio"],
            (None, episode["proprio"].shape[1]),
        )
        create_or_append_dataset(f, "reward", episode["reward"], (None,))
        create_or_append_dataset(f, "done", episode["done"], (None,))

        meta = {
            "ep_len": np.asarray([length], dtype=np.int32),
            "ep_offset": np.asarray([offset], dtype=np.int64),
            "task_index": np.asarray([task_id], dtype=np.int32),
            "init_state_id": np.asarray([init_state_id], dtype=np.int32),
            "seed": np.asarray([seed], dtype=np.int32),
            "success": np.asarray([episode["success"]], dtype=np.bool_),
            "mujoco_init_state": np.asarray(init_state, dtype=np.float32).reshape(1, -1),
        }
        for name, value in meta.items():
            create_or_append_dataset(f, name, value, (None, *value.shape[1:]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect LIBERO rollouts into LeWM HDF5 format."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to LIBERO .pth policy")
    parser.add_argument("--out", required=True, help="Output HDF5 path")
    parser.add_argument("--benchmark", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--save-img-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    control_seed(args.seed)

    algo, cfg = load_policy(args.checkpoint, args.device)
    benchmark, task, task_id, task_emb, env, init_states, task_info = prepare_task(
        cfg, args.benchmark, args.task_id, args.task_name
    )
    env.seed(args.seed)
    task_info.update(
        {
            "checkpoint": str(Path(args.checkpoint)),
            "max_steps": args.max_steps,
            "save_img_size": args.save_img_size,
        }
    )

    print(
        json.dumps(
            {
                "benchmark": benchmark.name,
                "task_id": task_id,
                "task": task.language,
                "episodes": args.episodes,
                "out": args.out,
                "save_img_size": args.save_img_size,
            },
            indent=2,
        )
    )

    try:
        for ep_idx in range(args.episodes):
            init_state_id = ep_idx % len(init_states)
            episode = rollout_episode(
                algo=algo,
                cfg=cfg,
                env=env,
                init_state=init_states[init_state_id],
                task_emb=task_emb,
                max_steps=args.max_steps,
                save_img_size=args.save_img_size,
            )
            append_episode(
                h5_path=args.out,
                episode=episode,
                task_id=task_id,
                init_state_id=init_state_id,
                init_state=init_states[init_state_id],
                seed=args.seed,
                attrs=task_info,
            )
            print(
                f"[{ep_idx + 1}/{args.episodes}] "
                f"frames={episode['pixels'].shape[0]} success={episode['success']}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
