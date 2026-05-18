import json
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import gymnasium
import h5py
import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from gymnasium import spaces
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


PROPRIO_DIM = 9  # concat(robot0_gripper_qpos[2], robot0_joint_pos[7]) — matches collect_libero


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _as_hwc_uint8(image):
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"expected image with 3 dims, got {image.shape}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = image.transpose(1, 2, 0)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _demo_sort_key(name: str):
    if name.startswith("demo_"):
        return int(name.split("_")[-1])
    return name


def _proprio_from_libero_obs(obs):
    """Must match collect_libero.proprio_from_obs so eval uses training-format 9-D proprio."""
    return np.concatenate(
        [
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
            np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32)


def infer_bddl_file(hdf5_path: str, override: str | None):
    if override:
        return override

    with h5py.File(hdf5_path, "r") as f:
        attrs = f["data"].attrs
        for key in ("bddl_file_name", "bddl_file"):
            if key in attrs:
                candidate = _decode_attr(attrs[key])
                if os.path.exists(candidate):
                    return candidate
                basename = os.path.basename(candidate)
                matches = list(Path(get_libero_path("bddl_files")).rglob(basename))
                if matches:
                    return str(matches[0])

        if "env_args" in attrs:
            env_args = json.loads(_decode_attr(attrs["env_args"]))
            candidate = env_args.get("bddl_file") or env_args.get("bddl_file_name")
            if candidate and os.path.exists(candidate):
                return candidate
            if candidate:
                basename = os.path.basename(candidate)
                matches = list(Path(get_libero_path("bddl_files")).rglob(basename))
                if matches:
                    return str(matches[0])

    raise ValueError(
        "Could not infer bddl_file_name from the HDF5 attrs. "
        "Pass world.bddl_file_name=/absolute/path/to/task.bddl"
    )


class SingleTaskDemoEnv(gymnasium.Env):
    """LIBERO env whose reset is tied to one demo's MuJoCo state and goal frame."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        demo_hdf5_path: str,
        bddl_file_name: str | None = None,
        camera_heights: int = 128,
        camera_widths: int = 128,
        goal_offset: int = 100,
        **kwargs,
    ):
        super().__init__()
        self.demo_hdf5_path = demo_hdf5_path
        self.bddl_file_name = infer_bddl_file(demo_hdf5_path, bddl_file_name)
        self.camera_heights = int(camera_heights)
        self.camera_widths = int(camera_widths)
        self.goal_offset = goal_offset
        self._eval_demo_id = 0
        self._goal_image = None
        self._last_obs = None

        with h5py.File(self.demo_hdf5_path, "r") as f:
            self.demo_names = sorted(f["data"].keys(), key=_demo_sort_key)

        self.env = OffScreenRenderEnv(
            bddl_file_name=self.bddl_file_name,
            camera_heights=self.camera_heights,
            camera_widths=self.camera_widths,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "agentview_image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.camera_heights, self.camera_widths, 3),
                    dtype=np.uint8,
                ),
                "proprio": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(PROPRIO_DIM,),
                    dtype=np.float32,
                ),
            }
        )

    def set_eval_demo(self, demo_id: int, goal_offset: int | None = None):
        self._eval_demo_id = int(demo_id)
        if goal_offset is not None:
            self.goal_offset = int(goal_offset)

    def _load_demo_reset_data(self, demo_id: int):
        demo_name = self.demo_names[demo_id]
        with h5py.File(self.demo_hdf5_path, "r") as f:
            demo = f["data"][demo_name]
            states = demo["states"]
            init_state = states[0].astype(np.float64)
            goal_idx = min(self.goal_offset, states.shape[0] - 1)
            goal = _as_hwc_uint8(demo["obs"]["agentview_rgb"][goal_idx])
            init_pixels = _as_hwc_uint8(demo["obs"]["agentview_rgb"][0])
        return demo_name, init_state, goal_idx, goal, init_pixels

    def _pack_obs(self, raw_obs):
        return {
            "agentview_image": _as_hwc_uint8(raw_obs["agentview_image"]),
            "proprio": _proprio_from_libero_obs(raw_obs),
        }

    def _info(self, demo_name, demo_id, goal_idx, init_state):
        return {
            "goal": self._goal_image,
            "init_state_id": int(demo_id),
            "demo_name": demo_name,
            "goal_frame_id": int(goal_idx),
            "mujoco_init_state": init_state,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        demo_id = self._eval_demo_id
        demo_name, init_state, goal_idx, goal, _ = self._load_demo_reset_data(demo_id)

        self.env.reset()
        raw_obs = self.env.set_init_state(init_state)
        self._last_obs = self._pack_obs(raw_obs)
        self._goal_image = goal
        return self._last_obs, self._info(demo_name, demo_id, goal_idx, init_state)

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        self._last_obs = self._pack_obs(raw_obs)
        info = dict(info)
        info["goal"] = self._goal_image
        info["init_state_id"] = int(self._eval_demo_id)
        return self._last_obs, reward, bool(done), False, info

    def render(self):
        if self._last_obs is None:
            return np.zeros((self.camera_heights, self.camera_widths, 3), dtype=np.uint8)
        return self._last_obs["agentview_image"]

    def close(self):
        self.env.close()


gymnasium.register(
    id="single-task-libero-demo-v0",
    entry_point=SingleTaskDemoEnv,
    max_episode_steps=600,
)


def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def get_normalization(cfg):
    """Compute action/proprio μ/σ from the training HDF5 — must match training-time stats."""
    dataset_path = Path(cfg.dataset.train_hdf5_path).parent
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    process = {}
    for col in cfg.dataset.keys_to_cache:
        scaler = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        scaler.fit(col_data)
        process[col] = scaler
    return process


def get_single_env(world):
    envs = world.envs.env.envs
    if len(envs) != 1:
        raise ValueError("single_task_eval.py expects world.num_envs=1")
    return envs[0].unwrapped


@hydra.main(version_base=None, config_path="./config/eval", config_name="single_task")
def run(cfg: DictConfig):
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }
    process = get_normalization(cfg)

    cfg.world.max_episode_steps = cfg.eval.max_steps
    world = swm.World(**cfg.world, image_shape=(224, 224))

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

    env = get_single_env(world)
    demo_count = len(env.demo_names)
    demo_ids = list(cfg.eval.demo_ids) if cfg.eval.demo_ids else list(range(demo_count))
    demo_ids = demo_ids[: cfg.eval.num_eval]

    out_dir = Path(__file__).parent / "videos" / cfg.eval.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    summary = []
    for i, demo_id in enumerate(demo_ids):
        env.set_eval_demo(demo_id, cfg.eval.goal_offset)
        demo_name, init_state, goal_idx, goal, init_pixels = env._load_demo_reset_data(demo_id)

        # Save reference frames to a sibling dir so record_video can't clobber them.
        Image.fromarray(init_pixels).save(frames_dir / f"ep_{i:03d}_{demo_name}_initial.png")
        Image.fromarray(goal).save(frames_dir / f"ep_{i:03d}_{demo_name}_goal_{goal_idx:03d}.png")

        ep_dir = out_dir / f"ep_{i:03d}_{demo_name}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        world.record_video(
            video_path=ep_dir,
            max_steps=cfg.eval.max_steps,
            fps=30,
            viewname="pixels",
            seed=cfg.seed + i,
        )
        summary.append(
            {
                "eval_index": i,
                "demo_id": int(demo_id),
                "demo_name": demo_name,
                "goal_frame_id": int(goal_idx),
                "init_state_norm": float(np.linalg.norm(init_state)),
            }
        )
        print(f"Episode {i}: {demo_name}, init=states[0], goal=frame {goal_idx}")

    with (out_dir / "summary.json").open("w") as f:
        json.dump(
            {"config": OmegaConf.to_container(cfg, resolve=True), "episodes": summary},
            f,
            indent=2,
        )
    print(f"\nSaved videos and reference frames to: {out_dir}")


if __name__ == "__main__":
    run()
