"""
Local stub for liberogymwrapper.v0.

Wraps LIBERO's OffScreenRenderEnv into a gymnasium-compatible interface.
obs["agentview_image"] is returned as (C, H, W) uint8 to match the
convention expected by LiberoGoalLEwMEnv.render().
"""
import os
import numpy as np
import torch
import gymnasium

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

WARMUP_STEPS = 5


class LiberoGoalEnv(gymnasium.Env):
    """
    Gymnasium wrapper around LIBERO OffScreenRenderEnv.

    Parameters
    ----------
    task_id : int
        Task index within the benchmark.
    benchmark_name : str
        LIBERO benchmark name, e.g. "LIBERO_Goal" or "LIBERO_90".
    task_order_index : int
        Task order index for the benchmark (usually 0).
    image_size : int
        Square image resolution for rendering.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task_id: int = 0,
        benchmark_name: str = "LIBERO_Goal",
        task_order_index: int = 0,
        image_size: int = 224,
        **kwargs,
    ):
        super().__init__()

        benchmark = get_benchmark(benchmark_name)(task_order_index)
        task = benchmark.get_task(task_id)

        bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        init_states_path = os.path.join(
            get_libero_path("init_states"), task.problem_folder, task.init_states_file
        )
        self._init_states = torch.load(init_states_path, weights_only=False)
        self._image_size = image_size
        self._last_obs = None

        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=image_size,
            camera_widths=image_size,
        )

        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.observation_space = gymnasium.spaces.Dict(
            {
                "agentview_image": gymnasium.spaces.Box(
                    low=0, high=255,
                    shape=(3, image_size, image_size),
                    dtype=np.uint8,
                )
            }
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def get_observations(self) -> dict:
        """Return obs dict with agentview_image in (C, H, W) uint8."""
        if self._last_obs is None:
            return {
                "agentview_image": np.zeros(
                    (3, self._image_size, self._image_size), dtype=np.uint8
                )
            }
        img = self._last_obs["agentview_image"]  # (H, W, C) from OffScreenRenderEnv
        return {"agentview_image": img.transpose(2, 0, 1)}  # → (C, H, W)

    # ── gymnasium interface ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self._env.reset()
        idx = int(seed or 0) % len(self._init_states)
        obs = self._env.set_init_state(self._init_states[idx])
        for _ in range(WARMUP_STEPS):
            obs, _, _, _ = self._env.step(np.zeros(7))
        self._last_obs = obs
        return self.get_observations(), {}

    def step(self, action):
        obs, reward, done, info = self._env.step(np.asarray(action, dtype=np.float64))
        self._last_obs = obs
        terminated = bool(done)   # LIBERO sets done=True on task success
        truncated = False          # timeout controlled externally via max_episode_steps
        return self.get_observations(), float(reward), terminated, truncated, info

    def render(self):
        """Return (H, W, 3) uint8 — called by swm.World's MegaWrapper."""
        if self._last_obs is None:
            return np.zeros((self._image_size, self._image_size, 3), dtype=np.uint8)
        return self._last_obs["agentview_image"]  # (H, W, C)

    def close(self):
        self._env.close()
