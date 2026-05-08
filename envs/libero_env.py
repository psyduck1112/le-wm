import numpy as np
import gymnasium
import h5py
from liberogymwrapper.v0 import LiberoGoalEnv as _LiberoGoalEnv


class LiberoGoalLEwMEnv(_LiberoGoalEnv):
    """LiberoGoal wrapped for LeWM eval.

    Adds:
      - render() -> (H,W,3) uint8 for AddPixelsWrapper
      - info["goal"] on reset/step for WorldModelPolicy
    """

    def __init__(self, demo_hdf5_path: str, **kwargs):
        self._demo_hdf5_path = demo_hdf5_path
        self._goal_image = None   # set on reset()
        self._goal_images = self._preload_goal_images()
        super().__init__(**kwargs)

    def _preload_goal_images(self) -> list[np.ndarray]:
        """Load last frame of each episode from training HDF5 as goal images."""
        goals = []
        with h5py.File(self._demo_hdf5_path, "r") as f:
            ep_offsets = f["ep_offset"][:]
            ep_lengths = f["ep_len"][:]
            pixels = f["pixels"]
            for offset, length in zip(ep_offsets, ep_lengths):
                last_frame = pixels[offset + length - 1]  # (H,W,3) uint8
                goals.append(last_frame)
        return goals

    def render(self):
        """Return (H,W,3) uint8 image for AddPixelsWrapper."""
        obs = self.get_observations()
        img = obs["agentview_image"]  # (C,H,W) from LiberoGymWrapper
        return img.transpose(1, 2, 0)  # → (H,W,3)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # pick a random goal image from preloaded demos
        idx = np.random.randint(len(self._goal_images))
        self._goal_image = self._goal_images[idx]  # (H,W,3) uint8
        info["goal"] = self._goal_image
        info["init_state_id"] = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["goal"] = self._goal_image
        info["init_state_id"] = 0  # keep same goal throughout episode
        return obs, reward, terminated, truncated, info


gymnasium.register(
    id="libero-goal-lewm-v0",
    entry_point="envs.libero_env:LiberoGoalLEwMEnv",
    max_episode_steps=600,
)
