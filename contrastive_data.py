"""对比学习用的数据集: HDF5Dataset + 配对采样 (Step 1).

在原 4 帧窗口 (anchor) 之外, 每个样本额外返回同一条 episode 的两帧:
  - 近帧  (|Δt| ≤ w_near):  z_task 正样本 / z_ctx 正样本
  - 远帧  (|Δt| ≥ w_far):   z_task 负样本

这是对比学习正负样本的来源 (见 project-contrastive-latent-split):
  * 同 ep 近帧  -> 任务进度相近   -> L_task 拉近
  * 同 ep 远帧  -> 任务进度不同   -> L_task 推开
  * 跨 ep (batch 里别的样本) -> 场景不同 -> L_ctx 推开 (白嫖, 不在这里取)

设计要点:
  * anchor / 近帧 / 远帧都走父类同一个 `_load_slice` -> 同一个 transform,
    图像预处理 (resize/normalize) 完全一致, 无需额外对齐代码.
  * `pair_sampling=False` (默认) 时完全退化为父类行为, 基线/物理头训练不受影响.
  * episode 短 (本数据 median 71 raw 帧) 时可能取不到合格远帧, 标 `has_far=False`,
    由 loss 屏蔽该样本的 L_task.
  * `ep_idx` (全局唯一) 仅用于 L_ctx 剔除同 batch 里碰巧同 ep 的假负样本, 不进任何 loss 当目标.
"""

import random

from stable_worldmodel.data.dataset import HDF5Dataset


class ContrastiveHDF5Dataset(HDF5Dataset):
    # 配对采样开关与超参. train.py 在需要时设置; 默认关闭 = 纯父类行为.
    pair_sampling: bool = False
    w_near: int = 5      # 近帧最大时间间隔 (raw 帧). ≈1 macro, "同进度"
    w_far: int = 20      # 远帧最小时间间隔 (raw 帧). ≈4 macro ≈ 30% episode, "不同进度"
    ep_base: int = 0     # ConcatDataset 多成员时的全局 ep 偏移, 保证 ep_idx 跨成员唯一

    def _load_one_frame(self, ep: int, k: int) -> dict:
        """取 ep 内原始第 k 帧 (单帧). 复用父类 _load_slice (含 transform)."""
        return self._load_slice(ep, k, k + 1)

    def _sample_near(self, t_a: int, L: int, rng: random.Random) -> int:
        lo, hi = max(0, t_a - self.w_near), min(L - 1, t_a + self.w_near)
        choices = [t for t in range(lo, hi + 1) if t != t_a]
        return rng.choice(choices) if choices else t_a

    def _sample_far(self, t_a: int, L: int, rng: random.Random) -> tuple[int, bool]:
        # 左段: 帧 ≤ t_a - w_far (|Δt| ≥ w_far); 右段: 帧 ≥ t_a + w_far.
        left = list(range(0, t_a - self.w_far + 1)) if t_a - self.w_far >= 0 else []
        right = list(range(t_a + self.w_far, L)) if t_a + self.w_far <= L - 1 else []
        cands = left + right
        if cands:
            return rng.choice(cands), True
        # ep 太短: 退而取最远的一端, 标 has_far=False (loss 屏蔽其 L_task).
        t_far = 0 if (t_a > L - 1 - t_a) else (L - 1)
        return t_far, False

    def __getitem__(self, idx: int) -> dict:
        out = super().__getitem__(idx)
        if not self.pair_sampling:
            return out

        ep, start = self.clip_indices[idx]
        L = int(self.lengths[ep])
        # anchor 窗口最后一帧的原始时刻 (用它的 emb 当 anchor).
        t_a = start + (self.num_steps - 1) * self.frameskip

        # 用 Python random: torch DataLoader 已为各 worker 设过 random 种子.
        rng = random.Random()
        t_near = self._sample_near(t_a, L, rng)
        t_far, has_far = self._sample_far(t_a, L, rng)

        near = self._load_one_frame(ep, t_near)
        far = self._load_one_frame(ep, t_far)
        out["pos_pixels"] = near["pixels"][0]
        out["pos_eye"] = near["eye_in_hand"][0]
        out["neg_pixels"] = far["pixels"][0]
        out["neg_eye"] = far["eye_in_hand"][0]
        out["ep_idx"] = self.ep_base + int(ep)
        out["has_far"] = has_far
        # 调试/可选加权用 (signed 间隔). 训练 step 可忽略.
        out["pos_dt"] = int(t_near - t_a)
        out["neg_dt"] = int(t_far - t_a)
        return out
