import argparse
import os

import mujoco
import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench.expert_policy import ExpertPolicy
from stable_worldmodel.envs.ogbench.scene_env import SceneEnv


def _patch_scene_env_drawer_only():
    """强制 OGBScene 在 data_collection mode 下永远把 target_task 锁在 'drawer'。

    替换原版 set_new_target 的 4-task 概率采样:
      - target_task 固定 = 'drawer'
      - target_drawer_pos 在 -0.16(开) 和 0.0(关) 之间翻转
      - oracle 完成一次就接下一次反向,episode 内形成开关循环 → 100% drawer 数据

    多进程 (num_envs > 1) 必须验证 worker 子进程也被 patch 了。
    验证方法: 在 set_new_target 内加 print(os.getpid()),
    应该看到 num_envs 个不同 PID。
    """
    if getattr(SceneEnv.set_new_target, "_drawer_only_patched", False):
        return

    def set_new_target(self, return_info=True, p_stack=0.5):
        assert self._mode == 'data_collection'
        self._target_task = 'drawer'
        cur = self._data.joint('drawer_slide').qpos[0]
        self._target_drawer_pos = -0.16 if cur >= -0.08 else 0.0
        self._model.site('drawer_handle_center_target').pos[1] = self._target_drawer_pos
        mujoco.mj_kinematics(self._model, self._data)
        if return_info:
            return self.compute_observation(), self.get_reset_info()

    set_new_target._drawer_only_patched = True
    SceneEnv.set_new_target = set_new_target


_patch_scene_env_drawer_only()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',     type=int,   default=500)
    parser.add_argument('--num-envs',     type=int,   default=8)
    parser.add_argument('--image-size',   type=int,   default=64)
    parser.add_argument('--seed',         type=int,   default=0)
    parser.add_argument('--dataset-name', type=str,   default='ogbench/scene_expert')
    parser.add_argument('--policy-type',  type=str,   default='markov_oracle',
                        choices=['markov_oracle', 'plan_oracle'])
    parser.add_argument('--action-noise', type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()

    # 打印任务概要，方便确认参数对不对
    print(f"Collecting {args.episodes} episodes with {args.num_envs} envs...")
    print(f"Dataset: {os.environ.get('STABLEWM_HOME', '~/.stable_worldmodel')}/{args.dataset_name}.h5")

    world = swm.World(
        'swm/OGBScene-v0',
        num_envs=args.num_envs,
        image_shape=(args.image_size, args.image_size),
        mode='data_collection',
        max_episode_steps=500,    # 给 oracle 足够 budget chain 多次开关循环 (~25 次)
        terminate_at_goal=False,  # 单次开关成功后不要终止 episode
    )

    policy = ExpertPolicy(
        policy_type=args.policy_type,
        action_noise=args.action_noise,
        seed=args.seed,
    )
    world.set_policy(policy)

    world.record_dataset(
        dataset_name=args.dataset_name,
        episodes=args.episodes,
        seed=args.seed,
    )

    print("Done.")


if __name__ == '__main__':
    main()

