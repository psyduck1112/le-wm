import argparse
import os

import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench.expert_policy import ExpertPolicy


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
