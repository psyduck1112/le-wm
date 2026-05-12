---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).


<a class="flex" href="https://huggingface.co/spaces/lerobot/visualize_dataset?path=lerobot/libero_goal_image">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl.svg"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl-dark.svg"/>
</a>


## Dataset Description



- **Homepage:** [More Information Needed]
- **Paper:** [More Information Needed]
- **License:** apache-2.0

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{
    "codebase_version": "v3.0",
    "robot_type": "panda",
    "total_episodes": 428,
    "total_frames": 52042,
    "total_tasks": 10,
    "chunks_size": 1000,
    "fps": 10,
    "splits": {
        "train": "0:428"
    },
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "features": {
        "observation.images.image": {
            "dtype": "image",
            "shape": [
                256,
                256,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
            ],
            "fps": 10
        },
        "observation.images.wrist_image": {
            "dtype": "image",
            "shape": [
                256,
                256,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
            ],
            "fps": 10
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                8
            ],
            "names": {
                "motors": [
                    "x",
                    "y",
                    "z",
                    "rx",
                    "ry",
                    "rz",
                    "rw",
                    "gripper"
                ]
            },
            "fps": 10
        },
        "action": {
            "dtype": "float32",
            "shape": [
                7
            ],
            "names": {
                "motors": [
                    "x",
                    "y",
                    "z",
                    "roll",
                    "pitch",
                    "yaw",
                    "gripper"
                ]
            },
            "fps": 10
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null,
            "fps": 10
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null,
            "fps": 10
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null,
            "fps": 10
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null,
            "fps": 10
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null,
            "fps": 10
        }
    },
    "data_files_size_in_mb": 100,
    "video_files_size_in_mb": 200
}
```


## Citation

**BibTeX:**

```bibtex
[More Information Needed]
```