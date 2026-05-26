# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LeWorldModel (LeWM)** is a research implementation of a Joint Embedding Predictive Architecture (JEPA) for learning world models from raw pixels. Key innovations:
- Two loss terms only: next-embedding prediction loss + Sketch Isotropic Gaussian Regularizer (SIGReg)
- ~15M trainable parameters, trainable on a single GPU in a few hours
- Used for planning in control tasks (PushT, DMControl Reacher, OGBench Cube, TwoRoom, OGBScene)

Paper: arXiv:2603.19312 — authors: Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero

## Server Layout (current machine, 2026-05-18)

| 用途 | 路径 | 说明 |
|------|------|------|
| 代码 | `/home/yikang/git/le-wm` | nvme 系统盘，git 管理 |
| venv | `/home/yikang/envs/lewm` | nvme 系统盘（hdd 无写权限） |
| 数据 / checkpoint | `/home/yikang/stable-wm` | nvme 系统盘，待管理员开通 hdd2 后迁移 |
| LIBERO 源码 | `/home/yikang/LIBERO` | nvme 系统盘 |

磁盘情况：nvme 3.4T（91% 满，剩 301G）；hdd1/hdd2 各 18T 但无写权限（需管理员 `chown`）。

环境变量在 `~/.bashrc` 中已配置，且 `source /home/yikang/envs/lewm/bin/activate` 已加入 `~/.bashrc` 自动激活，每次登录不需手动 source。

GPU：3× NVIDIA RTX PRO 6000 Blackwell（每张 ~96 GB），CUDA 13.1。

## Environment Setup

顺序敏感，不要乱调：

```bash
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# 2. 创建 venv（Python 3.10，放在大盘或 home）
uv venv --python=3.10 /home/yikang/envs/lewm
source /home/yikang/envs/lewm/bin/activate

# 3. 在 venv 里装 pip（uv venv 默认不含 pip）
uv pip install pip==23.3.2   # 必须 <24.1，gym 0.21 的 metadata 有残缺版本号，新版 pip 拒绝

# 4. gym==0.21 安装（顺序严格）
uv pip install "setuptools==65.7.0" "packaging==21.3" wheel
pip install "gym==0.21.0" --no-build-isolation   # 必须用 pip（不是 uv pip），uv pip 会拒绝残缺 metadata

# 5. 核心训练依赖
uv pip install "stable-pretraining==0.1.4"   # 新版删了 vit_hf
uv pip install "stable-worldmodel[train]"    # 不装 [env]，避免触发 gym 冲突

# 6. 训练常用包
pip install matplotlib datasets transformers lightning scikit-learn \
            zstandard huggingface_hub hf-xet

# 7. OGBScene-v0 依赖（新目标）
pip install ogbench dm_control
pip install pygame pymunk shapely
```

> **注意**：安装了 `gym==0.21.0` 之后，后续一律用 `pip`（venv 内的）而不是 `uv pip`。uv pip 每次解析依赖都会读已安装包的 metadata，遇到 gym 的残缺 metadata（`opencv-python>=3.`）就报错退出。

## Training

```bash
python train.py data=pusht
python train.py data=dmc
python train.py data=tworoom
python train.py data=ogb
python train.py data=libero_goal
```

Config entry point: `config/train/lewm.yaml`；dataset-specific overrides in `config/train/data/`。

`wandb entity` 在 yaml 里必须是带引号的字符串，例如 `entity: "934464091"`。

## Evaluation

```bash
# 标准 eval（PushT / Cube / Reacher / TwoRoom）
python eval.py --config-name=pusht.yaml policy=pusht/lewm
python eval.py --config-name=cube.yaml policy=cube/lewm
python eval.py --config-name=reacher.yaml policy=reacher/lewm
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm

# OGBScene drawer eval：两种模式
# 1) 子序列 eval（随机起点+offset，类似 cube 的做法）
MUJOCO_GL=egl EGL_DEVICE_ID=0 python ogb/ogb_eval.py policy=<ckpt 路径(不带_object.ckpt)>
# 2) 全任务 eval（每ep从step 0跑，目标=首次success帧，env terminated判定）
MUJOCO_GL=egl EGL_DEVICE_ID=0 python ogb/ogb_eval_full.py policy=<ckpt 路径>

# LIBERO eval
MUJOCO_GL=egl EGL_DEVICE_ID=0 python eval_libero.py \
    world.demo_hdf5_path=/home/yikang/stable-wm/libero_goal.h5 \
    world.task_id=0 eval.num_eval=10 policy=lewm

# LIBERO short-horizon eval
EGL_DEVICE_ID=0 python eval_short.py world.task_id=0 eval.goal_offset=50 eval.num_eval=5

# LIBERO single-task eval
EGL_DEVICE_ID=0 python single_task_eval.py eval.num_eval=1
```

## OGBScene-v0 使用方法

```python
import stable_worldmodel as swm

world = swm.World('swm/OGBScene-v0', num_envs=1, image_shape=(64, 64))
# Action space: Box(-1, 1, (1, 5))
# Observation space: Box(-inf, inf, (1, 40))
```

启动任何用到 mujoco / dm_control 渲染的脚本，都要加：
```bash
MUJOCO_GL=egl EGL_DEVICE_ID=0 python your_script.py
```

本机已同时有 `libEGL_mesa.so.0` 和 `libEGL_nvidia.so.0`，**不需要**手动替换系统 EGL 库（INSTALL.md §9.1 在本机可跳过）。

## Architecture

The codebase is small (~350 LOC across 5 core Python files):

| File | Role |
|------|------|
| `jepa.py` | Core `JEPA` class: `encode()`, `predict()`, `rollout()`, `get_cost()` |
| `module.py` | Building blocks: `SIGReg`, `ARPredictor`, `Embedder`, `MLP`, attention layers |
| `train.py` | Hydra-based training pipeline (Lightning Trainer) |
| `eval.py` | Planning evaluation with CEM solver (original envs) |
| `ogb/ogb_eval.py` | OGBScene 子序列 eval（随机起点 + 固定 goal_offset，patch set_state 接受 button_states 数组）|
| `ogb/ogb_eval_full.py` | OGBScene 全任务 eval（每ep从step 0跑，goal=该ep首次success帧，逐ep调用 evaluate_from_dataset）|
| `ogb/OGB_collect.py` | OGBScene 数据采集（drawer-only monkey patch + `terminate_at_goal=False`）|
| `ogb/filter_eval.py` | 从 collect 数据切出 eval 子集（裁列 + 重编号 ep_idx/step_idx）|
| `utils.py` | Callbacks, checkpoint helpers |
| `eval_libero.py` | LIBERO-specific eval: loads policy, runs World, records video |
| `eval_short.py` | Short-horizon eval: picks frame K of each demo as goal |
| `single_task_eval.py` | Single-task eval with deterministic per-demo init_state + frame-K goal |
| `envs/libero_env.py` | `LiberoGoalLEwMEnv`: preloads goal images from HDF5 |
| `utils/parquet2hdf5.py` | Converts LeRobot parquet → HDF5. **proprio = 8-D** |
| `utils/extract.py` | Decompress `.h5.zst` with zstandard stream_reader |
| `utils/h5check.py` | Inspect HDF5 structure |

### Data Flow

**Training:**
```
pixels (B,T,C,H,W) + actions (B,T,A)
  → encode() → emb (B,T,D) + act_emb (B,T,A_emb)
  → predict(ctx) → pred_emb
  → loss = MSE(pred_emb, target_emb) + λ·SIGReg(emb)
```

**Planning (evaluation):**
```
current pixels + goal pixels + action candidates
  → encode() → state_emb, goal_emb
  → rollout() → predicted embeddings (autoregressive)
  → get_cost() → MSE(pred_emb, goal_emb)
  → CEM solver minimizes cost over 300 candidates × 30 steps
```

### Key Design Details

- **Encoder**: ViT-tiny with patch size 14, input 224×224; CLS token used as the state embedding.
- **Predictor** (`ARPredictor`): 6-layer transformer, 16 heads, mlp_dim=2048; autoregressive over history.
- **history_size=3**: predictor sees the last 3 frames of context.
- **SIGReg**: enforces isotropic Gaussian latent distribution via random projections + Epps-Pulley statistic; weight λ=0.09 is the single tunable hyperparameter.
- **Checkpoint formats**: `_object.ckpt` (full serialized object) and `_weights.ckpt` (state dict only). Evaluation requires the `_object.ckpt` format.
- **Optimizer**: AdamW lr=5e-5, weight_decay=1e-3; linear warmup + cosine annealing.
- **Precision**: bf16 mixed precision.

## Install Pitfalls

- `gym==0.21` 必须用 `pip<24.1`（uv pip 和 pip≥24.1 都拒绝其残缺 metadata `opencv-python>=3.`）；安装前先装 `packaging==21.3` + `wheel`
- 安装 gym 后所有后续 pip 操作改用 venv 内的 `pip`，不用 `uv pip`（uv pip 每次都会触发 gym metadata 解析报错）
- `stable-pretraining` 必须 `==0.1.4`（新版删了 `vit_hf`）
- HuggingFace 如需国内访问：`export HF_ENDPOINT=https://hf-mirror.com`
- 解压 `.h5.zst` 用 `utils/extract.py` 的 stream_reader，避免 `/tmp` 撑爆
- Hydra `**dict` 展开前先 `pop("_target_", None)`
- OGBScene-v0 额外需要：`ogbench dm_control pygame pymunk shapely`

## LIBERO Eval Environment Setup

Eval stack: `eval_libero.py` → `envs/libero_env.py` → LiberoGymWrapper → LIBERO → robosuite → mujoco

### EGL Rendering（本机可跳过系统库替换）

本机已有 Mesa EGL，直接用环境变量即可：
```bash
MUJOCO_GL=egl EGL_DEVICE_ID=0 python ...
```

如果换到只有 NVIDIA EGL 的机器（`eglQueryString` 缺失），才需要替换系统库（见 INSTALL.md §9.1）。

### mujoco patch（每次重装 mujoco 都要重做）

```bash
sed -i 's/except ImportError:/except Exception:/g' \
    /home/yikang/envs/lewm/lib/python3.10/site-packages/mujoco/__init__.py
```

### Python dependencies

```bash
pip install numba bddl easydict gym imageio PyOpenGL
pip install "robosuite==1.4.0"
# robosuite 会覆盖 mujoco，重做上面的 patch
```

### Install LIBERO

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO /home/yikang/LIBERO
pip install -e /home/yikang/LIBERO
# pip editable 会生成 easy-install.pth，自动包含路径，不需要手动补 .pth
python -c "from libero.libero import benchmark; print('OK')"  # 第一次运行输入 N 跳过数据集路径
```

### Patches

```bash
# PyTorch 2.6+ weights_only fix
sed -i 's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/' \
    /home/yikang/LIBERO/libero/libero/benchmark/__init__.py

# LiberoGymWrapper import path fix
sed -i 's|from liberogymwrapper.libero_env import LiberoGoalEnv|from liberogymwrapper.v0 import LiberoGoalEnv|' \
    /home/yikang/git/le-wm/envs/libero_env.py

# numpy uint64 seed overflow — stable_worldmodel/world.py ~line 684
# 改为：
# seed_val = self.envs.envs[i].unwrapped.np_random_seed
# results['seeds'][ep_idx] = int(seed_val) % (2**31)
```

### Checkpoint symlink

```bash
ln -sf /home/yikang/stable-wm/outputs/lewm_libero_goal_epoch_97_object.ckpt \
       /home/yikang/stable-wm/lewm_object.ckpt
```

## Configuration System

Hydra is used for all configs:
```
config/
  train/
    lewm.yaml          # main training config
    data/              # dataset overrides (pusht, dmc, tworoom, ogb, libero_goal)
    launcher/local.yaml
  eval/
    pusht.yaml / cube.yaml / reacher.yaml / tworoom.yaml / libero_goal.yaml
    ogbscene_drawer.yaml         # ogb/ogb_eval.py 用（子序列 eval）
    ogbscene_drawer_full.yaml    # ogb/ogb_eval_full.py 用（全任务 eval, mode=data_collection）
    solver/cem.yaml
    launcher/local.yaml
```

WandB logging enabled by default; set `entity`/`project` in `lewm.yaml` (entity 必须是带引号的字符串).

## Research Goals

**Long-term:** Test LeWM on complex manipulation tasks, collect large-scale datasets, validate on real robot hardware.

**Current milestone (2026-05-18):** 目标已切换为使用 `swm/OGBScene-v0` 环境进行数据生成与训练。

## OGBScene Drawer Pipeline (2026-05-21)

### 数据采集 (`ogb/OGB_collect.py`)

```bash
MUJOCO_GL=egl EGL_DEVICE_ID=0 python ogb/OGB_collect.py \
    --episodes 100 --num-envs 8 \
    --dataset-name ogbench/scene_drawer_train
```

关键设计：
- **monkey-patch `SceneEnv.set_new_target`** → 强制 `_target_task='drawer'`，target 在 `-0.16(开)` 和 `0.0(关)` 之间翻转。这样所有 episode 都是 drawer 任务（不会混入 cube/button/window）。
- **`terminate_at_goal=False` + `max_episode_steps=500`** → 一个 episode 500 步，oracle 在内部 cycle ~10 次开关。
  - **为什么不能 `terminate_at_goal=True`**：`swm.World.record_dataset` 的执行顺序是 `step() → _dump_step_data() → check terminateds[i] → _reset_single_env(i)`。后两步把 `self.infos[i]` 重置为新 episode 的 step_idx=0，导致下一次 iteration 的 `get_action()` 看到的是 reset infos，oracle 永远没机会检测 `done=True` → `set_new_target()` 实际是死代码 → episode 在第一个 subtask 完成时就结束（平均 33 步而不是 500 步）。
- **数据列**：`pixels, qpos(25), qvel, button_states(2), action(5), observation(40)`，外加 `privileged_target_drawer_pos`, `privileged_drawer_pos`, `privileged_target_task`, `success` 等。

### Eval 数据准备 (`ogb/filter_eval.py`)

```bash
# 从训练数据切出 eval 子集（避免与训练重叠）
python ogb/filter_eval.py --src ogbench/scene_drawer_train \
                     --dst ogbench/scene_drawer_eval \
                     --offset 80 --n-episodes 20
```

只保留 9 列（pixels, qpos, qvel, button_states, privileged_target_drawer_pos, privileged_drawer_pos, action, observation, success），重新合成 `ep_idx`/`step_idx`（不能直接 copy 源里的，源的 episode 编号在新 dataset 里会越界）。

### 两种 Eval 模式

| | `ogb/ogb_eval.py`（子序列）| `ogb/ogb_eval_full.py`（全任务）|
|---|---|---|
| 起点 | 随机 step | 固定 step 0 |
| 目标 | `start + goal_offset_steps` | 该 ep 首次 `success=True` 的步 |
| 成功定义 | reach goal_emb (cost based) | env `terminated=True` |
| env mode | 默认 | `data_collection`（关键）|

### 关键：`mode='data_collection'` 在 eval 中的作用

`scene_env.py:1049-1062`：

```python
if self._mode == 'data_collection':
    self._success = drawer_success                # 只看 drawer
else:
    self._success = (
        all(cube_successes) and all(button_successes)
        and drawer_success and window_success     # 全部成功
    )
```

正常 eval 模式要求 cube/button/drawer/window **同时**满足。reset 后这些目标是随机设置的，几乎不可能匹配 → `terminated` 永远 False → 成功率永远 0。

→ `ogb/ogb_eval_full.py` 用 `mode='data_collection'` 让 success 只看 drawer。

### Eval 用法（drawer-only 全任务）

```bash
MUJOCO_GL=egl EGL_DEVICE_ID=0 python ogb/ogb_eval_full.py \
    eval.dataset_name=ogbench/scene_drawer_test \
    eval.num_eval=5 eval.eval_budget=100 \
    policy=/home/yikang/stable-wm/outputs/lewm_ogbscene_drawer_epoch_100
```

注意：`policy=` 路径**不要带** `_object.ckpt` 后缀，`AutoCostModel` 会自动加。

### Callable 链（`ogbscene_drawer_full.yaml`）

```yaml
callables:
  - method: set_state
    args:
      qpos:          {value: qpos}
      qvel:          {value: qvel}
      button_states: {value: button_states}        # 见下方 patch
  - method: set_target_drawer_pos
    args:
      target_pos:    {value: privileged_target_drawer_pos}
```

### Monkey-patch: `SceneEnv.set_state` 接受 `button_states` 数组

`scene_env.set_state(qpos, qvel, button_states)` 原本要求 `button_states` 拆成 `button_state_0`/`button_state_1` 两个标量 kwarg。但 Hydra callable 机制只能整列传 `(2,)` 数组。

`ogb/ogb_eval.py` / `ogb/ogb_eval_full.py` 顶层 patch：

```python
def set_state(self, qpos, qvel, button_states=None, **kwargs):
    if button_states is not None:
        arr = np.asarray(button_states).reshape(-1)
        kwargs["button_state_0"] = int(arr[0])
        kwargs["button_state_1"] = int(arr[1])
    orig_set_state(self, qpos, qvel, **kwargs)
```

### Eval budget 与 CEM 调用次数

```
eval_budget=N raw env steps
frame_skip=2, action_block=2  →  1 macro = 2 raw steps
receding_horizon=K macro      →  每 K macro 重 plan 一次
→ CEM 调用次数 = N / (K * 2)
```

例：`eval_budget=40, receding_horizon=5, action_block=2` → 4 次 CEM。每次 CEM (300 candidates × 30 iter) ~0.5s → 总 ~2s 一个 episode。这是正常速度。Drawer 子任务 oracle 用 ~15 raw steps 完成，policy 给 40-120 budget 比较合理。

## Training proprio layout (8-D, non-obvious)

`libero_goal.h5["proprio"]` comes from LeRobot `observation.state`:
```
[ eef_pos(3) | eef_axis_angle(3) | gripper_qpos(2) ]
```
At eval time:
```python
from robosuite.utils.transform_utils import quat2axisangle
proprio = concat(obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
```

LIBERO renders agentview in OpenGL convention (origin bottom-left); training pixels are top-left. Flip vertically (`img[::-1]`) in any new env using raw `OffScreenRenderEnv`.
