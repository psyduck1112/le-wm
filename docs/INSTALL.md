# LeWM 服务器安装文档

适用环境：Ubuntu / Debian Linux 服务器（含 NVIDIA GPU），用于训练 LeWM 模型 + 在 LIBERO 中做 eval。
本文档基于 AutoDL 上踩过的所有坑整理，按顺序执行可一次走通。

---

## 0. 路径规划：**程序 / 环境 / 数据 三处分离**

把 **代码**、**Python 环境（venv）**、**数据集 + checkpoint** 放在**三个独立目录**里，互不嵌套。这样做的理由：

| 类别 | 体积 | 是否需要 git | 是否可重建 | 适合放在 |
|------|------|--------------|------------|----------|
| **程序代码** | 小（< 100 MB） | 是 | 是（重新 clone） | 系统盘 / home |
| **Python 环境（venv）** | 中（5–10 GB） | 否，**绝不能进 git** | 是（重装一次） | 大盘 envs/ |
| **数据 + checkpoint** | 大（数十 GB ~ TB） | 否，**绝不能进 git** | 否（重新下载/重新训练代价高） | 大盘 data/ |

三处分离的好处：
- **代码可以频繁 `git clean` / 重新 clone** 而不会动到环境和数据
- **环境出问题可以整体删掉重建**，不影响数据
- **数据持久化**，跨项目 / 跨 venv 复用，不会被误删
- **备份策略不同**：代码靠 git，环境靠 `uv pip freeze`，数据靠快照/对象存储

### 物理布局示意

```
$HOME/                            # 系统盘，小
└── le-wm/                        # ← $REPO_DIR：代码（git 管理）

/data/                            # 大盘（df -h 找剩余最多的挂载点）
├── envs/                         # ← $ENV_HOME：所有项目的 venv 放这里
│   └── lewm/                     #     本项目的 venv
├── stable-wm/                    # ← $STABLEWM_HOME：数据集 + checkpoint
│   ├── libero_goal.h5
│   └── outputs/
├── LIBERO/                       # ← $LIBERO_DIR：第三方源码（含 task assets）
├── uv-cache/                     # 可选：wheel 缓存
└── pip-cache/
```

> **确认大盘挂在哪**：`df -h` 看剩余空间最多的挂载点（常见：`/data`、`/mnt/...`、AutoDL 的 `/root/autodl-tmp`）。

> **本机情况（2026-05-18 实测）**：
> - `/`（nvme）3.4T / 已用 91% → **不能再放 venv 或数据**
> - `/mnt/hdd1` 19T 剩 17T、`/mnt/hdd2` 19T 剩 17T → 都是大盘，**用 `/mnt/hdd2`**（剩余最多）
> - 多用户机器（uid 1002/1003/1006/1011 都有 session）→ 在 hdd2 下用 `yikang/` 子目录隔离

### 占位符（本机实际值）

| 变量 | 含义 | 取值 | 大小 |
|------|------|------|------|
| `$REPO_DIR` | 代码 | `/home/yikang/git/le-wm` | < 100 MB |
| `$ENV_HOME` | venv 父目录 | `/mnt/hdd2/yikang/envs` | 单个 venv 5–10 GB |
| `$STABLEWM_HOME` | 数据 + checkpoint | `/mnt/hdd2/yikang/stable-wm` | ≥ 50 GB |
| `$LIBERO_DIR` | LIBERO 源码 | `/mnt/hdd2/yikang/LIBERO` | ~5 GB |

把下面这几行写进 `~/.bashrc`：

```bash
# 三处分离：代码 / 环境 / 数据
export REPO_DIR=/home/yikang/git/le-wm              # 程序：nvme 系统盘（git 已 clone 在此）
export ENV_HOME=/mnt/hdd2/yikang/envs               # 环境：大盘 hdd2
export STABLEWM_HOME=/mnt/hdd2/yikang/stable-wm     # 数据：大盘 hdd2
export LIBERO_DIR=/mnt/hdd2/yikang/LIBERO           # 第三方：大盘 hdd2

# HuggingFace 直连（机器在国外，不需要镜像）
# 如果将来从国内访问，再加：export HF_ENDPOINT=https://hf-mirror.com

# 各种 cache 全部挪到大盘，避免占已经 91% 的系统盘
export UV_CACHE_DIR=/mnt/hdd2/yikang/uv-cache
export PIP_CACHE_DIR=/mnt/hdd2/yikang/pip-cache
export HF_HOME=/mnt/hdd2/yikang/hf-cache              # HuggingFace 数据集/模型缓存
export TORCH_HOME=/mnt/hdd2/yikang/torch-cache        # torch.hub / 预训练权重
export WANDB_DIR=/mnt/hdd2/yikang/wandb               # wandb run 日志（跑久了几十 GB）
```

第一次建目录：

```bash
mkdir -p /mnt/hdd2/yikang/{envs,stable-wm,LIBERO,uv-cache,pip-cache,hf-cache,torch-cache,wandb}
```

`source ~/.bashrc` 之后开始。最终 venv 路径 = `$ENV_HOME/lewm`（例如 `/data/envs/lewm`），激活命令固定为：

```bash
source $ENV_HOME/lewm/bin/activate
```

### 反例（不要这么做）

```
$REPO_DIR/.venv/            ❌ venv 进了项目目录，git/IDE 都会受影响
$REPO_DIR/datasets/         ❌ 数据进了项目目录，几十 GB 拖慢 git
/root/lewm/                 ❌ 全堆在系统盘根目录，几个 G wheel 就把盘塞满
$ENV_HOME/lewm/data/        ❌ 数据塞进 venv 里，重建环境就丢
```

---

## 1. 系统级依赖

```bash
sudo apt-get update
sudo apt-get install -y git build-essential cmake \
    libegl1 libgl1-mesa-dri libglvnd0 \
    libosmesa6-dev libglfw3 libglew-dev
```

GPU 驱动 + CUDA：本项目用 PyTorch bf16 训练，需要 CUDA 11.8 以上。`nvidia-smi` 能输出正常表格即可，不要求特定 CUDA toolkit 版本（PyTorch 自带）。

---

## 2. 安装 uv（如果还没装）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # 或 source $HOME/.local/bin/env
uv --version
```

---

## 3. 克隆代码

```bash
mkdir -p $(dirname $REPO_DIR)
git clone https://github.com/<你的fork>/le-wm.git $REPO_DIR
cd $REPO_DIR
```

---

## 4. 创建 Python 环境（**装在大盘上，不在项目目录里**）

先确认 `$ENV_HOME` 落在大盘上、至少剩 10 GB：

```bash
df -h $ENV_HOME 2>/dev/null || df -h $(dirname $ENV_HOME)
```

创建并激活 venv：

```bash
mkdir -p $ENV_HOME
uv venv --python=3.10 $ENV_HOME/lewm
source $ENV_HOME/lewm/bin/activate
```

之后每次开 shell 都需要：

```bash
source $ENV_HOME/lewm/bin/activate
```

> 一次创建多个项目 venv 时，统一放在 `$ENV_HOME/<项目名>` 下（`$ENV_HOME/lewm`、`$ENV_HOME/<另一个项目>` ……），不要互相嵌套。

---

## 5. 安装训练依赖

顺序敏感，不要乱调：

```bash
# 1) gym==0.21 必须用旧 setuptools 且关闭 build-isolation
pip install "setuptools==65.7.0"
pip install "gym==0.21.0" --no-build-isolation

# 2) stable-pretraining 锁版本（新版本删了 vit_hf）
uv pip install "stable-pretraining==0.1.4"

# 3) 装训练 extra；不要加 [env] extra，会强行升级 gym
uv pip install "stable-worldmodel[train]"

# 4) 训练 / 数据处理常用包
uv pip install matplotlib datasets transformers lightning scikit-learn \
               zstandard huggingface_hub hf-xet
```

验证：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import stable_pretraining, stable_worldmodel; print('OK')"
```

---

## 6. 准备数据

`.h5` 训练数据必须放在 `$STABLEWM_HOME` 下。

```bash
mkdir -p $STABLEWM_HOME
cd $STABLEWM_HOME
```

### 6a. 直接下 HuggingFace 上已转好的 `.h5`

```bash
hf download quentinll/lewm-libero_goal --local-dir $STABLEWM_HOME/hf_libero_goal
# 若有 .tar.zst 包：
tar --zstd -xvf <archive>.tar.zst
```

### 6b. 从 LeRobot parquet 自己转（如果 HF 上没有）

`utils/parquet2hdf5.py` 已经处理好 proprio 的 8-D 布局：
```
[eef_pos(3) | eef_axis_angle(3) | gripper_qpos(2)]
```

```bash
python $REPO_DIR/utils/parquet2hdf5.py  # 看脚本顶部调整输入输出路径
```

下载 `.h5.zst` 解压用 `zstandard stream_reader`（避免 `/tmp` 撑爆）：
```bash
python $REPO_DIR/utils/extract.py
```

---

## 7. （可选）下载预训练 checkpoint

```bash
hf download quentinll/lewm-libero_goal --local-dir $STABLEWM_HOME/hf_libero_goal
# 把 weights.pt + config.json 转成 _object.ckpt 格式（评测必需）
# 转换脚本在 utils/ 下，按需调用
```

---

## 8. 跑训练（先确认能起来）

```bash
cd $REPO_DIR
python train.py data=libero_goal
```

> `wandb entity` 字段在 yaml 里必须是带引号的字符串（例如 `entity: "934464091"`）。
> 用 Hydra config 时若手动 `**dict` 展开，记得过滤 `_target_`。

到这一步训练能起来就算训练环境完成。LIBERO eval 还需要继续往下做。

---

## 9. LIBERO 评测环境（headless server）

eval 调用栈：`eval_libero.py` → `envs/libero_env.py` → LiberoGymWrapper → LIBERO → robosuite → mujoco

### 9.1 修 EGL 渲染

服务器一般只带 NVIDIA 私有 EGL（缺 `eglQueryString`），要用 Mesa EGL 覆盖：

```bash
cd /tmp
apt-get download libegl1 libegl-mesa0 libgl1-mesa-dri libglvnd0
mkdir -p extracted
for deb in libegl1_*.deb libegl-mesa0_*.deb libgl1-mesa-dri_*.deb libglvnd0_*.deb; do
    dpkg -x "$deb" extracted/
done

sudo rm -f /usr/lib/x86_64-linux-gnu/libEGL.so /usr/lib/x86_64-linux-gnu/libEGL.so.1
sudo cp extracted/usr/lib/x86_64-linux-gnu/libEGL.so.1.1.0 /usr/lib/x86_64-linux-gnu/
sudo ln -sf /usr/lib/x86_64-linux-gnu/libEGL.so.1.1.0 /usr/lib/x86_64-linux-gnu/libEGL.so.1
sudo ln -sf /usr/lib/x86_64-linux-gnu/libEGL.so.1   /usr/lib/x86_64-linux-gnu/libEGL.so
sudo cp extracted/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0.0.0 /usr/lib/x86_64-linux-gnu/
sudo ln -sf /usr/lib/x86_64-linux-gnu/libGLdispatch.so.0.0.0 /usr/lib/x86_64-linux-gnu/libGLdispatch.so.0
sudo ldconfig

python -c "from OpenGL import EGL; print('OK')"
```

### 9.2 mujoco `__init__.py` patch（每次重装 mujoco 都要重做）

mujoco 只 catch `ImportError`，但 PyOpenGL 抛 `AttributeError`，会让 `import mujoco` 直接崩：

```bash
sed -i 's/except ImportError:/except Exception:/g' \
    $ENV_HOME/lewm/lib/python3.10/site-packages/mujoco/__init__.py
```

### 9.3 LIBERO 相关 Python 依赖

```bash
uv pip install numba bddl easydict gym imageio
uv pip install "robosuite==1.4.0"   # LIBERO 卡这个版本
# robosuite 装完会覆盖 mujoco，重做 9.2
sed -i 's/except ImportError:/except Exception:/g' \
    $ENV_HOME/lewm/lib/python3.10/site-packages/mujoco/__init__.py
```

### 9.4 安装 LIBERO

```bash
mkdir -p $(dirname $LIBERO_DIR)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO $LIBERO_DIR
cd $LIBERO_DIR
uv pip install -e .

# uv editable 安装不生成 .pth，手动补一条（注意：是 LIBERO 根目录，不是 LIBERO/libero）
echo "$LIBERO_DIR" > $ENV_HOME/lewm/lib/python3.10/site-packages/libero.pth

# 第一次 import 会问数据集路径，输入 N 跳过
python -c "from libero.libero import benchmark; print('OK')"
```

### 9.5 PyTorch 2.6+ weights_only 兼容

LIBERO 的 `init_states` 里有 numpy 对象，torch 默认 `weights_only=True` 会拒绝加载：

```bash
sed -i 's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/' \
    $LIBERO_DIR/libero/libero/benchmark/__init__.py
```

### 9.6 LiberoGymWrapper 导入路径修正

```bash
sed -i 's|from liberogymwrapper.libero_env import LiberoGoalEnv|from liberogymwrapper.v0 import LiberoGoalEnv|' \
    $REPO_DIR/envs/libero_env.py
```

### 9.7 numpy uint64 seed overflow

打开 `$ENV_HOME/lewm/lib/python3.10/site-packages/stable_worldmodel/world.py`，找到 `np_random_seed` 那一行（约 684 附近），改成：

```python
seed_val = self.envs.envs[i].unwrapped.np_random_seed
results['seeds'][ep_idx] = int(seed_val) % (2**31)
```

### 9.8 checkpoint 软链

`AutoCostModel` 找的是 `<cache_dir>/<policy_name>_object.ckpt`：

```bash
ln -sf $STABLEWM_HOME/outputs/lewm_libero_goal_epoch_97_object.ckpt \
       $STABLEWM_HOME/lewm_object.ckpt
```

---

## 10. 跑通三种 eval

```bash
cd $REPO_DIR

# 10a. 随机 goal eval
MUJOCO_GL=egl EGL_DEVICE_ID=0 python eval_libero.py \
    world.demo_hdf5_path=$STABLEWM_HOME/libero_goal.h5 \
    world.task_id=0 \
    eval.num_eval=10 \
    policy=lewm

# 10b. Short-horizon eval（goal = 每个 demo 的第 K 帧）
EGL_DEVICE_ID=0 python eval_short.py world.task_id=0 eval.goal_offset=50 eval.num_eval=5

# 10c. Single-task eval（per-demo init_state + frame-K goal）
EGL_DEVICE_ID=0 python single_task_eval.py eval.num_eval=1
```

---

## 11. 常见坑速查表

| 现象 | 原因 / 解决 |
|------|------|
| `gym` 装不上 / `setuptools` 报错 | 必须 `setuptools==65.7.0` + `--no-build-isolation` |
| `vit_hf` import 失败 | 装错版本，必须 `stable-pretraining==0.1.4` |
| HuggingFace 下载卡住 / xet 报错 | 设 `HF_ENDPOINT=https://hf-mirror.com` |
| `/tmp` 空间不够解压 `.h5.zst` | 用 `utils/extract.py` 的 `stream_reader` |
| `wandb` 报 entity 类型错误 | yaml 里写 `entity: "934464091"`（要带引号） |
| `Hydra` 实例化报 `_target_` 重复 | `**dict` 前先 `pop("_target_", None)` |
| `import mujoco` AttributeError | 重做 §9.2 sed patch |
| eval 报 `init_state_id` missing | `envs/libero_env.py` 的 reset/step 里塞 `info["init_state_id"] = 0` |
| eval seed C long overflow | §9.7 改 `int(seed_val) % (2**31)` |
| 新写 env 渲染图上下颠倒 | LIBERO agentview 是 OpenGL 约定（左下原点），训练数据是左上原点，需要 `img[::-1]` 翻一下 |
| 新 env proprio 维度对不上 | 用 LeRobot 8-D 布局：`concat(eef_pos, quat2axisangle(eef_quat), gripper_qpos)` |

---

## 12. 验证清单

- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → `True`
- [ ] `python -c "import stable_pretraining, stable_worldmodel"` 无报错
- [ ] `python train.py data=libero_goal` 能起来跑出第一个 step
- [ ] `python -c "from libero.libero import benchmark"` 无报错
- [ ] `python -c "from OpenGL import EGL"` 无报错
- [ ] `MUJOCO_GL=egl python eval_libero.py ... eval.num_eval=1` 跑完一个 episode 并存视频

全部通过即环境就绪。
