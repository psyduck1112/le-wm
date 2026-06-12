这是整理后的完整设计文档：

---

# LeWM × LIBERO 评估系统设计文档

## 一、设计思路

### 起点

目标：用 LeWM 训练好的世界模型，在 LIBERO 环境里跑评估。

LeWM 的 eval 系统已经存在（eval.py 跑通了 PushT/DMControl），LIBERO 环境也已经存在。问题变成：**让两者对接起来**。

不是凭空设计架构，而是从已有的两端往中间推。

---

### 第一步：读懂两端的接口

**LeWM 对环境的要求**（读 `eval.py` 和 `swm.World` 源码）：

```python
world = swm.World(env_name="pusht-v0", ...)
env = gym.make(env_name)
obs = env.reset()
obs, reward, done, info = env.step(action)
pixels = info["pixels"]   # 从这里取图像
goal   = info["goal"]     # 从这里取目标
```

整理成接口要求：
```
① 能 gym.make(env_name) 创建（需要注册到 gymnasium）
② reset() 返回的 info 里要有 goal
③ step()  返回的 info 里要有 pixels
④ terminated=True 表示任务成功
```

**LIBERO 提供了什么**（读 LiberoGymWrapper 源码）：

```python
obs, info = env.reset()             # info 里没有 goal，没有 pixels
obs, reward, done, info = env.step(action)  # 同样没有
```

---

### 第二步：找 gap

```
LeWM 需要          LIBERO 有
─────────────────────────────────────────────────
info["pixels"]  ←  env.render() 可以拿到图像，但没存进 info
info["goal"]    ←  完全没有，需要从 demo 数据里取
terminated      ←  LIBERO 原生有，不需要处理
gym.make()      ←  需要注册到 gymnasium
```

gap 就是需要填的东西，架构围绕这些 gap 展开。

---

### 第三步：决定在哪里填 gap

两种选择：

**选择 A：直接改 LiberoGymWrapper 源码** → 升级会丢失修改，污染第三方库

**选择 B：写子类，在子类里加** → 开闭原则，对扩展开放，对修改关闭

选 B，写 `LiberoGoalLEwMEnv` 继承 `LiberoGoalEnv`。

---

### 第四步：读源码发现 pixels 不需要自己处理

写 `info["pixels"] = self.render()` 时，发现 stable_worldmodel 内部已经有 `AddPixelsWrapper`：

```python
class AddPixelsWrapper:
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        info["pixels"] = self.env.render()  # 自动调 render
        return obs, reward, done, truncated, info
```

设计随之调整：

```
原来想的：LiberoGoalLEwMEnv 负责 pixels + goal
发现之后：LiberoGoalLEwMEnv 只负责 render() 格式 + goal
          pixels 交给 AddPixelsWrapper 自动处理
```

这是**读源码驱动设计**：不是先想好再写，而是边读边发现边调整。

---

### 第五步：确定每层边界

判断标准：**这个逻辑是谁关心的？**

```
goal 图像的来源    → LeWM eval 关心  → 放 LiberoGoalLEwMEnv
render() 格式转换  → MegaWrapper 关心 → 放 LiberoGoalLEwMEnv
pixels 存进 info   → MegaWrapper 自己处理，不用管
任务成功判断       → LIBERO 自己处理，不用管
```

---

### 第六步：从最小可运行开始

```
里程碑 1：policy=random 跑通
→ 验证环境能创建、reset、step、eval 循环能跑完

里程碑 2：policy=lewm 跑通
→ 验证 goal 格式、pixels 格式、归一化参数正确

里程碑 3：success_rate > 0
→ 验证整个 pipeline 能产生有意义的结果
```

---

## 二、系统架构

### 调用栈（从底层到顶层）

```
MuJoCo
  ↑ 物理引擎，管理关节、碰撞、渲染

robosuite
  ↑ 机器人仿真框架，封装 MuJoCo
  ↑ 提供 Panda 机械臂、相机、动作空间

libero（OffScreenRenderEnv）
  ↑ 在 robosuite 上加了任务定义（BDDL 文件）和初始状态管理

LiberoGymWrapper（LiberoGoalEnv）
  ↑ 把 libero 包装成标准 gymnasium.Env 接口

LiberoGoalLEwMEnv（我们写的）
  ↑ 填补 LIBERO 和 LeWM 之间的 gap
  ↑ 加 render() 和 goal 图像

MegaWrapper（swm 内部自动套上）
  AddPixelsWrapper      → 调 render()，把图像存进 info["pixels"]
  EverythingToInfoWrapper → 把 obs dict 所有 key 移进 info
  EnsureInfoKeysWrapper   → 检查必要的 key 存在
  ResizeGoalWrapper       → 处理 info["goal"] 的尺寸
  StackedWrapper          → 对所有 key 做 history 堆叠（history_size=3）

swm.World
  ↑ 管理多个并行 env、eval 循环
```

### 为什么要堆叠 3 帧

LeWM 的 predictor 预测下一个状态时，需要看过去 3 帧的历史（history_size=3），和训练时一致。单帧图像无法判断物体运动方向，多帧才能提供时序信息。

---

### LiberoEnv 需要满足的三个条件

```
① render() → 返回 (H, W, 3) uint8 的 agentview 图像（给 AddPixelsWrapper 用）
② reset() 的 info 里包含 goal（给 WorldModelPolicy 用）
③ terminated=True 当任务成功（给 evaluate() 判断成功用）
```

条件 ③ 不需要自己写，LIBERO 原生就会在任务完成时设 `terminated=True`。

---

### evaluate() 内部逻辑

```python
for 每个 episode:
    obs = env.reset()
    while not done:
        action = policy(obs)    # LeWM 用 CEM 规划出动作
        obs, reward, done, info = env.step(action)
    记录是否成功（terminated=True 就是成功）

返回 success_rate = 成功次数 / 总次数
```

### WorldModelPolicy 怎么产生动作

```
当前帧图像 → encode() → 当前状态 embedding
goal 图像  → encode() → 目标状态 embedding

CEM 规划器：
  随机采样 300 组动作序列
  对每组：rollout() 预测未来状态
          get_cost() 计算预测状态和目标状态的距离
  选距离最小的动作序列
  执行第一个动作
  重复（滚动规划）
```

这是 LeWM 区别于 BC 的核心：不是直接输出动作，而是在 latent space 里规划，找到能让预测状态接近目标的动作序列。

---

## 三、需要新写的文件

```
le-wm/
  envs/
    __init__.py
    libero_env.py        ← 核心：gymnasium wrapper
  config/eval/
    libero_goal.yaml     ← eval 配置
  eval_libero.py         ← eval 入口脚本
```

---

## 四、各文件详解

### 1. envs/libero_env.py

核心文件，包含三件事：

**① LiberoGoalLEwMEnv 类**

```python
class LiberoGoalLEwMEnv(LiberoGoalEnv):

    def __init__(self, demo_hdf5_path: str, **kwargs):
        self._demo_hdf5_path = demo_hdf5_path
        self._goal_images = self._preload_goal_images()
        self._goal_image = None
        super().__init__(**kwargs)

    def _preload_goal_images(self):
        """从训练 HDF5 取每个 episode 最后一帧作为 goal"""
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
        """返回 (H,W,3) uint8，给 AddPixelsWrapper 用"""
        obs = self.get_observations()
        img = obs["agentview_image"]  # LiberoGymWrapper 返回 (C,H,W)
        return img.transpose(1, 2, 0)  # → (H,W,3)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        idx = np.random.randint(len(self._goal_images))
        self._goal_image = self._goal_images[idx]
        info["goal"] = self._goal_image
        info["init_state_id"] = idx   # stable_worldmodel 内部要求
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["goal"] = self._goal_image
        info["init_state_id"] = 0     # 占位，不影响 success_rate
        return obs, reward, terminated, truncated, info
```

**② 注册到 gymnasium**

```python
gymnasium.register(
    id="libero-goal-lewm-v0",
    entry_point="envs.libero_env:LiberoGoalLEwMEnv",
    max_episode_steps=600,
)
```

---

### 2. config/eval/libero_goal.yaml

```yaml
world:
  env_name: libero-goal-lewm-v0
  num_envs: 1
  history_size: 3        # 必须和训练一致
  frame_skip: 1
  max_episode_steps: 600
  task_id: 0             # 0-9，libero_goal 共 10 个 task

eval:
  num_eval: 20           # 每个 task 跑多少次
  dataset_name: libero_goal

dataset:
  keys_to_cache:
    - action
    - proprio

plan_config:
  horizon: 5
  action_block: 1

solver:
  _target_: stable_worldmodel.solver.CEMSolver

policy: /path/to/checkpoint
```

---

### 3. eval_libero.py

```python
import os
os.environ["MUJOCO_GL"] = "egl"

import envs  # 触发 gymnasium.register
import hydra
import stable_worldmodel as swm

@hydra.main(config_path="./config/eval", config_name="libero_goal")
def run(cfg):
    # 归一化参数（从训练数据算均值方差）
    process = get_normalization(cfg)
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    # 创建环境
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # 加载策略
    policy_path = cfg.get("policy", "random")
    if policy_path != "random":
        model = swm.policy.AutoCostModel(policy_path)
        model = model.to("cuda").eval()
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )
    else:
        policy = swm.policy.RandomPolicy()

    world.set_policy(policy)

    # 跑评估
    metrics = world.evaluate(episodes=cfg.eval.num_eval, seed=cfg.seed)
    print(metrics)

    # 录制视频
    video_dir = Path(__file__).parent / "videos" / f"task_{cfg.world.task_id}_{policy_path}"
    video_dir.mkdir(parents=True, exist_ok=True)
    world.record_video(
        video_path=video_dir,
        max_steps=600,
        fps=30,
        viewname="pixels",
        seed=cfg.seed,
    )
    print(f"Videos saved to {video_dir}")
```

---

## 五、风险点

| 风险 | 说明 | 应对 |
|---|---|---|
| render() 格式 | AddPixelsWrapper 期望 HWC uint8，LiberoGymWrapper 返回 CHW | 在 render() 里做 `transpose(1,2,0)` |
| goal 图像格式 | ResizeGoalWrapper 期望 HWC | 同上，确认格式 |
| init_state_id 缺失 | stable_worldmodel 的 wrapper 在 step 时 assert 检查 | reset/step 的 info 里都要加这个 key |
| seed 溢出 | np_random_seed 是 uint64，转 C long 溢出 | `int(seed_val) % (2**31)` |
| 服务器 EGL | 容器无显示器，EGL 需要特殊配置 | `MUJOCO_GL=egl EGL_DEVICE_ID=0` |

---

## 六、执行顺序

```
① 本地写 envs/libero_env.py
② 本地写 config/eval/libero_goal.yaml
③ 本地写 eval_libero.py
④ 服务器配置 EGL 环境（见安装踩坑文档）
⑤ 服务器装 LIBERO + LiberoGymWrapper
⑥ push 代码到服务器
⑦ 跑 policy=random 验证里程碑 1
⑧ 跑 policy=lewm 验证里程碑 2
⑨ 确认 success_rate > 0（里程碑 3）
```

---

## 七、运行命令

```bash
MUJOCO_GL=egl EGL_DEVICE_ID=0 python eval_libero.py \
    world.demo_hdf5_path=/root/autodl-tmp/stable-wm/libero_goal.h5 \
    world.task_id=0 \
    eval.num_eval=10 \
    policy=lewm
```

MUJOCO_GL=egl EGL_DEVICE_ID=0 python eval_libero.py \
world.task_id=0 \
eval.num_eval=1 \
solver.num_samples=100 \
solver.n_steps=10 \
solver.topk=10 \
plan_config.receding_horizon=1