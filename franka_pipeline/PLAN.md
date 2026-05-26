# Franka MuJoCo Pipeline — Plan

目标：用 raw MuJoCo + Python 自建 Franka manipulation 数据采集流水线，作为 LeWM (SimDist 风格) 项目的 sim 端基础。
不依赖 robosuite / LIBERO / swm.World 等抽象层，每行代码自己懂。

## 顶层数据流

```
scene.xml + Franka MJCF
        │
        ▼
   ┌────────────┐    reset()        ┌──────────┐
   │   Task     │ ─────────────────▶│  MjData  │
   │            │    success()      │ (state)  │
   │            │    get_privileged │          │
   └────────────┘                   └──────────┘
        │                                ▲
        ▼                                │
   ┌────────────┐  EEF target            │
   │   Oracle   │ ─────────────────┐     │
   │ (state     │                  │     │ mj_step
   │  machine)  │                  ▼     │
   └────────────┘             ┌──────────┴───┐
                              │ Controller   │
                              │ (Jacobian IK)│
                              │ → data.ctrl  │
                              └──────────────┘
                                     │
                                     ▼
                              ┌──────────┐
                              │ Renderer │
                              │ → pixels │
                              └──────────┘
                                     │
                                     ▼
                              ┌──────────┐
                              │ Collect  │
                              │ → HDF5   │
                              └──────────┘
```

## 模块分工

| # | 文件 | 职责 | 行数估计 |
|---|---|---|---|
| 0 | `smoke_test.py` | 验证 mujoco 装好、能 load Franka、能 step、能渲染 | ~30 |
| 1 | `assets/scene.xml` | MJCF：Franka + 桌面 + 红方块 + 2 个相机 | ~60 |
| 2 | `renderer.py` | 包一层 `mujoco.Renderer`，支持多相机 | ~30 |
| 3 | `controller.py` | EEF 目标 pos → joint action（Jacobian 伪逆 IK）| ~80 |
| 4 | `task.py` | `PickAndPlaceTask`: reset / success / get_privileged | ~60 |
| 5 | `oracle.py` | State machine 专家: approach → grasp → lift → place | ~100 |
| 6 | `collect.py` | rollout loop + HDF5 写出 + 噪声注入 | ~150 |

## 写的顺序（依赖关系）

```
smoke_test  →  scene.xml  →  renderer  →  controller
                                              │
                                              ▼
                                            task
                                              │
                                              ▼
                                            oracle
                                              │
                                              ▼
                                            collect
```

每一步都跑一遍验证，再往下走。

## v0 任务定义

**Pick-and-place 红方块**:
- 起始：方块随机放在桌面 0.4-0.6m × ±0.1m 区域
- 目标：把方块放到固定位置 `(0.3, 0.3, 0.43)`，容差 3cm
- 成功判定：方块距目标 < 3cm 且静止 0.5s
- horizon: ~150 步（dt=0.02s → 3s 完成）

足够简单能跑通，足够复杂能看出 oracle 行为，且和 LIBERO 的 pick-place 性质对齐。

## 暂不做的事（避免范围爆炸）

- ❌ Domain randomization（v1 加）
- ❌ Multi-task（v0 只 pick-place 一个 task）
- ❌ 真机部署接口（v1 加 polymetis bridge）
- ❌ Value head 训练（在 collect 跑通之后做）
- ❌ 多进程并行采集（先单进程跑通）

## 输出 schema（HDF5）

和之前讨论的统一 schema 对齐：

```
data.h5
├─ ep_idx          (N_steps,)        int32
├─ step_idx        (N_steps,)        int64
├─ ep_len          (N_episodes,)     int32
├─ ep_offset       (N_episodes,)     int64
├─ pixels          (N_steps, H, W, 3)  uint8
├─ pixels_wrist    (N_steps, H, W, 3)  uint8    # 第二相机
├─ action          (N_steps, 5)      float32   # [dx, dy, dz, dyaw, gripper]
├─ proprio         (N_steps, 8)      float32   # [eef_pos(3), axis_angle(3), gripper(2)]
├─ success         (N_steps,)        bool
└─ privileged/
    ├─ qpos        (N_steps, nq)     float32
    ├─ qvel        (N_steps, nv)     float32
    ├─ box_pos     (N_steps, 3)      float32
    └─ box_quat    (N_steps, 4)      float32
```

action 维度故意和 OGB 对齐（5-D），这样下游训练代码改动最小。

## 依赖

```bash
pip install mujoco mujoco-menagerie h5py numpy
# Windows 本地能装能跑（无需 EGL）
# Linux server 跑大批量时：MUJOCO_GL=egl
```

注意：`mujoco_menagerie` 不是 pip 包，是 git repo。两个办法：
1. `git clone https://github.com/google-deepmind/mujoco_menagerie` 放任意位置，scene.xml 用相对路径 include
2. 手动 copy `franka_emika_panda/` 到 `assets/` 下

选 (1)，因为 menagerie 经常更新，git clone 后续好维护。
