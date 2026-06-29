# le-wm × LIBERO 诊断结果 (按 LIBERO_DIAGNOSTICS.md 执行)

**任务**: LIBERO_90 `KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet`
**被诊断模型**: `lewm_libero_bc_drawer_epoch_200`(192-D emb, ViT-tiny/patch14, history=3, num_preds=1)
**训练数据**: `libero_bc_drawer.h5`(5000 集 BC rollout;pixels/action(7)/proprio(8))
**le-wm MPC 基线成功率**: **10%**(2/20)
**日期**: 2026-06-02

脚本均在 `diagnostics/`,结果在 `diagnostics/results/`。全程冻结/旁路 le-wm,未改训练核心。

---

## 决策表(汇总定锅)

| 探测 | 结果 | 真正瓶颈 | "le-wm 能力有限" |
|---|---|---|---|
| 前置 SIGReg 谱 | eff-rank **55/192**,未塌缩 | encoder 学到了东西 | — |
| 前置 单步预测 | 比 copy-last 仅好 **11%** | 预测器偏弱 | ⚠ |
| **探测4 BC 对照** | **BC 95% vs MPC 10%** | **数据无罪,锅在 le-wm WM+规划** | **否** |
| 探测1 CEM 预算 | 6× → **10%→20%**(小幅) | 规划预算非主因 | 否 |
| **探测2 cost 单调** | Spearman **−0.25**,假谷 **24/条**,7% 单调 | **goal-image MSE cost 有根本缺陷** | 否(换 cost) |
| **探测3 线性探针** | eef R² **0.58**,夹爪 **0.18** | **表征对精细操作不足** | 是(根本性) |
| 探测5 多步开环 | err/copy **~0.7** 平稳,不发散 | 预测器中等偏弱,可缓解 | 部分 |
| **探测6 oracle 规划** | **le-wm cost 0% vs privileged 100%** | **cost(不是预测器)** | 否(预测器洗清) |

---

## 各探测详情

### 前置 sanity (`p0_sanity.py`)
- **SIGReg 未塌缩**:embedding 协方差有效秩 55/192,participation ratio 40,top-5 特征值占比 0.23。encoder 没退化。
- **单步 teacher-forcing**:模型 1-step L2 误差 0.54,copy-last 0.61,ratio **0.888**(只比"原地不动"好 11%);相对误差 3.9%。→ 预测器偏弱的第一信号。

### 探测4 — BC 对照 (`p4_bc_eval.py`) 【决定性】
- 同任务同数据训练的标准 LIBERO BCTransformerPolicy:**19/20 = 95%**,平均 ~70 步关好抽屉。
- le-wm MPC 仅 10%。→ **数据完全够用,失败 100% 在 le-wm 的世界模型 + 规划链路**,彻底排除数据/任务难度。

### 探测1 — CEM 预算 (`results/p1_budget6x.log`)
- 预算 6×(num_samples 300→600, n_steps 5→15):成功率 10% → **20%**。
- 小幅翻倍但远未解决(仍 80% 失败)→ **规划预算不是主瓶颈**,只是 cost 噪声大时多搜能偶然撞到好方向。
- 备注:`CEMSolver.batch_size` 语义 = 并行 env 数(**不是**候选批大小),保持 =1;放大只调 num_samples/n_steps。

### 探测2 — cost 单调性 (`p2_cost_mono.py`)
沿 BC 轨迹算 `cost_t = ||enc(f_t) − goal_emb||²`:
- **same**(本轨迹真末帧当 goal):Spearman(progress,cost) = **−0.30**,0% 轨迹干净单调。
- **cross**(别条 demo 末帧,真实 eval 情形):Spearman **−0.25**,平均 **24.3 个假谷/条**,仅 7% 单调。
- → goal-image 的 emb-MSE **不随任务进度单调下降**,作为规划 cost 有根本缺陷(目标图含大量任务无关像素,emb 欧氏距离 ≠ 任务距离)。

### 探测3 — 线性探针 (`p3_probe.py`)
冻结 encoder,从 emb 回归 proprio(8-D);按 episode 划分 train/test:

| 组 | Linear R² | MLP R² |
|---|---|---|
| eef_pos | 0.58 | 0.68 |
| axis_angle | 0.26 | 0.32 |
| gripper | 0.19 | 0.18 |
| overall | 0.36 | 0.42 |

- eef 位置部分可解(R²~0.6),但**朝向、尤其夹爪开合(0.18)几乎读不出**——夹爪在画面里可见却解不出,说明 encoder 丢失操作关键细节。
- 物体/抽屉 pose 因 h5 无真值未测(需 re-sim);以夹爪结果推测大概率也差。
- → **表征对精细操作不足**(ViT-tiny 容量)。cost 建在这种 emb 上,直接导致探测2 的坏 cost。

### 探测5 — 多步开环复合误差 (`p5_openloop.py`)
真起点 emb + 真动作自回归 rollout,对比 encode(真帧k):

| k | err | copy(原地) | err/copy | 相对误差 |
|---|---|---|---|---|
| 1 | 0.49 | 0.64 | 0.77 | 3.6% |
| 5 | 2.19 | 3.13 | 0.70 | 15.8% |
| 7 | 3.08 | 4.32 | 0.71 | 22% |
| 10 | 4.59 | 5.96 | 0.77 | 33% |

- `err/copy` 全程平稳 **~0.7**,**没有失控雪球**,每个步长都比原地不动好 ~30%。
- 但 horizon=7 处相对误差 22% → 预测器中等偏弱、有复合误差,属"可缓解"(num_preds>1 / TD-MPC),非主因。

### 探测6 — oracle 模拟器规划 (`p6_oracle.py`) 【决定性】
把 MPC 里**学到的预测器**换成**真 MuJoCo 模拟器**(存读档分叉 + 真步进 + 渲染真末帧 + le-wm 真 encoder),encoder/cost/CEM 不变:

| 模式(同完美物理、同 CEM N=100/n_iter=8/H=6) | 成功率 | 现象 |
|---|---|---|
| oracle 动力学 + **le-wm cost** | **0/5 = 0%** | 抽屉纹丝不动(q 停 −0.156),机械臂够不到、乱走 |
| oracle 动力学 + **privileged cost**(reach+close,验证框架) | **5/5 = 100%** | 16–18 步关好抽屉 |

- 唯一变量是 cost。**le-wm cost → 0%,完美 cost → 100%**。
- → **给了完美物理引擎,le-wm 的 goal-image cost 依然无法驱动规划** → 瓶颈是 **cost/encoder,不是学到的预测器**。预测器被洗清(换 TD-MPC 也救不了)。
- privileged 100% 证明 MPC 框架与任务本身没问题。

---

## 根因结论

le-wm MPC 在 LIBERO 关抽屉上只有 10%,**不是**因为:
- ❌ 数据不足(探测4:同数据 BC 95%)
- ❌ 学到的动力学差(探测6:换完美物理仍 0%)
- ❌ 规划预算(探测1:6× 只到 20%)

**主因(按致命程度)**:
1. **goal-image 的 emb-MSE cost 坏**(探测2 + 探测6)——非单调、假谷多,完美物理下也无法引导规划。**最大杠杆**。
2. **encoder 表征对精细操作不足**(探测3)——夹爪/朝向几乎不可解码;cost 建在这种 emb 上,坏 cost 的根在这。
3. **单步预测器中等偏弱**(探测5)——有复合误差但不发散,且被探测6 洗清为非主因。

## 改进方向(按性价比)
1. **换 cost / goal 表示**(探测2、6 主攻):放弃 goal-image emb-MSE,改用任务感知的距离 / 学到的 reward 或 value(TD-MPC 的 value 头可在此发力——但要先修 cost 表征,不是修动力学)。
2. **更强 encoder 表征**(探测3):ViT-tiny 丢失精细操作态;cost 只能和它所在的 emb 一样好。
3. **多步预测 / TD-MPC**(探测5):缓解预测器,但探测6 表明单独做这个救不了 le-wm,**cost 必须先修**。

> 关键启示:探测6 把"预测器烂"与"cost 烂"彻底切开。结论是 **cost 型瓶颈**(配合表征型),而非动力学型——因此盲目上 TD-MPC 后继价值意义不大,除非同时重做 cost/goal 表示。
