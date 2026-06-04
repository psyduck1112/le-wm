# 阶段 2（td-JEPA value 当 cost）进度总结

> 更新：2026-06-04　分支 `yikang-dev`

## 1. 背景与目标

诊断长期卡在 LIBERO 抽屉任务 ~10% 成功率。M1（双相机 encoder 重训，epoch 200）后：
- **探针3 PASS**：emb 已编码任务状态（drawer R²=0.77、gripper 0.48）。第一层 emb 盲区修好。
- **探针2 仍不过**：裸距离 cost `‖emb−goal_emb‖²` 沿轨迹非单调（cross Spearman −0.005，假谷 ~21）。

阶段 2 设想：用**学出来的 cost-to-go** `C(emb_t, goal_emb)` 替换裸距离 cost。
- `C` = 折扣化剩余步数（每步 reward −1，到 goal=0），天生单调，提供 shaping。
- 冻结 encoder + BC 数据离线 TD 训练，goal 侧永远纯视觉，detach 隔离，**不动 le-wm 本体**。

## 2. 已完成的工作（`value/` 包）

| 文件 | 作用 | 产物 |
|---|---|---|
| `cache_emb.py` | 冻结 encoder 把 36.8 万帧编码缓存 | `emb_cache.npz`（137MB）✅ |
| `value_head.py` | `GoalCost(emb_t,goal)→标量≥0`，输入 `[emb,goal,emb−goal]`，Softplus | 40 万参数 ✅ |
| `train_value.py` | 离线 TD（SmoothL1 + EMA target + MC warmup + 对角锚点） | `value_head.pt`，loss 0.02 收敛 ✅ |
| `p8_value_mono.py` | C 的单调性闸门（探针2 的 C 版） | 见下 ✅ |
| `p6_oracle.py`（改） | 加 `--cost value`，oracle 真物理 rollout + C 打分 | 见下 ✅ |

训练实现细节（已修正的关键点）：
- TD 目标 `target = 1 + γ·C̄(s', g)`，`reached` 时 bootstrap 项=0（**不是整个 target=0**）。
- 对角锚点：10% 概率 `goal=当前帧`，直接监督 `C(s,s)=0`，焊死零点尺度。
- 前 3k step 用 MC 目标 `(1−γ^k)/(1−γ)` 热身（纯监督），之后切 TD 自举。γ=0.98，τ=0.995。

## 3. 关键发现：旧"cost 是瓶颈"结论被框架混淆（已推翻）

**之前内存里"探针6 铁证 cost 是瓶颈"是错的。** 复核 privileged 控制组发现：

- **旧 MPC 配置（n_iter=3 / budget=100 / receding=5）下，privileged cost（完美物理 + 直接读抽屉关节真值）也是 0%**（`results/p6_priv_validate.log` 0/2）。
- privileged 是"验证框架本身能否解任务"的控制组，它失败 → **probe6 的 0% 无法区分"cost 烂"还是"框架太弱"**。
- `results/p4_bc.log`：BC 策略 95% 成功（~69 步）→ 任务可解、demo 是好的。
- **加强配置（n_iter=5 / budget=150 / receding=3）下，privileged 4/4 = 100%**（~20 步解决，`results/p6_priv_strong.log`）。

→ **结论：旧 CEM 框架太弱才是首要瓶颈；probe6 必须用加强配置才是有效测试。**

## 4. 判决实验：加强框架 + C 当 cost

同一套加强框架（n_iter=5 / budget=150 / receding=3）对比：

| cost | 成功率 | 行为 |
|---|---|---|
| privileged | **4/4 = 100%** | eef_dist 0.34→接触→拉抽屉，20 步解决 |
| **C (value)** | **0/4 = 0%** | eef_dist 全程 ~0.39 原地晃，drawer_q 卡在 −0.145 纹丝不动 |

→ **框架已修好的前提下，C 当 cost 仍不产生有用规划梯度。这才是对 cost 的干净隔离。**

C 的单调性闸门（`p8`，复核）：
- same goal：Spearman −1.000，假谷 0（平凡，它就这么训的，非真测试）。
- **cross goal：Spearman −0.005 → −0.78**（趋势大改善），**但假谷 33.8、frac_monotone 仅 0.62**（残余非单调）。

## 5. 待解决的核心问题

**C 在闭环规划里失效，病因 = OOD（off-manifold 平梯度）：**
- C 只在 BC demo 的 emb 上训过；demo 一开始就"有目的地接近抽屉"。
- CEM 从随机 init 出发，候选动作 rollout 的想象帧落在 demo 没覆盖的区域，C 在那儿**landscape 是平的** → 所有候选打分几乎一样 → CEM 分不出好坏 → 手臂乱晃。
- 这与 p8 的"强趋势 −0.78 但假谷 33.8"一致：on-distribution 单调，off-distribution 失效。

## 6. 下一步（按优先级）

1. **裸距离 cost 也上加强框架**（必做对照）：旧 lewm 0/8 同样被弱框架污染。需确认距离 cost 在强框架下到底行不行——可能根本不需要 C，也可能它也一样废。这决定后续方向。
2. **若确认 C 因 OOD 失效，对症修法：**
   - **负样本对**：训练时掺跨轨迹/不可达 goal → 监督高 cost，把 off-manifold 撑起来。
   - **β 距离兜底**：`cost = C + β·‖emb−goal‖²`，给 off-manifold 局部梯度（默认 β=0，此处开小量）。
   - **保守正则**：对 OOD emb 抬高 C（CQL 风格），防平梯度。
3. **MPC 配置本身需固化**：n_iter=5 / budget=150 / receding=3 是新的有效基线，正式 eval（`bc_drawer_eval`）也要同步。

## 7. 文件 / 日志索引

- 代码：`value/{cache_emb,value_head,train_value,p8_value_mono}.py`、`diagnostics/p6_oracle.py`（加 value 模式）
- 日志：`diagnostics/results/p6_priv_validate.log`（旧框架 priv 0%）、`p6_priv_strong.log`（强框架 priv 100%）、`p6_value_strong.log`（强框架 C 0%）、`p4_bc.log`（BC 95%）
- 训练日志：`value/train_value.log`、`value/cache_emb.log`
