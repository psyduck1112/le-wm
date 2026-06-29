# 周报 / Weekly Report — LeWM 关抽屉 cost 修复
### 2026-06-15 → 2026-06-23 · 中英对照 / Bilingual

> 任务 / Task：LIBERO `close_the_top_drawer`，双相机 JEPA 世界模型（encoder + predictor + cost + CEM planning）。
> 上下文 / Context：诊断已坐实**瓶颈是 cost**（完美物理 + 真状态 cost = 100%，换回原版 goal-image emb-L2 cost = 0%）。本周的全部工作都在"把 cost 做对"。完整诊断见 [`DIAGNOSTICS_REPORT.md`](DIAGNOSTICS_REPORT.md)，脚本清单见 [`../INDEX.md`](../INDEX.md)。
> *Diagnosis already pinned the bottleneck to the cost (perfect physics + true-state cost = 100%; original goal-image emb-L2 cost = 0%). This week is entirely about getting the cost right.*

---

## TL;DR

本周把上周失败的 cost（后处理冻结解码器 D_φ，闸2 = **0%**，会被 CEM 钻空子 reward-hack）换成**协同训练的 physics head**，并把训练数据从 5000 扩到 11000 条（加入扰动/谱探索覆盖 OOD）。训练到 epoch 200 收敛后，**闸2（oracle 动力学）成功率 0% → 62.5%**。新增 cost-curve 诊断证明 phys cost **诚实不撒谎**（成功 ρ≈1，失败也不变负），失败根因从"cost 撒谎"降级为"off-tube 覆盖不足"——下一步明确指向 DAgger 自采数据。

*This week we replaced last week's failing cost (post-hoc frozen decoder D_φ, gate-2 = 0%, exploited by CEM as reward-hacking) with a **co-trained physics head**, and expanded the training set 5000 → 11000 episodes (adding perturbation/spectrum exploration for OOD coverage). After training converged at epoch 200, **gate-2 (oracle dynamics) success went 0% → 62.5%**. A new cost-curve diagnostic shows the phys cost is **honest** (success ρ≈1, failures never go negative); the failure cause downgraded from "cost lies" to "off-tube coverage gap" — next step clearly points to DAgger.*

---

## 0. 起点：上周卡在哪 / Starting point: where we were stuck

上周的尝试是 **D_φ**（decode-then-cost）：在**冻结**的 emb 上事后训一个小解码器 `emb→(eef, drawer_q)`，再用 p6 那个 100% 的特权距离公式当 cost。

*Last week's attempt was D_φ (decode-then-cost): a post-hoc decoder trained on the FROZEN emb, reassembling p6's 100% privileged distance formula as the cost.*

- **闸1（数据上单调性）PASS**：在 BC 流形上 D_φ 解码 R²≈0.98，cost 随进度单调下降。
- **闸2（完美物理闭环）FAIL = 0%**：CEM 是优化器，把机械臂推到训练没见过的姿势后，D_φ 在那里乱估、被钻空子。`corr(真 reach, D_φ reach) = −0.557`——估计值与真相**反向**移动，正反馈环：越 OOD → cost 越撒谎 → 优化器越往假洼地走。

*Gate-1 (on-data monotonicity) PASSED (R²≈0.98), but gate-2 (perfect-physics closed loop) FAILED at 0%: CEM pushed the arm into unseen poses where D_φ mis-estimated and got exploited (corr(true,decoded reach) = −0.557, a positive feedback loop into OOD).*

**根因 / Root cause**：encoder 冻结 → D_φ 的梯度从不回流 encoder → latent 只是**碰巧**在专家管道上编码了物理量，一离开管道就崩。这是"固定/狭窄数据分布"的协变量漂移通病。

*Frozen encoder → D_φ's gradient never reached the encoder → the latent only accidentally encoded physics on the expert tube, collapsing off-tube. Classic covariate shift of a narrow data distribution.*

---

## 1. 本周改动 / Changes this week

### 1.1 OOD 可行性研究 → 否定"只靠检测器"路线 / OOD feasibility study → kills the detector-only route

先问一个便宜的问题：CEM 漂到 OOD 时，一个 sim-free 的 OOD 分数能不能**知道** D_φ 在撒谎？如果能，加悲观罚就够了，不必动数据。

*First a cheap question: when CEM drifts OOD, can a sim-free OOD score KNOW that D_φ is lying? If yes, a pessimism penalty suffices and we needn't touch the data.*

| 探针 / Probe | 问题 / Question | 结论 / Conclusion |
|---|---|---|
| `p10` OOD 检测器 | 在 emb 上建便宜检测器（maha_k / 残差）| 阈值建好 / detector built |
| `p11` OOD 探针 | OOD 分数能预报 D_φ 误差吗？ | maha 跟得上 q-幻觉 / cost 低报 ✓，**但跟不上 reach 谎 ✗** |
| `p12` 悲观罚 | 加 `β·relu(maha−thr)` 能修闸2吗？ | **所有 β 都 0%** → 全局悲观救不了 |
| `p13` 局部 kNN | 局部密度能看见 reach 谎吗？ | **否** → reach 不可检测 |
| `p14` 来源消融 | 失败归因 eef vs q（2×2）| 喂真 eef 就能成 → **decoded eef 是元凶** |

**结论 / Conclusion**：检测器（全局 maha、局部 kNN）能看到抽屉幻觉和 cost 低报，但**看不到 reach 谎**（解码器对"手臂很远"是均值坍缩，OOD 上无信号）。→ **仅靠"限制在分布内"救不了 reach，必须修数据侧（覆盖）和/或模型侧（让 encoder 真正编码物理）。** 这就否定了"D_φ + 悲观罚"，转向下面 1.2 + 1.3。

*Detectors see drawer hallucination and cost under-report but NOT the reach lie (the decoder mean-collapses on "arm far", no OOD signal). → Restricting-to-in-distribution cannot fix reach; we must fix the data side (coverage) and/or the model side. This kills "D_φ + pessimism" and motivates 1.2 + 1.3.*

### 1.2 数据扩张 / Data expansion: 5000 → 11000

专家 BC 数据只覆盖一条窄管道，且从不示范"从坏状态恢复"。本周加入两批探索数据填 (s,a) 空间：

*Expert BC data covers one narrow tube and never demonstrates recovery. Two exploration batches were added to fill the (s,a) space:*

- **perturb（5000）**：在 sim 里重放专家轨迹但注入随机扰动动作（action-repeat、随机中途起点），渲染真帧。
- **spectrum（1000）**：更宽的随机探索谱。
- 合计 **v2(5000) + perturb(5000) + spectrum(1000) = 11000 条**（fk5 下 ~599k 训练窗口），用 `ConcatDataset` 拼接。

*perturb (5000): replay expert trajectories with injected random/repeated actions; spectrum (1000): wider random exploration. Total 11000 episodes (~599k windows at fk5).*

### 1.3 核心改动：协同训练的 physics head / Core change: co-trained physics head

把"解码物理量"从**事后**改成**协同训练**，让监督信号**直接回流改造 encoder 和 predictor**。

*Turn physics-decoding from post-hoc into co-training, so the supervision back-props into the encoder AND predictor.*

- **结构 / Architecture**：`JEPA.task_head = MLP(192 → 256 → 9)`（LayerNorm + GELU），解码 proprio(8: eef_pos3 + axis-angle3 + gripper2) + drawer_qpos(1)。仅 **+52k 参数（0.28%）**，总 18.48M。
- **关键 / Key**：同一个 head **同时**作用在 `emb`（梯度→encoder）和 `pred_emb`（梯度→predictor）。辅助 loss `aux = MSE(head(emb), y) + MSE(head(pred_emb), y)`，加进主训练 `loss += λ_aux·aux`。这把"物理可解码"变成训练目标，而不是事后补的读出器。
- **cost 形式不变 / Cost unchanged**：仍是 `‖head_eef − cabinet‖ + 30·|head_q − closed|`，goal-image-free。
- **训练 / Training**：frameskip=5（对齐论文 OGBench；macro-action 35 维），~17–19 min/epoch，跑到 **epoch 200 收敛**（`outputs/phys_fk5/`）。

*One shared head applied to both emb and pred_emb; aux MSE added to the main loss (λ_aux). Same goal-image-free cost formula. fk5, trained to epoch 200.*

### 1.4 诊断目录深度整理 / Diagnostics deep-cleanup

- **D_φ 相关全部归档**（这条线不再用）：decode-then-cost（p2b/p6c/p6d）+ OOD/悲观簇（p10–p14）→ `diagnostics/archive/`。
- **同类型 probe 合并**：emb 解码 probe p3/p3b/p3c → 单个 `probe_decode.py`（`--target proprio|full`、`--compare-cam`），输出文件名保留以兼容 `make_report_figs`。
- 新增权威清单 `INDEX.md`；`DIAGNOSTICS_REPORT.md` 更新到 phys-head 期。

*Archived all D_φ-related scripts; merged decode-probes p3/p3b/p3c into one `probe_decode.py`; added `INDEX.md`; updated `DIAGNOSTICS_REPORT.md` to the phys-head era.*

---

## 2. 结果 / Results

### 2.1 闸2 成功率（oracle 动力学，epoch 200，n=8，同 seed）/ Gate-2 success (oracle dynamics, e200, n=8)

| Cost | 成功率 / Success | 说明 / Note |
|---|---|---|
| **privileged**（读 sim 真状态）| **100%** | 上界 / harness + encoder OK |
| **phys**（co-trained head）| **62.5% (5/8)** | ← 本周成果 / this week's result |
| **lewm**（原版 goal-image emb-L2）| **12.5% (1/8)** | 对照 / baseline cost |
| **D_φ**（上周，post-hoc）| **0% (0/8)** | 上周失败 / last week |
| 真机端到端 p9（参考）/ real e2e | ~10% | 历史部署数 / historical |

**phys 是 lewm 的 5 倍**（62.5% vs 12.5%）：在**同一个被重塑的 encoder** 上，显式解码 + shaped 距离 ≫ 原始 emb-L2。

*phys is 5× lewm on the SAME reshaped encoder: explicit decode + shaped distance ≫ raw emb-L2.*

### 2.2 cost-curve：cost 诚不诚实 / Cost honesty along the planner's actual trajectory

新工具 `p6_oracle.py --cost-curve`：沿 planner **实际走过的轨迹**逐帧记 planner-cost vs 真实状态 cost，算 Spearman ρ。ρ≈+1 = cost 诚实（觉得变好时真在变好）。

*New tool: along the planner's ACTUAL rollout, record planner-cost vs true-state cost per step, Spearman ρ. ρ≈+1 = honest.*

| 逐 episode（同 init）| phys 结果 / ρ | lewm 结果 / ρ |
|---|---|---|
| 成功的 / successes | ρ = **+0.95 ~ +1.00** | （ep6）ρ = +0.99 |
| 失败的 / failures | ρ = **+0.15 ~ +0.45（从不变负）** | ρ = +0.08 ~ +0.81 |
| 汇总 / overall | 5/8，全程诚实 | 1/8，正但散（mean ~0.58）|

**两点 / Two points**：
1. **phys 失败也不撒谎**：3 条失败全部**卡在 budget 停住**（不被推走），ρ 掉到 0 附近但**不变负**——cost 不 reward-hack，只是 off-tube 失去分辨率。对比上周 D_φ 的 −0.557（主动撒谎）是质变。
2. **lewm 在 e200 也不再撒谎了**：之前 e177 单条测到 ρ=−0.55（撒谎），但 e200 这 8 条 ρ 全为正（只是弱、散）。最可能原因：**co-trained head 把 encoder 重塑得够多，连用同一 emb 的 lewm cost 也跟着没那么病态**（与 p2 上 lewm on-tube ρ +0.02→−0.42 的副作用一致）。即 lewm 从"撒谎"退化成"诚实但弱"。

*1) phys failures STALL at budget (not pushed away), ρ near 0 but never negative — not reward-hacking, just off-tube resolution loss (vs D_φ's −0.557). 2) lewm at e200 also stopped lying (all positive ρ, just weak/scattered) — the co-trained head reshaped the encoder enough that even lewm's emb-L2 is less pathological; lewm degraded from "lying" to "honest but weak".*

### 2.3 probe3：emb 物理可解码性（phys 模型 e200）/ Decode-probe: physics decodable from the emb

冻结 phys 模型 encoder，用**弱读出器**（ridge 线性 + 2 层 MLP）从 192-d emb 回归特权物理量，看信息在不在 latent 里。episode-split，40 test eps / 3029 帧。

*Freeze the phys encoder, regress privileged physics from the 192-d emb with a deliberately weak readout (ridge + 2-layer MLP). Episode-split, 40 test eps / 3029 frames.*

| 物理量 / group | linear R² | MLP R² | RMSE（原单位）|
|---|---|---|---|
| eef_pos | 0.986 | 0.953 | 0.0095 (~1cm) |
| axis_angle | 0.919 | 0.792 | 0.0503 |
| gripper | 0.940 | 0.981 | 0.0010 |
| drawer_qpos | 0.967 | 0.969 | 0.0090 (~9mm) |
| **overall** | **0.952** | 0.907 | 0.0297 |

对照旧 v2 模型（无 task_head）proprio overall ≈ **0.77** → phys **0.95**。且 **linear R² > MLP**：co-trained head 把物理量训得近乎 emb 的**线性函数**，抽屉单独 0.967——这是"物理被显式编码进 latent"的直接表征级证据（注：probe3 高 R² 部分是 by construction，aux loss 的训练目标就是让 emb 可解码物理量，它证明 aux 生效、信息进了 latent，但不直接等于 planning 成功）。

*v2 (no head) proprio overall ≈0.77 → phys 0.95; linear R² > MLP means physics is near-linear in the emb (drawer 0.967). Direct representation-level evidence the co-trained head pushed physics into the latent (high R² is partly by-construction — it confirms the aux loss worked, not planning success).*

---

## 3. 解读 / Interpretation

- **表征变好了，度量还没好 / Representation improved, the metric did not.** head 让任务方向在 latent 里份量更重 → 物理可解码、cost 不再撒谎（表征层面的胜利）。但原始 emb-L2（lewm）仍只有 12.5%——一个 latent 服务多任务，**没有单一 L2 是所有任务的正确距离**，必须把任务显式注入（这正是 phys 做的，也是终点 `V(emb; z)` 要做的）。
- **失败模式升级了 / The failure mode upgraded.** 从"cost 撒谎、臂被推走"（reward-hacking，难治）变成"off-tube 看不清、臂停住"（覆盖问题，可治）。这把问题从模型侧推到了**数据侧**。
- **下一步明确 / Clear next step.** 既然失败是 off-tube 覆盖不足、且检测器侧（1.1）已证明救不了 reach，正解就是 **DAgger**：planner 自跑 → 收集真正走到的 OOD 状态 → sim 打真标签 → 重训 head + predictor，让训练分布 = 部署分布。

*The head made the representation better (physics decodable, cost honest) but not the metric (raw emb-L2 still 12.5%; one latent serves many tasks, no single L2 is right for all → must inject the task, which phys does and the eventual V(emb; z) will do). The failure mode upgraded from "cost lies, arm walks away" (reward-hacking, hard) to "off-tube blur, arm stalls" (coverage, fixable) → push from model side to DATA side → DAgger.*

---

## 4. 下一步 / Next steps

1. **闸3 真机端到端 / Gate-3 real MPC**：当前 62.5% 是 **oracle 动力学下的 cost 质量上界**，不是部署数。需把 CEM 改成 fk=5 的 35 维 macro-action，跑 `p9_real_mpc.py` 拿真机数。**当前阻塞点。**
2. **DAgger 数据循环 / DAgger loop**：planner 自跑采 OOD 状态 → sim 真标签 → 重训，对齐部署分布。
3. **distill 学到的 cost-to-go / Learned value**：把手写距离升级成 distill 自 100% 特权 cost 的通用 `V(emb; z)`，系统正式成 model-based RL。

*1) Gate-3 real MPC (current 62.5% is the oracle-dynamics cost-quality upper bound, not the deployment number) — needs CEM rewired to fk5 35-dim macro-actions; current blocker. 2) DAgger data loop. 3) Distill the hand-written distance into a general learned cost-to-go V(emb; z).*

---

## 附：本周关键产物 / Appendix: key artifacts

| 内容 / Item | 路径 / Path |
|---|---|
| phys eval（62.5%）| `results/phys_e200.log`, `results/phys_e200/` |
| lewm eval（12.5%）| `results/lewm_e200.log`, `results/lewm_e200/` |
| cost-curve 图 / figures | `results/phys_e200/costcurve_*.png`, `results/lewm_e200/costcurve_*.png` |
| OOD 研究产物 / OOD study | `results/archive/p1[0-4]_*`（已归档）|
| 模型 / checkpoint | `/home/yikang/stable-wm/outputs/phys_fk5/lewm_libero_bc_drawer_phys_epoch_200_object.ckpt` |
| 完整诊断 / full diagnosis | [`DIAGNOSTICS_REPORT.md`](DIAGNOSTICS_REPORT.md) · 脚本清单 [`../INDEX.md`](../INDEX.md) |
