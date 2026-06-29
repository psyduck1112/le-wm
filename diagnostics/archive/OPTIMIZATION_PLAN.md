# le-wm 优化方案（施工图）

> 配套诊断见 `ANALYSIS_REPORT.md`。本文是可执行的实施计划。
> **任务**：`libero-bc-drawer`（关抽屉，goal-image 条件 CEM-MPC）。baseline 成功率 10%。

---

## 0. TL;DR

诊断已钉死(探测6 铁证):le-wm 失败的**唯一根因是规划 cost(goal图 emb 距离)失效**,根因的根因是 **emb 不编码任务关键状态(抽屉/夹爪)**。动力学、规划搜索、任务难度/数据全部排除(探测1/4/5/6)。

**战略约束**:**世界模型(encode→predict→CEM 规划)是核心卖点,不退化成反应式 BC**。故所有改动只修 cost、保留规划范式;且**不碰 le-wm 两损失核心(next-emb MSE + SIGReg),只做加法/隔离**。重训 le-wm 便宜,故"修 emb"是首选而非重手段。

**主线三步**:① 重训 encoder **把腕部相机加进视觉 emb**(修 emb 盲区,第1层;proprio 只当旁路条件,**不进 goal 匹配的 emb**——否则破坏 goal-image cost)→ ② 加 **td_jepa 后继 value** 当规划 cost(修 shaping,第2层,td_jepa 正位,白送 zero-shot)→ ③(可选消融)把 value 折进重训塑 metric emb(改造C)。

---

## 1. 诊断结论（压缩）

```
唯一根因 = 规划 cost(goal图 emb 距离)失效
   ├─ 探测6:完美物理 + 此cost 仍 0%;换能读真实状态的cost 立刻 100%   ← 铁证
   ├─ 探测4:BC 95% vs MPC 10%                                       ← 排除任务/数据
   ├─ 探测2:此cost 沿真轨迹非单调、假谷成片(症状)
   ├─ 探测3:emb 不编码抽屉/夹爪(R²=0.18)(病因)
   └─ 探测5+6:动力学温和 / 完美物理也败                            ← 排除 predictor
```
cost 坏在**两层**,必须分别修:

| 层 | 病 | 修法 |
|---|---|---|
| 第1层 | emb **看不见**抽屉/夹爪 | 重训 encoder:**腕部相机进视觉 emb**(proprio 只当旁路) |
| 第2层 | 即使看得见,"到终点的裸几何距离"**没 shaping** | 学一个 shaped value-to-go |

⚠️ **关键**:抽屉是**物体状态**,不在机器人 proprio 里 → 修抽屉只能靠**视觉(腕部相机)**。而 **proprio 不能进 goal 匹配的 emb**——goal 是图、没有 goal proprio,烤进去会破坏 cost(详见阶段1)。proprio 若要用,只当当前态旁路条件喂 predictor/value。

---

## 2. 总体架构（改动落点）

```
        ┌──── 改动1:进 emb 的只有视觉 ────┐
agentview + eye_in_hand              proprio ┐
        │                                    │(旁路条件,不进 goal 匹配 emb)
   [encoder ViT]  ←─ 改动3(可选):加辅助metric loss,塑emb
        │ emb (纯视觉,D)                     │
   [predictor]  ←─ 不动(可选吃 proprio 旁路)─┤
        │ 想象未来 emb                        │
   [cost]  ←─ 改动2:goal图距离 → V(emb, [proprio,] goal_emb) ┘
        │           (goal 一侧永远纯视觉)
   [CEM-MPC]  ←─ 不动(只换它调用的cost)
```
**两损失核心(next-emb MSE + SIGReg)在改动1下不变;改动2 的 TD/EMA 严格隔离;改动3 是消融。**

---

## 3. 阶段总览与依赖

```
阶段0 数据准备 ──→ 阶段1 重训encoder ──┬─→ 阶段2 td_jepa value(主线)
(双相机+proprio   (修第1层,闸门:        │   (修第2层,白送zero-shot)
 +抽屉qpos标签)    探测3/2/6)            ├─→ 阶段2′ 子目标规划(零学习,先白嫖)
                                        └─→ 阶段3 改造C metric化(可选消融)
                                                          ⋯⋯ 远期:online(TD-MPC2式)
```

---

## 阶段 0 — 数据准备（前提）

**目标**:产出带双相机 + proprio + 抽屉真值的数据集,供重训和验证用。

**做什么**:
1. 重新采集/重渲染 demo,每帧存:`agentview_image` + `eye_in_hand_image` + `proprio(夹爪+关节)` + `action`。腕部相机仿真本来就能渲(探测4 的 BC 就在用)。
2. **额外存抽屉关节 qpos**(用 `p6_oracle.py:find_drawer_joint` 的方式读 `env.sim.data.qpos[drawer_addr]`)——**专门给探测3 当回归靶子**,验证"emb 修好后是否编码抽屉"。这是当前最大数据缺口(探测3 缺抽屉位姿)。

**产出**:`libero_bc_drawer_v2.h5`,新增 `eye_in_hand` 像素 + `drawer_qpos` 列。

---

## 阶段 1 — 重训 encoder：腕部相机进 emb，proprio 当旁路（修第1层）

**目标**:让 emb 编码抽屉/夹爪状态,使任何建在 emb 上的 cost 有救。

### 🔑 核心约束:被 goal 匹配的 emb 必须"纯视觉"
cost = `‖emb_t − goal_emb‖²`(或 `V(emb, goal_emb)`),而 `goal_emb = encode(goal 图像)`——**目标是一张图,没有 "goal proprio"**。若把 proprio 烤进 emb,goal 那侧的 proprio 维**没东西可匹配**(只能填垃圾值,或强行对齐"末态夹爪姿势"——而你根本不在乎夹爪最后停哪)→ **直接破坏 goal-image cost**。所以**被匹配的 emb 必须纯视觉**,才能和 goal 图对称。

### 两种信号分开放
| 信号 | 进 encoder(emb)吗 | 为什么 |
|---|---|---|
| **腕部相机 eye_in_hand** | ✅ **进** | 它是**视觉**,goal 图也有(腕部视角的关好抽屉),两边对称可匹配——抽屉的核心修法 |
| **proprio(夹爪/关节)** | ❌ **不进 goal 匹配的 emb** | 非视觉,goal 图没有它 → 进了破坏 cost |

**proprio 想用,只当"当前态旁路条件"**(规划时当前帧永远拿得到),喂给:① predictor `predict(emb, proprio, action)` 帮动力学;② value 头 `V(emb_t, proprio_t, goal_emb)`——**只喂当前态一侧,goal 一侧仍纯视觉**。即 proprio 是"条件",不是"被匹配的表征"。

### 改动(`Embedder` 输入)
- **双相机进 emb**:agentview + eye_in_hand 各过 ViT(可共享权重)→ token 序列**拼接**。**emb 保持纯视觉。**
- **两损失原样保留**(next-emb MSE + SIGReg)——**只改视觉输入模态,不加 loss**,核心稳定性声明不受影响。
- proprio **不进 emb**;若后续要用,作旁路条件(见上)。

**合规**:✅ 不碰两损失结构。

### 先只加腕部相机,proprio 很可能冗余
腕部相机**近距离能看见夹爪手指**——探测3 的 gripper R²=0.18 是单 agentview **远看**的结果,腕部近看会大幅改善。所以:
- **先只加 eye_in_hand 重训** → 跑探测3 看夹爪 R²。
- 升了 → proprio 对 emb **冗余,连旁路都不用加**,最干净。
- 没升 → 再把 proprio 当旁路补给 predictor/value(仍不进 goal 匹配 emb)。

**验证闸门(必须跑)**:
| 探测 | 看什么 | 通过标准 |
|---|---|---|
| 探测3(加 drawer_qpos 靶子) | 抽屉/夹爪 R² | 抽屉 R² 明显 >0,夹爪从 0.18 上去 |
| 探测2 | cost 单调性 | cross ρ 更负、假谷数下降 |
| 探测6-lewm | 完美物理+新cost 成功率 | 从 0% 上来 |

**三种闸门结果 → 分支**:
- R² 升 **且** cost 单调 → 第2层不严重,**纯世界模型可能就够**,可暂缓阶段2。
- R² 升 **但** cost 仍不单调 → 第1层 OK、第2层卡着 → **进阶段2**。
- R² **没升** → 两损失没把抽屉挤进 emb → 需阶段3 的辅助头。

**风险**:next-emb MSE + SIGReg 不显式奖励"编码抽屉",腕部相机只是**间接**逼它学 → 不保证,故必须验。

---

## 阶段 2 — td_jepa 后继 value 当规划 cost（修第2层，td_jepa 正位）

**目标**:用学出来的 shaped value-to-go 替换裸几何距离,给 CEM 一路下坡的梯度。

**模块**:`GoalValue(emb_t, [proprio_t,] goal_emb) → V`(标量 value-to-go;或后继特征 ψ + goal 导出的 z)。**goal 一侧永远纯视觉**(goal_emb=encode(goal图));proprio 只作**当前态**旁路条件,可选。

**训练(TD bootstrap,离线)**:
```
td_target = r + γ · V_target(emb_{t+1}, goal_emb)        # 未到目标
          = 0                                            # 到目标
r 取每步 -1(或代价+1)→ V 即"折扣化剩余步数",天生单调
```
- **EMA target 网络** `V_target` 稳定自举。
- **goal relabeling(hindsight)**:同一轨迹里把未来某帧当 goal,正/负样本对。
- **建议 SARSA-style on-dataset 动作**(不对动作取 max)→ V ≈ 专家 cost-to-go,**避开离线 OOD 高估**,简单稳定;够当 cost 用(策略是 CEM,不靠 V 选动作)。
- **encoder detach**:emb 来自阶段1 冻结的 encoder,TD 梯度**不回传**→ 两损失核心不污染。

**接进 CEM(只换打分)**:
```
原:cost(候选) = ‖encode(想象末帧) − goal_emb‖²
新:cost(候选) = V(想象末帧 emb, goal_emb)
```
CEM、predictor rollout **全不变**。

**为什么是 shaping**:V 递归定义为"剩余步数",每接近目标一步降~1,沿任意通往目标的路单调——正是探测6 privileged(reach+close)那种 shaping,但**学出来**。

**白送 zero-shot**:`M` 是 goal_emb 的函数 → **任意新目标 encode 后即插即用规划,无需重训**(td_jepa 的回报)。仅 goal 条件(le-wm 无 reward)。

**必须在阶段1 之后**:value 头建在盲 emb 上会继承盲区(探测6 教训)。

**验证**:探测2(V 沿真轨迹单调吗)+ 探测6式(V 当 cost 的成功率)+ 固定 CEM 预算下 vs baseline。

---

## 阶段 2′ — 子目标规划（零学习替代，先白嫖）

**目标**:不学任何东西就修第2层 shaping,作为阶段2 的廉价对照/前哨。

**做什么**:
- 检索一条参考专家 demo(其末帧=任务目标的那条,或最近邻)。
- 子目标 = 参考 demo 里**比当前进度靠前 Δ≈H 步**的那帧 emb(或"最近点+前瞻":先找参考里 emb 离当前最近的帧,再 +Δ)。
- CEM 朝**近子目标**规划,保留现有 `‖emb−goal‖²` cost;走着推进子目标指针。

**为什么有效**:近子目标 → emb 距离**局部**有效,绕过 same-goal −0.30 的远目标问题。

**同样依赖阶段1**:盲 emb 上近子目标也看不见抽屉。

**建议**:阶段1 后**先白嫖这个**(零成本),看 shaping 是不是就够;不够再上阶段2。

---

## 阶段 3 — 改造 C：value 目标折进重训，塑 metric emb（可选，消融）

**触发条件**:阶段2 的 value 在 emb 上**难学**(emb 是为 MSE+SIGReg 塑的,非 value 可解码)。

**做什么**:重训时加辅助 loss,让训 V 的梯度**回传进 encoder**,把 emb 塑成"距离=进度"的度量空间(metric/temporal-contrastive/value-decodability loss),第1+2层一锅端、V 也好学。

**代价**:加第三个 loss = **碰两损失纯净性**。重训便宜 → 做**"带/不带辅助 loss"消融**,对比稳定性与成功率。**不是默认,是实验。**

---

## 4. 否决清单（明确不做）

| 方向 | 否决理由 |
|---|---|
| 治复合误差(num_preds>1、引 td_jepa 治 compounding) | 探测5+6 排除动力学 |
| 加大 CEM 预算 | 探测1+6 证非瓶颈 |
| 改造 A(goal-cond BC actor) | 可行(BC 95%)但退化成反应式、放弃世界模型卖点,与战略不符 |
| 时间距离回归 cost(直接监督剩余步数) | 策略相关(学死专家节奏)、离分布无标签、时间≠任务度量;正是改造B 优于它的理由 |

---

## 5. 远期研究 — online（边跑边训）

- 与 zero-shot **方向相反**(zero-shot=测试不训;online=测试继续训)。把 le-wm+value 推成 **TD-MPC2 式在线智能体**,补"离分布 value 不准"的缺口(CEM 专钻没见过的状态)。
- 代价:请回 le-wm 回避的不稳定性 + 探索/安全。**留作阶段3 之后;value 在线更新严格隔离;不与当前 cost 修复混做。**

---

## 6. 评测协议（统一,避免误判）

- **固定 CEM 预算**下比成功率,**不只比速度**(探测1 教训:加预算能小幅提升,会污染对比)。
- 各条件**共享 goal 和初始状态**(参照 `p6_oracle.py` 的 `goal_eps`/`init_idx` 做法)。
- `n_eval ≥ 20`;报成功率 + 完成步数。
- 每个阶段都回跑相关探测当回归测试(探测2/3/6)。

---

## 7. 接口障碍（来自 CLAUDE.md，提前规划）

1. **无 reward** → FB 式 goal-conditioning,z 由 goal_emb 导出,**不做 reward_inference lstsq**。
2. **frameskip × action_dim** 适配 le-wm 的 `Embedder`。
3. 栈是 **Lightning + Hydra + stable_pretraining/stable_worldmodel**:加模块/loss,**不要移植到本仓(td_jepa)的 Pydantic 配置**。
4. **两损失核心隔离**:改动2 的 TD/EMA 单独 optimizer + 单独 target 网络,encoder detach;改动3 才允许梯度进 encoder,且仅作消融。

---

## 8. 里程碑顺序

```
M0 数据:双相机+proprio+drawer_qpos 重采集                    → 产出 v2.h5
M1 重训 encoder(腕部相机进视觉emb;proprio只当旁路,同两损失) → 闸门 探测3/2/6
M2 (先)子目标规划白嫖  +  (后)td_jepa value 当 cost          → 固定预算评测 vs 10%
   └ 白送:zero-shot 跨目标泛化验证
M3 (条件触发)改造C metric化 + 消融
M4 (远期)online / TD-MPC2
```
**起点 = M0+M1**:风险最低、且是整条路绕不开的前提。
