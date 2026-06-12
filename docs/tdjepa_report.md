# TD-JEPA 讲解 + 结合 LeWM 代码/数据的落地改造方案（中英对照）
# TD-JEPA Explainer + Code-/Data-Grounded Integration Plan for LeWM (Bilingual)

> 论文 / Paper：*TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning*，arXiv:2510.00739（2025-10），Bagatella, Pirotta 等，Meta。ICLR 2026 **Oral**。
> 官方代码 / Official code：`github.com/facebookresearch/td_jepa`（PyTorch, uv, CC BY-NC 4.0, 无预训练 checkpoint / no pretrained checkpoints）。
> 写于 / Written：2026-06-11，对照本仓库 `jepa.py` / `module.py` / `train.py` / `value/` / `diagnostics/`。

---

# Group-Meeting TL;DR (one page)
<sub>组会速览 —— 英文为投影/口述用；中文小字仅供我自己理解</sub>

**In one sentence.** TD-JEPA pretrains on **offline, reward-free** data so that, given **any** reward at test time, it returns a near-optimal policy with **no training and no planning** (zero-shot RL).
<sub>离线无奖励预训练 → 测试时任意 reward 直接零样本出策略。</sub>

**Why zero-shot is possible — the one key equation.** For any policy and any reward, the Q-value factorizes into "dynamics" × "task":

$$Q^\pi_r(s,a) = F^\pi(s,a)^\top\, z_r$$

- `F^π` — **successor features**: the expensive-to-learn, **reward-agnostic long-horizon dynamics** representation.
- `z_r` — a **low-dim vector encoding only the reward**. New task = new z → hence zero-shot.

<sub>解耦：动态 F 与任务 z 分离，换任务只换 z。这是 zero-shot 的根。</sub>

**How F is learned — the core loss (a Bellman / TD bootstrap in latent space).**

$$\mathcal L = \big\| \,T(\phi(s),a,z) - \underbrace{\mathrm{sg}[\psi(s')]}_{\text{immediate}} - \underbrace{\gamma\,\mathrm{sg}[T(\phi(s'),a',z)]}_{\text{discounted future}}\, \big\|^2,\quad a'\sim\pi_z$$

`sg` = stop-gradient. Current successor feature = current task feature + discounted next step. **Captures long-horizon value without any rollout.**
<sub>latent 空间的 TD 自举，不靠 rollout 拿长 horizon。</sub>

**How it's used at test time — two steps.** ① linear-regress $z_r=(\psi^\top\psi)^{-1}\psi^\top r$; ② run the pretrained policy $\pi_{z_r}$ directly. **No CEM, no planning.**
<sub>回归 z_r → 跑预训练策略。</sub>

**Four components.** state encoder φ, task encoder ψ, policy-conditioned multi-step predictor T (≈ F), and a family of policies {π_z}.
<sub>φ / ψ / T / π_z 四件套。</sub>

**Results in one line.** Dominant on locomotion/control (DMC pixels 628.8; Walker nearly doubles FB); only on-par with BYOL-γ on manipulation goal-reaching (OGBench).
<sub>强在 control，弱在 manipulation goal-reaching —— 而 drawer 属后者。</sub>

**Relevance to LeWM (one line).** It replaces our pain point — **hand-written cost + a test-time CEM that reward-hacks in OOD** — with a **TD-learned value F^⊤z + a pretrained policy**; exactly the "learned V(emb;z) + model-based RL" route in CLAUDE.md. **But it does not fix offline data coverage (the OOD root cause) — that still needs DAgger self-collected data.**
<sub>把"手写 cost + CEM 在 OOD reward hacking"换成"学到的 value + 预训练策略"；但不解决数据覆盖，OOD 根因还得 DAgger。</sub>

---

# Part 1 · TD-JEPA 原理 / How TD-JEPA Works

## 1.1 解决的问题：zero-shot RL / The problem: zero-shot RL

**中文**：在**离线、无奖励**数据上预训练表征 + 一族策略，使测试时给**任意** reward 函数，**不再训练、不再 planning** 就能给出近优策略。理论根基是 successor measure / forward-backward（FB）：任意策略 π、任意奖励 r 的 Q 值可写成 `Q^π_r(s,a) = F^π(s,a)ᵀ z_r`。`F^π`（后继特征，与奖励无关的长期动态表征）与 `z_r`（只编码奖励的低维向量）解耦——换任务只换 `z_r`，这就是 zero-shot 的来源。

**EN**: Pretrain representations + a family of policies on **offline, reward-free** data so that, given **any** reward at test time, a near-optimal policy is produced with **no further training and no planning**. The foundation is the successor measure / forward-backward (FB) framework: for any policy π and reward r, the Q-value factorizes as `Q^π_r(s,a) = F^π(s,a)ᵀ z_r`. The reward-agnostic long-horizon dynamics `F^π` (successor features) and the reward-only low-dim vector `z_r` are decoupled — switching tasks only swaps `z_r`, which is exactly where zero-shot comes from.

## 1.2 四个组件 / The four components

| 组件 / Component | 符号 / Symbol | 作用 / Role |
|---|---|---|
| State encoder | φ: S→ℝ^dφ | 观测→latent / observation → latent |
| Task encoder | ψ: S→ℝ^dψ | 定义奖励/特征空间 / defines the reward/feature space |
| Policy-conditioned 多步预测器 / multi-step predictor | T(φ(s),a,z) | 预测随策略 z 变化的长期 latent 动态 ≈ F^{π_z} / predicts policy-z-dependent long-horizon latent dynamics ≈ successor features |
| 一族参数化策略 / family of policies | {π_z} | 每个任务向量一个策略，直接在 latent 里学 / one policy per task vector, learned directly in latent space |

后继测度双分解 / Successor-measure factorization: `M^{π_z} ≈ φ Tz ψᵀ`（多策略、多步 / multi-policy, multi-step）。

## 1.3 核心：TD latent-predictive loss（Eq. 9）/ The core loss

```
L_TD-JEPA = E_{(s,a,s')~D, a'~π_z} ‖ T(φ(s),a,z) − sg[ψ(s')] − γ·sg[T(φ(s'),a',z)] ‖²
```

**中文**：`sg[·]` 是 stop-gradient / target 网络。直觉：latent 空间的 Bellman 方程——当前后继特征 = 当前 task 特征 + 折扣后的下一步后继特征。**不靠 rollout 就拿到折扣意义下的长 horizon 信息**。策略侧用 TD3 风格 actor：π_z 最大化 `T(φ(s),a,z)ᵀz`。

**EN**: `sg[·]` is stop-gradient / a target network. Intuition: a Bellman equation in latent space — current successor feature = current task feature + discounted next-step successor feature. It **captures discounted long-horizon information without any rollout**. The policy side uses a TD3-style actor: π_z maximizes `T(φ(s),a,z)ᵀz`.

## 1.4 防坍缩 / Avoiding collapse

**中文**：正交归一正则 `L_REG`（惩罚 φ,ψ 协方差非对角项）+ identity 初始化 + target network。Thm 2：合适初始化下协方差沿训练保持常数，最优预测器恢复后继测度低秩分解。（对应 LeWM 的 SIGReg 角色，但更贴 FB 结构。）

**EN**: Orthonormality regularizer `L_REG` (penalizes off-diagonal terms of the φ, ψ covariances) + identity initialization + target network. Theorem 2: with proper init the covariances stay constant over training and the optimal predictor recovers the low-rank factorization of the successor measure. (Plays the role LeWM's SIGReg does, but tailored to the FB structure.)

## 1.5 测试时 zero-shot / Zero-shot at test time

**中文**：给 reward r + 一小批 `{(s,r)}`：① 线性回归 `z_r = (ψᵀψ)⁻¹ ψᵀ r`；② 直接部署 `π_{z_r}`。无 CEM、无 planning。Thm 4：策略评估误差被后继测度近似损失上界控制。

**EN**: Given a reward r and a small batch `{(s,r)}`: ① linear-regress `z_r = (ψᵀψ)⁻¹ ψᵀ r`; ② deploy `π_{z_r}` directly. No CEM, no planning. Theorem 4: the policy-evaluation error is upper-bounded by the successor-measure approximation loss.

## 1.6 与相邻方法 / Versus neighboring methods

| 方法 / Method | state enc | task enc | 目标 / Objective |
|---|---|---|---|
| **TD-JEPA** | 学 φ / learned | 学 ψ / learned | policy-conditioned 多步 TD latent 预测 / multi-step TD latent prediction |
| FB (Touati&Ollivier'21) | 无 / none | 学 / learned | 对比式双线性分解 / contrastive bilinear decomposition |
| HILP (Park'24) | 无 / none | 学 / learned | 保时序距离的 goal-reaching / temporal-distance-preserving goal-reaching |
| BYOL-γ (Lawson'25) | 学 / learned | 无 / none | 多步行为动态（不 condition 策略）/ multi-step behavioral dynamics (policy-unconditional) |

卖点 = **policy-conditioned** / Key edge = **policy-conditioned**：预测任意 z 策略的动态，不止数据里行为策略 / predicts dynamics of any policy z, not just the behavioral policy in the data.

## 1.7 实验与数字 / Experiments and numbers

**中文**：13 数据集 / 65 任务，ExORL + OGBench，含 from-pixels。
- DMC RGB（归一 1000）：TD-JEPA **628.8** > BYOL-γ 582.4 > FB 456.2 > Laplacian 293.1。Walker 738.9 vs FB 324.4。
- OGBench RGB 成功率：TD-JEPA **41.34** ≈ BYOL-γ 41.58（**略输**）> FB 39.89。
- 控制类统治性强；**纯 goal-reaching 上和 BYOL-γ 打平甚至略输**。
- Fast adaptation：zero-shot 策略当初始化再 TD3 微调，比从零快很多到同等渐近性能。

**EN**: 13 datasets / 65 tasks, ExORL + OGBench, including from-pixels.
- DMC RGB (normalized to 1000): TD-JEPA **628.8** > BYOL-γ 582.4 > FB 456.2 > Laplacian 293.1. Walker 738.9 vs FB 324.4.
- OGBench RGB success rate: TD-JEPA **41.34** ≈ BYOL-γ 41.58 (**slightly behind**) > FB 39.89.
- Dominant on control; **roughly tied with, or slightly behind, BYOL-γ on pure goal-reaching**.
- Fast adaptation: using the zero-shot policy as init and fine-tuning with TD3 reaches the same asymptotic performance much faster than from scratch.

## 1.8 缺陷 / 局限 / Flaws and limitations

**中文**：
1. 理论假设强（正交归一、对称转移、均匀状态分布），实操只是近似。
2. **不消灭离线分布漂移**——把"测试时优化器钻漏洞"换成"离线 RL 协变量漂移"，仍依赖数据行为覆盖度。窄数据（专家 BC、无恢复演示）下 zero-shot 策略一样退化。
3. **线性奖励假设**（z_r 线性回归）→ 复杂/非线性奖励拟合不出，表达力天花板。
4. 训练复杂度远高于 LeWM：4 网络 + actor + 一族策略联合训，要对付 TD 致命三件套、target net、正则权衡——放弃 LeWM "两 loss、15M 参数、单卡几小时" 的极简。
5. 纯 goal-reaching 优势不明显。

**EN**:
1. Strong theoretical assumptions (orthonormality, symmetric transitions, uniform state distribution); in practice only approximate.
2. **Does not eliminate offline distribution shift** — it swaps "the test-time optimizer exploiting holes" for "offline-RL covariate shift," still depending on the data's behavioral coverage. With narrow data (expert BC, no recovery demos) the zero-shot policy degrades just the same.
3. **Linear-reward assumption** (z_r via linear regression) → complex/non-linear rewards can't be fit; an expressivity ceiling.
4. Far higher training complexity than LeWM: jointly trains 4 networks + an actor + a family of policies, fighting the deadly triad, target nets, and regularizer trade-offs — abandoning LeWM's "two losses, 15M params, single GPU in hours" simplicity.
5. Advantage on pure goal-reaching is marginal.

## 1.9 复现情况 / Reproduction status

**中文**：官方开源、条件好：`uv sync --all-extras`；自带 baseline FB/HILP/ICVF/Laplacian/RLDP/BYOL/BYOL-γ；支持状态 + RGB；ExORL/DMC + OGBench 自动下载处理脚本；本地或 slurm/exca。**无预训练权重**，需自行预训练；license CC BY-NC（非商用）。ICLR 2026 Oral。

**EN**: Officially open-sourced with good ergonomics: `uv sync --all-extras`; ships baselines FB/HILP/ICVF/Laplacian/RLDP/BYOL/BYOL-γ; supports state + RGB; auto download/processing scripts for ExORL/DMC + OGBench; runs locally or via slurm/exca. **No pretrained weights** — you must pretrain yourself; license is CC BY-NC (non-commercial). ICLR 2026 Oral.

---

# Part 2 · 结合 LeWM 实际代码与数据的落地方案 / Code-/Data-Grounded Integration Plan

## 2.0 现状对位 / Mapping to the current codebase

**中文**：LeWM 当前管线（见 `jepa.py`）：

**EN**: LeWM's current pipeline (see `jepa.py`):

```
encode(): pixels(+eye_in_hand) → ViT-tiny CLS ×N相机/cams → projector(MLP) → emb (B,T,192)
predict(): ARPredictor(emb, act_emb) 一步预测下一 emb / one-step next-emb prediction   # module.py
rollout(): history_size=3, 自回归滚 T 步 / autoregressive T-step rollout               # jepa.py:71
get_cost()/criterion(): cost = ‖rollout 末帧 emb − goal_emb‖² / MSE of final emb to goal # jepa.py:122-167  ← 瓶颈/bottleneck
CEM: eval.py / p6_oracle.py 在 cost 上搜动作 / searches actions on this cost
```

**中文**：`train.py:lejepa_forward` 训练 loss 只有两项：

**EN**: `train.py:lejepa_forward` has only two loss terms:

```python
output["pred_loss"]   = (pred_emb - tgt_emb).pow(2).mean()
output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
output["loss"]        = pred_loss + λ·sigreg_loss          # train.py:40-42
```

**中文**：你在 `CLAUDE.md` 已坐实的结论链：
- bottleneck = **cost**，不是 predictor/数据（p6：oracle dynamics + privileged cost = 100%，+ lewm cost = fail）。
- CEM 是优化器，在 OOD 上 **reward hacking** 你手写的 `decoded_cost`（`value/state_decoder.py`），`corr(true_reach, D_φ_reach) = −0.557`。
- 目标路线：把手写 distance cost 升级成 **distill 自 p6 privileged cost 的学到 `V(emb; z)`**，z = 任务向量 → 正式成 model-based RL。TD-JEPA 正面命中这条路线。

**EN**: The conclusion chain you already nailed down in `CLAUDE.md`:
- The bottleneck is the **cost**, not the predictor/data (p6: oracle dynamics + privileged cost = 100%, + lewm cost = fail).
- CEM is an optimizer that **reward-hacks** your hand-written `decoded_cost` (`value/state_decoder.py`) in OOD regions: `corr(true_reach, D_φ_reach) = −0.557`.
- Target trajectory: upgrade the hand-written distance cost into a **learned `V(emb; z)` distilled from p6's privileged cost**, with z a task vector → formally turning this into model-based RL. TD-JEPA hits exactly this trajectory.

## 2.1 Stage 1（推荐先做）：TD value head，frozen encoder，drop-in 换 cost / TD value head, frozen encoder, drop-in cost swap

**中文**：保持 LeWM world model 冻结（沿用 `_common.load_jepa` / epoch-200 双相机 ckpt），只在 frozen 192-d emb 上新增 TD-JEPA 式 value 头，替换 CEM 的 cost。这是把 `value/state_decoder.py` 的 `decoded_cost` 升级成"TD 自举的学到 value"，最贴 `CLAUDE.md` 的"distill V(emb;z)"，且最省算力（复用 `value/emb_cache.npz`，不碰 ViT）。

**EN**: Keep the LeWM world model frozen (reuse `_common.load_jepa` / the epoch-200 dual-camera ckpt) and add only a TD-JEPA-style value head on the frozen 192-d emb to replace the CEM cost. This upgrades `value/state_decoder.py`'s `decoded_cost` into a "TD-bootstrapped learned value," closest to `CLAUDE.md`'s "distill V(emb;z)" and cheapest in compute (reuse `value/emb_cache.npz`, never touch the ViT).

### 数据 / 标签来源（已有）/ Data and labels (already available)

**中文**：
- 离线 transition：`(emb_t, action_t, emb_{t+1})`，从 `emb_cache.npz`（frozen 双相机 emb，含 `ep_offset/ep_len`）+ `DrawerH5` 的 `action`（LIBERO 7-D；OGBScene 5-D）按 episode 切相邻帧。
- **即时奖励**：用 p6 的 shaped reward `r = −(reach + 30·|q−closed|)`（`p6_oracle.privileged_cost` 的相反数）。LIBERO 用 `DrawerH5.drawer_qpos` + `proprio[:,0:3]`(eef)；OGBScene 用列 `privileged_drawer_pos` / `privileged_target_drawer_pos` / `success`。

**EN**:
- Offline transitions `(emb_t, action_t, emb_{t+1})` from `emb_cache.npz` (frozen dual-cam emb with `ep_offset/ep_len`) + `DrawerH5`'s `action` (LIBERO 7-D; OGBScene 5-D), sliced into adjacent frames per episode.
- **Immediate reward**: p6's shaped reward `r = −(reach + 30·|q−closed|)` (the negation of `p6_oracle.privileged_cost`). LIBERO: `DrawerH5.drawer_qpos` + `proprio[:,0:3]` (eef); OGBScene: columns `privileged_drawer_pos` / `privileged_target_drawer_pos` / `success`.

### 最小实现（单任务，先固定 z）/ Minimal implementation (single task, fixed z for now)

新增 / Add `value/td_value.py`:

```python
class TDValue(nn.Module):
    """V(emb) 头 / head：frozen 192-emb → 标量 cost-to-go。先单任务 / single-task first，z 省略 / z omitted."""
    def __init__(self, emb_dim=192, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))
    def forward(self, emb):                      # (...,192) -> (...,)
        return self.net(emb).squeeze(-1)
```

TD 训练（仿 `train_state_decoder.py` 的 emb_cache + episode-split，新增 target net）/ TD training (mirror `train_state_decoder.py`'s emb_cache + episode-split, add a target net):

```
# r_t = -(reach_t + 30*|q_t - CLOSED|)        离线即时奖励 / offline immediate reward
# V_target = r_t + γ * V_tgt(emb_{t+1})        末帧 bootstrap 截断 / truncate bootstrap at last frame
# L = ( V(emb_t) - sg[V_target] )²
# 部署 / deploy: cost = -V(emb)   （CEM 最小化 cost = 最大化 value / minimizing cost == maximizing value）
```

**中文 — 关键诚实点**：这步**没有引入策略**，仍走 CEM，本质是"更好的 cost"。单纯换成学到的 V **不会自动**消除 reward hacking——CEM 仍会在 OOD 钻 V 的洞。所以 Stage 1 必须配下面之一，否则只是把 `decoded_cost` 的洞换个形状：
- **悲观正则**（Phase 1，MOPO/MOReL 式）：训 V 的 ensemble，部署 cost = `-mean(V) + β·std(V)`。
- **CEM trust region**（Phase 3）：`p6_oracle.oracle_cem` 里限制 `cand` 单步偏离流形的幅度。

**EN — key honesty note**: This step introduces **no policy** and still runs CEM, so it's fundamentally "a better cost." Simply swapping in a learned V does **not** automatically remove reward hacking — CEM will still exploit holes in V under OOD. So Stage 1 must be paired with one of the following, or it just reshapes `decoded_cost`'s holes:
- **Pessimism regularization** (Phase 1, MOPO/MOReL-style): train an ensemble of V, deploy cost = `-mean(V) + β·std(V)`.
- **CEM trust region** (Phase 3): in `p6_oracle.oracle_cem`, cap how far `cand` can step off the manifold per step.

### 嫁接点（改动很小）/ Hook points (tiny diff)

**中文**：
1. `value/td_value.py`（新）+ `value/train_td_value.py`（仿 `train_state_decoder.py`）。
2. `diagnostics/p6_oracle.py`：`--cost` 增加 `"sf"` 分支，仿现有 `decoded_costs()`（`p6_oracle.py:94-98`）：
3. 验证用你已有的 cost-agnostic 三闸。

**EN**:
1. `value/td_value.py` (new) + `value/train_td_value.py` (mirror `train_state_decoder.py`).
2. `diagnostics/p6_oracle.py`: add a `"sf"` branch to `--cost`, mirroring `decoded_costs()` (`p6_oracle.py:94-98`):
3. Validate with your existing cost-agnostic three gates.

```python
def sf_costs(model, value, frames, eye_frames):
    emb = encode_frames(model, np.stack(frames), np.stack(eye_frames)).cuda()
    return (-value(emb)).cpu().numpy()           # cost = -V
# 在 oracle_cem 的 cost 选择处 / in oracle_cem's cost selection:
# elif cost_mode == "sf": costs = sf_costs(...)
```

**验证阶梯 / Validation ladder**（cost-agnostic，定义见 `CLAUDE.md` / defined in `CLAUDE.md`）：
- 闸1 单调性 / Gate 1 monotonicity：仿 `diagnostics/p2b_decoded_mono.py`，看 V 沿专家轨迹单调性、held-out cross ρ / check V's monotonicity along expert trajectories and held-out cross ρ.
- 闸2 oracle-dynamics CEM / Gate 2：`p6_oracle.py --cost sf`，对照同 config 的 `privileged`=100%（`decoded` 在此崩过）/ compare against `privileged`=100% under the same config (`decoded` collapsed here). **这是关键 go/no-go / This is the key go/no-go.**
- 闸3 真机 MPC / Gate 3：`diagnostics/p9_real_mpc.py`，n≥20，对照 baseline ~9–10% / against the ~9–10% baseline.

> **中文**：Stage 1 诚实定位——它把"手写 distance"升级成"TD 自举 + 可加悲观"的 value，是 `decoded_cost` 的正统替代；但只要还在跑 CEM，**OOD reward hacking 的根（离线数据覆盖）没动**。要真正拔掉，需要 Stage 3 的 amortized policy 或 DAgger 自采数据。
> **EN**: Honest positioning of Stage 1 — it upgrades "hand-written distance" to a "TD-bootstrapped + optionally pessimistic" value, a legitimate replacement for `decoded_cost`; but as long as CEM runs, **the root of OOD reward hacking (offline data coverage) is untouched**. Truly removing it needs Stage 3's amortized policy or DAgger-collected data.

## 2.2 Stage 2：引入 task encoder ψ 与任务向量 z / Introduce task encoder ψ and task vector z

**中文**：当目标从单一 drawer-close 扩到多 drawer 状态 / OGBScene 多子任务时，把 Stage 1 的固定单任务升级成 `F^{π}(s,a)ᵀ z`：
- 新增 `ψ`（小 MLP on frozen emb，dψ≈32–64）与后继特征头 `F(emb, act_emb) → ℝ^{dψ}`（可复用 `module.ARPredictor` 的 AdaLN 条件机制，把 action emb 当 condition）。
- TD 目标改为 Eq.9 latent 形式：`F(emb_t,a_t) ≈ sg[ψ(emb_{t+1})] + γ·sg[F(emb_{t+1},a_{t+1})]`。
- 测试时 `z_r = (ΨᵀΨ)⁻¹ Ψᵀ r`，用一小批 `(emb, r)` 拟合；cost = `−F(emb,a)ᵀ z_r`。
- ψ/F 用正交归一正则 + identity init 防坍缩；与 LeWM 的 SIGReg 并存（emb 端 SIGReg，ψ 端 TD-JEPA REG）。
- 数据列：OGBScene `observation(40)` / `privileged_*` 做 ψ 监督或 r 来源；LIBERO 沿用 `proprio` + `drawer_qpos`。

**EN**: When goals expand from single drawer-close to many drawer states / OGBScene subtasks, upgrade Stage 1's fixed single task into `F^{π}(s,a)ᵀ z`:
- Add `ψ` (small MLP on frozen emb, dψ≈32–64) and a successor-feature head `F(emb, act_emb) → ℝ^{dψ}` (reuse `module.ARPredictor`'s AdaLN conditioning, feeding the action emb as the condition).
- Change the TD target to the latent form of Eq. 9: `F(emb_t,a_t) ≈ sg[ψ(emb_{t+1})] + γ·sg[F(emb_{t+1},a_{t+1})]`.
- At test time `z_r = (ΨᵀΨ)⁻¹ Ψᵀ r`, fit on a small batch `(emb, r)`; cost = `−F(emb,a)ᵀ z_r`.
- Regularize ψ/F with orthonormality + identity init to avoid collapse; coexists with LeWM's SIGReg (SIGReg on the emb side, TD-JEPA REG on the ψ side).
- Data columns: OGBScene `observation(40)` / `privileged_*` as ψ supervision or reward source; LIBERO reuses `proprio` + `drawer_qpos`.

## 2.3 Stage 3（最 faithful，最重）：完整 TD-JEPA，amortized policy 替掉 CEM / Full TD-JEPA, amortized policy replaces CEM

**中文**：这是真正消除"CEM 钻 cost 洞"的一步——**测试时不再优化**：
- 在 latent 上加一族策略 `π_z(emb)`（TD3 actor），训练 `π_z` 最大化 `F(emb,a)ᵀz`。
- 部署：`z_r` 由 reward 回归 → 直接 `a = π_{z_r}(emb)`，闭环执行，无 rollout、无 CEM。
- LeWM 的 `predict/rollout`（`jepa.py:57-120`）此时退居二线，仅用于 fast-adaptation 或 model-based 数据增广。

**诚实警告**：
- 论文 1.8 局限 2：amortized policy 仍是**离线**学的，BC 数据（专家、窄、无恢复）下覆盖不足处照样退化。TD-JEPA 不替你解决协变量漂移。
- 你 `CLAUDE.md` 正确路线 = world model + 学到 value + **DAgger 自采数据**。TD-JEPA 给"学到 value/policy"那块的成熟实现；"DAgger 自采数据"（planner 自跑 → sim 打真标签 → 重训）仍要自己做，且是消 OOD 的关键。
- 工程上放弃 LeWM 极简性，引入 TD 致命三件套调参；建议**只有在 Stage 1/2 闸2 通过**后再上。

**EN**: This is the step that truly removes "CEM exploiting cost holes" — **no optimization at test time**:
- Add a family of policies `π_z(emb)` (TD3 actor) in latent space; train `π_z` to maximize `F(emb,a)ᵀz`.
- Deploy: regress `z_r` from reward → directly `a = π_{z_r}(emb)`, closed-loop, no rollout, no CEM.
- LeWM's `predict/rollout` (`jepa.py:57-120`) then steps back, used only for fast-adaptation or model-based data augmentation.

**Honest warning**:
- Paper limitation 2 (§1.8): the amortized policy is still learned **offline**; with BC data (expert, narrow, no recovery) it degrades wherever coverage is thin. TD-JEPA does not solve covariate shift for you.
- The correct route in your `CLAUDE.md` = world model + learned value + **DAgger-collected data**. TD-JEPA supplies the mature implementation of the "learned value/policy" piece; the "DAgger self-collection" piece (planner rolls out → sim labels truth → retrain) is still yours to build, and it is the key to killing OOD.
- Engineering-wise it abandons LeWM's simplicity and brings in deadly-triad tuning; recommend doing it **only after Stage 1/2 pass Gate 2**.

## 2.4 训练侧 loss 嫁接点（端到端联合训）/ Training-side loss hook (end-to-end joint training)

**中文**：若要把 TD loss 直接进 `train.py:lejepa_forward`（而非离线 emb_cache 路线），在 `output["loss"]` 处并入：

**EN**: If you want the TD loss directly inside `train.py:lejepa_forward` (instead of the offline emb_cache route), fold it into `output["loss"]`:

```python
# 现有 / existing: pred_loss + λ·sigreg_loss        (train.py:40-42)
# 新增 TD 项 / new TD term (needs action_{t+1} 与 reward/privileged 列 / columns in batch):
td = (F(emb_t, act_t) - (psi(emb_tp1).detach()
        + gamma * F_tgt(emb_tp1, act_tp1).detach())).pow(2).mean()
output["loss"] = pred_loss + λ·sigreg_loss + μ·td
```

**中文**：需要 ① `swm.data.HDF5Dataset` 的 `keys_to_load` 纳入 privileged/reward 列（`train.py:58` 的 image/数值分流已能处理新数值列）；② 维护 target 网络 EMA（LeWM 现无，需新增）；③ 对齐 `action_{t+1}`（注意 `train.py:26` 的 `nan_to_num` 边界处理）。
**EN**: You need ① `swm.data.HDF5Dataset`'s `keys_to_load` to include the privileged/reward columns (`train.py:58`'s image/numeric split already handles new numeric columns); ② maintain a target-network EMA (LeWM has none, must add); ③ align `action_{t+1}` (mind the `nan_to_num` boundary handling at `train.py:26`).

## 2.5 推荐执行顺序（务实）/ Recommended order of execution (pragmatic)

**中文**：
1. **Stage 1 + ensemble 悲观**，复用 `emb_cache.npz`，`p6_oracle.py --cost sf` 过闸2——最便宜最快验证"学到 value 是否真比 decoded 强"，一两天出 go/no-go。
2. 闸2 过 → 上 Stage 2 的 ψ/z；闸2 仍崩 → 问题确在数据覆盖，直接转 **DAgger 自采数据**（Phase 2），别在 cost 形式上继续打磨。
3. 验证 OK 后再考虑 Stage 3 的 amortized policy / 参考官方 `td_jepa` repo 的 OGBench `scene` 配置对齐细节。

**EN**:
1. **Stage 1 + ensemble pessimism**, reuse `emb_cache.npz`, pass Gate 2 via `p6_oracle.py --cost sf` — the cheapest, fastest test of "is the learned value actually better than decoded," go/no-go in a day or two.
2. Gate 2 passes → move to Stage 2's ψ/z; Gate 2 still collapses → the problem really is data coverage, pivot straight to **DAgger self-collection** (Phase 2) rather than polishing the cost form further.
3. After validation, consider Stage 3's amortized policy / align details with the official `td_jepa` repo's OGBench `scene` configs.

## 2.6 一句话定位 / One-line positioning

**中文**：TD-JEPA = "把 cost 从**测试时优化**变成**训练时学到的后继特征 value**，把**现场 CEM 搜索**变成**预训练好的策略**"。它给你 `CLAUDE.md` 目标路线里"learned `V(emb;z)` + model-based RL"那块成熟的实现与理论；但**离线数据覆盖度这个真正的 OOD 根因它不替你解决**——那仍需 DAgger / 在线自采数据。

**EN**: TD-JEPA = "turn the cost from **test-time optimization** into a **training-time learned successor-feature value**, and turn **on-the-fly CEM search** into a **pretrained policy**." It gives you the mature implementation and theory for the "learned `V(emb;z)` + model-based RL" piece of your `CLAUDE.md` route; but **the real OOD root cause — offline data coverage — it does not solve for you**, which still needs DAgger / online self-collected data.

---

## Sources / 来源
- arXiv:2510.00739 — https://arxiv.org/abs/2510.00739 ｜ HTML 全文 / full text https://arxiv.org/html/2510.00739v1
- OpenReview (ICLR 2026) — https://openreview.net/forum?id=SzXDuBN8M1
- 官方代码 / Official code — https://github.com/facebookresearch/td_jepa
- ICLR 2026 Oral — https://iclr.cc/virtual/2026/oral/10009367
- 本仓库锚点 / Repo anchors：`jepa.py`（encode/predict/rollout/get_cost）、`module.py`（ARPredictor/SIGReg）、`train.py:lejepa_forward`、`value/state_decoder.py`、`value/train_state_decoder.py`、`diagnostics/p6_oracle.py`、`diagnostics/_common.py`
