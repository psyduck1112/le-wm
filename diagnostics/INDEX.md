# Diagnostics Index / 诊断脚本权威清单

> 最后整理 / Last organized: **2026-06-22**
> 任务 / Task: LIBERO drawer-close, dual-cam JEPA (encoder + predictor + cost + CEM).
> 完整叙事见 / Full narrative: [`results/DIAGNOSTICS_REPORT.md`](results/DIAGNOSTICS_REPORT.md).

**状态标签 / Status tags**
- ✅ **现役 / live** — phys-head 期仍在跑，会持续维护。
- ✓ **结论已落 / settled** — 一次性探针，结论已写进 REPORT/memory，一般不再重跑。
- ⚠️ **证据基 / evidence base** — OOD/pessimism 调查，结论支撑当前方向（DAgger/悲观），保留作根据。
- 📦 **已归档 / archived** — 被取代，移入 `archive/`（脚本）或 `results/archive/`（产物），可逆。

**约定 / Conventions**
- 所有脚本 `import _common`、用 `./results` 相对路径 → **保持扁平，勿移动现役脚本**（移动会断 import / 产物路径）。
- GPU0 专用：`CUDA_VISIBLE_DEVICES=0`（GPU1/2 是别人的任务）。渲染需 `MUJOCO_GL=egl EGL_DEVICE_ID=0`。
- phys ckpt / decoder 加载必须 `weights_only=False`。

---

## 共享基建 / Shared infra

| File | Role |
|------|------|
| `_common.py` | 共用 helper（模型/数据加载、env 构建、编码）。几乎所有 pN 都 import 它。 |
| `make_report_figs.py` | 从 `results/*.npz` 重生成 `results/report_figs/fig1..5.png`（可复现）。 |
| `INDEX.md` | 本文件。 |
| `results/DIAGNOSTICS_REPORT.md` | 组会版完整报告（中英对照），phys-head 期已更新。 |

---

## 现役工具 / Live tools ✅

| Script | 干什么 / What | 关键产物 / Output | 最新结论 / Latest |
|--------|---------------|-------------------|-------------------|
| `p2_cost_mono.py` | cost 单调性（沿专家轨迹 on-tube）。统合旧 p2/p2b/p17。`--model phys\|v2 --cost lewm,phys,truth` | `results/cost_mono_phys.png` | phys ρ≈−0.93≈truth；lewm on-tube 由 +0.02→**−0.42**（被 head 间接抬升但仍不可用，6条仅1条<−0.8）。 |
| `p6_oracle.py` | 决定性实验：predictor→真 MuJoCo，只比 cost。`--cost lewm\|phys\|privileged`，`--cost-curve` 沿 planner 实轨记 plan-cost vs true-cost。 | `results/p6_oracle_phys.{log,npz}`、`results/costcurve_e177/` | privileged 100%；**phys 闸2 PASS**（e177 1/1, e173 1/3）；lewm 0%。cost-curve：phys ρ(plan,true)=**+1.00**（诚实），lewm **−0.55**（撒谎，复刻 D_φ −0.557）。 |
| `p15_attention.py` | ViT CLS attention rollout：state-emb 看图像哪块。双相机。`--ep` | `results/p15_attention_phys_ep0.png` | phys 版已出图（早→晚帧看注意力是否移到臂/抽屉）。 |

---

## 结论已落的一次性探针 / Settled probes ✓ (v1/v2 期)

| Script | 问题 / Question | 结论 / Conclusion | 产物 |
|--------|------------------|-------------------|------|
| `check_collapse.py` | 编码器坍缩？(纯 emb_cache，免 load) | 无坍缩（有效秩/谱正常）。 | — |
| `p0_sanity.py` | 模型练好了吗？单步TF误差 + 谱 | 单步 2.3%、有效秩 69/192 → 健康。 | `results/p0_sanity.npz` |
| `probe_decode.py` | **统合 p3/p3b/p3c** 的 emb→物理量解码 probe（ridge+MLP, R²）。`--target proprio\|full`、`--compare-cam`(mono vs dual ΔR²) | proprio R²≈0.77；reach R²=0.71、full priv 0.75；腕相机 ΔR² 全正、reach 最大 +0.091。 | `results/p3_probe.*`, `results/p3b_reach.*`, `results/p3c_cam_ablation.*` + `results/p3c_wrist_camera_report.md` |
| `p4_bc_eval.py` | 数据够吗？(BC baseline) | BC 95% → 数据足，瓶颈在 WM+planning。 | `results/p4_bc.log` |
| `p5_openloop.py` | 多步想象误差爆炸吗？ | 第10步~33%，线性增长不发散，始终优于 copy-last。 | `results/p5_openloop.{png,npz}` |
| `p7_subgoal.py` | 拆路标能救坏 cost 吗？ | 完美物理+4路标仍 0/8 → 救不了。 | `results/p7_subgoal.{log,npz}` |
| `p9_real_mpc.py` | 真实端到端成功率？ | ~10%（坏 cost 下的部署真值）。**注**：fk5 后 CEM 35维 macro-action 未接通，闸3 暂阻塞。 | `results/p9_real_mpc*.log`、`results/p9_videos/` |

---

## 数据扩张 / Data-expansion (perturb) — 产物

`utils/perturb_*` 系列（在 repo `utils/`，非本目录）的可视化产物落在这里：
`results/perturb_diversity*.png`、`results/perturb_montage_ep*.png`、`results/perturb_*.mp4`、`results/replay_ep2540.mp4`。
背景见 memory `project-data-expansion-perturb`。

---

## 已归档 / Archived 📦

> **D_φ 相关全部归档**（这条线大概率不再用；结论已写进 REPORT §4.2）。
> 注：`value/ood_detector.npz`（p10 产物）在 `value/` 目录、不在本目录，未动。

**`archive/`（脚本+旧文档）**
- **D_φ decode-then-cost 线**：`p2b_decoded_mono.py`(已 git 删除，并入 `p2_cost_mono.py`)、`p6c_video.py`、`p6d_video_dual.py`(D_φ 视频，被 `p6_oracle.py --cost-curve` + phys 视频取代)。
- **D_φ OOD/pessimism 簇**（Jun15，结论=detector 侧救不了 reach→必须 DAgger）：`p10_ood_detector.py`、`p11_ood_probe.py`、`p11_plot.py`、`p12_pessimistic.py`、`p13_knn.py`、`p14_source_ablation.py`。
- **被 `probe_decode.py` 取代的原始 decode-probe**：`p3_probe.py`、`p3b_reach.py`、`p3c_cam_ablation.py`（产生 REPORT 引用数字的原始脚本，保留备查）。
- **旧文档**：`DIAGNOSIS_RESULTS.md` / `OPTIMIZATION_PLAN.md`（2026-06-02 最早，被 REPORT 取代）。

**`results/archive/`（死蔵产物）**
- value-head 期：`p6_value*.log`、`p6_priv_*.log`
- p2b 孤儿：`p2b_decoded_mono.{log,npz,png}`
- D_φ 视频产物：`p6c_video.log`、`p6d_dual.{log,npz}`
- D_φ OOD 簇产物：`p10_ood_detector.npz`、`p11_ood_probe.{npz,png,log}`、`p11_rerun_emb.log`、`p12_pessimistic.{npz,log}`、`p13_knn.npz`、`p14_source_ablation.{npz,log}`
- 旧 p6 verbose run：`p6_epoch200_lewm.log`、`p6_lewm_strong.log`、`p6_oracle_full.log`、`p6_oracle.npz`(旧)
- 无对应脚本的旧 run：`p1_budget*.log`、`p1_timing.log`、`p6b_*.log`、`p7_run.out`

> `make_report_figs.py` 的 `load()` 已加 archive 回退：归档的 `p2b_decoded_mono.npz` / `p6d_dual.npz` 仍被 fig3/fig5 读取，图可正常重生。

---

## 视频目录 / Video dirs

| Dir | 内容 |
|-----|------|
| `results/p6c_videos/` | privileged SUCCESS / decoded FAIL / dual-overlay（D_φ 期，经典对照）。 |
| `results/p16_videos/` | phys oracle：lewm fail / phys SUCCESS / privileged。 |
| `results/p9_videos/` | p9 真实端到端 rollout。 |
| `results/costcurve_e177/` | e177 cost-curve：phys SUCCESS / lewm fail + 两张 costcurve png。 |
| `results/phys_e173/`, `phys_e177/`, `lewm_e177_b200/`, `lewm_e89/`, `oracle_phys_e89/` | 各 epoch/cost 的 oracle eval 录像。 |
| `results/report_figs/` | `fig1..5.png`（由 `make_report_figs.py` 生成）。 |
