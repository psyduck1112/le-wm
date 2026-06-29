"""Step 1 验证: ContrastiveHDF5Dataset 配对采样是否正确.

检查: 近帧 |Δt|≤w_near, 远帧(has_far时) |Δt|≥w_far, pos/neg 形状=(C,H,W),
      ep_idx 在范围内, has_far 比例合理.  transform=None (只验采样, 不验图像预处理).
"""
import sys, os                                          # sys 用于改 import 路径; os 用于拼路径
# 把仓库根目录 (本文件的上上级) 加进 import 搜索路径, 这样能 import 根目录的 contrastive_data
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                      # 批量统计 (min/max/mean/比例) 用
import torch                                            # dataset 返回 torch 张量, 取 shape 用
from contrastive_data import ContrastiveHDF5Dataset     # Step 1 写的子类

# 要加载的列: 双相机图像 + 动作 + 物理量 (和训练 config 一致)
KEYS = ["pixels", "eye_in_hand", "action", "proprio", "drawer_qpos"]
# 构造数据集对象 (单个 h5 成员, 不拼 ConcatDataset, 验采样足够)
ds = ContrastiveHDF5Dataset(
    name="libero_bc_drawer_v2",                         # h5 文件名 (无扩展名)
    cache_dir="/home/yikang/git/le-wm/data",            # 文件所在目录 (软链 -> hdd1)
    frameskip=5, num_steps=4,                           # 与训练一致: 跨20帧抽4帧
    keys_to_load=KEYS,                                  # 加载上面这些列
    keys_to_cache=["action", "proprio", "drawer_qpos"], # 小的数值列整列进内存, 读得快
    transform=None,                                     # 不套图像预处理, 只验采样逻辑+原始形状
)
ds.pair_sampling = True                                 # 打开配对采样 (否则退化为父类)
# 取出近/远阈值供断言用. w_near/w_far 是类属性 (定义在 contrastive_data.py 类体里),
# 通过实例 ds 可直接访问, 拿到 5 / 20; 这样断言和类定义同一套数, 不写死.
W_NEAR, W_FAR = ds.w_near, ds.w_far
# 打印数据集规模: 总窗口数 / episode 数 / 两个阈值
print(f"len(ds)={len(ds)}  n_episodes={len(ds.lengths)}  w_near={W_NEAR} w_far={W_FAR}")

# ---- 单样本细看 ----
s = ds[0]                                               # 取第 0 个样本 (一个 dict)
ep, start = ds.clip_indices[0]                          # 第 0 个窗口属于哪条 ep、起点在哪
t_a = start + (ds.num_steps - 1) * ds.frameskip         # anchor 窗口最后一帧的原始时刻
print(f"\n[sample idx=0] ep={ep} start={start} t_a={t_a} L={int(ds.lengths[ep])}")
# anchor 是 4 帧窗口, 形状应是 (4,C,H,W); 双相机各一份
print(f"  anchor pixels {tuple(s['pixels'].shape)}  eye {tuple(s['eye_in_hand'].shape)}")
# pos/neg 是单帧, 形状应是 (C,H,W)
print(f"  pos_pixels {tuple(s['pos_pixels'].shape)}  neg_pixels {tuple(s['neg_pixels'].shape)}")
# 打印两个间隔 + has_far + 全局 ep 编号
print(f"  pos_dt={s['pos_dt']}  neg_dt={s['neg_dt']}  has_far={s['has_far']}  ep_idx={s['ep_idx']}")

# ---- 批量统计 2000 个随机样本 ----
N = 2000                                                # 抽样个数
rng = np.random.default_rng(0)                          # 固定种子, 结果可复现
idxs = rng.integers(0, len(ds), size=N)                 # 随机抽 N 个窗口下标
pos_abs, neg_abs, far_flags = [], [], []                # 分别收集近间隔/远间隔/has_far
bad = 0                                                 # 形状异常计数器
for i in idxs:                                          # 遍历每个抽到的下标
    s = ds[int(i)]                                      # 取样本
    pd, nd, hf = abs(s["pos_dt"]), abs(s["neg_dt"]), bool(s["has_far"])  # 取绝对间隔+标志
    pos_abs.append(pd); neg_abs.append(nd); far_flags.append(hf)        # 收集
    # pos/neg 必须是单帧 (3,224,224); 不对就计数
    if tuple(s["pos_pixels"].shape) != (3, 224, 224): bad += 1
    if tuple(s["neg_pixels"].shape) != (3, 224, 224): bad += 1
    # 断言: 近帧间隔必须落在 [1, w_near] (≥1 排除 anchor 自己, ≤w_near 是"近")
    assert 1 <= pd <= W_NEAR, f"near gap {pd} 违反 [1,{W_NEAR}]"
    # 断言: 只要标了 has_far, 远帧间隔必须 ≥ w_far
    if hf: assert nd >= W_FAR, f"far gap {nd} < {W_FAR} 但 has_far=True"

pos_abs, neg_abs = np.array(pos_abs), np.array(neg_abs) # 转 numpy 便于统计
far_flags = np.array(far_flags)                         # 同上
print(f"\n[stats over {N} samples]")
# 近间隔的分布: 应全在 [1,5]
print(f"  pos |Δt|: min={pos_abs.min()} max={pos_abs.max()} mean={pos_abs.mean():.2f}")
# 远间隔的分布: has_far 的应 ≥20
print(f"  neg |Δt|: min={neg_abs.min()} max={neg_abs.max()} mean={neg_abs.mean():.2f}")
# has_far=True 比例: 多数应为 True, 少数短 ep 为 False
print(f"  has_far=True 比例: {far_flags.mean()*100:.1f}%  (False={int((~far_flags).sum())} 条, 短 ep)")
# 形状异常数: 必须 0
print(f"  形状异常计数: {bad}  (应为 0)")
# 看看降级 (has_far=False) 时实际取到的远间隔有多大 (验证降级也尽量取远)
print(f"  has_far=False 时 neg|Δt| 范围: "
      f"{neg_abs[~far_flags].min() if (~far_flags).any() else '-'}.."
      f"{neg_abs[~far_flags].max() if (~far_flags).any() else '-'}")
# 总判定: 形状没问题 + 上面断言没抛 = PASS
print("\nPASS ✅ (所有断言通过)" if bad == 0 else "\nFAIL ❌ 形状异常")
