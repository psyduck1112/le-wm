import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def masked_mse(pred, target):
    """逐元素 MSE, 忽略 target 中的 NaN (序列边界帧). pred/target: (..., P)."""
    mask = ~torch.isnan(target)                       # 有效位置=True
    target = torch.nan_to_num(target, 0.0)            # NaN->0 (被 mask 乘掉, 不影响)
    se = (pred - target).pow(2) * mask                # 无效位置的平方误差清零
    return se.sum() / mask.sum().clamp_min(1)         # 只对有效元素求均值


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    # physics aux loss: 共享 head 解码 emb 与 pred_emb 的物理量, 梯度回流塑形 latent.
    if self.model.task_head is not None:
        aux_w = cfg.loss.aux.weight
        # 标签已被 transform 逐维 z-score; 按 keys 顺序拼成 (B,T,P)
        y = torch.cat([batch[k] for k in cfg.loss.aux.label_keys], dim=-1)
        phys_enc = self.model.predict_state(emb)            # (B,T,P)  -> 回流 encoder
        phys_pred = self.model.predict_state(pred_emb)      # (B,Tc,P) -> 回流 predictor
        output["aux_enc_loss"] = masked_mse(phys_enc, y)
        output["aux_pred_loss"] = masked_mse(phys_pred, y[:, n_preds:])
        output["aux_loss"] = output["aux_enc_loss"] + output["aux_pred_loss"]
        output["loss"] = output["loss"] + aux_w * output["aux_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    # 主数据集 + extra_datasets (同 schema 的 .h5, 经 data/ 软链) 纵向拼接, 扩宽 (s,a) 分布.
    members = [swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)]
    for extra in cfg.data.get("extra_datasets", []) or []:
        kw = dict(cfg.data.dataset)
        kw["name"] = extra["name"]
        if extra.get("cache_dir"):
            kw["cache_dir"] = extra["cache_dir"]
        members.append(swm.data.HDF5Dataset(**kw, transform=None))

    if len(members) == 1:
        dataset = members[0]
    else:
        dataset = swm.data.ConcatDataset(members)
        dataset.get_dim = members[0].get_dim  # ConcatDataset 无 get_dim; schema 相同, 委托
        print(f"ConcatDataset: {len(members)} members "
              f"sizes={[len(m) for m in members]} total={len(dataset)}")

    # 图像列: pixels(agentview) + 可选 eye_in_hand(腕部相机, M1). 二者都走图像预处理,
    # 不能进下面的数值标准化循环 (会被当向量算 mean/std -> 毁图/崩溃).
    image_cols = [c for c in cfg.data.dataset.keys_to_load
                  if c.startswith("pixels") or c == "eye_in_hand"]
    transforms = [get_img_preprocessor(source=c, target=c, img_size=cfg.img_size)
                  for c in image_cols]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col in image_cols:
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    # ConcatDataset.__getitem__ 委托给子数据集, 用的是子数据集各自的 transform,
    # 所以要设到每个成员上 (而不是只设 concat 本身).
    for m in members:
        m.transform = transform
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim # 5帧动作拼接

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    # 多相机: 每路 ViT 出一个 hidden_dim 的 CLS, 在 encode() 里拼接后过 projector.
    # 单相机数据集 (pusht/dmc/...) n_cams=1 -> input_dim 不变, 向后兼容.
    n_cams = len(image_cols)
    projector = MLP(
        input_dim=n_cams * hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    # physics task head: emb -> 物理量 (归一化空间). 联合训练, 梯度回流塑形 encoder+predictor.
    # 仅当 cfg.loss.aux 存在时启用 (其它数据集无 proprio/drawer 标签).
    task_head = None
    if cfg.loss.get("aux") is not None:
        phys_dim = sum(dataset.get_dim(k) for k in cfg.loss.aux.label_keys)
        task_head = MLP(
            input_dim=embed_dim,
            output_dim=phys_dim,
            hidden_dim=cfg.loss.aux.hidden_dim,
        )  # 默认 LayerNorm+GELU; 逐帧解码, 不用 BatchNorm1d
        print(f"task_head: {embed_dim} -> {cfg.loss.aux.hidden_dim} -> {phys_dim}  "
              f"(label_keys={list(cfg.loss.aux.label_keys)})")

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        task_head=task_head,
    )

    # opt-in: 从已有权重初始化 (微调). cube co-train 从官方 object ckpt 起步;
    # task_head 不在源权重里 -> strict=False -> 保持随机初始化. drawer 默认 init_from 缺省 -> 跳过.
    init_from = OmegaConf.select(cfg, "init_from")
    if init_from:
        obj = torch.load(init_from, map_location="cpu", weights_only=False)
        src_sd = obj.state_dict() if hasattr(obj, "state_dict") else obj
        missing, unexpected = world_model.load_state_dict(src_sd, strict=False)
        head_missing = [k for k in missing if k.startswith("task_head")]
        print(f"[init_from] {init_from}: missing={len(missing)} unexpected={len(unexpected)}")
        print(f"[init_from] task_head stays random ({len(head_missing)} keys); "
              f"non-head missing={[k for k in missing if not k.startswith('task_head')][:5]}")
        assert not unexpected, f"unexpected keys when loading init_from: {unexpected[:5]}"

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(cfg.get("output_dir", swm.data.utils.get_cache_dir()), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
