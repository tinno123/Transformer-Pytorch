import os
import argparse
import random

import numpy as np
import yaml
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime

from Transformer.model import Transformer
from utils.dataset import BucketBatchSampler,MyDataSet,collate_fn
from utils.train_tools import NoamLR, GetOptimizer
import torch
from torch.amp import autocast
from utils.loss import Loss


TOKENS_PER_STEP = 500000
def train(model,train_dataloader,valid_dataloader,LR,optimizer,epochs,device,fp16=True,noam=None,valid_tokens= 0):
    scaler = GradScaler(device) if fp16 else None
    best_loss = float('inf')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    target_tokens = epochs
    valid_tokens = valid_tokens
    pending = 0
    checkpoint = {}
    save_stage = valid_tokens +1000000
    if not os.path.exists(f'checkpoint/{timestamp}'):  os.makedirs(f'checkpoint/{timestamp}')
    pbar = tqdm(total=target_tokens, desc=f'Epoch 1/1')
    pbar.update(valid_tokens)



    while valid_tokens < target_tokens:
        model.train()
        train_loss = 0
        valid_loss = 0
        loss = Loss(37000, padding_idx=0, smoothing=0.1)
        lr = 0

        for step , batch in enumerate(train_dataloader):
            src_ids, target_ids, labels_ids, src_mask, labels_mask = batch
            src_ids, target_ids, labels_ids, src_mask, labels_mask = src_ids.to(device), target_ids.to(device), labels_ids.to(device), src_mask.to(device), labels_mask.to(device)


            #计算有效token
            current_token = torch.sum(target_ids != 0).item() + torch.sum(src_ids != 0).item()
            valid_tokens += current_token
            pending += current_token


            if fp16:
                with autocast("cuda"):
                    output = model(src_ids, target_ids, src_mask, None)
                    l = loss(output, labels_ids)
                scaler.scale(l*(current_token / TOKENS_PER_STEP)).backward()
                if pending >= TOKENS_PER_STEP:
                    pending = 0
                    lr =  noam.step(target_tokens, valid_tokens)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()


                train_loss += l.item()
                pbar.set_postfix({
                     'total_loss': f"{train_loss / (step + 1):.4f}",
                    'lr': lr,
                })

            else:
                output = model(src_ids, target_ids, src_mask, None)
                l = loss(output, labels_ids)
                (l*(current_token /TOKENS_PER_STEP)).backward()
                if pending >= TOKENS_PER_STEP:
                    pending = 0
                    lr = noam.step(target_tokens, valid_tokens)
                    optimizer.step()
                    optimizer.zero_grad()
                train_loss += l.item()
                pbar.set_postfix({
                    'total_loss': f"{train_loss / (step + 1):.4f}",
                    'lr': lr,
                })
            pbar.update(current_token)
            checkpoint = {
                'epoch': valid_tokens,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'noam': noam.state_dict() if noam is not None else None,
                'loss': valid_loss
            }
            if valid_tokens > save_stage:
                torch.save(checkpoint, f'checkpoint/{timestamp}/last_model.pth')
                save_stage += 1000000
                with open(f"checkpoint/{timestamp}/log.txt", "a") as f:
                    f.write(f"[Train] valid_tokens:{valid_tokens}, train_loss:{train_loss / (step + 1):.4f}\n")


        model.eval()
        for step, batch in enumerate(valid_dataloader):
                src_ids, target_ids, labels_ids, src_mask, labels_mask = batch
                src_ids, target_ids, labels_ids, src_mask, labels_mask = src_ids.to(device), target_ids.to(
                    device), labels_ids.to(device), src_mask.to(device), labels_mask.to(device)
                with torch.no_grad():
                    output = model(src_ids, target_ids, src_mask, None)
                    l = loss(output, labels_ids)
                    valid_loss += l.item()
        with open(f"checkpoint/{timestamp}/log.txt", "a") as f:
            f.write(f"[Valid] valid_tokens:{valid_tokens}, valid_loss:{valid_loss / (step + 1):.4f}\n")
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(checkpoint, f'checkpoint/{timestamp}/best_model.pth')
    pbar.close()




def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEFAULT_CONFIG = "config.yaml"


def load_config(path=DEFAULT_CONFIG):
    """读取 YAML 配置。

    文件缺失或内容不是 dict 时直接抛错，避免静默用错超参跑完一整轮训练。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件内容不是一个 dict：{path}")
    return cfg


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Transformer 训练")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML 配置文件路径")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # 全局：随机种子与设备
    seed = cfg.get("seed", 11)
    seed_everything(seed)

    device_cfg = cfg.get("device", "auto")
    if device_cfg in (None, "", "auto"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    # 数据集
    data_cfg = cfg["data"]
    train_src_path = data_cfg["train_src"]
    train_target_path = data_cfg["train_target"]
    valid_src_path = data_cfg["valid_src"]
    valid_target_path = data_cfg["valid_target"]
    cache_path = data_cfg.get("cache", None)

    # 训练循环
    train_cfg = cfg.get("training", {}) or {}
    target_tokens = train_cfg.get("target_tokens")
    batch_size = train_cfg.get("batch_size", 8)
    valid_batch_size = train_cfg.get("valid_batch_size") or batch_size // 2
    num_workers = train_cfg.get("num_workers", 8)
    valid_num_workers = train_cfg.get("valid_num_workers", 4)
    prefetch_factor = train_cfg.get("prefetch_factor", 4)
    fp16 = train_cfg.get("fp16", True)

    # 模型
    model_cfg = cfg["model"]
    N = model_cfg.get("N", 6)
    heads = model_cfg.get("heads", 8)
    dims = model_cfg.get("dims", 512)
    des_vocab_size = model_cfg.get("des_vocab_size", 37000)
    tie_weights = model_cfg.get("tie_weights", True)

    # 论文 §5.3 的优化器超参
    opt_cfg = cfg.get("optimizer", {}) or {}
    sch_cfg = cfg.get("scheduler", {}) or {}
    optimizer_type = opt_cfg.get("type", "adam")
    base_lr = opt_cfg.get("lr", 0.0)
    betas = tuple(opt_cfg.get("betas", (0.9, 0.98)))   # YAML 读出来是 list，Adam 需要 tuple
    eps = float(opt_cfg.get("eps", 1e-9))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    d_model = sch_cfg.get("d_model", dims)
    warmup_steps = sch_cfg.get("warmup_steps", 4000)
    factor = float(sch_cfg.get("factor", 1.0))

    # 断点续训
    resume_cfg = cfg.get("resume", {}) or {}
    resume_path = resume_cfg.get("checkpoint") or ""
    valid_tokens = resume_cfg.get("start_tokens", 0)
    train_data = MyDataSet(train_src_path, train_target_path, cache_path)
    train_loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
    if num_workers > 0:   # num_workers=0 时这两项不合法
        train_loader_kwargs.update(persistent_workers=True, prefetch_factor=prefetch_factor)
    train_dataloader = DataLoader(
        train_data,
        sampler=BucketBatchSampler(train_data),
        **train_loader_kwargs,
    )

    valid_loader_kwargs = dict(
        batch_size=valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=valid_num_workers,
        pin_memory=True,
    )
    if valid_num_workers > 0:
        valid_loader_kwargs.update(persistent_workers=True, prefetch_factor=prefetch_factor)
    valid_dataloader = DataLoader(
        MyDataSet(valid_src_path, valid_target_path, mode="valid"),
        **valid_loader_kwargs,
    )








    #模型
    model  = Transformer(N=N, heads=heads, dims=dims, des_vocab_size=des_vocab_size, tie_weights=tie_weights).train().to(device)


    optimizer = GetOptimizer(model, optimizer_type, lr=base_lr, betas=betas, eps=eps, weight_decay=weight_decay)
    noam = NoamLR(optimizer, d_model=d_model, warmup_steps=warmup_steps, factor=factor)
    if resume_path != "" and os.path.exists(resume_path):
        state = torch.load(resume_path)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        valid_tokens = state['epoch']
    train(model,train_dataloader,valid_dataloader,None,optimizer,target_tokens,device,fp16=fp16,noam=noam,valid_tokens=valid_tokens)


