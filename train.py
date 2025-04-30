# train.py

import os
import yaml
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchmetrics

from dataset import PointCloudDataset
from model import PointNetPP

def load_config():
    # 自动定位当前脚本所在目录
    root = os.path.dirname(os.path.abspath(__file__))
    # 假设你的 YAML 在 project_root/configs/train.yaml
    cfg_path = os.path.join(root, "configs", "train.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def simple_collate(batch):
    return batch[0]

def main():
    # 1. 读配置
    # cfg 已经是一个普通 dict，你可以通过 cfg["lr"]、cfg["train_batch_size"] 等访问
    set_seed(int(cfg.get("random_seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 准备文件列表
    labels_json = os.path.join("data", "labels.json")
    with open(labels_json, "r", encoding="utf-8") as f:
        all_tiles = json.load(f)
    point_dir = os.path.join("data", "pointclouds")
    train_files = [os.path.join(point_dir, x["tile_name"]) for x in all_tiles if x["split"] == "train"]
    val_files   = [os.path.join(point_dir, x["tile_name"]) for x in all_tiles if x["split"] == "val"]

    # 3. Dataset & DataLoader
    normalize_coord = 1.0 / float(cfg.get("voxel_size", 0.05))
    train_ds = PointCloudDataset(train_files, cfg["output_mapping"], normalize_coord)
    val_ds   = PointCloudDataset(val_files,   cfg["output_mapping"], normalize_coord)

    train_loader = DataLoader(
        train_ds,
        batch_size=1,  # 从 4→1
        shuffle=True,
        num_workers=0,  # 从 4→0
        collate_fn=simple_collate
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # 从 4→1
        shuffle=False,
        num_workers=0,
        collate_fn=simple_collate
    )

    # 4. 模型、优化器、损失
    num_classes = int(cfg["num_classes"])
    model = PointNetPP(num_classes=num_classes).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4))
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg.get("step_size", 20)),
        gamma=float(cfg.get("gamma", 0.1))
    )
    ignore_idx = int(cfg.get("ignore_index", -100))
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)

    # 5. 验证指标
    iou_metric = torchmetrics.JaccardIndex(
        num_classes=num_classes,
        task="multiclass",
        ignore_index=ignore_idx
    ).to(device)
    f1_metric = torchmetrics.F1Score(
        num_classes=num_classes,
        task="multiclass",
        average="macro",
        ignore_index=ignore_idx
    ).to(device)

    # 6. 训练循环
    best_miou = 0.0
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    max_epochs = int(cfg.get("max_epochs", cfg.get("max_epochs", 100)))

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for coords, feats, labels in tqdm(train_loader, desc=f"Train {epoch}/{max_epochs}"):
            coords, feats, labels = coords.to(device), feats.to(device), labels.to(device)
            optimizer.zero_grad()
            scores = model(coords, feats)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch} Train Loss: {np.mean(losses):.4f}")

        model.eval()
        iou_metric.reset(); f1_metric.reset()
        with torch.no_grad():
            for coords, feats, labels in tqdm(val_loader, desc=f" Val  {epoch}/{max_epochs}"):
                coords, feats, labels = coords.to(device), feats.to(device), labels.to(device)
                preds = model(coords, feats).argmax(dim=1)
                iou_metric.update(preds, labels)
                f1_metric.update(preds, labels)
        class_iou = iou_metric.compute()
        miou = class_iou[1:].mean().item()
        mf1  = f1_metric.compute().item()
        print(f"Epoch {epoch} Val mIoU: {miou:.4f}, Val F1: {mf1:.4f}")

        scheduler.step()
        torch.cuda.empty_cache()

        if miou > best_miou:
            best_miou = miou
            path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(model.state_dict(), path)
            print(f"Saved best model (mIoU={miou:.4f}) to {path}")

    print(f"Training finished! Best mIoU: {best_miou:.4f}")

if __name__ == "__main__":
    main()
