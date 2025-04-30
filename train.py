import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence

import fire
import laspy
import numpy as np
import torch
import torchmetrics
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from MinkowskiEngine import MinkowskiAlgorithm, SparseTensorQuantizationMode, TensorField

from functional import compose_transforms_from_list
from model import MinkUNet14C


def las2pyg(las: laspy.LasData, path: Path) -> Data:
    gt_key = "classification" if "classification" in set(las.point_format.dimension_names) else "raw_classification"
    return Data(
        xyz=torch.from_numpy(las.xyz.copy()),
        intensity=torch.from_numpy(las.intensity.astype(np.int64)),
        classification=torch.from_numpy(las[gt_key]).long(),
        return_number=torch.from_numpy(np.asarray(las.return_number)).long(),
        number_of_returns=torch.from_numpy(np.asarray(las.number_of_returns)).long(),
        edge_of_flight_line=torch.from_numpy(np.asarray(las.edge_of_flight_line)),
        instance_id=(
            torch.from_numpy(np.asarray(las.instance).copy().astype(np.int64)).long()
            if hasattr(las, "instance")
            else torch.full((len(las.return_number),), fill_value=-1, dtype=torch.long)
        ),
        rgb=torch.stack(
            [
                torch.from_numpy(las.red.astype(np.int64)),
                torch.from_numpy(las.green.astype(np.int64)),
                torch.from_numpy(las.blue.astype(np.int64)),
            ],
            dim=-1,
        ).long()
        if hasattr(las, "red")
        else None,
        filename=path,
    )


def collate_custom(batch):
    items = [item[:3] for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(items)


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, config, fnames: Sequence[Path], transforms):
        self.fnames = fnames
        self.transforms = compose_transforms_from_list(transforms)

    def __getitem__(self, index: int) -> Data:
        tile_path = self.fnames[index]
        las = laspy.read(tile_path)
        data = las2pyg(las, str(tile_path))
        if self.transforms:
            data = self.transforms(data)
        return data

    def __len__(self) -> int:
        return len(self.fnames)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_training(
    dataset_path: str = None,
    model_output_dir: str = "./checkpoints",
    **kwargs,
):
    from_cli = OmegaConf.create(kwargs)
    base_conf = OmegaConf.load("./configs/train.yaml")
    conf = OmegaConf.merge(base_conf, from_cli)

    seed_everything(conf.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load split information
    label_file = Path(dataset_path) / "labels.json"
    with open(label_file, encoding="utf-8") as f:
        all_tiles: List[Dict] = json.load(f)

    train_tiles = [Path(dataset_path) / "pointclouds" / x["tile_name"] for x in all_tiles if x["split"] == "train"]
    val_tiles = [Path(dataset_path) / "pointclouds" / x["tile_name"] for x in all_tiles if x["split"] == "val"]

    # Datasets and loaders
    train_dataset = PointCloudDataset(conf, train_tiles, conf.train_transforms)
    val_dataset = PointCloudDataset(conf, val_tiles, conf.val_transforms)

    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=conf.train_batch_size,
        shuffle=True,
        collate_fn=collate_custom,
        pin_memory=True
    )
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=conf.val_batch_size,
        shuffle=False,
        collate_fn=collate_custom,
        pin_memory=True
    )

    # Model, optimizer, scheduler, loss
    model = MinkUNet14C(conf.num_features, conf.num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=conf.lr,
        weight_decay=getattr(conf, "weight_decay", 0)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=conf.step_size, gamma=conf.gamma)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=conf.ignore_index)

    # Metrics
    train_metrics = torchmetrics.MetricCollection({
        "train_f1": torchmetrics.classification.MulticlassF1Score(
            num_classes=conf.num_classes, ignore_index=conf.ignore_index
        )
    }).to(device)

    val_metrics = torchmetrics.MetricCollection({
        "val_f1": torchmetrics.classification.MulticlassF1Score(
            num_classes=conf.num_classes, ignore_index=conf.ignore_index
        )
    }).to(device)

    best_val_f1 = 0.0
    os.makedirs(model_output_dir, exist_ok=True)

    for epoch in range(1, conf.max_epochs + 1):
        model.train()
        train_metrics.reset()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{conf.max_epochs} [Train]"):
            coords, feats, batch_idx = batch.pos / conf.voxel_size, batch.x, batch.batch
            coords = torch.cat([batch_idx.unsqueeze(1), coords], dim=1).to(device)
            feats = feats.to(device)
            y = batch.classification.long().to(device)

            # Sparse input
            in_field = TensorField(
                features=feats,
                coordinates=coords,
                quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=MinkowskiAlgorithm.MEMORY_EFFICIENT,
            )
            sinput = in_field.sparse()
            # 放在 `sinput = in_field.sparse()` 之后、`soutput = model(sinput)` 之前
            # print(f"[DEBUG] batch.x shape = {feats.shape}, expected = {conf.num_features}")
            # assert feats.size(1) == conf.num_features, \
            #     f"Feature dim mismatch: got {feats.size(1)}, expected {conf.num_features}"

            # Forward
            soutput = model(sinput)
            out_field = soutput.slice(in_field).F

            # Compute loss (ignoring index in loss_fn)
            loss = loss_fn(out_field, y)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = out_field.max(dim=1).indices
            train_metrics.update(preds, y)

        scheduler.step()
        train_res = train_metrics.compute()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_metrics.reset()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{conf.max_epochs} [Val]"):
                coords, feats, batch_idx = batch.pos / conf.voxel_size, batch.x, batch.batch
                coords = torch.cat([batch_idx.unsqueeze(1), coords], dim=1).to(device)
                feats = feats.to(device)
                y = batch.classification.long().to(device)

                in_field = TensorField(
                    features=feats,
                    coordinates=coords,
                    quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                    minkowski_algorithm=MinkowskiAlgorithm.MEMORY_EFFICIENT,
                )
                sinput = in_field.sparse()
                soutput = model(sinput)
                out_field = soutput.slice(in_field).F

                preds = out_field.max(dim=1).indices
                val_metrics.update(preds, y)

        val_res = val_metrics.compute()
        val_f1 = val_res["val_f1"].item()

        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Train F1: {train_res['train_f1']:.4f} | Val F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(model_output_dir, f"best_model_epoch_{epoch}.pth"))
            print(f"New best model saved at epoch {epoch} with Val F1: {val_f1:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    fire.Fire(run_training)
