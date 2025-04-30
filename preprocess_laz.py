# preprocess_laz.py
import os
import json
import torch
import laspy
import numpy as np
from tqdm import tqdm

# 路径配置（建议修改为你实际本地路径）
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
PCD_DIR = os.path.join(DATA_DIR, "pointclouds")
LABEL_PATH = os.path.join(DATA_DIR, "labels.json")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

SPLIT_DIRS = {
    "train": os.path.join(DATA_DIR, "train"),
    "val": os.path.join(DATA_DIR, "val"),
    "test": os.path.join(DATA_DIR, "test")
}
for d in SPLIT_DIRS.values():
    os.makedirs(d, exist_ok=True)


# 读取标注文件
with open(LABEL_PATH, "r") as f:
    label_entries = json.load(f)

approved_tiles = [
    item for item in label_entries
    if item["review_category"] == "approved" and item["tile_name"].endswith(".laz")
]

def to_numpy(field):
    return field.array if hasattr(field, "array") else field

def laz_to_tensor_dict(path):
    with laspy.open(path) as fh:
        las = fh.read()

        xyz = np.vstack([
            to_numpy(las.x),
            to_numpy(las.y),
            to_numpy(las.z)
        ]).T.astype(np.float32)

        intensity = to_numpy(las.intensity).astype(np.int32)
        rn = to_numpy(las.return_number).astype(np.int32)
        nor = to_numpy(las.number_of_returns).astype(np.int32)
        classification = to_numpy(las.classification).astype(np.int32)

        try:
            rgb = np.vstack([
                to_numpy(las.red),
                to_numpy(las.green),
                to_numpy(las.blue)
            ]).T.astype(np.int32)
        except AttributeError:
            rgb = np.zeros((xyz.shape[0], 3), dtype=np.int32)

        return {
            "xyz": torch.from_numpy(xyz),
            "intensity": torch.from_numpy(intensity),
            "return_number": torch.from_numpy(rn),
            "number_of_returns": torch.from_numpy(nor),
            "rgb": torch.from_numpy(rgb),
            "classification": torch.from_numpy(classification)
        }



# 批量转换
for entry in tqdm(approved_tiles, desc="Converting .laz to .pth"):
    laz_path = os.path.join(PCD_DIR, entry["tile_name"])
    split = entry["split"]
    if split not in SPLIT_DIRS:
        print(f"[WARNING] Unknown split '{split}' in {entry['tile_name']}, skipping.")
        continue
    out_dir = SPLIT_DIRS[split]
    out_path = os.path.join(out_dir, entry["tile_name"].replace(".laz", ".pth"))

    try:
        tensor_dict = laz_to_tensor_dict(laz_path)
        torch.save(tensor_dict, out_path)
    except Exception as e:
        print(f"[ERROR] Failed to process {entry['tile_name']}: {e}")
