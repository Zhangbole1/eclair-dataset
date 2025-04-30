import laspy
import numpy as np
import torch
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    def __init__(self, file_list, class_mapping, normalize_coord=10.0):
        self.files = file_list
        self.class_mapping = class_mapping  # e.g. dict mapping LAS classes to new labels
        self.normalize_coord = normalize_coord

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Read the LAS/LAZ file
        filepath = self.files[idx]
        las = laspy.read(filepath)
        # Raw coordinates and features
        coords = np.vstack((las.x, las.y, las.z)).T  # shape (N,3)
        intensity = las.intensity.astype(np.float32)  # (N,)
        return_num = las.return_number  # (N,) uint
        num_returns = las.number_of_returns  # (N,) uint
        # Optional color
        has_color = hasattr(las, "red")
        if has_color:
            rgb = np.vstack((las.red, las.green, las.blue)).T.astype(np.float32)
        else:
            rgb = None

        # Remap classification labels
        raw_labels = las.classification
        labels = np.copy(raw_labels)  # to avoid modifying original
        if self.class_mapping:
            # Initialize all labels as 0 (default)
            new_labels = np.zeros_like(labels, dtype=np.int64)
            for src, dst in self.class_mapping.items():
                new_labels[labels == src] = dst
            labels = new_labels
        labels = torch.from_numpy(labels.astype(np.int64))

        # Normalize coordinates (center and scale)
        coords = coords.astype(np.float32)
        centroid = coords.mean(axis=0)
        coords -= centroid
        coords /= self.normalize_coord

        # Normalize intensity to [0,1] then shift to [-0.5,0.5]
        intensity = intensity / np.iinfo(las.intensity.dtype).max - 0.5

        # One-hot encode return_number and number_of_returns (clamped max 5)
        # (Assume return_num and num_returns are 1-indexed)
        max_returns = 5
        ret_clamped = np.clip(return_num - 1, 0, max_returns-1)  # 0-indexed
        nr_clamped = np.clip(num_returns - 1, 0, max_returns-1)
        return_onehot = np.eye(max_returns, dtype=np.float32)[ret_clamped]      # shape (N,5)
        numret_onehot = np.eye(max_returns, dtype=np.float32)[nr_clamped]      # shape (N,5)

        # Normalize RGB to [0,1] and shift to [-0.5,0.5] if present
        if has_color:
            rgb = rgb / np.iinfo(np.uint16).max - 0.5  # assuming 16-bit colors
            features = np.concatenate([intensity[:, None], return_onehot, numret_onehot, rgb], axis=1)
        else:
            features = np.concatenate([intensity[:, None], return_onehot, numret_onehot], axis=1)
        features = torch.from_numpy(features.astype(np.float32))
        coords = torch.from_numpy(coords.astype(np.float32))
        return coords, features, labels
