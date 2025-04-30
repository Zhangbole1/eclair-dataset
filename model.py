from torch_geometric.nn import fps, knn

import torch
import torch.nn as nn



def safe_fps(x, ratio):
    """
    Farthest Point Sampling，但至少采一个点。
    x: Tensor [M,3]
    ratio: float  采样比例
    """
    M = x.size(0)
    # 目标采样数至少 1
    num = max(int(M * ratio), 1)
    # 调用导入的 fps，而不是 safe_fps
    idx = fps(x, batch=None, ratio=ratio)
    # 如果 fps 返回的点少于目标数，就随机补足
    if idx.numel() < num:
        # 随机打乱再取前 num
        idx = torch.randperm(M, device=x.device)[:num]
    return idx


def safe_knn(x, y, k):
    """
    k-NN 查询，但 k 不超过 x 点数，且至少 1。
    x: Tensor [M,3] 被查询点集
    y: Tensor [N,3] 查询中心点集
    k: 原始请求的邻居数
    """
    M = x.size(0)
    k_eff = min(max(k, 1), M)
    return knn(x=x, y=y, k=k_eff)


def MLP(channels, batch_norm=True):
    """Creates a sequential MLP network given a list of channel sizes."""
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Linear(channels[i], channels[i + 1], bias=not batch_norm))
        if batch_norm:
            layers.append(nn.BatchNorm1d(channels[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class PointNetPP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Set Abstraction layers
        # SA1: from N -> N1 points, input feature 14 -> 64 -> 128 dims
        self.sa1_mlp = MLP([14 + 3, 64, 128])  # +3 for relative coords
        # SA2: from N1 -> N2 points, input feature 128 -> 128 -> 256 dims
        self.sa2_mlp = MLP([128 + 3, 128, 256])
        # (You can add more SA layers for larger networks)
        # Feature Propagation layers
        # FP1: propagate from SA2 (N2 points, 256-dim) to SA1 (N1 points, 128-dim)
        self.fp1_mlp = MLP([256 + 128, 128, 128])  # concat SA2 feat (interp) with SA1 feat
        # FP2: propagate from SA1 (N1, 128-dim) to original N points (14-dim input skip)
        self.fp2_mlp = MLP([128 + 14, 128, 128])  # concat SA1 interp with original features
        # Segmentation head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, coords, features):
        """
        coords: Tensor [N, 3] (XYZ coordinates for N points)
        features: Tensor [N, 14] (intensity, returns, color features per point)
        """
        N = coords.size(0)
        # 1. Set Abstraction Layer 1
        # Sample a subset of points (e.g., 50% of points for SA1)
        idx1 = safe_fps(coords, ratio=0.5)
        coords1 = coords[idx1]  # coordinates of sampled points
        # Find neighbors of each sampled point within a radius or k-NN
        neighbor_idx1 = safe_knn(coords, coords1, k=32)
        # knn returns index pairs (index_in_x, index_in_y) for each neighbor relationship
        # Separate indices for clarity
        src_idx = neighbor_idx1[0]  # indices of neighbors in original set
        dst_idx = neighbor_idx1[1]  # indices of corresponding centroid (in coords1)
        # Compute relative coordinates and get features for neighbors
        neighbor_coords = coords[src_idx] - coords1[dst_idx]  # [total_neighbors, 3]
        neighbor_feat = features[src_idx]  # [total_neighbors, 14]
        # Combine neighbor relative coords and features
        neighbor_input = torch.cat([neighbor_coords, neighbor_feat], dim=1)  # [?, 17]
        # Apply PointNet (shared MLP + max pool) on neighbor features for each centroid
        feat1 = self.sa1_mlp(neighbor_input)  # produces [?, 128] for all neighbor points
        # We need to aggregate (max) for each group of neighbors belonging to the same centroid
        # We can use scatter_max (from torch_scatter) or group by dst_idx:
        num_centroids = coords1.size(0)
        # Initialize tensor for aggregated features per centroid
        feat1_max = torch.zeros((num_centroids, 128), device=feat1.device)
        # Scatter max manually: group by dst_idx
        feat1_max.index_add_(0, dst_idx, feat1)  # using index_add as a workaround for max
        # (In practice, use torch_scatter.scatter to get max per group)
        # After aggregation, feat1_max is [N1, 128]

        # 2. Set Abstraction Layer 2 (on the output of SA1)
        idx2 = safe_fps(coords1, ratio=0.5)
        coords2 = coords1[idx2]
        neighbor_idx2 = safe_knn(coords1, coords2, k=32)
        src_idx2 = neighbor_idx2[0];
        dst_idx2 = neighbor_idx2[1]
        neighbor_coords2 = coords1[src_idx2] - coords2[dst_idx2]  # relative coords in SA1 frame
        neighbor_feat2 = feat1_max[src_idx2]  # SA1 features of neighbors
        neighbor_input2 = torch.cat([neighbor_coords2, neighbor_feat2], dim=1)  # [?, 131]
        feat2 = self.sa2_mlp(neighbor_input2)  # [?, 256] for neighbor points
        # Aggregate to get features for each coords2 centroid
        feat2_max = torch.zeros((coords2.size(0), 256), device=feat2.device)
        feat2_max.index_add_(0, dst_idx2, feat2)  # (using sum as placeholder for max)
        # feat2_max shape: [N2, 256]

        # 3. Feature Propagation Layer 1 (from SA2 back to SA1)
        # We have coords2 (N2 points) with feat2_max (256-dim), and coords1 (N1 points) with feat1_max (128-dim).
        # Interpolate features from coords2 to coords1:
        # For simplicity, use nearest neighbor interpolation:
        # Find 3 nearest coords2 for each coords1
        interp_idx = safe_knn(coords2, coords1, k=3)
        src_i = interp_idx[0];
        dst_i = interp_idx[1]
        # Get neighbor features and distances
        diff = coords2[src_i] - coords1[dst_i]
        dist2 = (diff ** 2).sum(dim=1).add(1e-9)  # squared distances (avoid zero)
        inv_w = 1.0 / dist2  # inverse distance weights
        # Normalize weights per target point (sum to 1 for the 3 neighbors)
        # (Sum weights for each dst_i group)
        w_sum = torch.zeros(coords1.size(0), device=coords1.device)
        w_sum.index_add_(0, dst_i, inv_w)
        weights = inv_w / (w_sum[dst_i] + 1e-9)
        # Weighted sum of neighbor features
        interp_feat = torch.zeros((coords1.size(0), 256), device=feat2_max.device)
        interp_feat.index_add_(0, dst_i, feat2_max[src_i] * weights.unsqueeze(1))
        # Now interp_feat is the interpolated 256-dim feature for each of the N1 points.
        # Concatenate with skip connection from SA1 (feat1_max)
        fp1_input = torch.cat([interp_feat, feat1_max], dim=1)  # [N1, 256+128]
        feat1_fp = self.fp1_mlp(fp1_input)  # [N1, 128] upsampled features for SA1 points

        # 4. Feature Propagation Layer 2 (from SA1 back to original points)
        interp_idx2 = safe_knn(coords1, coords, k=3)
        src_j = interp_idx2[0];
        dst_j = interp_idx2[1]
        diff2 = coords1[src_j] - coords[dst_j]
        dist2b = (diff2 ** 2).sum(dim=1).add(1e-9)
        inv_w2 = 1.0 / dist2b
        w_sum2 = torch.zeros(N, device=coords.device)
        w_sum2.index_add_(0, dst_j, inv_w2)
        weights2 = inv_w2 / (w_sum2[dst_j] + 1e-9)
        interp_feat2 = torch.zeros((N, 128), device=feat1_fp.device)
        interp_feat2.index_add_(0, dst_j, feat1_fp[src_j] * weights2.unsqueeze(1))
        # Concatenate with original point features (the 14-dim input features as skip)
        fp2_input = torch.cat([interp_feat2, features], dim=1)  # [N, 128+14]
        feat0_fp = self.fp2_mlp(fp2_input)  # [N, 128]

        # 5. Per-point classification
        scores = self.classifier(feat0_fp)  # [N, num_classes]
        return scores
