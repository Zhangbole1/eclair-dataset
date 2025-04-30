import numpy as np
from sklearn.neighbors import NearestNeighbors

class Compose:
    """将多个点云变换和特征生成操作按顺序组合的类。"""
    def __init__(self, transforms):
        """
        参数:
            transforms (list): 包含一系列变换对象的列表，这些对象都实现了__call__方法。
        """
        self.transforms = transforms

    def __call__(self, coords, feats):
        """
        按顺序对coords和feats应用每个变换。

        参数:
            coords (np.ndarray): 点坐标数组，形状为(N,3)。
            feats (np.ndarray or None): 点特征数组，形状为(N,C)。如果为None，将在需要时创建。

        返回:
            (coords, feats): 变换后的坐标和特征。
        """
        for t in self.transforms:
            coords, feats = t(coords, feats)
        return coords, feats

class RandomRotate:
    """随机旋转点云，用于数据增广。默认绕竖直轴(z轴)旋转。"""
    def __init__(self, axis='z', angle=360):
        """
        参数:
            axis (str): 选择旋转轴，可为 'x', 'y', 'z'。默认 'z'。
            angle (float): 旋转角度范围(度)。将在[0, angle)区间内均匀采样旋转角度。默认360度表示全方位随机旋转。
        """
        self.axis = axis.lower()
        self.angle_rad = np.deg2rad(angle)  # 将角度转换为弧度

    def __call__(self, coords, feats):
        """
        对点云坐标进行随机旋转。

        参数:
            coords (np.ndarray): 原始点坐标数组，形状(N,3)。
            feats (np.ndarray or None): 原始点特征数组。

        返回:
            (coords, feats): 旋转后的坐标和原始特征(不修改)。
        """
        if self.axis not in ['x', 'y', 'z']:
            raise ValueError("axis 参数应为 'x', 'y' 或 'z'")
        # 在指定范围内采样旋转角度
        theta = np.random.rand() * self.angle_rad  # [0, angle_rad)
        c, s = np.cos(theta), np.sin(theta)
        # 根据选择的轴构造旋转矩阵
        if self.axis == 'z':
            rot_matrix = np.array([[c, -s, 0],
                                   [s,  c, 0],
                                   [0,  0, 1]], dtype=float)
        elif self.axis == 'x':
            rot_matrix = np.array([[1,  0,  0],
                                   [0,  c, -s],
                                   [0,  s,  c]], dtype=float)
        elif self.axis == 'y':
            rot_matrix = np.array([[ c, 0, s],
                                   [ 0, 1, 0],
                                   [-s, 0, c]], dtype=float)
        # 对所有点坐标进行矩阵乘法完成旋转
        coords = coords.dot(rot_matrix.T)  # (N,3) x (3,3)
        # 注意: 如果 feats 中包含法向量等方向特征，也需要相应旋转。同一变换应应用于法向量特征以保持一致。
        return coords, feats  # 仅改变坐标，特征保持不变

class ComputeNormals:
    """计算每个点的法向量，将其作为特征附加到点特征中。"""
    def __init__(self, k=16):
        """
        参数:
            k (int): 计算法向量时使用的近邻点数。默认16个邻居。
        """
        self.k = k

    def __call__(self, coords, feats):
        """
        计算点云中每个点的法向量，并将法向量添加为特征。

        参数:
            coords (np.ndarray): 点坐标，形状(N,3)。
            feats (np.ndarray or None): 点特征，形状(N,C)。

        返回:
            (coords, feats): 未改变的坐标，以及附加了法向量特征的特征数组。
        """
        N = coords.shape[0]
        if N == 0:
            return coords, feats  # 空点云直接返回
        # 使用sklearn快速查找每个点的k近邻
        nbrs = NearestNeighbors(n_neighbors=self.k+1, algorithm='auto').fit(coords)
        # 对每个点查询k+1个邻居（包括点自身）
        distances, indices = nbrs.kneighbors(coords)
        normals = np.zeros((N, 3), dtype=float)
        for i in range(N):
            # 排除自身，只取邻居点坐标
            neighbor_idxs = indices[i, 1:]  # 跳过indices[i,0]（即点自身）
            neighbor_points = coords[neighbor_idxs]
            center = coords[i]
            # 计算邻居相对坐标
            rel_coords = neighbor_points - center  # 形状(k,3)
            # 计算协方差矩阵 (3x3)
            cov = rel_coords.T.dot(rel_coords)
            # 求协方差矩阵的特征值和特征向量
            # eigh求解对称矩阵特征值，返回的特征值是升序排序
            eigvals, eigvecs = np.linalg.eigh(cov)
            # 取最小特征值对应的特征向量作为法向量（对应邻域平面）
            normal = eigvecs[:, 0]
            # 归一化法向量方向
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            normals[i] = normal
        # 如果原始feats为空，则用法向量初始化特征，否则将法向量拼接到已有特征最后
        if feats is None:
            feats = normals.astype(np.float32)
        else:
            feats = np.hstack([feats, normals.astype(np.float32)])
        return coords, feats

class HeightAboveGround:
    """计算每个点相对地面的高度，将其作为额外特征。"""
    def __call__(self, coords, feats):
        """
        计算并添加高度特征。假设地面近似与XY平面重合，采用最低点高度作为地面参考。

        参数:
            coords (np.ndarray): 点坐标，形状(N,3)。
            feats (np.ndarray or None): 点特征，形状(N,C)。

        返回:
            (coords, feats): 原始坐标，和附加了高度特征的特征数组。
        """
        if coords.shape[0] == 0:
            return coords, feats
        # 取点云中最小的Z坐标作为地面高度
        ground_z = np.min(coords[:, 2])
        heights = coords[:, 2] - ground_z  # 每个点的相对高度
        heights = heights.reshape(-1, 1).astype(np.float32)
        if feats is None:
            feats = heights
        else:
            feats = np.hstack([feats, heights])
        return coords, feats
