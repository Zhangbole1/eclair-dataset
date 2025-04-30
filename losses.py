import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss 实现，用于处理类别不平衡，关注难分类样本。"""
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, reduction='mean'):
        """
        参数:
            alpha: 平衡因子。可以是None、标量(float)或每个类别的权重列表(list/tuple/Tensor)。
                   - 如果为None，则对各类别不使用额外权重(除了ignore_index)。
                   - 如果为浮点数且是二分类问题，会自动将alpha解释为正负类权重。
                   - 如果为列表或Tensor，则应为每个类别指定一个权重。
            gamma (float): 难易样本调节参数gamma。gamma越大，越侧重难分类样本。默认2.0。
            ignore_index (int): 忽略的类别标签，不计算损失。默认-100 (与PyTorch默认相同)。
            reduction (str): {'mean','sum','none'} 指定输出如何聚合。默认'mean'取平均。
        """
        super(FocalLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                # 将alpha列表转换为Tensor注册为buffer，方便移动到设备
                alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
                self.register_buffer('alpha', alpha_tensor)
            elif isinstance(alpha, torch.Tensor):
                self.register_buffer('alpha', alpha.to(torch.float32))
            elif isinstance(alpha, (float, int)):
                # 单一浮点alpha的情况：如果用于二分类，则构造 [1-alpha, alpha] 权重
                # 多类别情况下，单一alpha将被忽略（视为不使用权重）
                self.alpha = float(alpha)
                if self.alpha < 0:
                    raise ValueError("alpha应为非负数")
            else:
                raise TypeError("alpha应为None、float、list或torch.Tensor类型")
        else:
            self.alpha = None
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, target):
        """
        计算 Focal Loss。

        参数:
            logits (Tensor): 预测的logits张量，形状为(N, C)，其中C为类别数。
            target (Tensor): 目标标签张量，形状为(N)，值为类别索引。

        返回:
            Tensor: 根据 reduction 设置返回标量损失或每个样本的损失向量。
        """
        # 使用交叉熵计算每个样本的基准损失 (不做reduction，以便逐元素调整)
        if hasattr(self, 'alpha') and isinstance(self.alpha, torch.Tensor):
            weight = self.alpha
            # 将权重移到与logits相同的设备
            if weight.device != logits.device:
                weight = weight.to(logits.device)
        elif hasattr(self, 'alpha') and isinstance(self.alpha, float):
            weight = None
            # 如果是二分类且提供标量alpha，则构造权重tensor
            num_classes = logits.shape[1]
            if num_classes == 2:
                alpha_val = self.alpha
                alpha_tensor = torch.tensor([1.0 - alpha_val, alpha_val], dtype=torch.float32, device=logits.device)
                weight = alpha_tensor
        else:
            weight = None
        # 计算交叉熵损失（逐元素），考虑ignore_index和类别权重
        ce_loss = F.cross_entropy(logits, target, weight=weight, reduction='none', ignore_index=self.ignore_index)
        # 计算每个样本的pt（预测正确概率），用于调整权重 (pt = exp(-ce_loss))
        pt = torch.exp(-ce_loss)  # pt = prob[target]，交叉熵的定义保证此公式成立
        # 基于pt计算focal调制因子 (1-pt)^gamma，并与ce_loss相乘形成focal loss
        focal_loss = ce_loss * ((1 - pt) ** self.gamma)
        # 如果提供了alpha并且是每类权重或二分类alpha，也将其应用到损失上
        # （注：直接在cross_entropy中已经应用了weight，对应每个样本根据真实类别乘以alpha权重，
        # 因此此处无需再次乘alpha。如果alpha是标量二分类，已在上面构造对应weight。）
        # 汇总损失
        if self.reduction == 'mean':
            # 忽略被ignore_index标记的样本：它们的ce_loss在PyTorch实现中为0，但仍计算在pt中。
            # 为安全起见，可以基于target过滤，但因为这些样本ce_loss为0，focal_loss也为0，不影响均值计算。
            # 这里直接求均值（总损失/有效元素数）
            # 计算有效元素数用于mean（忽略ignore_index对应项）
            if self.ignore_index is not None:
                mask = (target != self.ignore_index)
                if mask.any():
                    return focal_loss[mask].mean()
                else:
                    return torch.tensor(0.0, device=logits.device)  # 全部忽略
            else:
                return focal_loss.mean()
        elif self.reduction == 'sum':
            if self.ignore_index is not None:
                mask = (target != self.ignore_index)
                return focal_loss[mask].sum()
            else:
                return focal_loss.sum()
        else:  # 'none'
            return focal_loss

class DiceLoss(nn.Module):
    """Dice Loss 实现，用于语义分割以衡量预测与真实的区域重叠。"""
    def __init__(self, ignore_index=-100, reduction='mean', smooth=1e-5):
        """
        参数:
            ignore_index (int): 忽略的类别标签。这些标签的像素不参与损失计算。默认-100。
            reduction (str): {'mean','sum','none'} 指定输出的聚合方式。默认'mean'。
            smooth (float): 平滑项，防止除零。默认1e-5。
        """
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, logits, target):
        """
        计算 Dice Loss。

        参数:
            logits (Tensor): 模型预测的logits张量，形状为(N, C)。
            target (Tensor): 真实标签张量，形状为(N)，值为类别索引。

        返回:
            Tensor: 损失值（根据reduction为标量或逐元素张量）。
        """
        # 将logits通过softmax转为每类的预测概率
        probs = F.softmax(logits, dim=1)  # (N, C)
        # 如果存在ignore_index，则过滤掉这些标签的样本
        if self.ignore_index is not None:
            mask = (target != self.ignore_index)
            if not mask.any():
                # 如果所有样本都被忽略，则损失为0
                return torch.tensor(0.0, device=logits.device)
            probs = probs[mask]         # 只保留未忽略样本的预测
            target = target[mask]       # 只保留未忽略样本的标签
        # 将目标转换为one-hot编码，形状(N, C)
        target_one_hot = F.one_hot(target, num_classes=probs.shape[1]).to(probs.dtype)  # (N, C)
        # 计算每个类别的交集和并集（或说预测总和和真实总和）
        dims = (0,)  # 我们按所有样本总和，也可以按batch中每个样本，但这里将整个batch视作集合
        intersection = torch.sum(probs * target_one_hot, dim=dims)  # 按类别求交集概率和
        pred_sum = torch.sum(probs, dim=dims)
        target_sum = torch.sum(target_one_hot, dim=dims)
        # 根据Dice系数公式计算每个类别的dice系数
        dice_per_class = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        # 对各类别的dice系数取平均，得到总体dice（如果需要可以排除背景类，这里不额外排除）
        mean_dice = torch.mean(dice_per_class)
        # Dice Loss = 1 - 平均Dice系数
        dice_loss = 1 - mean_dice
        if self.reduction == 'mean' or self.reduction is None:
            # mean模式下直接返回标量
            return dice_loss
        elif self.reduction == 'sum':
            # sum模式下，将标量乘以未忽略样本数（只是为了接口一致，这里mean_dice本身是平均后的标量，因此sum同mean效果相同）
            return dice_loss *  target_one_hot.shape[0]
        else:  # 'none'
            # 如果需要逐样本dice损失，可以按样本计算dice。但通常DiceLoss用于整体评价，此处返回每个类别dice供参考。
            return 1 - dice_per_class  # 返回每个类别的Dice损失向量 (C维)
        
class CombinedLoss(nn.Module):
    """组合损失，将Focal Loss和Dice Loss结合，用于训练。支持OHEM在线难例挖掘。"""
    def __init__(self, focal_alpha=None, focal_gamma=2.0, dice_weight=1.0, focal_weight=1.0, 
                 ohem_thresh=None, ohem_min_kept=None, ignore_index=-100):
        """
        参数:
            focal_alpha: FocalLoss的alpha参数（同FocalLoss定义）。
            focal_gamma: FocalLoss的gamma参数。
            dice_weight (float): 在总损失中DiceLoss的权重系数。默认1.0。
            focal_weight (float): 在总损失中FocalLoss的权重系数。默认1.0。
            ohem_thresh (float): Online Hard Example Mining的概率阈值。如果设定，则仅利用预测概率低于该值的样本计算损失。
            ohem_min_kept (int 或 float): OHEM保留的最少样本数。如果为整数，则至少保留该数量的最难样本；
                                         如果为0~1之间float，则表示按比例保留(top比例%)的难样本。
            ignore_index (int): 忽略的标签，不参与损失计算。默认-100。
        """
        super(CombinedLoss, self).__init__()
        # 内部包含FocalLoss和DiceLoss模块
        self.focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_index=ignore_index, reduction='none')
        self.dice_loss_fn = DiceLoss(ignore_index=ignore_index, reduction='none')
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ohem_thresh = ohem_thresh
        self.ohem_min_kept = ohem_min_kept
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        """
        计算组合损失。如果启用了OHEM，则根据阈值筛选难样本计算损失。

        参数:
            logits (Tensor): 模型预测的logits，形状(N, C)。
            target (Tensor): 真实标签，形状(N)。

        返回:
            Tensor: 组合后的总损失标量。
        """
        N = target.shape[0]
        # 如果使用OHEM，则筛选出困难样本的索引
        if self.ohem_thresh is not None or self.ohem_min_kept is not None:
            # 先排除ignore_index标签的点
            if self.ignore_index is not None:
                valid_mask = (target != self.ignore_index)
            else:
                valid_mask = torch.ones_like(target, dtype=torch.bool)
            if valid_mask.sum() == 0:
                # 全部是ignore，不计算损失
                return torch.tensor(0.0, device=logits.device)
            # 计算每个点的预测概率（针对其真实类别）
            probs = F.softmax(logits[valid_mask], dim=1)  # 仅针对有效点
            target_valid = target[valid_mask]
            # 获取每个有效点对应真实类别的概率
            target_probs = probs[range(target_valid.shape[0]), target_valid]
            # 初始化选中样本的掩码（针对整个N长度）
            select_mask = torch.zeros_like(target, dtype=torch.bool)
            select_mask[valid_mask] = False  # 先全部设为False
            if self.ohem_thresh is not None:
                # 基于概率阈值筛选
                hard_mask_valid = target_probs < self.ohem_thresh  # 有效点中难样本的布尔掩码
                # 将这些难样本对应回全局mask
                hard_indices_global = valid_mask.nonzero(as_tuple=False).squeeze(1)[hard_mask_valid]
                select_mask[hard_indices_global] = True
                selected_count = hard_mask_valid.sum().item()
            else:
                selected_count = 0
            # 确定需要保留的最少样本数
            if self.ohem_min_kept is not None:
                if isinstance(self.ohem_min_kept, float) and 0 < self.ohem_min_kept < 1:
                    min_kept = int(self.ohem_min_kept * valid_mask.sum().item())
                else:
                    min_kept = int(self.ohem_min_kept)
                min_kept = max(min_kept, 1)  # 至少保留1个
            else:
                min_kept = 0
            # 如果基于阈值选取的样本数少于min_kept，则补充剩余难样本
            if min_kept > 0 and selected_count < min_kept:
                # 需要额外选择 (min_kept - selected_count) 个难样本
                # 计算所有有效点的逐点损失（用简单交叉熵衡量难度，不含ignore部分）
                ce_losses = F.cross_entropy(logits[valid_mask], target_valid, reduction='none')
                # 如果已经基于阈值选了一些，将它们的loss考虑进去或先排除? 
                # 这里对于未满足min_kept的情况，通常阈值严格导致hard样本少，
                # 我们直接根据loss排序选取 top (min_kept) 硬样本。
                num_valid = ce_losses.shape[0]
                if num_valid > 0:
                    # 获得交叉熵损失最大的 min_kept 个样本索引（在有效mask空间内）
                    topk = min(min_kept, num_valid)
                    _, hard_idx_valid = torch.topk(ce_losses, k=topk, largest=True)
                    hard_idx_global = valid_mask.nonzero(as_tuple=False).squeeze(1)[hard_idx_valid]
                    select_mask[hard_idx_global] = True
                    selected_count = topk  # 更新选择的样本数
            # 如果没有指定阈值，仅指定了min_kept，则按loss选取
            elif self.ohem_thresh is None and self.ohem_min_kept is not None:
                ce_losses = F.cross_entropy(logits[valid_mask], target_valid, reduction='none')
                num_valid = ce_losses.shape[0]
                topk = min(min_kept, num_valid)
                _, hard_idx_valid = torch.topk(ce_losses, k=topk, largest=True)
                hard_idx_global = valid_mask.nonzero(as_tuple=False).squeeze(1)[hard_idx_valid]
                select_mask[hard_idx_global] = True
                selected_count = topk
            # 若最终未选择任何样本，则默认使用所有有效样本
            if selected_count == 0:
                select_mask[valid_mask] = True
        else:
            # 未使用OHEM，则使用所有样本
            select_mask = None

        # 计算FocalLoss和DiceLoss
        if select_mask is not None:
            # 仅对选中的困难样本计算损失
            logits_sel = logits[select_mask]
            target_sel = target[select_mask]
            focal = self.focal_loss_fn(logits_sel, target_sel)
            dice = self.dice_loss_fn(logits_sel, target_sel)
        else:
            # 对所有样本计算损失
            focal = self.focal_loss_fn(logits, target)
            dice = self.dice_loss_fn(logits, target)
        # 此时 focal 和 dice 根据设置为逐元素张量，需要根据reduction取均值
        # 我们希望最终得到标量损失用于反向传播，因此在这里手动聚合
        if focal.dim() > 0:
            # 忽略未选中或ignore的样本，其损失为0
            focal_loss_val = focal.mean()
        else:
            focal_loss_val = focal  # 已经是标量
        if dice.dim() > 0:
            dice_loss_val = dice.mean()
        else:
            dice_loss_val = dice
        # 按权重组合两种损失
        total_loss = self.focal_weight * focal_loss_val + self.dice_weight * dice_loss_val
        return total_loss
