"""
MLDA 域适应损失函数
包含:
    1. 类条件 MMD (ICD loss): 最小化类内域差异，最大化类间域差异
    2. Wasserstein 距离 (批平均 1D EMD): 用于域间分布对齐

参考论文: "Multi-level domain adaptation for improved generalization in
electroencephalogram-based driver fatigue detection" (EAAI 2025)
"""
import torch
import numpy as np


# ========================================================================== #
#  类条件域差异 (Intra/Inter-Class Domain Discrepancy, ICD)                   #
# ========================================================================== #

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """多核最大均值差异 (MK-MMD)

    Args:
        source: (n, d) 源域特征
        target: (m, d) 目标域特征
        kernel_mul: 核带宽的乘数因子
        kernel_num: 核数量
        fix_sigma: 固定带宽 (None 则使用均值启发式)

    Returns:
        (n+m, n+m) 合并核矩阵
    """
    n = source.size(0)
    total = torch.cat([source, target], dim=0)

    # 计算成对 L2 距离矩阵
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    # 带宽计算: 均值启发式
    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        n_samples = total.size(0)
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    # 多尺度核
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]

    return sum(kernel_val)


def _coefficient(category_1, category_2, label_1, label_2):
    """类别指示器矩阵: 为类别对 (c1, c2) 创建布尔外积矩阵

    Args:
        category_1: 第一个类别标签
        category_2: 第二个类别标签
        label_1: (n,) 源域标签
        label_2: (m,) 目标域标签

    Returns:
        (n+m, n+m) 指示器矩阵 (float, 与输入同设备)
    """
    cls_bool1 = (label_1 == category_1).int()
    cls_bool2 = (label_2 == category_2).int()
    total_cls = torch.cat([cls_bool1, cls_bool2], dim=0).float()
    return torch.outer(total_cls, total_cls)


def idcd_loss(source, target, source_label, target_label,
              kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """类内/类间域差异损失 (Intra/Inter-Class Domain Discrepancy)

    最小化类内域差异 (同类样本跨域应尽量接近)，
    最大化类间域差异 (不同类样本跨域应尽量远离)。

    L_intra = mean(intra_class) - mean(inter_class)

    Args:
        source: (n, d) 源域特征
        target: (m, d) 目标域特征
        source_label: (n,) 源域真实标签
        target_label: (m,) 目标域伪标签
        kernel_mul, kernel_num, fix_sigma: 核参数

    Returns:
        标量损失
    """
    n = source.size(0)

    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    num_class = 2
    intra_class_vals = []
    inter_class_vals = []

    for c1 in range(num_class):
        for c2 in range(num_class):
            coef = _coefficient(c1, c2, source_label, target_label)

            e_ss = (coef[:n, :n] * XX).sum() / (coef[:n, :n].sum() + 1e-5)
            e_tt = (coef[n:, n:] * YY).sum() / (coef[n:, n:].sum() + 1e-5)
            e_st = (coef[:n, n:] * XY).sum() / (coef[:n, n:].sum() + 1e-5)
            e_ts = (coef[n:, :n] * YX).sum() / (coef[n:, :n].sum() + 1e-5)

            # MMD² = E[SS] + E[TT] - E[ST] - E[TS]
            mmd_val = e_ss + e_tt - e_st - e_ts

            if c1 == c2:
                intra_class_vals.append(mmd_val)
            else:
                inter_class_vals.append(mmd_val)

    # L_intra = mean(intra_class) - mean(inter_class)
    loss = (sum(intra_class_vals) / len(intra_class_vals)
            - sum(inter_class_vals) / len(inter_class_vals))
    return loss


# ========================================================================== #
#  Wasserstein 距离 (域间分布对齐)                                             #
# ========================================================================== #

def compute_wasserstein_distance(src_projected, tar_projected):
    """批平均 1D Wasserstein 距离 (Earth Mover's Distance)

    对投影后的特征在每个维度上计算 1D EMD，然后取所有维度的平均值。

    实现策略:
        - 优先使用 scipy.stats.wasserstein_distance (与原论文实现保持一致)
        - 若 scipy 不可用，回退到纯 PyTorch 排序实现 (完全可微)

    梯度流说明:
        在调用前应使用 .detach() 断开特征对主模型的梯度，
        梯度仅通过 U/V 投影网络回传。

    Args:
        src_projected: (B, d) U(src_feature)
        tar_projected: (B, d) V(tar_feature)

    Returns:
        Python float 标量距离
    """
    try:
        import scipy.stats
        src_np = src_projected.detach().cpu().numpy()
        tar_np = tar_projected.detach().cpu().numpy()
        d = src_np.shape[1]

        dist = 0.0
        for dim_idx in range(d):
            dist += scipy.stats.wasserstein_distance(
                src_np[:, dim_idx], tar_np[:, dim_idx]
            )
        return dist / d
    except ImportError:
        # 纯 PyTorch 回退: 排序法计算 1D Wasserstein 距离
        src_sorted, _ = torch.sort(src_projected, dim=0)
        tar_sorted, _ = torch.sort(tar_projected, dim=0)
        return torch.abs(src_sorted - tar_sorted).mean().item()


# ========================================================================== #
#  MMD 损失 (用于 DAEEGViT CLS token 域适应)                                  #
# ========================================================================== #

def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Maximum Mean Discrepancy (MMD) 损失

    使用多核高斯核 (MK-MMD) 度量源域和目标域特征分布差异。
    用于 DAEEGViT 在 CLS token 特征上的域适应。

    L_mmd = MMD²(P_s, P_t) = E[k(s,s')] + E[k(t,t')] - 2*E[k(s,t)]

    Args:
        source: (n, d) 源域 CLS 特征
        target: (m, d) 目标域 CLS 特征
        kernel_mul: 核带宽乘数因子
        kernel_num: 核数量
        fix_sigma: 固定带宽 (None 使用均值启发式)

    Returns:
        标量损失 (可微)
    """
    n = source.size(0)
    m = target.size(0)
    total = torch.cat([source, target], dim=0)

    # 成对 L2 距离矩阵
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    # 带宽: 均值启发式
    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        n_samples = total.size(0)
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    # 多尺度高斯核
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernels = sum([torch.exp(-L2_distance / bw) for bw in bandwidth_list])

    # MMD² = E[k(s,s)] + E[k(t,t)] - 2*E[k(s,t)]
    XX = kernels[:n, :n].sum() / (n * n)
    YY = kernels[n:, n:].sum() / (m * m)
    XY = kernels[:n, n:].sum() / (n * m)
    YX = kernels[n:, :n].sum() / (m * n)

    return XX + YY - XY - YX
