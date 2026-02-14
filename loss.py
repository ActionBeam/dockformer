from torch.nn.modules.loss import _Loss, MSELoss
import torch
from scipy.spatial.transform import Rotation
from utils import quat_to_mat, softmax_cross_entropy, get_pair_dis_one_hot
import numpy as np
import random


def lddt(
    lig_lig_atom_pred_dis: torch.Tensor,
    lig_lig_atom_label_dis: torch.Tensor,
    rec_lig_atom_pred_dis: torch.Tensor,
    rec_lig_atom_label_dis: torch.Tensor,
    cutoff: float = 8.0,
    eps: float = 1e-10,
) -> torch.Tensor:

    n = lig_lig_atom_pred_dis.shape[0]
    # 计算真实距离小于cut off的元素
    holo_dists_to_score = (
        (lig_lig_atom_label_dis < 100.0)
        * (1.0 - torch.eye(n, device=lig_lig_atom_pred_dis.device))
    )

    # 计算真实距离和预测距离的误差
    holo_dist_l1 = torch.abs(lig_lig_atom_label_dis - lig_lig_atom_pred_dis)

    holo_score = (    # 计算距离是否小于这些阈值，如果小于就是1，假设有一个距离是1.5，那么score矩阵的值为2
        (holo_dist_l1 < 0.5).type(holo_dist_l1.dtype)
        + (holo_dist_l1 < 1.0).type(holo_dist_l1.dtype)
        + (holo_dist_l1 < 2.0).type(holo_dist_l1.dtype)
        + (holo_dist_l1 < 4.0).type(holo_dist_l1.dtype)
    )
    holo_score = holo_score * 0.25

    # holo_norm = 1.0 / (eps + torch.sum(holo_dists_to_score, dim=-1))  # 1/真实原子间距离小于15的原子数
    # holo_score = holo_norm * (eps + torch.sum(holo_dists_to_score * holo_score, dim=-1))  # 在小于15的原子中，求出score矩阵的值并求和

    cross_dists_to_score = rec_lig_atom_label_dis < cutoff

    cross_dist_l1 = torch.abs(rec_lig_atom_label_dis - rec_lig_atom_pred_dis)

    cross_score = (    # 计算距离是否小于这些阈值，如果小于就是1，假设有一个距离是1.5，那么score矩阵的值为2
        (cross_dist_l1 < 0.5).type(cross_dist_l1.dtype)
        + (cross_dist_l1 < 1.0).type(cross_dist_l1.dtype)
        + (cross_dist_l1 < 2.0).type(cross_dist_l1.dtype)
        + (cross_dist_l1 < 4.0).type(cross_dist_l1.dtype)
    )
    cross_score = cross_score * 0.25

    # cross_norm = 1.0 / (eps + torch.sum(cross_dists_to_score, dim=-1))  # 1/真实原子间距离小于15的原子数
    # cross_score = cross_norm * (eps + torch.sum(cross_dists_to_score * cross_score, dim=-1))  # 在小于15的原子中，求出score矩阵的值并求和

    return holo_score, cross_score, holo_dists_to_score.to(torch.bool), cross_dists_to_score.to(torch.bool)


def lddt_loss(
    holo_logits: torch.Tensor,
    cross_logits: torch.Tensor,
    lig_lig_atom_pred_dis: torch.Tensor,
    lig_lig_atom_label_dis: torch.Tensor,
    rec_lig_atom_pred_dis: torch.Tensor,
    rec_lig_atom_label_dis: torch.Tensor,
    cutoff: float = 8.0,
    no_bins: int = 50,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:

    holo_score, cross_score, holo_dists_to_score, cross_dists_to_score = lddt(
        lig_lig_atom_pred_dis,
        lig_lig_atom_label_dis,
        rec_lig_atom_pred_dis,
        rec_lig_atom_label_dis,
        cutoff=cutoff,
        eps=eps
    )

    holo_score, cross_score = holo_score.detach(), cross_score.detach()   # 不计算梯度

    bin_index = torch.floor(holo_score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    holo_one_hot = torch.nn.functional.one_hot(
        bin_index, num_classes=no_bins
    )

    errors = softmax_cross_entropy(holo_logits[holo_dists_to_score], holo_one_hot[holo_dists_to_score])
    holo_loss = torch.mean(errors)

    bin_index = torch.floor(cross_score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    cross_one_hot = torch.nn.functional.one_hot(
        bin_index, num_classes=no_bins
    )

    errors = softmax_cross_entropy(cross_logits[cross_dists_to_score], cross_one_hot[cross_dists_to_score])
    cross_loss = torch.mean(errors)

    loss = (holo_loss + cross_loss) / 2

    return loss


def compute_plddt(holo_logits: torch.Tensor, cross_logits) -> torch.Tensor:
    num_bins = holo_logits.shape[-1]  # no_bins
    bin_width = 1.0 / num_bins  # 得到每个bins的宽度
    bounds = torch.arange(  # [no_bins]
        start=0.5 * bin_width, end=1.0, step=bin_width, device=holo_logits.device
    )
    holo_probs = torch.nn.functional.softmax(holo_logits, dim=-1)  # [num_res, no_bins]
    holo_pred_lddt = torch.sum(   # 对应算法的第5行每个概率和对应的bins相乘，最后把残基内的所有乘法结果求和
        holo_probs * bounds.view(*((1,) * len(holo_probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    cross_probs = torch.nn.functional.softmax(cross_logits, dim=-1)  # [num_res, no_bins]
    cross_pred_lddt = torch.sum(   # 对应算法的第5行每个概率和对应的bins相乘，最后把残基内的所有乘法结果求和
        cross_probs * bounds.view(*((1,) * len(cross_probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    pred_lddt = (holo_pred_lddt.mean() + cross_pred_lddt.mean()) / 2

    return pred_lddt * 100

class TestLoss(_Loss):
    def __init__(self, device):
        super(TestLoss, self).__init__()
        self.rank = device
        self.dist_threshold = 8

    def forward(self, distance_predict_batch, distance_target_batch, lig_distance_predict_batch,
                lig_distance_target_batch, lig_pos_predict_batch, target_lig_coords_batch, rmsd_loss_batch,
                holo_logits_batch, cross_logits_batch, rec_coords_batch, docking):
        coord_loss = []
        mseloss = torch.nn.MSELoss()
        huber_loss = torch.nn.SmoothL1Loss()
        for i in range(len(distance_predict_batch)):
            if docking:
                rmsd_losss = rmsd_loss_batch[i]
                coord_loss_tmp = rmsd_losss
                coord_loss.append(coord_loss_tmp)
            else:
                coord_loss.append(1e-8)

        loss = [z for z in coord_loss]
        min_loss = min(loss)
        min_loss_index = loss.index(min_loss)

        if docking:
            print('\nplddt:',compute_plddt(holo_logits_batch[min_loss_index], cross_logits_batch[min_loss_index]).item())
        return min_loss, {'coord_loss':coord_loss[min_loss_index], "min_loss_index": min_loss_index}