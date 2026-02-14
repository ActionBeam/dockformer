import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import hashlib

from tqdm import tqdm
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import glob
import torch
from torch import nn

from datetime import datetime
import logging
from io import StringIO
import sys
from scipy.spatial.transform import Rotation
from torch.utils.tensorboard import SummaryWriter


class generateCoord():
    def __init__(self, rank):
        super(generateCoord, self).__init__()
        self.rank = rank
        self.writer = SummaryWriter(f'./Output/coordRun')

    def compute_RMSD(self, a, b):
        return torch.sqrt((((a - b) ** 2).sum(axis=-1)).mean())

    def distance_loss_function(self, epoch, y_pred, x, protein_nodes_xyz, compound_pair_dis_constraint):
        protein_nodes_xyz = protein_nodes_xyz.to(self.rank)

        dis = torch.cdist(x, protein_nodes_xyz, compute_mode='donot_use_mm_for_euclid_dist')  # 蛋白质和drug的随机距离图
        dis_clamp = torch.clamp(dis, max=10)
        # dis_clamp = torch.clamp(dis, max=4.5)

        interaction_loss = ((dis_clamp - y_pred).abs()).sum()
        config_dis = torch.cdist(x, x).to(self.rank)
        # configuration_loss = 1 * (((config_dis-compound_pair_dis_constraint).abs())[LAS_distance_constraint_mask]).sum()
        configuration_loss = 1 * (((config_dis - compound_pair_dis_constraint).abs())).sum()
        # basic exlcuded-volume. the distance between compound atoms should be at least 1.22Å
        configuration_loss += 2 * ((1.22 - config_dis).relu()).sum()

        if epoch < 500:
            loss = interaction_loss
        else:
            loss = 1 * (interaction_loss + 5e-3 * (epoch - 500) * configuration_loss)

        return loss, (interaction_loss.item(), configuration_loss.item())

    def tensorboard_log(self, metrics, data_split: str, step: int, log_hparam: bool = False):
        logs = {}
        for key, metric in metrics.items():
            metric_name = f'{key}/{data_split}'
            logs[metric_name] = metric
            self.writer.add_scalar(metric_name, metric, step)

    def distance_optimize_compound_coords(self, coords, y_pred, protein_nodes_xyz, compound_pair_dis_constraint,
                                          LAS_mask, total_epoch=5000):
        c_pred = protein_nodes_xyz.mean(axis=0).to(self.rank)
        x = (5 * (2 * torch.rand(coords.shape).to(self.rank) - 1)) + c_pred.reshape(1, 3).detach()
        x.requires_grad = True
        optimizer = torch.optim.Adam([x], lr=0.1)
        # optimizer = torch.optim.LBFGS([x], lr=1.0)
        LAS_lig_pair = torch.cdist(coords, coords, compute_mode='donot_use_mm_for_euclid_dist').to(self.rank)
        compound_pair_dis_constraint[LAS_mask] = LAS_lig_pair[LAS_mask].to(torch.float32)

        res = {"loss": 0, "interaction_loss": 0, "configuration_loss": 0, "rmsd": 0}
        for epoch in range(total_epoch):
            # LBFGS优化器
            # def closure():
            #     optimizer.zero_grad()
            #     loss, (interaction_loss, configuration_loss) = self.distance_loss_function(epoch, y_pred, x, protein_nodes_xyz, compound_pair_dis_constraint, LAS_distance_constraint_mask)
            #     loss_value = loss.item()
            #     loss.backward()
            #     return loss_value
            # loss_v = optimizer.step(closure)
            # res["loss"] = loss_v

            # Adam优化器
            optimizer.zero_grad()
            loss, (interaction_loss, configuration_loss) = self.distance_loss_function(epoch, y_pred, x,
                                                                                       protein_nodes_xyz,
                                                                                       compound_pair_dis_constraint)
            loss.backward()
            optimizer.step()

            res['loss'] = loss.item()
            res["configuration_loss"] = configuration_loss
            res["interaction_loss"] = interaction_loss
            res["rmsd"] = self.compute_RMSD(coords, x.detach())

            self.tensorboard_log(res, data_split='generate_coords', step=epoch)
        return x

    def get_info_pred_distance(self, coords, y_pred, protein_nodes_xyz, compound_pair_dis_constraint, n_repeat,
                               LAS_mask):

        for repeat in range(n_repeat):
            x = self.distance_optimize_compound_coords(coords, y_pred, protein_nodes_xyz.to(torch.float32),
                                                       compound_pair_dis_constraint, LAS_mask)
        return x.detach()
