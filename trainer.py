import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import list_detach, move_to_device
from scipy.spatial.transform import Rotation
from rdkit import Chem
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import write_with_new_coords
from typing import Dict, Callable
import os
import random
from torch.utils.tensorboard import SummaryWriter
from MyGenerateCoords import generateCoord
import pandas as pd
from transformers.optimization import get_linear_schedule_with_warmup


class Trainer():
    def __init__(self, model, rank, world_size, metrics: Dict[str, Callable] = None, main_metric: str = "BindingLoss",
                 main_metric_goal: str = 'min', loss_func=torch.nn.MSELoss(), mode: int = 0, docking: bool = False, name=""):
        self.device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
        self.world_size = world_size
        self.rank = rank
        self.model = model.to(self.device)
        # 模型并行化
        self.mode = mode
        # if self.mode == 0:
        #     # self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model) #将BatchNorm层转换为SyncBatchNorm层
        #     self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
        self.loss_func = loss_func(device=self.device)  # loss函数device的初始化
        if docking:
            self.optim = torch.optim.AdamW(self.model.parameters(), lr=5.0e-6, weight_decay=1.0e-6)
        else:
            self.optim = torch.optim.AdamW(self.model.parameters(), lr=5.0e-4, weight_decay=1.0e-6)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[10], gamma=0.1)
        self.num_epochs = 300  # epoch次数
        self.warmup_ratio = 0.1
        self.iter = 0
        self.epoch = 0
        self.batch_loss = []
        self.log = open(r'./Output/log', 'a+')
        self.log_rmsd = open(r'./Output/rmsd_log', 'a+')
        self.sdf_file = f'./Output/SDF/sdf/{name}'
        if not os.path.exists(self.sdf_file):
            os.makedirs(self.sdf_file)
        # 新添加的评估指标
        self.metrics = metrics
        self.main_metric = main_metric
        self.main_metric_goal = main_metric_goal
        if not docking:
            self.writer = SummaryWriter(f'./Output/Running/run')
        else:
            self.writer = SummaryWriter(f'./Output/Running/run/docking')
        self.outshow = './Output/showtest.csv'
        self.best_val_score = -np.inf if self.main_metric_goal == 'max' else np.inf
        self.generateSample = generateCoord(self.device)
        self.patience = 20  # 用于早停法，连续20个epoch没有上升时，停止训练
        self.counter = 0
        self.docking = docking


    def forward_pass(self, batch):
        lig_batch, lig_name_batch, lig_coords_batch, lig_feature_batch, lig_edge_type_batch, rec_coords_batch, rec_feature_batch, rec_edge_type_batch, target_lig_coords_batch, idx_batch, LAS_mask_batch, lig_spatial_pos_batch, lig_edge_input_batch = batch

        distance_predict_batch, lig_distance_predict_batch, lig_pos_predict_batch, rmsd_loss_batch, holo_logits_batch, cross_logits_batch = self.model(lig_batch, lig_coords_batch, lig_feature_batch, \
                                                                        lig_edge_type_batch, rec_coords_batch,
                                                                        rec_feature_batch, rec_edge_type_batch,
                                                                        target_lig_coords_batch, idx_batch,
                                                                        lig_spatial_pos_batch, lig_edge_input_batch,
                                                                        self.docking)  # foward the rest of the batch to the model
        # 计算rec-lig距离的target矩阵
        distance_target_batch = []
        lig_distance_target_batch = []
        for i in range(len(distance_predict_batch)):
            lig_rec_distance = torch.cdist(target_lig_coords_batch[i], rec_coords_batch[i],
                                           compute_mode='donot_use_mm_for_euclid_dist').to(
                torch.float32)  # 计算rec-lig相互距离，不使用矩阵乘法计算欧式距离
            # lig_rec_distance[lig_rec_distance > 15] = 15
            distance_target_batch.append(lig_rec_distance)

            lig_distance = torch.cdist(target_lig_coords_batch[i], target_lig_coords_batch[i],
                                       compute_mode='donot_use_mm_for_euclid_dist').to(torch.float32)

            lig_distance_target_batch.append(lig_distance)

        loss, loss_components = self.loss_func(distance_predict_batch, distance_target_batch, lig_distance_predict_batch, lig_distance_target_batch, lig_pos_predict_batch, target_lig_coords_batch, rmsd_loss_batch, LAS_mask_batch, holo_logits_batch, cross_logits_batch, rec_coords_batch, self.docking)
        conf_parameter = [distance_predict_batch, lig_distance_predict_batch, lig_pos_predict_batch]
        # if loss_func does not return any loss_components, we turn the empty list into None
        return loss, (loss_components if loss_components != {} else None), conf_parameter

    def reduce_loss(self, loss, average=True):
        world_size = self.world_size
        if world_size < 2:  # 单GPU的情况
            return loss
        with torch.no_grad():
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)  # 对不同设备之间的value求和
            if average:  # 如果需要求平均，获得多块GPU计算loss的均值
                loss /= world_size
        return loss

    def process_batch(self, batch, optim):
        loss, loss_components, conf_parameter = self.forward_pass(batch)
        if optim != None:
            loss.backward()
            # 是否要加入梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2)
            self.optim.step()
            self.optim.zero_grad()
            if not self.docking:
                self.scheduler.step()
            self.iter = self.iter + 1

        loss = self.reduce_loss(loss, average=True)

        return loss.item(), loss_components, list_detach(conf_parameter)

    def evaluate_metrics(self, predictions, targets, batch=None, val=False) -> Dict[str, float]:
        metrics = {}
        for key, metric in self.metrics.items():
            if not hasattr(metric, 'val_only') or val:
                metrics[key] = metric(predictions, targets).item()
        return metrics

    def tensorboard_log(self, metrics, data_split: str, step: int, log_hparam: bool = False):
        metrics['epoch'] = self.epoch
        logs = {}
        for key, metric in metrics.items():
            metric_name = f'{key}/{data_split}'
            logs[metric_name] = metric
            self.writer.add_scalar(metric_name, metric, step)

    def evaluation(self, test_loader: DataLoader, data_split: str = ''):
        self.model.eval()
        total_metrices = {k: 0 for k in list(self.metrics.keys()) + ['Testloss']}
        # 写入csv文件，展示最终的结果
        show_rmsd = []
        show_loss = []
        show_name = []
        for i, batch in tqdm(enumerate(test_loader), desc=f'test ...'):
            batch = move_to_device(list(batch), self.device)
            with torch.no_grad():
                loss, loss_components, conf_parameter = self.process_batch(batch, optim=None)
            target_lig_coords_batch = batch[8]
            rec_coords_batch = batch[5]
            LAS_mask_batch = batch[10]
            compound_pair_dis_batch = []
            new_lig_coords_batch = []
            target_lig = batch[0]
            lig_name_batch = batch[1]
            n_repeat = 1
            distance_predict_batch = conf_parameter[0]
            lig_predict_distance_batch = conf_parameter[1]
            lig_pos_predict_batch = conf_parameter[2]
            min_index = loss_components["min_loss_index"]
            # v2.0替换为网络预测的结果
            compound_pair_dis_batch = lig_predict_distance_batch

            metrics = self.evaluate_metrics([lig_pos_predict_batch[min_index]], [target_lig_coords_batch[min_index]])
            new_lig_coords_batch.append(lig_pos_predict_batch[min_index])
            for new_coord in new_lig_coords_batch:
                write_with_new_coords(target_lig[min_index], new_coord, f'{self.sdf_file}/{lig_name_batch[min_index]}.sdf')

            metrics["Testloss"] = loss
            show_rmsd.append(metrics['mean_rmsd'])
            show_loss.append(metrics['Testloss'])
            show_name.append(lig_name_batch[0])

            for key, value in metrics.items():
                total_metrices[key] += value
            self.tensorboard_log(metrics, data_split='test', step=i)
            # print(lig_name_batch)

        for key, value in total_metrices.items():
            total_metrices[key] = value / len(test_loader)
            print(f"{key}: {total_metrices[key]}")

        # 按照预测坐标的rmsd大小从小到大排序并写入csv文件
        showdata = {'name': show_name, 'rmsd': show_rmsd, 'loss': show_loss}
        df = pd.DataFrame(showdata)
        df_sorted = df.sort_values('rmsd')
        df_sorted.to_csv(self.outshow, index=False)

