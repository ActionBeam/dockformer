import numpy as np
import torch
from torch.utils.data import DataLoader
# from pdbbindDataset import PDBBind, collate_revised
from Dataset import PDBBind, collate_revised
from DeepDock import DeepDock
from loss import TestLoss
from rdkit import Chem
import os
from tqdm import tqdm
import argparse
import sys
from distributed_utils import dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from trainer import Trainer
from my_metrics import Rsquared, MAE, RMSD, RMSDfraction, CentroidDist, CentroidDistFraction, RMSDmedian, \
    CentroidDistMedian, KabschRMSD
from utils import seed_all



def test(opt, name):
    # 测试阶段
    seed_value = 42
    seed_all(seed_value)
    batch_size = 10
    device = int(opt.device)
    test_names = name
    metrics_dict = {'rsquared': Rsquared(),
                    'mean_rmsd': RMSD(),
                    'mean_centroid_distance': CentroidDist(),
                    'rmsd_less_than_2': RMSDfraction(2),
                    'rmsd_less_than_5': RMSDfraction(5),
                    'rmsd_less_than_10': RMSDfraction(10),
                    'rmsd_less_than_20': RMSDfraction(20),
                    'rmsd_less_than_50': RMSDfraction(50),
                    'median_rmsd': RMSDmedian(),
                    'median_centroid_distance': CentroidDistMedian(),
                    'centroid_distance_less_than_2': CentroidDistFraction(2),
                    'centroid_distance_less_than_5': CentroidDistFraction(5),
                    'centroid_distance_less_than_10': CentroidDistFraction(10),
                    'centroid_distance_less_than_20': CentroidDistFraction(20),
                    'centroid_distance_less_than_50': CentroidDistFraction(50),
                    'kabsch_rmsd': KabschRMSD(),
                    'mae': MAE()
                    }

    test_metrics_name = ['mean_rmsd', 'rmsd_less_than_2', 'rmsd_less_than_5', 'mean_centroid_distance',
                         'centroid_distance_less_than_2', 'centroid_distance_less_than_5']
    test_metrics = {metric: metrics_dict[metric] for metric in test_metrics_name}

    # 读取保存的模型参数
    checkpoint = torch.load(
        os.path.join('./Output/parameter/best/best_parameter_docking.pt'), map_location=f"cuda:{device}")
    model_test = DeepDock(device, rec_feature_dims=79, lig_feature_dims=110).to(f"cuda:{device}")
    # new_state_dict = collections.OrderedDict()
    # for k, v in checkpoint.items():
    #     name = k[7:]  # remove "module."
    #     new_state_dict[name] = v
    # model_test.load_state_dict(new_state_dict)
    model_test.load_state_dict(checkpoint, strict=False)

    trainer_test = Trainer(model_test, device, 1, test_metrics, loss_func=TestLoss, mode=1, docking=True, name=name)

    # 数据
    test = PDBBind(test_names, rank=device, seed=seed_value)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate_revised)
    test_dataloader.dataset.set_epoch(0)
    trainer_test.evaluation(test_dataloader, data_split=name)

if __name__ == '__main__':
    """
        world_size: 所有的进程数量
        rank: 全局的进程id
    """
    parser = argparse.ArgumentParser(description='simple distributed training job')
    # parser.add_argument('--path', default='/home/usr/dockformer', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--world-size', default=8, type=int, help='number of distributed processes')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()
    world_size = opt.world_size

    # os.chdir(opt.path)
    # casf2016
    # test(opt, 'test')

    # posebuster
    test(opt, 'pose_buster')
    
    # DockGen
    #test(opt, 'DockGen')




 

