import torch
from torch import nn
from binding import binding
from utils import Linear
import torch.nn.functional as F

class DeepDock(nn.Module):
    def __init__(self, rank, rec_feature_dims, lig_feature_dims):
        super(DeepDock, self).__init__()
        self.device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
        self.binding = binding(self.device, rec_feature_dims, lig_feature_dims)
        # self.distance_mlp = torch.nn.Sequential(Linear(emb_dims, 64), torch.nn.LeakyReLU(), Linear(64, 1), torch.nn.Sigmoid())
        # self.distance_mlp = torch.nn.Sequential(Linear(emb_dims, emb_dims), torch.nn.LeakyReLU(), Linear(emb_dims, 1))

       

    def forward(self, lig_batch, lig_coords_batch, lig_feature_batch,\
                                            lig_edge_type_batch, rec_coords_batch, rec_feature_batch, rec_edge_type_batch, target_lig_coords_batch, idx_batch, lig_spatial_pos_batch, lig_edge_input_batch, docking):
        # 进入模型前向传播

        pair_matrix_batch, lig_distance_predict, lig_pos_predict_batch, rmsd_loss_batch, holo_logits_batch, cross_logits_batch = self.binding(lig_batch, lig_coords_batch, lig_feature_batch,\
                                            lig_edge_type_batch, rec_coords_batch, rec_feature_batch, rec_edge_type_batch, lig_spatial_pos_batch, lig_edge_input_batch, target_lig_coords_batch, docking)
        
        # distance_map_batch = []
        # for i in range(len(pair_matrix_batch)):
        #     # distance_map = self.distance_mlp(pair_matrix_batch[i]).squeeze(-1)
        #     # # distance_map = distance_map * 15
        #     # distance_map = F.elu(distance_map) + 1.0
        #     distance_map_batch.append(distance_map.to(torch.float32))

        return pair_matrix_batch, lig_distance_predict, lig_pos_predict_batch, rmsd_loss_batch, holo_logits_batch, cross_logits_batch

    def __repr__(self):
        return "DeepDock " + str(self.__dict__)