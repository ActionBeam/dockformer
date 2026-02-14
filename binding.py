import torch
from torch import nn
from torch.nn import init
from utils import get_pair_dis_one_hot, Linear, random_rotation_translation
from torch.autograd import Variable
import math
from torch.autograd import Variable
import random
from multi_attention import pair_Transition
from transformer_encoder_layer import EncoderBlock
from feature_utils import get_mol_bond
import torch.nn.functional as F
import os
from unimolDataset import Dictionary
import numpy as np
from structure_module import StructureModule, RecyclingEmbedder, Gate_Block

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, self.K)
        # print(x.shape)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class DistanceHead(nn.Module):
    def __init__(self, heads, activation_fn):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_porj = nn.Linear(heads, 1)
        self.activation_fn = nn.ReLU()

    def forward(self, x):
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_porj(x)
        x = x.view(-1, x.size(1), x.size(1))
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
            self,
            input_dim,
            out_dim,
            hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        # self.bn1 = nn.BatchNorm1d(64)   暂时不需要BN,后续改代码可能会需要改
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(1024)
        # self.bn4 = nn.BatchNorm1d(512)
        # self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        x = x.unsqueeze(0).permute(0, 2, 1)  # (1, dim, N)
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1).to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        x = x.squeeze(0)
        return x



class binding(nn.Module):
    def __init__(self, device, rec_feature_dims, lig_feature_dims, emb_dim=512, ffn_embed_dim=2048, \
                 attention_head=64, encoder_layer=15, cross_encoder_layer=4, dropout=0.1, **kwargs):
        super(binding, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.num_head = attention_head
        self.final_layer_norm_lig = nn.LayerNorm(self.emb_dim)
        self.final_layer_norm_rec = nn.LayerNorm(self.emb_dim)
        self.mol_dictionary = Dictionary.load(os.path.join("./lmdb", "dict_mol.txt"))
        self.pocket_dictionary = Dictionary.load(os.path.join("./lmdb", "dict_pkt.txt"))

        self.rec_feature_dims = rec_feature_dims
        self.lig_feature_dims = lig_feature_dims
        # v1.0
        # self.lig_embedder = Linear(lig_feature_dims, self.emb_dim)
        # self.rec_embedder = Linear(rec_feature_dims, self.emb_dim)
        # v2.0
        self.lig_embedder = nn.Embedding(len(self.mol_dictionary), self.emb_dim)
        self.rec_embedder = nn.Embedding(len(self.pocket_dictionary), self.emb_dim)
        self.div_term = torch.exp(torch.arange(0, self.emb_dim, 2) *  # 计算公式中10000**（2i/d_model)
                                  -(math.log(10000.0) / self.emb_dim)).to(self.device)
        self.lig_pair_dis_embedder = torch.nn.Sequential(Linear(128, 64), torch.nn.LeakyReLU(),
                                                         Linear(64, self.num_head), nn.LayerNorm(self.num_head))
        self.rec_pair_dis_embedder = torch.nn.Sequential(Linear(128, 64), torch.nn.LeakyReLU(),
                                                         Linear(64, self.num_head), nn.LayerNorm(self.num_head))
        self.rec_pos_embed = Linear(3 * self.emb_dim, self.emb_dim)
        self.lig_pos_embed = Linear(3 * self.emb_dim, self.emb_dim)
        # self.lig_bond_bias_embedder = torch.nn.Sequential(Linear(3, attention_head))
        self.lig_bond_bias_embedder = torch.nn.Embedding(512 * 3, self.num_head)
        self.lig_bias_embedder = Linear(128 + 3, self.num_head)
        # v1.0
        # self.gbf = GaussianLayer(128, self.device)
        # v2.0
        self.lig_gbf = GaussianLayer(128, len(self.mol_dictionary) * len(self.mol_dictionary))
        self.rec_gbf = GaussianLayer(128, len(self.pocket_dictionary) * len(self.pocket_dictionary))
        self.lig_gbf_proj = NonLinearHead(128, int(self.num_head/2))
        self.rec_gbf_proj = NonLinearHead(128, self.num_head)
        self.recyling = 3
        self.rec_emb_layers = nn.Sequential()
        for i in range(encoder_layer):
            self.rec_emb_layers.add_module("block" + str(i),
                                           EncoderBlock(emb_dims=self.emb_dim, ffn_embed_dims=self.ffn_embed_dim,
                                                        attention_head=self.num_head, \
                                                        dropout=dropout))

        self.lig_emb_layers = nn.Sequential()
        for i in range(encoder_layer):
            self.lig_emb_layers.add_module("block" + str(i),
                                           EncoderBlock(emb_dims=self.emb_dim, ffn_embed_dims=self.ffn_embed_dim,
                                                        attention_head=self.num_head, \
                                                        dropout=dropout, is_cross=False, is_talking=True))
        self.cross_attention_layers = nn.Sequential()
        for i in range(cross_encoder_layer):
            self.cross_attention_layers.add_module("block" + str(i),
                                                   EncoderBlock(emb_dims=self.emb_dim,
                                                                ffn_embed_dims=self.ffn_embed_dim,
                                                                attention_head=self.num_head, \
                                                                dropout=dropout, is_cross=True))

        self.pair_Transition = pair_Transition(embedding_channels=self.num_head + 2 * self.emb_dim, out_dims=1)
        self.lig_bias_project = DistanceHead(self.emb_dim + attention_head, "relu")
        self.num_edges = 512 * 3
        self.num_spatial = 512
        self.num_edge_dis = 128
        self.num_kernel = 128
        self.multi_hop_max_dist = 5
        self.spatial_pos_encoder = nn.Embedding(self.num_spatial, int(self.num_head/2))
        self.edge_encoder = nn.Embedding(self.num_edges + 1, int(self.num_head/2))
        self.edge_dis_encoder = nn.Embedding(self.num_edge_dis * int(self.num_head/2) * int(self.num_head/2), 1)
        self.sm_lig_rec_pair_Transition = Linear(self.num_head, self.num_head)
        self.sm_lig_lig_pair_Transition = Linear(self.num_head, self.num_head)
        self.structure_module = StructureModule(c_hidden=self.emb_dim, no_heads=self.num_head,
                                                ffn_embed_dim=self.ffn_embed_dim, dropout=dropout)
        self.recyling_embedder = RecyclingEmbedder(no_head=self.num_head)
        self.atom_gate_layer = Gate_Block(self.emb_dim)
        self.lig_pair_gate_layer = Gate_Block(self.num_head)
        self.lig_rec_pair_gate_layer = Gate_Block(self.num_head)
        self.holo_plddt_layer = nn.Sequential(nn.LayerNorm(self.emb_dim + attention_head),
                                              Linear(self.emb_dim + attention_head, 128, init="relu"), nn.ReLU(),
                                              Linear(128, 128, init="relu"), nn.ReLU(),
                                              Linear(128, 50, init="final"))
        self.cross_plddt_layer = nn.Sequential(nn.LayerNorm(2 * self.emb_dim + attention_head),
                                               Linear(2 * self.emb_dim + attention_head, 128, init="relu"), nn.ReLU(),
                                               Linear(128, 128, init="relu"), nn.ReLU(),
                                               Linear(128, 50, init="final"))




    def forward(self, lig_batch, lig_coords_batch, lig_feature_batch, lig_edge_type_batch, rec_coords_batch,
                rec_feature_batch, rec_edge_type_batch, lig_spatial_pos_batch, lig_edge_input_batch, target_lig_coords_batch, docking):
        pair_matrix_batch = []
        lig_distance_predict_batch = []
        lig_pos_predict_batch = []
        rmsd_loss_batch = []
        holo_logits_batch = []
        cross_logits_batch = []
        for i in range(len(lig_batch)):
            # 特征的embedding
            lig_feature = lig_feature_batch[i]
            lig_feat = self.lig_embedder(lig_feature)  # [N, emb_dim]
            rec_feature = rec_feature_batch[i]
            rec_feat = self.rec_embedder(rec_feature)  # [M, emb_dim]

            # position embedding(sin)
            random_orientation, random_position = random_rotation_translation(translation_distance=5, device=self.device)
            lig_coord = lig_coords_batch[i]
            lig_noise_coords = (random_orientation @ lig_coord.T).T + random_position
            lig_pos_x = lig_noise_coords[:, 0].unsqueeze(
                -1) * self.div_term  # (B, N_view, H, W, num_feats)      [pos_view/10000^(0/128), pos_view/10000^(0/128), pos_view/10000^(2/128), pos_view/10000^(2/128), ...]
            lig_pos_y = lig_noise_coords[:, 1].unsqueeze(
                -1) * self.div_term  # (B, N_view, H, W, num_feats)      [pos_x/10000^(0/128), pos_x/10000^(0/128), pos_x/10000^(2/128), pos_x/10000^(2/128), ...]
            lig_pos_z = lig_noise_coords[:, 2].unsqueeze(
                -1) * self.div_term  # (B, N_view, H, W, num_feats)      [pos_y/10000^(0/128), pos_y/10000^(0/128), pos_y/10000^(2/128), pos_y/10000^(2/128), ...]
            lig_pe_x = torch.zeros(lig_pos_x.shape[0], self.emb_dim, device=self.device)
            lig_pe_x[:, 0::2] = torch.sin(lig_pos_x)  # 计算偶数维度的pe值
            lig_pe_x[:, 1::2] = torch.cos(lig_pos_x)  # 计算奇数维度的pe值
            lig_pe_y = torch.zeros(lig_pos_y.shape[0], self.emb_dim, device=self.device)
            lig_pe_y[:, 0::2] = torch.sin(lig_pos_y)  # 计算偶数维度的pe值
            lig_pe_y[:, 1::2] = torch.cos(lig_pos_y)  # 计算奇数维度的pe值
            lig_pe_z = torch.zeros(lig_pos_z.shape[0], self.emb_dim, device=self.device)
            lig_pe_z[:, 0::2] = torch.sin(lig_pos_z)  # 计算偶数维度的pe值
            lig_pe_z[:, 1::2] = torch.cos(lig_pos_z)  # 计算奇数维度的pe值

            lig_posemb = torch.cat((lig_pe_x, lig_pe_y, lig_pe_z), dim=-1)
            lig_pos_embedding = self.lig_pos_embed(lig_posemb)

            rec_coord = rec_coords_batch[i]
            noise = rec_coord.mean(dim=0)
            rec_noise_coords = rec_coord - noise
            random_orientation, random_position = random_rotation_translation(translation_distance=5,
                                                                              device=self.device)
            rec_noise_coords = (random_orientation @ rec_noise_coords.T).T + random_position
            rec_pos_x = rec_noise_coords[:, 0].unsqueeze(
                -1) * self.div_term  # (B, N_view, H, W, num_feats)      [pos_view/10000^(0/128), pos_view/10000^(0/128), pos_view/10000^(2/128), pos_view/10000^(2/128), ...]
            rec_pos_y = rec_noise_coords[:, 1].unsqueeze(
                -1) * self.div_term  # (B, N_view, H, W, num_feats)      [pos_x/10000^(0/128), pos_x/10000^(0/128), pos_x/10000^(2/128), pos_x/10000^(2/128), ...]
            rec_pos_z = rec_noise_coords[:, 2].unsqueeze(
                -1) * self.div_term  # (B, N_view, H, W, num_feats)      [pos_y/10000^(0/128), pos_y/10000^(0/128), pos_y/10000^(2/128), pos_y/10000^(2/128), ...]
            rec_pe_x = torch.zeros(rec_pos_x.shape[0], self.emb_dim, device=self.device)
            rec_pe_x[:, 0::2] = torch.sin(rec_pos_x)  # 计算偶数维度的pe值
            rec_pe_x[:, 1::2] = torch.cos(rec_pos_x)  # 计算奇数维度的pe值
            rec_pe_y = torch.zeros(rec_pos_y.shape[0], self.emb_dim, device=self.device)
            rec_pe_y[:, 0::2] = torch.sin(rec_pos_y)  # 计算偶数维度的pe值
            rec_pe_y[:, 1::2] = torch.cos(rec_pos_y)  # 计算奇数维度的pe值
            rec_pe_z = torch.zeros(rec_pos_z.shape[0], self.emb_dim, device=self.device)
            rec_pe_z[:, 0::2] = torch.sin(rec_pos_z)  # 计算偶数维度的pe值
            rec_pe_z[:, 1::2] = torch.cos(rec_pos_z)  # 计算奇数维度的pe值

            rec_posemb = torch.cat((rec_pe_x, rec_pe_y, rec_pe_z), dim = -1)
            rec_pos_embedding = self.rec_pos_embed(rec_posemb)


            # 获取原子距离, 得到的bias加入到attention的过程中
            lig_distance = torch.cdist(lig_noise_coords, lig_noise_coords,
                                       compute_mode='donot_use_mm_for_euclid_dist')  # [N, N]
            rec_distance = torch.cdist(rec_noise_coords, rec_noise_coords,
                                       compute_mode='donot_use_mm_for_euclid_dist')  # [M, M]

            # 2D bias
            spatial_pos = lig_spatial_pos_batch[i]
            edge_input = lig_edge_input_batch[i]
            # 初始化graph_attn_bias
            graph_attn_bias = torch.zeros([lig_feat.size(0), lig_feat.size(0)], dtype=torch.float32).to(self.device)
            graph_attn_bias = graph_attn_bias.unsqueeze(0).repeat(self.num_head, 1, 1)  # [num_head, N, N]
            spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(2, 0, 1)  # [N, N] -> [N, N, num_head/2] ->[num_head/2, N, N]
            graph_attn_bias[:int(self.num_head/2), :, :] = graph_attn_bias[:int(self.num_head/2), :, :] + spatial_pos_bias
            spatial_pos_ = spatial_pos.clone()
            # 最短路径长度大于1的，减去1
            spatial_pos_[spatial_pos_ == 0] = 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :self.multi_hop_max_dist, :]  # [N, N, self.multi_hop_max_dist, 3]
            edge_input = self.edge_encoder(edge_input).mean(
                -2)  # [N, N, self.multi_hop_max_dist, 3, num_head] -> [N, N, self.multi_hop_max_dist, num_head]
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(2, 0, 1, 3).reshape(max_dist, -1, int(self.num_head/2))

            edge_input_flat = torch.matmul(edge_input_flat,
                                           self.edge_dis_encoder.weight.reshape(-1, int(self.num_head/2), int(self.num_head/2))[
                                           :max_dist, :, :])
            # [max_dist, N, N, num_head] -> [N, N, max_dist, num_head]
            edge_input = edge_input_flat.reshape(max_dist, lig_feature.size(0), lig_feature.size(0),
                                                 int(self.num_head/2)).permute(1, 2, 0, 3)
            edge_input = (edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))).permute(2, 0, 1)
            graph_attn_bias[:int(self.num_head/2), :, :] = graph_attn_bias[:int(self.num_head/2), :, :] + edge_input

            # v2.0 高斯编码
            lig_edge_type = lig_edge_type_batch[i]
            rec_edge_type = rec_edge_type_batch[i]
            lig_distance_bias = self.lig_gbf(lig_distance, lig_edge_type).to(torch.float32)  # [N, N, 128]
            rec_distance_bias = self.rec_gbf(rec_distance, rec_edge_type).to(torch.float32)  # [M, M, 128]
            lig_distance_bias = self.lig_gbf_proj(lig_distance_bias).permute(2, 0, 1)
            rec_distance_bias = self.rec_gbf_proj(rec_distance_bias).permute(2, 0, 1)

            # attention_bias维度的转换  [n, n, no_bin] -> [n, n, num_head] -> [num_head, n, n]
            # lig_distance_bias = self.lig_pair_dis_embedder(lig_distance_bias).permute(2, 0, 1)
            # rec_distance_bias = self.rec_pair_dis_embedder(rec_distance_bias).permute(2, 0, 1)

            # v1.0使用键的信息
            # lig_bond_bias = get_mol_bond(lig_batch[i]).to(self.device) #[N, N, 3]
            # lig_bond_bias = self.lig_bond_bias_embedder(lig_bond_bias).mean(-2).permute(2, 0, 1) #[num_head, N, N]
            # lig_bias = lig_distance_bias + lig_bond_bias
            # v2.0不使用键的信息
            graph_attn_bias[int(self.num_head/2):, :, :] = graph_attn_bias[int(self.num_head/2):, :, :] + lig_distance_bias

            lig_self_attention = lig_feat + lig_pos_embedding  # 存储第一次的特征，并作为残差的累积项
            rec_self_attention = rec_feat + rec_pos_embedding    # 存储第一次的特征，并作为残差的累积项

            # 使用transformer的encoder更新特征
            lig_att_weight = graph_attn_bias
            rec_att_weight = rec_distance_bias
            for j, layer in enumerate(self.lig_emb_layers):  # 一共有16层
                lig_self_attention, lig_att_weight = layer(lig_self_attention, lig_self_attention, lig_att_weight)
            for j, layer in enumerate(self.rec_emb_layers):
                rec_self_attention, rec_att_weight = layer(rec_self_attention, rec_self_attention, rec_att_weight)

            lig_self_attention = self.final_layer_norm_lig(lig_self_attention)
            rec_self_attention = self.final_layer_norm_rec(rec_self_attention)

            # v1.0进行cross attention
            # lig_cross_attention = lig_self_attention
            # for j, layer in enumerate(self.cross_attention_layers):
            #     lig_cross_attention, pair_matrix = layer(lig_cross_attention, rec_self_attention, None)
            # lig_cross_attention = self.final_layer_norm(lig_cross_attention)

            # v2.0拼接配体和蛋白质特征
            mol_sz = lig_self_attention.size(0)
            pocket_sz = rec_self_attention.size(0)
            attn_bs = lig_att_weight.size(0)
            cross_feature = torch.cat([lig_self_attention, rec_self_attention],
                                      dim=-2)  # [mol_sz + pocket_sz, hidden_dim]
            cross_bias = torch.zeros(attn_bs, mol_sz + pocket_sz, mol_sz + pocket_sz).type_as(cross_feature)
            cross_bias[:, :mol_sz, :mol_sz] = lig_att_weight
            cross_bias[:, -pocket_sz:, -pocket_sz:] = rec_att_weight

            cross_att_weight = cross_bias
            cross_self_attention = cross_feature
            for j in range(self.recyling):
                for k, layer in enumerate(self.cross_attention_layers):
                    cross_self_attention, cross_att_weight = layer(cross_self_attention, cross_self_attention,
                                                                   cross_att_weight)

            cross_att_weight = cross_att_weight.permute(1, 2, 0)
            lig_cross_attention = cross_self_attention[:mol_sz]
            pocket_cross_attneiton = cross_self_attention[mol_sz:]

            lig_pair_att_weight = cross_att_weight[:mol_sz, :mol_sz, :]
            lig_pocket_att_weight = (cross_att_weight[:mol_sz, mol_sz:, :] + cross_att_weight[mol_sz:, :mol_sz,
                                                                             :].transpose(0, 1)) / 2.0

            lig_pocket_att_weight[lig_pocket_att_weight == float("-inf")] = 0

            if docking:
                # structure module
                pocket_center = target_lig_coords_batch[i].mean(dim=0)
                sm_ligand_coord = lig_coord - lig_coord.mean(dim=0) + pocket_center
                sm_lig_pocket_feat = lig_pocket_att_weight
                sm_lig_distance_feat = lig_pair_att_weight
                lig_feat = lig_cross_attention
                # rmsd_losses = torch.tensor([], device=self.device)
                lig_rec_update, lig_lig_update = None, None
                rmsd_loss = None
                # num_recycling = random.choice(range(1, self.recyling))
                num_recycling = 1  # self.recyling
                is_grad_enabled = torch.is_grad_enabled()
                for cycle_no in range(num_recycling):
                    is_final_iter = cycle_no == (num_recycling - 1)
                    with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                        sm_lig_pocket_feat = self.sm_lig_rec_pair_Transition(sm_lig_pocket_feat).permute(2, 0, 1)
                        sm_lig_distance_feat = self.sm_lig_lig_pair_Transition(sm_lig_distance_feat).permute(2, 0, 1)
                        sm_ligand_coord, lig_feat, sm_lig_pocket_feat, sm_lig_distance_feat, rmsd_loss = self.structure_module(lig_feat, pocket_cross_attneiton, sm_lig_pocket_feat, sm_lig_distance_feat, sm_ligand_coord, target_lig_coords_batch[i], rec_coords_batch[i])
                        # rmsd_losses = torch.cat([rmsd_losses, rmsd_loss], dim=1)
                        # recycling机制
                        # 将lig特征赋值为新的lig特征
                        # cross_self_attention[:mol_sz] = cross_self_attention[:mol_sz] + lig_cross_attention_update
                        # pair特征更新
                        lig_rec_update, lig_lig_update = self.recyling_embedder(sm_ligand_coord, rec_coords_batch[i])
                        sm_lig_pocket_feat = sm_lig_pocket_feat.permute(1, 2, 0) + lig_rec_update
                        sm_lig_distance_feat = sm_lig_distance_feat.permute(1, 2, 0) + lig_lig_update
                        lig_feat = self.atom_gate_layer(lig_feat, lig_cross_attention)
                        sm_lig_pocket_feat = self.lig_rec_pair_gate_layer(sm_lig_pocket_feat, lig_pocket_att_weight)
                        sm_lig_distance_feat = self.lig_pair_gate_layer(sm_lig_distance_feat, lig_pair_att_weight)

                lig_pos_predict_batch.append(sm_ligand_coord)
                rmsd_loss_batch.append(rmsd_loss[:, -1])

                lig_pair_att_weight = sm_lig_distance_feat
                lig_pocket_att_weight = sm_lig_pocket_feat

                # modified
                lig_cross_attention = lig_cross_attention + lig_feat
                # modifiedv2
                # lig_cross_attention = lig_feat
            else:
                lig_pos_predict_batch.append(target_lig_coords_batch[i])

            pair_matrix = torch.cat([
                lig_pocket_att_weight,
                lig_cross_attention.unsqueeze(-2).repeat(1, pocket_sz, 1),
                pocket_cross_attneiton.unsqueeze(-3).repeat(mol_sz, 1, 1)
            ],
                dim=-1
            )

            # 将lig_bias拼接上ligand原子的特征之后投影到1维返回求loss
            lig_distance_predict = torch.cat(
                (lig_pair_att_weight, lig_cross_attention.unsqueeze(-2).repeat(1, len(lig_cross_attention), 1)), dim=-1)
            lig_distance_predict_tmp = self.lig_bias_project(lig_distance_predict).squeeze(0)
            lig_distance_predict_batch.append(lig_distance_predict_tmp)

            # 得到相互作用矩阵
            # pair_matrix = pair_matrix.permute(1, 2, 0)  # (num_lig, num_res, h)
            pair_matrix_tmp = self.pair_Transition(pair_matrix).squeeze(-1)
            pair_matrix_batch.append(pair_matrix_tmp)

            # plddt
            if docking:
                holo_plddt = self.holo_plddt_layer(lig_distance_predict)
                cross_plddt = self.cross_plddt_layer(pair_matrix)
                holo_logits_batch.append(holo_plddt)
                cross_logits_batch.append(cross_plddt)
            else:
                holo_logits_batch.append(None)
                cross_logits_batch.append(None)

        return pair_matrix_batch, lig_distance_predict_batch, lig_pos_predict_batch, rmsd_loss_batch, holo_logits_batch, cross_logits_batch