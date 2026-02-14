import math
import torch
from torch import nn
from utils import Linear, permute_final_dims, flatten_final_dims
from transformer_encoder_layer import EncoderBlock


class Gate_Block(nn.Module):
    def __init__(self, dim_tmp, drop_rate=0.1):
        super().__init__()
        self.gate_layer = nn.Sequential(
            nn.Linear(3 * dim_tmp, dim_tmp),
            nn.Dropout(p=drop_rate))
        self.norm = nn.LayerNorm(dim_tmp)

    def forward(self, f1, f2):
        g = torch.sigmoid(self.gate_layer(torch.cat([f2, f1, f2 - f1], dim=-1)))
        f2 = self.norm(g * f2 + f1)
        return f2


class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransitionLayer, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class StructureModule(nn.Module):
    def __init__(
        self,
        c_hidden: int,
        no_heads: int,
        ffn_embed_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = c_hidden
        self.num_heads = no_heads
        self.ffn_embed_dim = ffn_embed_dim
        self.head_dim = self.embed_dim // self.num_heads
        self.lig_rec_pair_dim = no_heads
        self.lig_lig_pair_dim = no_heads

        self.no_blocks = 8
        self.lig_rec_attn = nn.ModuleList([EncoderBlock(emb_dims=self.embed_dim,
                                                        ffn_embed_dims=self.ffn_embed_dim,
                                                        attention_head=self.num_heads,
                                                        dropout=dropout, is_cross=True) for i in range(self.no_blocks)]
                                         )
        self.lig_rec_attn_Linear = nn.ModuleList([nn.Sequential(nn.LayerNorm(self.num_heads),
                                                                Linear(self.num_heads, self.num_heads//2),
                                                                nn.Dropout(p=dropout),
                                                                nn.LeakyReLU(),
                                                                Linear(self.num_heads//2, 1, init="final")
                                                                ) for i in range(self.no_blocks)]
                                                )
        # self.feat2rigid = Linear(self.embed_dim, 6, init="final")
        # self.lig_lig_ipa = InvariantPointAttention(c_s=self.embed_dim,
        #                                            c_z=self.num_heads,
        #                                            no_heads=self.num_heads)
        self.lig_lig_attn = nn.ModuleList([EncoderBlock(emb_dims=self.embed_dim,
                                                        ffn_embed_dims=self.ffn_embed_dim,
                                                        attention_head=self.num_heads,
                                                        dropout=dropout, is_cross=False) for i in range(self.no_blocks)]
                                         )
        # self.ipa_dropout = nn.Dropout(dropout)
        # self.transition = StructureModuleTransitionLayer(self.embed_dim)
        self.lig_lig_attn_Linear = nn.ModuleList([nn.Sequential(nn.LayerNorm(self.num_heads),
                                                                Linear(self.num_heads, self.num_heads//2),
                                                                nn.Dropout(p=dropout),
                                                                nn.LeakyReLU(),
                                                                Linear(self.num_heads//2, 1, init="final")
                                                                ) for i in range(self.no_blocks)]
                                                )
        self.eps = 1e-8

    def forward(
        self,
        lig_feat,
        rec_feat,
        lig_rec_pair,
        lig_lig_pair,
        sm_ligand_coord,
        target_lig_coords,
        rec_coords
    ):
        # rigid = (torch.tensor([1., 0., 0., 0.], dtype=lig_feat.dtype, device=lig_feat.device, requires_grad=self.training), torch.tensor([0., 0., 0.], dtype=lig_feat.dtype, device=lig_feat.device, requires_grad=self.training))
        cross_rec_feat = rec_feat.clone()
        pred_lig_pos = sm_ligand_coord
        # delta_pos = torch.zeros(lig_feat.size(0), 3, device=lig_feat.device)
        rmsd_loss = torch.tensor([], device=lig_feat.device)
        for i in range(self.no_blocks):
            # 获取每次的单位向量
            lig_coord_pair = pred_lig_pos.unsqueeze(0) - pred_lig_pos.unsqueeze(1)
            lig_coord_norms = torch.norm(lig_coord_pair, dim=-1, keepdim=True) + 1e-6
            lig_norm_coord_pair = lig_coord_pair / lig_coord_norms
            lig_rec_coord_pair = rec_coords.unsqueeze(0) - pred_lig_pos.unsqueeze(1)
            lig_rec_coord_norms = torch.norm(lig_rec_coord_pair, dim=-1, keepdim=True) + 1e-6
            lig_rec_norm_coord_pair = lig_rec_coord_pair / lig_rec_coord_norms
            # lig-rec attention 获取rigids和平移权重
            lig_feat, lig_rec_pair = self.lig_rec_attn[i](lig_feat, cross_rec_feat, lig_rec_pair)
            cross_attn_score = self.lig_rec_attn_Linear[i](lig_rec_pair.permute(1, 2, 0))  # (N, N, no_heads) -> (N, N)
            # cross_lig_rec_pair = torch.nn.functional.softmax(cross_lig_rec_pair, dim=-1)  # 求出相互作用权重
            weight_lig_rec_diff = lig_rec_norm_coord_pair * cross_attn_score
            delta_lig_rec_pos = torch.sum(weight_lig_rec_diff, dim=1)
            # lig-lig ipa 获取平移权重
            # lig_feat, attn_score = self.lig_lig_ipa(lig_feat, lig_lig_pair, rigid)
            lig_feat, lig_lig_pair = self.lig_lig_attn[i](lig_feat, lig_feat, lig_lig_pair)
            attn_score = self.lig_lig_attn_Linear[i](lig_lig_pair.permute(1, 2, 0))
            # lig_feat = self.ipa_dropout(lig_feat)
            # lig_feat = self.transition(lig_feat)
            # new_rigids = self.feat2rigid(torch.max(lig_feat, dim=0)[0])
            # quaternion_vec, translation_vec = new_rigids[..., :3], new_rigids[..., 3:]
            # quaternion_vec = torch.cat([torch.tensor([1.], dtype=lig_feat.dtype, device=lig_feat.device), quaternion_vec], dim=-1)
            # new_rigid = (normalize_quaternion(quaternion_vec), translation_vec)
            # rigid = quaternion_translation_compose(new_rigid, rigid)  # 顺序很重要，是在rigid的基础上，有进行了一个新的new_rigid
            # attn_score = torch.nn.functional.softmax(attn_score, dim=-1)
            weight_lig_lig_diff = lig_norm_coord_pair * attn_score
            delta_lig_lig_pos = torch.sum(weight_lig_lig_diff, dim=1)
            delta_pos = delta_lig_rec_pos + delta_lig_lig_pos
            # 整体在空间平移和旋转
            # pred_lig_pos = quaternion_translation_apply(rigid[0], rigid[1] * self.trans_scale_factor, lig_pos)
            # 坐标进行平移微调
            pred_lig_pos = pred_lig_pos + delta_pos
            # preds = {"positions": pred_lig_pos, "rigid": rigid}
            rmsd = torch.sqrt(  # 获取两个局部坐标系下所有原子的相互距离 算法28的第三行
                torch.mean(torch.sum((pred_lig_pos - target_lig_coords) ** 2, dim=-1) + self.eps)
            ).view(-1, 1)
            # error_dist = torch.clamp(error_dist, min=0, max=self.l1_clamp_distance)
            rmsd_loss = torch.cat([rmsd_loss, rmsd], dim=1)
            # if i < (self.no_blocks - 1):
            #     detach_rigid = (rigid[0].detach(), rigid[1])
            #     rigid = detach_rigid

        return pred_lig_pos, lig_feat, lig_rec_pair, lig_lig_pair, rmsd_loss


class RecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """

    def __init__(
        self,
        no_head: int,
        min_bin: float = -1e-6,
        max_bin: float = 8.,
        no_bins: int = 64,
        inf: float = 1e8,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            no_bins:
                Number of distogram bins
        """
        super(RecyclingEmbedder, self).__init__()

        self.no_head = no_head
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = inf

        self.lig_rec_linear = Linear(self.no_bins, self.no_head)
        self.lig_lig_linear = Linear(self.no_bins, self.no_head)


    def forward(
        self,
        pred_lig_pos,
        rec_coord,
    ):
        """
        Args:
            m:
                First row of the MSA embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted C_beta coordinates
        Returns:
            m:
                [*, N_res, C_m] MSA embedding update
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        bins = torch.linspace(   # 在指定范围内生成指定数量的bins
            self.min_bin,
            self.max_bin,
            self.no_bins,
            dtype=pred_lig_pos.dtype,
            device=pred_lig_pos.device,
            requires_grad=False,
        )
        upper = torch.cat(  # 会前面直接拷贝squared_bins的元素，并在最后一个维度增加一个极大的阈值
            [bins[1:], bins.new_tensor([self.inf])], dim=-1
        )

        lig_rec_dis = torch.cdist(pred_lig_pos, rec_coord, compute_mode='donot_use_mm_for_euclid_dist').unsqueeze(-1)
        lig_lig_dis = torch.cdist(pred_lig_pos, pred_lig_pos, compute_mode='donot_use_mm_for_euclid_dist').unsqueeze(-1)

        lig_rec_onehot = ((lig_rec_dis > bins) * (lig_rec_dis < upper)).type(pred_lig_pos.dtype)
        lig_lig_onehot = ((lig_lig_dis > bins) * (lig_lig_dis < upper)).type(pred_lig_pos.dtype)

        lig_rec_update = self.lig_rec_linear(lig_rec_onehot)   # (N, M, C_z)
        lig_lig_update = self.lig_lig_linear(lig_lig_onehot)   # (N, N, C_z)

        return lig_rec_update, lig_lig_update
