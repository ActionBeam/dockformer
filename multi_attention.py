import numpy as np
import torch
from torch import nn
from torch.nn import init
from typing import Optional, Callable, List, Tuple, Sequence
import math
from utils import Linear


class pair_Transition(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels, out_dims, n=1):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.linear_1 = Linear(embedding_channels, n * embedding_channels, init="relu")
        self.linear_2 = Linear(n * embedding_channels, out_dims, init="relu")
        self.linear_3 = Linear(embedding_channels, out_dims, init="normal")
        self.activate = torch.nn.LeakyReLU()

        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, z):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        z = self.layernorm(z)
        z = self.linear_1(z)
        z = self.activate(z)
        z = self.linear_2(z)
        # z = self.activate(z)
        # z = self.linear_3(z)
        return z


@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
        Softmax, but without automatic casting to fp32 when the input is of
        type bfloat16
    """
    s = torch.nn.functional.softmax(t, dim=dim)

    return s


class SelfMultiheadAttention(nn.Module):
    def __init__(
            self,
            c_hidden: int,
            no_heads: int,
            dropout: float = 0.1,
            is_talking: bool = False
    ):
        super().__init__()
        self.embed_dim = c_hidden
        self.num_heads = no_heads
        self.is_talking = is_talking
        self.dropout = nn.Dropout(dropout)
        self.head_dim = self.embed_dim // self.num_heads
        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.in_proj = Linear(self.embed_dim, self.embed_dim * 3, init="glorot")
        self.out_proj = Linear(self.embed_dim, self.embed_dim)
        if self.is_talking:
            self.pre_softmax_talking_heads = torch.nn.Conv2d(self.num_heads, self.num_heads, 1, bias=False)
            self.post_softmax_talking_heads = torch.nn.Conv2d(self.num_heads, self.num_heads, 1, bias=False)

    def forward(
            self,
            q_x: torch.Tensor,
            attention_bias: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        q, k, v = self.in_proj(q_x).chunk(3, dim=-1)

        # [batch_size, q, num_head, head_dim] -> [batch_size * num_head, 1, head_dim]
        q = (q.view(-1, self.num_heads, self.head_dim)
             .transpose(0, 1)
             .contiguous()
             .view(self.num_heads, -1, self.head_dim)
             )

        k = (k.view(-1, self.num_heads, self.head_dim)
             .transpose(0, 1)
             .contiguous()
             .view(self.num_heads, -1, self.head_dim)
             )

        v = (v.view(-1, self.num_heads, self.head_dim)
             .transpose(0, 1)
             .contiguous()
             .view(self.num_heads, -1, self.head_dim)
             )

        att_weight = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
            att_weight = att_weight.masked_fill_(attention_mask, -np.inf)  # 在mask==1的位置上填充对应的value值(-np.inf)

        if attention_bias is not None:
            att_weight += attention_bias

        if self.is_talking:
            attn = self.pre_softmax_talking_heads(att_weight.unsqueeze(0))
            attn = attn.squeeze(0)
        else:
            attn = att_weight

        attn = softmax_no_cast(attn, -1)  # 对最后一个维度进行softmax

        if self.is_talking:
            attn = self.post_softmax_talking_heads(attn.unsqueeze(0))
            attn = attn.squeeze(0)

        o = torch.matmul(attn, v)  # attention和v做矩阵乘法
        o = (
            o.view(self.num_heads, -1, self.head_dim)
            .transpose(0, 1)
            .contiguous()
            .view(-1, self.embed_dim)
        )

        o = self.out_proj(o)
        return o, att_weight


class CrossMultiheadAttention(nn.Module):
    def __init__(
            self,
            c_hidden: int,
            no_heads: int,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = c_hidden
        self.num_heads = no_heads

        self.dropout = nn.Dropout(dropout)
        self.head_dim = self.embed_dim // self.num_heads
        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = Linear(self.embed_dim, self.embed_dim)
        self.k_proj = Linear(self.embed_dim, self.embed_dim)
        self.v_proj = Linear(self.embed_dim, self.embed_dim)
        self.out_proj = Linear(self.embed_dim, self.embed_dim)

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: torch.Tensor,
            v_x: torch.Tensor,
            attention_bias: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        q = self.q_proj(q_x)
        k = self.k_proj(k_x)
        v = self.v_proj(v_x)

        # [*, q, num_head, head_dim] -> [num_head, 1, head_dim]
        q = (q.view(-1, self.num_heads, self.head_dim)
             .transpose(0, 1)
             .contiguous()
             .view(self.num_heads, -1, self.head_dim))

        k = (k.view(-1, self.num_heads, self.head_dim)
             .transpose(0, 1)
             .contiguous()
             .view(self.num_heads, -1, self.head_dim))

        v = (v.view(-1, self.num_heads, self.head_dim)
             .transpose(0, 1)
             .contiguous()
             .view(self.num_heads, -1, self.head_dim))

        att_weight = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
            att_weight = att_weight.masked_fill_(attention_mask, -np.inf)  # 在mask==1的位置上填充对应的value值(-np.inf)

        # 距离矩阵的更新
        if attention_bias is not None:
            att_weight += attention_bias
        attn = softmax_no_cast(att_weight, -1)  # 对最后一个维度进行softmax

        o = torch.matmul(attn, v)  # attention和v做矩阵乘法
        o = (
            o.view(self.num_heads, -1, self.head_dim)
            .transpose(0, 1)
            .contiguous()
            .view(-1, self.embed_dim)
        )

        o = self.out_proj(o)
        return o, att_weight

