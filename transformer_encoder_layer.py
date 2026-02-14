import torch
from torch import nn
from multi_attention import SelfMultiheadAttention, CrossMultiheadAttention


class PositionWiseFFN(nn.Module):
    """基于位置的前馈⽹络"""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """残差连接后进⾏层规范化"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    """transformer编码器块"""

    def __init__(self, emb_dims=128, ffn_embed_dims=512, attention_head=8, dropout=0.1, is_cross=False, is_talking=False):
        super(EncoderBlock, self).__init__()
        self.atten_layer_norm = nn.LayerNorm(emb_dims)
        self.dropout = nn.Dropout(dropout)
        self.is_cross = is_cross
        if is_cross:
            self.attention = CrossMultiheadAttention(emb_dims, attention_head, dropout=dropout)
        else:
            self.attention = SelfMultiheadAttention(emb_dims, attention_head, dropout=dropout, is_talking=is_talking)
        self.addnorm1 = AddNorm(emb_dims, dropout=dropout)
        self.ffn = PositionWiseFFN(emb_dims, ffn_embed_dims, emb_dims)
        self.addnorm2 = AddNorm(emb_dims, dropout=dropout)

    def forward(self, X, Y, attention_bias=None, attention_mask=None):
        if self.is_cross:
            Z, att_weight = self.attention(self.atten_layer_norm(X), self.atten_layer_norm(Y), self.atten_layer_norm(Y),
                                           attention_bias, attention_mask)
            Z = self.addnorm1(X, Z)
        else:
            Z, att_weight = self.attention(self.atten_layer_norm(X), attention_bias, attention_mask)
            Z = self.addnorm1(X, Z)
        return self.addnorm2(Z, self.ffn(self.atten_layer_norm(Z))), att_weight

