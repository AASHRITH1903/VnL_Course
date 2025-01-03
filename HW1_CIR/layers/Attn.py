import torch
from torch import nn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_hidden = d_model // n_heads

        self.Q_proj = nn.Linear(self.d_model, self.n_heads * self.d_hidden)
        self.K_proj = nn.Linear(self.d_model, self.n_heads * self.d_hidden)
        self.V_proj = nn.Linear(self.d_model, self.n_heads * self.d_hidden)

        self.out_proj = nn.Linear(self.n_heads * self.d_hidden, self.d_model)


    def forward(self, Q, K, V, attn_mask=None):

        # Q -> (B, Sq, d_model)
        # K -> (B, Sk, d_model)

        B, Sq, _ = Q.shape
        _, Sk, _ = K.shape

        # Projections: (B, Sq, d_model) -> (B, Sq, n_heads, d_hidden) -> (B, n_heads, Sq, d_hidden)
        Q = (
            self.Q_proj(Q)
            .view(B, Sq, self.n_heads, self.d_hidden)
            .transpose(1, 2)
        )
        K = (
            self.K_proj(K)
            .view(B, Sk, self.n_heads, self.d_hidden)
            .transpose(1, 2)
        )
        V = (
            self.V_proj(V)
            .view(B, Sk, self.n_heads, self.d_hidden)
            .transpose(1, 2)
        )

        # Scaled Dot Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_hidden ** 0.5)

        if attn_mask != None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        attn = nn.Softmax(dim=-1)(scores)

        context = (
            torch.matmul(attn, V)
            .transpose(1, 2)
            .reshape(B, Sq, self.n_heads * self.d_hidden)
        )

        context = self.out_proj(context)
        
        return context

























