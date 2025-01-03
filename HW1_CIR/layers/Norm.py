import torch
from torch import nn


class AddNorm(nn.Module):

    def __init__(self, d_model):
        super(AddNorm, self).__init__()

        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, C):

        assert Q.shape == C.shape

        sum = Q + C
        out = self.norm(sum)

        return out