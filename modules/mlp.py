import torch.nn.functional as F
from torch import nn
import torch


class MLPLayer(nn.Module):

    def __init__(self, input_size, output_size, non_linearity=torch.sigmoid):
        super().__init__()
        self.lin1 = nn.Linear(input_size, input_size // 2)
        self.lin2 = nn.Linear(input_size // 2, output_size)
        self.non_lin = non_linearity

    def forward(self, x):
        out = self.non_lin(self.lin1(x))
        return self.non_lin(self.lin2(out))
