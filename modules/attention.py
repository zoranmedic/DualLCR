import torch
from torch import nn


class AttentionLayer(nn.Module):

    def __init__(self, query_dim, value_dim, device='cpu', query_as_parameter=False):
        super().__init__()
        self.device = device
        self.W = nn.Linear(query_dim + value_dim, query_dim + value_dim, bias=False)
        self.v = nn.Parameter(torch.randn(1, query_dim + value_dim))
        if query_as_parameter:
            self.query = nn.Parameter(torch.randn(1, query_dim))
        self.tanh = nn.Tanh()

    def _attention_mask(self, lens):
        """
        Returns a boolean mask with values set to True for those fields that should be kept
        """
        batch_size = lens.shape[0]
        max_len = lens.max()
        item_idx = torch.arange(1, max_len + 1).repeat(batch_size, 1, 1).to(self.device)
        seq_len = lens.view(-1, 1, 1).repeat(1, 1, max_len).to(self.device)
        return item_idx <= seq_len   # B x 1 x max_len

    def forward(self, query, value, lens):
        """
        query: [B, 1, d]
        value: [B, n, d]
        lens: B (lengths of each sequence in batch)
        return: 
        """
        if query is None:
            query = self.query.repeat((value.shape[0], 1)).unsqueeze(1)
        if value.shape[0] > query.shape[0]: # if B of query is different of B of value
            query = query.repeat((1, value.shape[0] // query.shape[0], 1)).view(value.shape[0], 1, -1)
        query = query.repeat((1, value.shape[1], 1))    # [B, n, d]
        query = torch.cat((query, value), dim=-1)       # [B, n, 2d]
        scores = self.tanh(self.W(query))
        scores = torch.bmm(self.v.repeat(query.shape[0], 1, 1), scores.transpose(-1, 1))    # [B, 1, n]
        mask = self._attention_mask(lens)
        scores[~mask] = -99999
        return torch.softmax(scores, dim=-1)
