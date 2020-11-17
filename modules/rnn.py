import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LSTMLayer(nn.Module):

    def __init__(self, num_layers, hidden_size, input_size, device='cpu'):
        super().__init__()
        self.device = device

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            bidirectional=True, 
            batch_first=True
        )

    def _init_hidden(self, batch_size):
        return (
            Variable(  # init hidden state
                torch.zeros(2 * self.num_layers, batch_size, self.hidden_size)
            ).to(self.device),   
            Variable(  # init cell state
                torch.zeros(2 * self.num_layers, batch_size, self.hidden_size)
            ).to(self.device)    
        )

    def forward(self, x, lens=None):
        if lens is not None:
            x = pack_padded_sequence(x, lens.tolist(), batch_first=True, enforce_sorted=False)
            x, _ = self.rnn(x, self._init_hidden(lens.shape[0]))
            return pad_packed_sequence(x, batch_first=True)[0]
        else:
            return self.rnn(x)