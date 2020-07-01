"""RNN models using PyTorch"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        sequence_length,
        num_layers,
        bias=True,
        bidirectional=False,
        nonlinearity="tanh",  # 'relu' is also available
        dropout=0,
        batch_first=True,
        device="cpu",
    ):
        super(RNN, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.bidirectional = bidirectional
        self.num_dimensions = 2 if bidirectional else 1  # TODO: check if it is correct
        self.device = device

        self.model = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            nonlinearity=nonlinearity,
        )

        self.model.to(self.device)

    def init_hidden(self):
        self.hidden = (
            torch.autograd.Variable(
                torch.zeros(1, self.batch_size, self.hidden_dim).cuda()
            ),
            torch.autograd.Variable(
                torch.zeros(1, self.batch_size, self.hidden_dim).to(self.device)
            ),
        )

    def forward(self, x):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size) for batch_first=True
        hidden_0 = Variable(
            torch.zeros(
                self.num_layers * self.num_dimensions, x.size(0), self.hidden_size
            )
        ).to(self.device)

        # Reshape input
        # Input: (batch, seq_len, input_size)
        # x.view(x.size(0), self.sequence_length, self.input_size)

        # Propagate input through RNN
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        output, hidden = self.model(x, hidden_0)
        return output  # .view(-1, num_classes)
