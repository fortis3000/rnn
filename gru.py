"""GRU models using PyTorch"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class GRU(torch.nn.Module):
    """
    Example:

    model = GRU(
        input_size=2,
        hidden_size=2,
        sequence_length=20,#x_data.shape[1],
        num_layers=2,
        batch_first=True,
        bidirectional=False,
        device=device,
    )
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        sequence_length,
        bias=True,
        batch_first=True,
        dropout=0,
        bidirectional=False,
        device="cpu",
    ):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.sequence_length = sequence_length
        self.num_dimensions = 2 if bidirectional else 1
        self.device = device

        self.model = torch.nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=self.bias,
            batch_first=self.batch_first,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        self.model.to(self.device)

    def forward(self, x):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size) for batch_first=True
        hidden_0 = Variable(
            torch.ones(
                self.num_layers * self.num_dimensions,
                x.size(0),
                self.hidden_size,
                # device=self.device,
            )
        ).to(self.device)

        # hidden_0.unsqueeze_(1)
        # print("HIDDEN", hidden_0[0].size())
        # print(x.size())

        # Reshape input
        # Input: (batch, seq_len, input_size)
        # x.view(x.size(0), self.sequence_length, self.input_size)

        # hidden_0 = self.init_hidden(1024)
        # Propagate input through RNN
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        output, _ = self.model(x, hidden_0)
        return output  # .view(-1, num_classes)


class biGRU(torch.nn.Module):
    """Bidirectional GRU with simple attention layer

    Arguments:
        torch {[type]} -- [description]

    Example:
    
    model = biGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        sequence_length=seq_len,  # x_data.shape[1],
        out_sequence_length,
        num_layers=2,
        batch_first=True,
        bidirectional=True,
        device=device,
    )
    
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        sequence_length,
        out_sequence_length,
        bias=True,
        batch_first=True,
        dropout=0,
        bidirectional=True,
        device="cpu",
    ):
        super(biGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.sequence_length = sequence_length
        self.out_sequence_length = out_sequence_length
        self.num_dimensions = 2 if bidirectional else 1
        self.device = device

        self.model = torch.nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=self.bias,
            batch_first=self.batch_first,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )
        self.fcl1 = torch.nn.Linear(
            self.hidden_size * self.num_dimensions, self.hidden_size
        )
        self.activation1 = torch.nn.ReLU()

        self.model.to(self.device)
        self.fcl1.to(self.device)
        self.activation1.to(self.device)

    def forward(self, x):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size) for batch_first=True
        hidden_0 = Variable(
            torch.ones(
                self.num_layers * self.num_dimensions, x.size(0), self.hidden_size,
            )
        ).to(self.device)

        output, _ = self.model(x, hidden_0)
        output = self.fcl1(output[:, -self.out_sequence_length :, :])
        output = self.activation1(output)
        return output


def init_bigru(
    train_ds,
    num_layers,
    batch_first=True,
    dropout=0,
    bidirectional=True,
    device=torch.device("cpu"),
):

    input_size = train_ds.X_shape[2]
    hidden_size = train_ds.Y_shape[2]
    seq_len = train_ds.X_shape[1]

    logging.info(f"Init model with params: {input_size, hidden_size, seq_len}")

    return biGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        sequence_length=seq_len,
        out_sequence_length=train_ds.Y_shape[1],
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout,
        bidirectional=True,
        device=device,
    )
