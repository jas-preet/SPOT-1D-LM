import torch
import torch.nn as nn
from torch.nn import functional as F

PADDING_VALUE=0



class Network(nn.Module):
    def __init__(self, hidden_dim=1024):
        super(Network, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm1 = nn.LSTM(input_size=2324, hidden_size=self.hidden_dim, num_layers=2, batch_first=True,
                             bidirectional=True, dropout=0.50)
        self.drop1 = nn.Dropout(p=0.5)

        self.linear1 = nn.Linear(in_features=2*self.hidden_dim, out_features=1000)
        self.linear2 = nn.Linear(in_features=1000, out_features=1000)

        self.linear3 = nn.Linear(in_features=1000, out_features=11)


    def forward(self, x, seq_lens):
        #### creating a mask of shape [B, L]

        x = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True)
        x, (hidden, cell) = self.lstm1(x)
        x, y = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=PADDING_VALUE)
        x = self.drop1(x)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x