from torch.nn import Module, Sequential, Linear, ReLU, Dropout, LogSoftmax
from rgcn.layers import RelationalGraphConvolution
import torch.nn.functional as F


class RGCN(Module):
    def __init__(self, nfeat, nhid, dropout, support, num_basis, featureless=True):
        super(RGCN, self).__init__()
        self.dropout = dropout
        self.gc1 = RelationalGraphConvolution(nfeat, nhid, support, num_basis, featureless, dropout)
        self.gc2 = RelationalGraphConvolution(nhid, nhid, support, num_basis, False, dropout)

        self.fc1 = Sequential(
            Linear(nhid, nhid),
            ReLU(),
            Dropout(dropout))
        self.fc2 = Sequential(
            Linear(nhid, 4),
            LogSoftmax())

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
