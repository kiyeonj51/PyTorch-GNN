import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import math


class RelationalGraphConvolution(Module):
    def __init__(self, in_features, out_features, support, num_basis, featureless, dropout):
        super(RelationalGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.support = support
        self.num_basis = num_basis
        self.featureless = featureless
        self.dropout = dropout
        if num_basis > 0:
            self.basis = Parameter(torch.FloatTensor(in_features * num_basis, out_features))
            self.coef = Parameter(torch.FloatTensor(support, num_basis))
        else:
            self.basis = Parameter(torch.FloatTensor(in_features * support, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.basis.size(1))
        self.basis.data.uniform_(-stdv, stdv)
        if self.num_basis > 0:
            stdv = 1. / math.sqrt(self.coef.size(1))
            self.coef.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, input, adjs):
        adj_mat = []
        node_num = adjs.shape[1]
        for i, adj in enumerate(adjs):
            if not self.featureless:
                adj_mat.append(torch.spmm(adj, input))
            else:
                adj_mat.append(adj)
        adj_mat = torch.cat(adj_mat, dim=1)
        if self.num_basis > 0:
            weight = torch.matmul(self.coef, torch.reshape(self.basis, (self.num_basis, self.in_features, self.out_features)).permute(1, 0, 2))
            weight = torch.reshape(weight, (self.in_features * self.support, self.out_features))
            output = torch.spmm(adj_mat, weight)
        else:
            output = torch.spmm(adj_mat, self.basis)
        if self.featureless:
            temp = torch.ones(node_num)
            temp_drop = F.dropout(temp, self.dropout)
            output = temp_drop.reshape(-1, 1) * output
        return output + self.bias
