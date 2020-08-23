#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-26 20:53:30
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

import os
import argparse
import numpy as np
import scipy.sparse as sp

from torch import nn
from torch import optim
import torch

from rgcn.utils import load_data, compute_accuracy, normalize
from rgcn.models import RGCN


def to_sparse_tensor(sparse_array):
    if len(sp.find(sparse_array)[-1]) > 0:
        vals = torch.FloatTensor(sp.find(sparse_array)[-1])
        idxs = torch.LongTensor(sparse_array.nonzero())
        sparse_tensor = torch.sparse.FloatTensor(idxs, vals, torch.Size(sparse_array.shape))
    else:
        sparse_tensor = torch.sparse.FloatTensor(sparse_array.shape)
    return sparse_tensor


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="aifb", help="Dataset string ('aifb', 'mutag', 'bgs', 'am')")
parser.add_argument("--n_epochs", type=int, default=100, help="Number training epochs")
parser.add_argument("--hidden", type=int, default=16, help="Number hidden units")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0., help="weight decay")
parser.add_argument("-best_loss", type=float, default=np.inf, help="best loss")
parser.add_argument("--bases", type=int, default=-1, help="Number of bases used (-1: all)")
parser.add_argument("-l2",type=float, default=0., help="L2 normalization of input weights")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--num_basis', type=int, default=40, help='the number of basis.')

opt = parser.parse_args()

# cuda = torch.cuda.is_available()
opt.cuda = not opt.no_cuda and torch.cuda.is_available()

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

adj_list, labels, idx_train, idx_test = load_data()


labels = labels.todense()
labels = np.argmax(labels, axis=1)
labels = torch.LongTensor(labels).squeeze()

# saved_models = "saved_models_aifb"
# os.makedirs(saved_models, exist_ok=True)
# vertex_features_dim = adj_list[0].shape[0]

adjs = []
for adj in adj_list:
    adj = normalize(adj)
    if len(adj.nonzero()[0]) > 0:
        adj = to_sparse_tensor(adj)
        adjs.append(adj)
adjs = torch.stack(adjs)
init_input = None
num_node = adjs.shape[1]

model = RGCN(nfeat=num_node,
             nhid=opt.hidden,
             dropout=opt.dropout,
             support=adjs.shape[0],
             num_basis=opt.num_basis)

optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
criterion = nn.CrossEntropyLoss()

if opt.cuda:
    model.cuda()
    criterion.cuda()
    init_input = init_input.cuda()
    adjs = adjs.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_test = idx_test.cuda()

for epoch in range(opt.n_epochs):
    #########
    # Train #
    #########
    model.train()
    optimizer.zero_grad()
    output = model(init_input, adjs)
    loss = criterion(output[idx_train], labels[idx_train])
    train_loss = loss.item()
    loss.backward()
    optimizer.step()

    ###########
    # Testing #
    ###########
    output = model(init_input, adjs)
    loss = criterion(output[idx_test], labels[idx_test])
    val_loss = loss.item()
    val_acc = compute_accuracy(output.detach().cpu().numpy(), labels.cpu().numpy())

    print(f"[Epoch {epoch:04d}]"
          f"[train loss: {train_loss:.4f}]"
          f"[val loss: {val_loss:.4f}]"
          f"[val acc: {val_acc:.4f}]")
