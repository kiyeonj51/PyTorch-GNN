from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from gcn.utils import load_data, accuracy,download_data
from gcn.models import GCN

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,  help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
# parser.add_argument('--path', type=str, default='../data/cora/', help='data path')
parser.add_argument('--dataset', type=str, default='cora', help='dataset name')
parser.add_argument('--sub_dataset', type=str, default='', help='dataset name')
opt = parser.parse_args()
opt.cuda = not opt.no_cuda and torch.cuda.is_available()

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

# Download data
download_data(opt.dataset)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(opt.dataset,opt.sub_dataset)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=opt.hidden,
            nclass=labels.max().item() + 1,
            dropout=opt.dropout)
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

if opt.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

#########
# Train #
#########
t_total = time.time()
for epoch in range(opt.n_epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not opt.fastmode:
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print(f'[Epoch {epoch+1:04d}/{opt.n_epochs}]'
          f'[Train loss: {loss_train.item():.4f}]'
          f'[Train accuracy: {acc_train.item():.4f}]'
          f'[Validation loss: {loss_val.item():.4f}]'
          f'[Validation accuracy: {acc_val.item():.4f}]'
          f'[Time: {time.time() - t:.4f}s]')

###########
# Testing #
###########
model.eval()
output = model(features, adj)
loss_test = F.nll_loss(output[idx_test], labels[idx_test])
acc_test = accuracy(output[idx_test], labels[idx_test])
print("[Test result]"
      f"[Test loss: {loss_test.item():.4f}]"
      f"[Test accuracy: {acc_test.item():.4f}]")

