import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from ggnn.models import GGNN
from ggnn.dataset import bAbIDataset
from ggnn.dataset import bAbIDataloader

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--task_id', type=int, default=4, help='bAbI task id')
parser.add_argument('--question_id', type=int, default=0, help='question types')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--state_dim', type=int, default=4, help='GGNN hidden state size')
parser.add_argument('--n_steps', type=int, default=5, help='propogation steps number of GGNN')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')
opt = parser.parse_args()
print(opt)

opt.cuda = not opt.no_cuda and torch.cuda.is_available()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

opt.dataroot = f'../data/babi/processed_1/train/{opt.task_id}_graphs.txt'

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# dataloader
train_dataset = bAbIDataset(opt.dataroot, opt.question_id, True)
train_dataloader = bAbIDataloader(train_dataset, batch_size=opt.batchSize,
                                  shuffle=True)

test_dataset = bAbIDataset(opt.dataroot, opt.question_id, False)
test_dataloader = bAbIDataloader(test_dataset, batch_size=opt.batchSize,
                                 shuffle=False)

opt.annotation_dim = 1  # for bAbI
opt.n_edge_types = train_dataset.n_edge_types
opt.n_node = train_dataset.n_node

# model and optimizer
model = GGNN(opt)
model.double()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

if opt.cuda:
    model.cuda()
    criterion.cuda()

###############
# Train model #
###############

for epoch in range(0, opt.n_epochs):
    model.train()
    for i, (adj_matrix, annotation, target) in enumerate(train_dataloader, 0):
        model.zero_grad()

        padding = torch.zeros(len(annotation), opt.n_node, opt.state_dim - opt.annotation_dim).double()
        init_input = torch.cat((annotation, padding), 2)
        if opt.cuda:
            init_input = init_input.cuda()
            adj_matrix = adj_matrix.cuda()
            annotation = annotation.cuda()
            target = target.cuda()
        init_input = Variable(init_input)
        adj_matrix = Variable(adj_matrix)
        annotation = Variable(annotation)
        target = Variable(target)
        output = model(init_input, annotation, adj_matrix)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if (i % int(len(train_dataloader) / 10 + 1)) == 0 and opt.verbal:
            print('[Epoch %d/%d][Batch %d/%d] [Train loss: %.4f]' % (epoch, opt.n_epochs, i, len(train_dataloader), loss.item()))
    ###########
    # Testing #
    ###########
    test_loss = 0
    correct = 0
    model.eval()
    for i, (adj_matrix, annotation, target) in enumerate(test_dataloader, 0):
        padding = torch.zeros(len(annotation), opt.n_node, opt.state_dim - opt.annotation_dim).double()
        init_input = torch.cat((annotation, padding), 2)
        if opt.cuda:
            init_input = init_input.cuda()
            adj_matrix = adj_matrix.cuda()
            annotation = annotation.cuda()
            target = target.cuda()
        init_input = Variable(init_input)
        adj_matrix = Variable(adj_matrix)
        annotation = Variable(annotation)
        target = Variable(target)
        output = model(init_input, annotation, adj_matrix)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_dataloader.dataset)
    print('[Test set] [Test loss: {:.4f}], [Test accuracy: {}/{} ({:.0f}%)]'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))
