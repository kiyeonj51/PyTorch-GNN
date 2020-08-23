from __future__ import print_function

from rgcn.data_utils import *
import pickle as pkl
import os
import glob
import sys
import time
import argparse
import wget

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="mutag",
                    help="Dataset string ('aifb', 'mutag', 'bgs', 'am')")
opt = parser.parse_args()
print(opt)
# args = vars(parser.parse_args())
# print(args)


# Define parameters
dataset = opt.dataset
os.makedirs(f'../data/{dataset}',exist_ok=True)
files = ['completeDataset.tsv','strip_targets.py','testSet.tsv','trainingSet.tsv']

for file in files:
    url = f'https://raw.githubusercontent.com/tkipf/relational-gcn/master/rgcn/data/{dataset}/{file}'
    # url = f'https://github.com/tkipf/relational-gcn/blob/master/rgcn/data/{dataset}/{file}'
    path = f'../data/{dataset}/{file}'
    if not os.path.isfile(path):
        wget.download(url, path)


# NUM_GC_LAYERS = 2  # Number of graph convolutional layers

# Get data
A, X, y, labeled_nodes_idx, train_idx, test_idx, rel_dict, train_names, test_names = load_data(dataset)

rel_list = list(range(len(A)))
for key, value in rel_dict.items():
    if value * 2 >= len(A):
        continue
    rel_list[value * 2] = key
    rel_list[value * 2 + 1] = key + '_INV'


num_nodes = A[0].shape[0]
A.append(sp.identity(A[0].shape[0]).tocsr())  # add identity matrix

support = len(A)

print("Relations used and their frequencies" + str([a.sum() for a in A]))

print("Calculating level sets...")
t = time.time()
# Get level sets (used for memory optimization)
bfs_generator = bfs_relational(A, labeled_nodes_idx)
lvls = list()
lvls.append(set(labeled_nodes_idx))
lvls.append(set.union(*next(bfs_generator)))
print("Done! Elapsed time " + str(time.time() - t))

# Delete unnecessary rows in adjacencies for memory efficiency
todel = list(set(range(num_nodes)) - set.union(lvls[0], lvls[1]))
for i in range(len(A)):
    csr_zero_rows(A[i], todel)

data = {'A': A,
        'y': y,
        'train_idx': train_idx,
        'test_idx': test_idx
        }

dirname = os.path.dirname(os.path.realpath(sys.argv[0]))

with open(f'../data/{dataset}/{dataset}.pickle', 'wb') as f:
    pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)

path = f'../data/{dataset}/'
files = []
for ending in ['*.npz','*.gz','*.npy','*.pkl']:
    for file in  glob.glob(path+ending):
        os.remove(file)