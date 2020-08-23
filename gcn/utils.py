import numpy as np
import scipy.sparse as sp
import torch
import wget, os, tarfile, sys


def delete_no_feature_node(edges_unordered,nodes):
    if not bool(set(edges_unordered.flatten()) - set(nodes)):
        return edges_unordered
    no_feature_nodes = []
    for ii, edge in enumerate(edges_unordered):
        inter = set(edge) - set(nodes)
        if bool(inter):
            no_feature_nodes.append(ii)
    edges_unordered = np.delete(edges_unordered, no_feature_nodes, axis=0)
    return edges_unordered


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(dataset="cora", sub_dataset=""):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    path = f"../data/{dataset}/"
    if sub_dataset != "":
        dataset=sub_dataset

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}

    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.dtype(str))

    edges_unordered = delete_no_feature_node(edges_unordered,idx)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(int(len(idx_map) * 0.6))
    idx_val = range(int(len(idx_map) * 0.6), int(len(idx_map) * 0.8))
    idx_test = range(int(len(idx_map) * 0.8), len(idx_map))

    # idx_train = range(int(len(idx_map) *0.1))
    # idx_val = range(int(len(idx_map)*0.15), int(len(idx_map)*0.35))
    # idx_test = range(int(len(idx_map)*0.35),len(idx_map))

    # idx_train = range(140)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def download_data(dataset='cora'):
    if os.path.isdir(f'../data/{dataset}'):
        return
    url = f'https://linqs-data.soe.ucsc.edu/public/lbc/{dataset}.tgz'
    path = f'../data/{dataset}.tgz'
    os.makedirs(f'../data/{dataset}',exist_ok=True)
    if not os.path.isfile(path):
        try:
            wget.download(url, '../data')
        except Exception as e:
            print(e)
            sys.exit(0)
    tar = tarfile.open(path, 'r')
    tar.extractall('../data/')
    tar.close()
    os.remove(path)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)