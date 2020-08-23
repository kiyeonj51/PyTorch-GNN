from __future__ import print_function

import os, re, sys, gzip
import numpy as np
import scipy.sparse as sp
import rdflib as rdf
import glob
import pandas as pd
import wget
import pickle as pkl
from collections import Counter

np.random.seed(123)


def csr_zero_rows(csr, rows_to_zero):
    """Set rows given by rows_to_zero in a sparse csr matrix to zero.
    NOTE: Inplace operation! Does not return a copy of sparse matrix."""
    rows, cols = csr.shape
    mask = np.ones((rows,), dtype=np.bool)
    mask[rows_to_zero] = False
    nnz_per_row = np.diff(csr.indptr)

    mask = np.repeat(mask, nnz_per_row)
    nnz_per_row[rows_to_zero] = 0
    csr.data = csr.data[mask]
    csr.indices = csr.indices[mask]
    csr.indptr[1:] = np.cumsum(nnz_per_row)
    csr.eliminate_zeros()
    return csr


def csc_zero_cols(csc, cols_to_zero):
    """Set rows given by cols_to_zero in a sparse csc matrix to zero.
    NOTE: Inplace operation! Does not return a copy of sparse matrix."""
    rows, cols = csc.shape
    mask = np.ones((cols,), dtype=np.bool)
    mask[cols_to_zero] = False
    nnz_per_row = np.diff(csc.indptr)

    mask = np.repeat(mask, nnz_per_row)
    nnz_per_row[cols_to_zero] = 0
    csc.data = csc.data[mask]
    csc.indices = csc.indices[mask]
    csc.indptr[1:] = np.cumsum(nnz_per_row)
    csc.eliminate_zeros()
    return csc


def sp_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (dim, 1)
    data = np.ones(len(idx_list))
    row_ind = list(idx_list)
    col_ind = np.zeros(len(idx_list))
    return sp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return sp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors."""
    sp_nodes = sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(sp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def bfs(adj, roots):
    """
    Perform BFS on a graph given by an adjaceny matrix adj.
    Can take a set of multiple root nodes.
    Root nodes have level 0, first-order neighors have level 1, and so on.]
    """
    visited = set()
    current_lvl = set(roots)
    while current_lvl:
        for v in current_lvl:
            visited.add(v)

        next_lvl = get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference
        yield next_lvl

        current_lvl = next_lvl


def bfs_relational(adj_list, roots):
    """
    BFS for graphs with multiple edge types. Returns list of level sets.
    Each entry in list corresponds to relation specified by adj_list.
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = list()
    for rel in range(len(adj_list)):
        next_lvl.append(set())

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        for rel in range(len(adj_list)):
            next_lvl[rel] = get_neighbors(adj_list[rel], current_lvl)
            next_lvl[rel] -= visited  # set difference

        yield next_lvl

        current_lvl = set.union(*next_lvl)


def bfs_sample(adj, roots, max_lvl_size):
    """
    BFS with node dropout. Only keeps random subset of nodes per level up to max_lvl_size.
    'roots' should be a mini-batch of nodes (set of node indices).
    NOTE: In this implementation, not every node in the mini-batch is guaranteed to have
    the same number of neighbors, as we're sampling for the whole batch at the same time.
    """
    visited = set(roots)
    current_lvl = set(roots)
    while current_lvl:

        next_lvl = get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        for v in next_lvl:
            visited.add(v)

        yield next_lvl

        current_lvl = next_lvl


def get_splits(y, train_idx, test_idx, validation=True):
    # Make dataset splits
    # np.random.shuffle(train_idx)
    if validation:
        idx_train = train_idx[len(train_idx) / 5:]
        idx_val = train_idx[:len(train_idx) / 5]
        idx_test = idx_val  # report final score on validation set for hyperparameter optimization
    else:
        idx_train = train_idx
        idx_val = train_idx  # no validation
        idx_test = test_idx

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)

    y_train[idx_train] = np.array(y[idx_train].todense())
    y_val[idx_val] = np.array(y[idx_val].todense())
    y_test[idx_test] = np.array(y[idx_test].todense())

    return y_train, y_val, y_test, idx_train, idx_val, idx_test


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten())
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten())
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def binary_crossentropy(preds, labels):
    return np.mean(-labels*np.log(preds) - (1-labels)*np.log(1-preds))


def two_class_accuracy(preds, labels, threshold=0.5):
    return np.mean(np.equal(labels, preds > 0.5))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def evaluate_preds_sigmoid(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(binary_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(two_class_accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


class RDFReader:
    __graph = None
    __freq = {}

    def __init__(self, file):

        self.__graph = rdf.Graph()

        if file.endswith('nt.gz'):
            with gzip.open(file, 'rb') as f:
                self.__graph.parse(file=f, format='nt')
        else:
            self.__graph.parse(file, format=rdf.util.guess_format(file))

        # See http://rdflib.readthedocs.io for the rdflib documentation

        self.__freq = Counter(self.__graph.predicates())

        print("Graph loaded, frequencies counted.")

    def triples(self, relation=None):
        for s, p, o in self.__graph.triples((None, relation, None)):
            yield s, p, o

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__graph.destroy("store")
        self.__graph.close(True)

    def subjectSet(self):
        return set(self.__graph.subjects())

    def objectSet(self):
        return set(self.__graph.objects())

    def relationList(self):
        """
        Returns a list of relations, ordered descending by frequenecy
        :return:
        """
        res = list(set(self.__graph.predicates()))
        res.sort(key=lambda rel: - self.freq(rel))
        return res

    def __len__(self):
        return len(self.__graph)

    def freq(self, relation):
        """
        The frequency of this relation (how many distinct triples does it occur in?)
        :param relation:
        :return:
        """
        if relation not in self.__freq:
            return 0
        return self.__freq[relation]


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'], dtype=np.float32)


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_data(dataset_str='aifb', limit=-1):
    """
    :param dataset_str:
    :param rel_layers:
    :param limit: If > 0, will only load this many adj. matrices
        All adjacencies are preloaded and saved to disk,
        but only a limited a then restored to memory.
    :return:
    """

    print('Loading dataset', dataset_str)

    dirname = os.path.dirname(os.path.realpath(sys.argv[0]))

    if dataset_str == 'am':
        data_url = 'https://www.dropbox.com/s/htisydfgwxmrx65/am_stripped.nt.gz?dl=1'
        graph_file = '../data/am/am_stripped.nt.gz'

        if not os.path.isfile(graph_file):
            print('Downloading AM data.')
            wget.download(data_url, graph_file)

        task_file = '../data/am/completeDataset.tsv'
        train_file = '../data/am/trainingSet.tsv'
        test_file = '../data/am/testSet.tsv'
        label_header = 'label_cateogory'
        nodes_header = 'proxy'

    elif dataset_str == 'aifb':
        data_url = 'https://www.dropbox.com/s/fkvgvkygo2gf28k/aifb_stripped.nt.gz?dl=1'
        # The RDF file containing the knowledge graph
        graph_file = '../data/aifb/aifb_stripped.nt.gz'
        if not os.path.isfile(graph_file):
            print('Downloading AIFB data.')
            wget.download(data_url, graph_file)

        # The TSV file containing the classification task
        task_file = '../data/aifb/completeDataset.tsv'
        # The TSV file containing training indices
        train_file = '../data/aifb/trainingSet.tsv'
        # The TSV file containing test indices
        test_file = '../data/aifb/testSet.tsv'
        label_header = 'label_affiliation'
        nodes_header = 'person'

    elif dataset_str == 'mutag':
        data_url = 'https://www.dropbox.com/s/qy8j3p8eacvm4ir/mutag_stripped.nt.gz?dl=1'
        graph_file = '../data/mutag/mutag_stripped.nt.gz'
        if not os.path.isfile(graph_file):
            print('Downloading MUTAG data.')
            wget.download(data_url, graph_file)
        task_file = '../data/mutag/completeDataset.tsv'
        train_file = '../data/mutag/trainingSet.tsv'
        test_file = '../data/mutag/testSet.tsv'
        label_header = 'label_mutagenic'
        nodes_header = 'bond'

    elif dataset_str == 'bgs':
        data_url = 'https://www.dropbox.com/s/uqi0k9jd56j02gh/bgs_stripped.nt.gz?dl=1'
        graph_file = '../data/bgs/bgs_stripped.nt.gz'
        if not os.path.isfile(graph_file):
            print('Downloading BGS data.')
            wget.download(data_url, graph_file)
        task_file = '../data/bgs/completeDataset_lith.tsv'
        train_file = '../data/bgs/trainingSet(lith).tsv'
        test_file = '../data/bgs/testSet(lith).tsv'
        label_header = 'label_lithogenesis'
        nodes_header = 'rock'

    else:
        raise NameError('Dataset name not recognized: ' + dataset_str)

    adj_fprepend = '../data/' + dataset_str + '/adjacencies_'
    labels_file = '../data/' + dataset_str + '/labels.npz'
    train_idx_file = '../data/' + dataset_str + '/train_idx.npy'
    test_idx_file = '../data/' + dataset_str + '/test_idx.npy'
    train_names_file = '../data/' + dataset_str + '/train_names.npy'
    test_names_file = '../data/' + dataset_str + '/test_names.npy'
    rel_dict_file = '../data/' + dataset_str + '/rel_dict.pkl'
    nodes_file = '../data/' + dataset_str + '/nodes.pkl'

    graph_file = dirname + '/' + graph_file
    task_file = dirname + '/' + task_file
    train_file = dirname + '/' + train_file
    test_file = dirname + '/' + test_file
    adj_fprepend = dirname + '/' + adj_fprepend
    labels_file = dirname + '/' + labels_file
    train_idx_file = dirname + '/' + train_idx_file
    test_idx_file = dirname + '/' + test_idx_file
    train_names_file = dirname + '/' + train_names_file
    test_names_file = dirname + '/' + test_names_file
    rel_dict_file = dirname + '/' + rel_dict_file
    nodes_file = dirname + '/' + nodes_file

    adj_files = glob.glob(adj_fprepend + '*.npz')

    if adj_files != [] and os.path.isfile(labels_file) and \
            os.path.isfile(train_idx_file) and os.path.isfile(test_idx_file):

        # load precomputed adjacency matrix and labels

        adj_files.sort(
            key=lambda f: int(re.search('adjacencies_(.+?).npz', f).group(1)))

        if limit > 0:
            adj_files = adj_files[:limit * 2]

        adjacencies = [load_sparse_csr(file) for file in adj_files]
        adj_shape = adjacencies[0].shape

        print('Number of nodes: ', adj_shape[0])
        print('Number of relations: ', len(adjacencies))

        labels = load_sparse_csr(labels_file)
        labeled_nodes_idx = list(labels.nonzero()[0])

        print('Number of classes: ', labels.shape[1])

        train_idx = np.load(train_idx_file)
        test_idx = np.load(test_idx_file)
        train_names = np.load(train_names_file)
        test_names = np.load(test_names_file)

        relations_dict = pkl.load(open(rel_dict_file, 'rb'))

    else:

        # loading labels of nodes
        labels_df = pd.read_csv(task_file, sep='\t', encoding='utf-8')
        labels_train_df = pd.read_csv(train_file, sep='\t', encoding='utf8')
        labels_test_df = pd.read_csv(test_file, sep='\t', encoding='utf8')

        with RDFReader(graph_file) as reader:

            relations = reader.relationList()
            subjects = reader.subjectSet()
            objects = reader.objectSet()

            print([(rel, reader.freq(rel)) for rel in relations[:limit]])

            nodes = list(subjects.union(objects))
            adj_shape = (len(nodes), len(nodes))

            print('Number of nodes: ', len(nodes))
            print('Number of relations in the data: ', len(relations))

            relations_dict = {rel: i for i, rel in enumerate(list(relations))}
            nodes_dict = {node: i for i, node in enumerate(nodes)}

            assert len(nodes_dict) < np.iinfo(np.int32).max

            adjacencies = []

            for i, rel in enumerate(
                    relations if limit < 0 else relations[:limit]):

                print(
                    u'Creating adjacency matrix for relation {}: {}, frequency {}'.format(
                        i, rel, reader.freq(rel)))
                edges = np.empty((reader.freq(rel), 2), dtype=np.int32)

                size = 0
                for j, (s, p, o) in enumerate(reader.triples(relation=rel)):
                    if nodes_dict[s] > len(nodes) or nodes_dict[o] > len(nodes):
                        print(s, o, nodes_dict[s], nodes_dict[o])

                    edges[j] = np.array([nodes_dict[s], nodes_dict[o]])
                    size += 1

                print('{} edges added'.format(size))

                row, col = np.transpose(edges)

                data = np.ones(len(row), dtype=np.int8)

                adj = sp.csr_matrix((data, (row, col)), shape=adj_shape,
                                    dtype=np.int8)

                adj_transp = sp.csr_matrix((data, (col, row)), shape=adj_shape,
                                           dtype=np.int8)

                save_sparse_csr(adj_fprepend + '%d.npz' % (i * 2), adj)
                save_sparse_csr(adj_fprepend + '%d.npz' % (i * 2 + 1),
                                adj_transp)

                if limit < 0:
                    adjacencies.append(adj)
                    adjacencies.append(adj_transp)

        # Reload the adjacency matrices from disk
        if limit > 0:
            adj_files = glob.glob(adj_fprepend + '*.npz')
            adj_files.sort(key=lambda f: int(
                re.search('adjacencies_(.+?).npz', f).group(1)))

            adj_files = adj_files[:limit * 2]
            for i, file in enumerate(adj_files):
                adjacencies.append(load_sparse_csr(file))
                print('%d adjacency matrices loaded ' % i)

        nodes_u_dict = {np.unicode(to_unicode(key)): val for key, val in
                        nodes_dict.items()}

        labels_set = set(labels_df[label_header].values.tolist())
        labels_dict = {lab: i for i, lab in enumerate(list(labels_set))}

        print('{} classes: {}', len(labels_set), labels_set)

        labels = sp.lil_matrix((adj_shape[0], len(labels_set)))
        labeled_nodes_idx = []

        print('Loading training set')

        train_idx = []
        train_names = []
        for nod, lab in zip(labels_train_df[nodes_header].values,
                            labels_train_df[label_header].values):
            nod = np.unicode(to_unicode(nod))  # type: unicode
            if nod in nodes_u_dict:
                labeled_nodes_idx.append(nodes_u_dict[nod])
                label_idx = labels_dict[lab]
                labels[labeled_nodes_idx[-1], label_idx] = 1
                train_idx.append(nodes_u_dict[nod])
                train_names.append(nod)
            else:
                print(u'Node not in dictionary, skipped: ',
                      nod.encode('utf-8', errors='replace'))

        print('Loading test set')

        test_idx = []
        test_names = []
        for nod, lab in zip(labels_test_df[nodes_header].values,
                            labels_test_df[label_header].values):
            nod = np.unicode(to_unicode(nod))
            if nod in nodes_u_dict:
                labeled_nodes_idx.append(nodes_u_dict[nod])
                label_idx = labels_dict[lab]
                labels[labeled_nodes_idx[-1], label_idx] = 1
                test_idx.append(nodes_u_dict[nod])
                test_names.append(nod)
            else:
                print(u'Node not in dictionary, skipped: ',
                      nod.encode('utf-8', errors='replace'))

        labeled_nodes_idx = sorted(labeled_nodes_idx)
        labels = labels.tocsr()

        save_sparse_csr(labels_file, labels)

        np.save(train_idx_file, train_idx)
        np.save(test_idx_file, test_idx)

        np.save(train_names_file, train_names)
        np.save(test_names_file, test_names)

        pkl.dump(relations_dict, open(rel_dict_file, 'wb'))
        pkl.dump(nodes, open(nodes_file, 'wb'))

    features = sp.identity(adj_shape[0], format='csr')

    return adjacencies, features, labels, labeled_nodes_idx, train_idx, test_idx, relations_dict, train_names, test_names


def parse(symbol):
    if symbol.startswith('<'):
        return symbol[1:-1]
    return symbol


def to_unicode(input):
    if isinstance(input, str):
        return input
    # if isinstance(input, unicode):
    #     return input
    # elif isinstance(input, str):
    #     return input.decode('utf-8', errors='replace')
    return str(input).decode('utf-8', errors='replace')