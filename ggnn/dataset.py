from ggnn.utils import load_graphs_from_file
from ggnn.utils import find_max_edge_id, find_max_node_id, find_max_task_id
from ggnn.utils import split_set, data_convert, create_adjacency_matrix
from torch.utils.data import DataLoader


class bAbIDataset():
    """
    Load bAbI tasks for GGNN
    """

    def __init__(self, path, task_id, is_train):
        all_data = load_graphs_from_file(path)
        self.n_edge_types = find_max_edge_id(all_data)
        self.n_tasks = find_max_task_id(all_data)
        self.n_node = find_max_node_id(all_data)

        all_task_train_data, all_task_val_data = split_set(all_data)

        if is_train:
            all_task_train_data = data_convert(all_task_train_data, 1)
            self.data = all_task_train_data[task_id]
        else:
            all_task_val_data = data_convert(all_task_val_data, 1)
            self.data = all_task_val_data[task_id]

    def __getitem__(self, index):
        am = create_adjacency_matrix(self.data[index][0], self.n_node, self.n_edge_types)
        annotation = self.data[index][1]
        target = self.data[index][2] - 1
        return am, annotation, target

    def __len__(self):
        return len(self.data)


class bAbIDataloader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(bAbIDataloader, self).__init__(*args, **kwargs)