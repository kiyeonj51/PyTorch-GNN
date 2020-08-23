import os
import numpy as np
from pyvis.network import Network

path= os.getcwd()+ "/data/cora/"
dataset="cora"

idx_features_labels = np.genfromtxt(f"{path}{dataset}.content", dtype=np.dtype(str))
edges_unordered = np.genfromtxt(f"{path}{dataset}.cites",dtype=int)

idx = np.array(idx_features_labels[:, 0], dtype=int)
idx_map = {j: i for i, j in enumerate(idx)}
labels = list(idx_features_labels[:, -1])

edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=int).reshape(edges_unordered.shape)

color = ['#f09494', '#eebcbc', '#72bbd0', '#91f0a1', '#629fff', '#bcc2f2', '#eebcbc']
color_map = {label: color[i] for i, label in enumerate(set(labels))}
colors = np.array([color_map[label] for label in labels])

num_node = 300
net = Network(heading="cora")
net.add_nodes(list(range(len(labels[:num_node]))),
              label=labels[:num_node],
              color=colors[:num_node])

for edge in edges:
    if edge[0]<num_node and edge[1]<num_node:
        net.add_edge(int(edge[0]),int(edge[1]))
net.show_buttons(filter_=['edges'])
net.show("visualize/cora.html")