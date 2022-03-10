import os
import sys
import time
import torch
import random
import numpy as np
import networkx as nx
from torch import nn
import matplotlib.pyplot as plt



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def draw_ground_truth(graph,
                      save_path,
                      node_size=3,
                      edge_width=.1,
                      dpi=500,
                      edgecolors='k'
                      ):
    G = nx.Graph()
    if graph.pos is None:
        graph.pos = nx.random_layout(nx.path_graph(graph.num_nodes))

    nx.draw_networkx_nodes(G, pos=graph.pos,
                           nodelist=range(graph.num_nodes),
                           node_size=node_size,
                           node_color=graph.y.cpu().numpy(),
                           cmap='bwr',
                           edgecolors=edgecolors,
                           linewidths=.1)

    nx.draw_networkx_edges(G, pos=graph.pos,
                           edgelist=list(graph.edge_index.cpu().numpy().T),
                           width=edge_width)
    plt.savefig(save_path, dpi=dpi, pad_inches=0)



def PrintGraph(graph):

    if graph.name:
        print("Name: %s" % graph.name)
    print("# Nodes:%6d      | # Edges:%6d |  Class: %2d" \
          % (graph.num_nodes, graph.num_edges, graph.y))

    print("# Node features: %3d| # Edge feature(s): %3d" \
          % (graph.num_node_features, graph.num_edge_features))


def print_to_file(path='log/', fileName=None):

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass
    if not os.path.exists(path):
            os.makedirs(path)
    # cover old log file
    if os.path.exists(os.path.join(path, fileName)):
        os.remove(os.path.join(path, fileName))
    if not fileName:
        fileName = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))
    sys.stdout = Logger(fileName, path=path)

    print(fileName.center(50, '*'))
