import os
import sys
import time
import torch
import random
import numpy as np
import networkx as nx
from torch import nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# general function to train Node classification tasks
# under single graph.
def Ntrain(args, graph, model):
    model.train()
    idx = [i for i in range(graph.num_nodes)]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    nodes_train = random.sample(idx, int(args.train_ratio * graph.num_nodes))
    nodes_test = [i for i in idx if i not in nodes_train]

    t1 = time.time()
    count = 0
    last_test_acc = 0

    for epoch in range(1, args.epoch + 1):

        optimizer.zero_grad()
        output = model(graph.x, graph.edge_index, graph.edge_attr)
        loss = criterion(output[nodes_train], graph.y[nodes_train])
        loss.backward()
        optimizer.step()

        if epoch % args.verbose == 0:
            train_loss, train_acc = Ntest(nodes_train, graph, model)
            test_loss, test_acc = Ntest(nodes_test, graph, model)
            t2 = time.time()

            print('Epoch {:5d}[{:.2f}s], Train Loss: {:.7f}, Test Loss: {:.7f},  Train Acc: {:.7f}, Test Acc: {:.7f}' \
                  .format(epoch, t2 - t1, train_loss, test_loss, train_acc, test_acc))
            t1 = time.time()

            # early stopping
            if last_test_acc < test_acc:
                count = 0
            else:
                count += 1

            if count == 5:
                break

            last_test_acc = test_acc


def Ntest(node_idx, graph, model):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    output = model(graph.x, graph.edge_index, graph.edge_attr)
    loss = criterion(output[node_idx], graph.y[node_idx])
    acc = float(model.readout[node_idx].argmax(dim=1).eq(graph.y[node_idx]).sum().item()) / len(node_idx)
    return loss, acc


# General function for training graph classification(regresion) task and
# node classification task under multiple graphs.
def Gtrain(train_loader,
           model,
           optimizer,
           criterion=nn.MSELoss()
           ):
    model.train()
    loss_all = 0
    criterion = criterion

    for data in train_loader:
        data.to(device)
        optimizer.zero_grad()
        out = model(data.x,
                    data.edge_index,
                    data.edge_attr,
                    data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_loader.dataset)


def Gtest(test_loader,
          model,
          criterion=nn.L1Loss(reduction='mean'),
          ):

    model.eval()
    error = 0
    correct = 0

    with torch.no_grad():

        for data in test_loader:
            data = data.to(device)
            output = model(data.x,
                           data.edge_index,
                           data.edge_attr,
                           data.batch,
                           )

            error += criterion(output, data.y) * data.num_graphs
            correct += float(output.argmax(dim=1).eq(data.y).sum().item())

        return error / len(test_loader.dataset), correct / len(test_loader.dataset)


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
