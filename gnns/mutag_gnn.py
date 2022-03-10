import sys
import time
import random
import argparse
import os
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import ModuleList
from torch.nn import Sequential as Seq, ReLU, Linear as Lin, Softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINEConv, BatchNorm, global_mean_pool

sys.path.append('..')
from gnns.overloader import overload
from datasets.mutag_dataset import Mutagenicity
from utils import set_seed, Gtrain, Gtest


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mutag Model")

    parser.add_argument('--data_path', nargs='?', default=osp.join(osp.dirname(__file__), '..', 'data', 'MUTAG'),
                        help='Input data path.')
    parser.add_argument('--model_path', nargs='?', default=osp.join(osp.dirname(__file__), '..', 'param', 'gnns'),
                        help='path for saving trained model.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='GPU device.')
    parser.add_argument('--epoch', type=int, default=300, 
                        help='Number of epoch.')
    parser.add_argument('--lr', type=float, default= 1e-3,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Interval of evaluation.')
    parser.add_argument('--num_unit', type=int, default=2,
                        help='number of Convolution layers(units)')
    parser.add_argument('--random_label', type=bool, default=False,
                        help='train a model under label randomization for sanity check')

    return parser.parse_args()


class MutagNet(torch.nn.Module):

    def __init__(self, conv_unit=2):
        super(MutagNet, self).__init__()

        self.node_emb = Lin(14, 32)
        self.edge_emb = Lin(3, 32)
        self.relu_nn = ModuleList([ReLU() for i in range(conv_unit)])

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()

        for i in range(conv_unit):
            conv = GINEConv(nn=Seq(Lin(32, 75), self.relu_nn[i], Lin(75, 32)))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(32))
            self.relus.append(ReLU())

        self.lin1 = Lin(32, 16)
        self.relu = ReLU()
        self.lin2 = Lin(16, 2)
        self.softmax = Softmax(dim=1)

    @overload
    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return self.get_pred(graph_x)

    @overload
    def get_node_reps(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        for conv, batch_norm, ReLU in \
                zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_attr)
            x = ReLU(batch_norm(x))
        node_x = x
        return node_x
        
    @overload
    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_pred(self, graph_x):
        pred = self.relu(self.lin1(graph_x))
        pred = self.lin2(pred)
        self.readout = self.softmax(pred)
        return pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)

if __name__ == '__main__':

    set_seed(0)
    args = parse_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    test_dataset = Mutagenicity(args.data_path, mode='testing')
    val_dataset = Mutagenicity(args.data_path, mode='evaluation')
    train_dataset = Mutagenicity(args.data_path, mode='training')
    if args.random_label:
        for dataset in [test_dataset, val_dataset, train_dataset]:
            for g in dataset:
                g.y.fill_(random.choice([0, 1]))

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False
                             )
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False
                            )
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True
                              )
    model = MutagNet(args.num_unit).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr
                                 )
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.8,
                                  patience=10,
                                  min_lr=1e-4
                                  )
    min_error = None
    for epoch in range(1, args.epoch + 1):

        t1 = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']

        loss = Gtrain(train_loader,
                      model,
                      optimizer,
                      device=device,
                      criterion=nn.CrossEntropyLoss()
                      )

        _, train_acc = Gtest(train_loader,
                             model,
                             device=device,
                             criterion=nn.CrossEntropyLoss()
                             )

        val_error, val_acc = Gtest(val_loader,
                                   model,
                                   device=device,
                                   criterion=nn.CrossEntropyLoss()
                                   )
        test_error, test_acc = Gtest(test_loader,
                                     model,
                                     device=device,
                                     criterion=nn.CrossEntropyLoss()
                                     )
        scheduler.step(val_error)
        if min_error is None or val_error <= min_error:
            min_error = val_error

        t2 = time.time()

        if epoch % args.verbose == 0:
            test_error, test_acc = Gtest(test_loader,
                                         model,
                                         device=device,
                                         criterion=nn.CrossEntropyLoss()
                                         )
            t3 = time.time()
            print('Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Test Loss: {:.5f}, '
                  'Test acc: {:.5f}'.format(epoch, t3 - t1, lr, loss, test_error, test_acc))
            continue

        print('Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Train acc: {:.5f}, Validation Loss: {:.5f}, '
              'Validation acc: {:5f}'.format(epoch, t2 - t1, lr, loss, train_acc, val_error, val_acc))

    if args.random_label:
        save_path = 'mutag_net_rd.pt'
    else:
        save_path = 'mutag_net.pt'
    
    if not osp.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(model.cpu(), osp.join(args.model_path, save_path))