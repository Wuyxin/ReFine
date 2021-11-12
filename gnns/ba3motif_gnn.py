import sys
import time
import random
import argparse
import os
import os.path as osp

import torch
from torch.nn import Linear, ReLU, ModuleList, Softmax, CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import global_mean_pool ,LEConv
from torch_geometric.data import DataLoader
import torch.nn.functional as F

from overloader import overload

sys.path.append('..')
from utils import set_seed, Gtrain, Gtest
from datasets.ba3motif_dataset import BA3Motif



def parse_args():
    parser = argparse.ArgumentParser(description="Train BA-3Motif Model")

    parser.add_argument('--data_path', nargs='?', default=osp.join(osp.dirname(__file__), '..', 'data', 'BA3'),
                        help='Input data path.')
    parser.add_argument('--model_path', nargs='?', default=osp.join(osp.dirname(__file__), '..', 'param', 'gnns'),
                        help='path for saving trained model.')
    parser.add_argument('--epoch', type=int, default=100,
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

class BA3MotifNet(torch.nn.Module):
    def __init__(self, num_unit):
        super().__init__()

        self.num_unit = num_unit

        self.node_emb = Linear(4, 64)

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()

        for i in range(num_unit):
            conv = LEConv(in_channels=64, out_channels=64)
            self.convs.append(conv)
            self.relus.append(ReLU())

        self.lin1 = Linear(64, 64)
        self.relu = ReLU()
        self.lin2 = Linear(64, 3)
        self.softmax = Softmax(dim=1)

    @overload
    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return self.get_pred(graph_x)

    @overload
    def get_node_reps(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        x = F.dropout(x, p=0.4)
        for conv, ReLU in \
                zip(self.convs, self.relus):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = ReLU(x)
        x = F.dropout(x, p=0.4)
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
        self.readout = pred
        return pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)

if __name__ == '__main__':

    set_seed(0)
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = BA3Motif(args.data_path, mode='testing')
    val_dataset = BA3Motif(args.data_path, mode='evaluation')
    train_dataset = BA3Motif(args.data_path, mode='training')
    if args.random_label:
        for dataset in [test_dataset, val_dataset, train_dataset]:
            for g in dataset:
                g.y.fill_(random.choice([0, 1, 2]))

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
    model = BA3MotifNet(args.num_unit).to(device)

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
                      criterion=CrossEntropyLoss()
                      )

        _, train_acc = Gtest(train_loader,
                             model,
                             criterion=CrossEntropyLoss()
                             )

        val_error, val_acc = Gtest(val_loader,
                                   model,
                                   criterion=CrossEntropyLoss()
                                   )
        test_error, test_acc = Gtest(test_loader,
                                     model,
                                     criterion=CrossEntropyLoss()
                                     )
        scheduler.step(val_error)
        if min_error is None or val_error <= min_error:
            min_error = val_error

        t2 = time.time()

        if epoch % args.verbose == 0:
            test_error, test_acc = Gtest(test_loader,
                                         model,
                                         criterion=CrossEntropyLoss()
                                         )
            t3 = time.time()
            print('Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Test Loss: {:.5f}, '
                  'Test acc: {:.5f}'.format(epoch, t3 - t1, lr, loss, test_error, test_acc))
            continue

        print('Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Train acc: {:.5f}, Validation Loss: {:.5f}, '
              'Validation acc: {:5f}'.format(epoch, t2 - t1, lr, loss, train_acc, val_error, val_acc))

        torch.cuda.empty_cache()
    if args.random_label:
        save_path = 'ba3_net_rd.pt'
    else:
        save_path = 'ba3_net.pt'
    if not osp.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(model.cpu(), osp.join(args.model_path, save_path))
