# from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/mnist_voxel_grid.py
import os
import argparse
import os.path as osp

import torch
from functools import wraps
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv, voxel_grid, max_pool_x
from torch_geometric.data import  Batch

import sys
sys.path.append('..')
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train MNIST Spuer-Pixel Model")

    parser.add_argument('--data_path', nargs='?', default=osp.join(osp.dirname(__file__), '..', 'data', 'MNIST'),
                        help='Input data path.')
    parser.add_argument('--model_path', nargs='?', default=osp.join(osp.dirname(__file__), '..', 'param', 'gnns'),
                        help='path for saving trained model.')
    parser.add_argument('--epoch', type=int, default=21,
                        help='Number of epoch.')
    parser.add_argument('--lr', type=float, default= 0.01,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')

    return parser.parse_args()

def overload(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        if len(args) +  len(kargs) == 2:
            # for inputs like model(g)
            if len(args) == 2:
                g = args[1]
            # for inputs like model(graph=g)
            else:
                g = kargs['data']
            return func(args[0], g.to(device))
        elif len(args) +  len(kargs) == 6:
            # for inputs like model(x, ..., batch, pos)
            if len(args) == 6:
                _, x, edge_index, edge_attr, batch, pos = args
            # for inputs like model(x=x, ..., batch=batch, pos=pos)
            else:
                x, edge_index = kargs['x'], kargs['edge_index']
                edge_attr, batch = kargs['edge_attr'], kargs['batch']
                pos = kargs['pos']
            row, col = edge_index
            ground_tuth_mask = (x[row] > 0).view(-1).bool() * (x[col] > 0).view(-1).bool()
            g = Batch(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
                pos=pos,
                ground_tuth_mask=ground_tuth_mask
            ).to(x.device)
            return func(args[0], g)
        else:
            raise TypeError
    return wrapper

class MNISTNet(torch.nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=2, kernel_size=5)
        self.fc1 = torch.nn.Linear(4 * 64, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        import torch_geometric.transforms as T
        self.transform = T.Cartesian(cat=False, max_value=9)
        
    @overload
    def forward(self, data):
        graph_x = self.get_graph_rep(data)
        pred = self.get_pred(graph_x)
        self.readout = pred.softmax(dim=1)
        return pred
    
    @overload
    def get_graph_rep(self, data):
        from torch.autograd import Variable
        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv3(x, data.edge_index, data.edge_attr))
        cluster = voxel_grid(data.pos, data.batch, size=14, start=0, end=27.99)
        x, _ = max_pool_x(cluster, x, data.batch, size=4)
        graph_x = x.view(-1, self.fc1.weight.size(1))
        return graph_x

    def get_pred(self, graph_x):
        graph_x = F.elu(self.fc1(graph_x))
        graph_x = F.dropout(graph_x, training=self.training)
        graph_x = self.fc2(graph_x)
        return F.log_softmax(graph_x, dim=1)
    
def train(epoch):
    model.train()

    if epoch == 6:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        F.nll_loss(model(data), data.y).backward()
        optimizer.step()


def test():
    model.eval()
    correct = 0

    for data in test_loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(test_dataset)
if __name__ == '__main__':
    
    set_seed(0)
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Cartesian(cat=False, max_value=9)
    train_dataset = MNISTSuperpixels(args.data_path, True, transform=transform)
    test_dataset = MNISTSuperpixels(args.data_path, False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    d = train_dataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epoch):
        train(epoch)
        test_acc = test()
        print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))

    if not osp.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(model.cpu(), osp.join(args.model_path, 'mnist_net.pt'))

    
