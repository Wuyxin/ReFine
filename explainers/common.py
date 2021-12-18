from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import ModuleList, Linear as Lin
from torch_geometric.nn import BatchNorm, ARMAConv


class MLP(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, act=nn.Tanh()):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(OrderedDict([
                ('lin1', Lin(in_channels, hidden_channels)),
                ('act', act),
                ('lin2', Lin(hidden_channels, out_channels))
                ]))
     
    def forward(self, x):
        return self.mlp(x)


class EdgeMaskNet(torch.nn.Module):

    def __init__(self,
                 n_in_channels,
                 e_in_channels,
                 hid=72, n_layers=3):
        super(EdgeMaskNet, self).__init__()

        self.node_lin = Lin(n_in_channels, hid)
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(n_layers):
            conv = ARMAConv(in_channels=hid, out_channels=hid)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hid))

        if e_in_channels > 1:
            self.edge_lin1 = Lin(2 * hid, hid)
            self.edge_lin2 = Lin(e_in_channels, hid)
            self.mlp = MLP(2 * hid, hid, 1)
        else:
            self.mlp = MLP(2 * hid, hid, 1)
        self._initialize_weights()
        
    def forward(self, x, edge_index, edge_attr):

        x = torch.flatten(x, 1, -1)
        x = F.relu(self.node_lin(x))
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(conv(x, edge_index))
            x = batch_norm(x)

        e = torch.cat([x[edge_index[0, :]], x[edge_index[1, :]]], dim=1)

        if edge_attr.size(-1) > 1:
            e1 = self.edge_lin1(e)
            e2 = self.edge_lin2(edge_attr)
            e = torch.cat([e1, e2], dim=1)  # connection

        return self.mlp(e)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight) 
