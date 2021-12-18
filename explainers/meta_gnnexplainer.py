"""
Modified based on torch_geometric.nn.models.GNNExplainer
which generates explainations in node prediction tasks.

Citation:
Ying et al. GNNExplainer: Generating Explanations for Graph Neural Networks.
"""

from math import sqrt
import torch
from torch_geometric.nn import MessagePassing

EPS = 1e-15


class MetaGNNGExplainer(torch.nn.Module):

    coeffs = {
        'edge_size': 0.05,
        'edge_ent': 0.5,
    }

    def __init__(self, model, epochs=100, lr=0.01, log=True):
        super(MetaGNNGExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.log = log

    def __set_masks__(self, x, edge_index, init="normal"):

        N = x.size(0)
        E = edge_index.size(1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.edge_mask = None

    def __loss__(self, log_logits, pred_label):

        # pred = log_logits.softmax(dim=1)[0, pred_label]
        # loss = -torch.log2(pred+ EPS) + torch.log2(1 - pred+ EPS)

        loss = -log_logits[0, pred_label]
        m = self.edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()
        return loss

    def explain_graph(self, graph, **kwargs):

        self.__clear_masks__()

        # get the initial prediction.
        with torch.no_grad():
            log_logits = self.model(graph)
            pred_label = log_logits.argmax(dim=-1)

        self.__set_masks__(graph.x, graph.edge_index)
        self.to(graph.x.device)

        optimizer = torch.optim.Adam([self.edge_mask], lr=self.lr)

        for epoch in range(1, self.epochs + 1):

            optimizer.zero_grad()
            log_logits = self.model(graph)
            loss = self.__loss__(log_logits, pred_label)
            loss.backward()
            optimizer.step()

        edge_mask = self.edge_mask.detach().sigmoid()
        self.__clear_masks__()

        return edge_mask

    def __repr__(self):
        return f'{self.__class__.__name__}()'
