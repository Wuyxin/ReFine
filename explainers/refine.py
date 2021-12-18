import copy
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_geometric.nn import MessagePassing
from explainers.base import Explainer
from .common import EdgeMaskNet

EPS = 1e-6

class ReFine(Explainer):
    coeffs = {
        'edge_size': 1e-4,
        'edge_ent': 1e-2,
    }

    def __init__(self, device, gnn_model,
                 n_in_channels=14,
                 e_in_channels=3,
                 hid=50, n_layers=2,
                 n_label=2, gamma=1
                 ):
        super(ReFine, self).__init__(device, gnn_model)

        self.edge_mask = nn.ModuleList([
            EdgeMaskNet(
                n_in_channels,
                e_in_channels,
                hid=hid,
                n_layers=n_layers) for _ in range(n_label)
        ]).to(device)
        self.gamma = gamma

    def __set_masks__(self, mask, model):
    
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = mask

    def __clear_masks__(self, model):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None

    def __reparameterize__(self, log_alpha, beta=1, training=True):
        
        if training:
            random_noise = torch.rand(log_alpha.size()).to(self.device)
            gate_inputs = torch.log2(random_noise) - torch.log2(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta + EPS
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()
            
        return gate_inputs

    def fidelity_loss(self, log_logits, mask, pred_label):

        idx = [i for i in range(len(pred_label))]
        loss = -log_logits.softmax(dim=1)[idx, pred_label.view(-1)].sum()

        loss = loss + self.coeffs['edge_size'] * mask.mean()
        ent = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()
        return loss

    # batch version
    def pack_subgraph(self, graph, imp, top_ratio=0.2):

        if abs(top_ratio - 1.0) < EPS:
            return graph, imp

        exp_subgraph = copy.deepcopy(graph)
        top_idx = torch.LongTensor([])
        graph_map = graph.batch[graph.edge_index[0, :]]

        # extract ego graph
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
            Gi_n_edge = len(edge_indicator)
            topk = max(math.ceil(top_ratio * Gi_n_edge), 1)

            Gi_pos_edge_idx = np.argsort(-imp[edge_indicator])[:topk]
            top_idx = torch.cat([top_idx, edge_indicator[Gi_pos_edge_idx]])

        exp_subgraph.edge_attr = graph.edge_attr[top_idx]
        exp_subgraph.edge_index = graph.edge_index[:, top_idx]
        exp_subgraph.x, exp_subgraph.edge_index, exp_subgraph.batch, _ = \
            self.__relabel__(exp_subgraph, exp_subgraph.edge_index)

        return exp_subgraph, imp[top_idx]

    def get_contrastive_loss(self, c, y, batch, tau=0.1):
        
        c = c / c.norm(dim=1, keepdim=True)
        mat = F.relu(torch.mm(c, c.T))
        unique_graphs = torch.unique(batch)

        ttl_scores = torch.sum(mat, dim=1)
        pos_scores = torch.tensor([mat[i, y == y[i]].sum() for i in unique_graphs]).to(c.device)
        
        # contrastive_loss = - torch.log(torch.sum(pos_scores / ttl_scores, dim=0))
        contrastive_loss = - torch.logsumexp(pos_scores / (tau * ttl_scores), dim=0)
        
        return contrastive_loss

    def get_mask(self, graph):
        # batch version
        graph_map = graph.batch[graph.edge_index[0, :]]
        mask = torch.FloatTensor([]).to(graph.x.device)
        for i in range(len(graph.y)):
            edge_indicator = (graph_map == i).bool()
            G_i_mask = self.edge_mask[graph.y[i]](
                graph.x,
                graph.edge_index[:, edge_indicator],
                graph.edge_attr[edge_indicator, :]
            ).view(-1)
            mask = torch.cat([mask, G_i_mask])
        return mask

    def get_pos_edge(self, graph, mask, ratio):

        num_edge = [0]
        num_node = [0]
        sep_edge_idx = []
        graph_map = graph.batch[graph.edge_index[0, :]]
        pos_idx = torch.LongTensor([])
        mask = mask.detach().cpu()
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
            Gi_n_edge = len(edge_indicator)
            topk = max(math.ceil(ratio * Gi_n_edge), 1)

            Gi_pos_edge_idx = np.argsort(-mask[edge_indicator])[:topk]

            pos_idx = torch.cat([pos_idx, edge_indicator[Gi_pos_edge_idx]])
            num_edge.append(num_edge[i] + Gi_n_edge)
            num_node.append(
                num_node[i] + (graph.batch == i).sum().long()
            )
            sep_edge_idx.append(Gi_pos_edge_idx)

        return pos_idx, num_edge, num_node, sep_edge_idx

    def explain_graph(
        self, graph, ratio=1.0, fine_tune=False,
        lr=1e-4, epoch=50, draw_graph=0, vis_ratio=0.2):
        
        if not fine_tune:
            edge_mask = self.get_mask(graph)
            edge_mask = self.__reparameterize__(edge_mask, training=False)
            imp = edge_mask.detach().cpu().numpy()
            self.last_result = (graph, imp)
            
            if draw_graph:
                self.visualize(graph, imp, vis_ratio=vis_ratio)
            return imp

        mask_net = copy.deepcopy(self.edge_mask[graph.y.item()])
        optimizer = torch.optim.Adam(mask_net.parameters(), lr=lr)
        for _ in range(epoch):
            optimizer.zero_grad()
            edge_mask = mask_net(
                graph.x,
                graph.edge_index,
                graph.edge_attr
            ).view(-1)
            edge_mask = self.__reparameterize__(edge_mask, training=False)

            pos_idx, _, _, _ = self.get_pos_edge(graph, edge_mask, ratio)
            pos_edge_mask = edge_mask[pos_idx]
            pos_edge_index = graph.edge_index[:, pos_idx]
            pos_edge_attr = graph.edge_attr[pos_idx, :]
            self.__set_masks__(pos_edge_mask, self.model)
            G1_x, G1_pos_edge_index, G1_batch, G1_pos = self.__relabel__(graph, pos_edge_index)

            log_logits = self.model(
                x=G1_x,
                edge_index=G1_pos_edge_index,
                edge_attr=pos_edge_attr,
                batch=G1_batch,
                pos=G1_pos
            )
            self.__clear_masks__(self.model)
            fid_loss = self.fidelity_loss(log_logits, edge_mask, graph.y)
            fid_loss.backward()
            optimizer.step()
        
        imp = edge_mask.detach().cpu().numpy()
        self.last_result = (graph, imp)
        
        if draw_graph:
            self.visualize(graph, imp, vis_ratio=vis_ratio)
        return imp

    def pretrain(
        self, graph, model=None,
        ratio=1.0, reperameter=False, **kwargs
        ):

        if model == None:
            model = self.model

        ori_mask = self.get_mask(graph)
        edge_mask = self.__reparameterize__(ori_mask, training=reperameter)
        
        # ----------------------------------------------------
        # (1) compute fidelity loss
        self.__set_masks__(edge_mask, self.model)
        log_logits = self.model(graph)
        fid_loss = self.fidelity_loss(log_logits, edge_mask, graph.y)
        self.__clear_masks__(self.model)
        
        # ----------------------------------------------------
        # (2) compute contrastive loss
        pos_idx, _, _, _ = self.get_pos_edge(graph, edge_mask, ratio)
        pos_edge_mask = edge_mask[pos_idx]
        pos_edge_index = graph.edge_index[:, pos_idx]
        pos_edge_attr = graph.edge_attr[pos_idx, :]
        self.__set_masks__(pos_edge_mask, self.model)
        G1_x, G1_pos_edge_index, G1_batch, G1_pos = self.__relabel__(graph, pos_edge_index)
        graph_rep = self.model.get_graph_rep(
            x=G1_x,
            edge_index=G1_pos_edge_index,
            edge_attr=pos_edge_attr,
            batch=G1_batch,
            pos=G1_pos
        )
        cts_loss = self.get_contrastive_loss(graph_rep, graph.y, graph.batch)
        self.__clear_masks__(self.model)

        loss =  fid_loss + self.gamma * cts_loss
        
        return loss
    
    def remap_device(self, device):
        self.device = device
        self.model = self.model.to(device)
        self.edge_mask = self.edge_mask.to(device)