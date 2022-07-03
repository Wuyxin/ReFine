import copy
import math
import numpy as np
from torch_geometric.nn import MessagePassing

import torch
from torch_geometric.nn import MessagePassing
from explainers.base import Explainer
from .common import EdgeMaskNet

EPS = 1e-6


class PGExplainer(Explainer):
    coeffs = {
        'edge_size': 1e-4,
        'edge_ent': 1e-2,
    }

    def __init__(self, device, gnn_model,
                 n_in_channels=14,
                 e_in_channels=3,
                 hid=64, n_layers=2,
                 n_label=2
                 ):
        super(PGExplainer, self).__init__(device, gnn_model)

        self.device = device
        self.edge_mask=EdgeMaskNet(
            n_in_channels,
            e_in_channels,
            hid=hid,
            n_layers=n_layers).to(self.device)

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

    def __setup_target_label__(self, graph):
        
        if not hasattr(graph, 'hat_y'):
            graph.hat_y = self.model(graph).argmax(-1).to(graph.x.device)
        return graph

    def __reparameterize__(self, log_alpha, beta=0.1, training=True):

        if training:
            random_noise = torch.rand(log_alpha.size()).to(self.device)
            gate_inputs = torch.log2(random_noise) - torch.log2(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta + EPS
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def __loss__(self, log_logits, mask, pred_label):

        # loss = criterion(log_logits, pred_label)
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

    def get_mask(self, graph):
        # batch version
        graph_map = graph.batch[graph.edge_index[0, :]]
        mask = torch.FloatTensor([]).to(self.device)
        for i in range(len(graph.y)):
            edge_indicator = (graph_map == i).bool()
            G_i_mask = self.edge_mask(
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

    def explain_graph(self, graph, model=None,
                      temp=0.1, ratio=0.1,
                      draw_graph=0, vis_ratio=0.2,
                      train_mode=False, supplement=False
                      ):
        
        if not model == None:
            self.model = model
            
        graph = self.__setup_target_label__(graph)
        ori_mask = self.get_mask(graph)
        edge_mask = self.__reparameterize__(ori_mask, training=train_mode, beta=temp)
        imp = edge_mask.detach().cpu().numpy()

        if train_mode:
            # ----------------------------------------------------
            # (1) batch version: get positive edge index(G_s) for ego graph
            self.__set_masks__(edge_mask, self.model)
            log_logits = self.model(graph)
            loss = self.__loss__(log_logits, edge_mask, graph.hat_y)
            
            self.__clear_masks__(self.model)
            return loss

        if draw_graph:
            self.visualize(graph, imp, self.name, vis_ratio=vis_ratio)
        self.last_result = (graph, imp)

        return imp
