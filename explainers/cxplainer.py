

import torch
from torch_geometric.nn import NNConv
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Softmax
from explainers.base import Explainer

import warnings
warnings.filterwarnings("ignore")


class CXplainer(Explainer):
    
    def __init__(self, device, gnn_model_path):
        super(CXplainer, self).__init__(device, gnn_model_path)
        
    def explain_graph(self, graph,
                      model=None,
                      epoch=100,
                      lr=0.01,
                      draw_graph=0,
                      vis_ratio=0.2):

        y = graph.y
        if model == None:
            model = self.model
        model(graph)
        orig_pred = model.readout[0, y]

        granger_imp = []
        for e_id in range(graph.num_edges):
            edge_mask = torch.ones(graph.num_edges, dtype=torch.bool)
            edge_mask[e_id] = False
            tmp_g = graph.clone()
            tmp_g.edge_index = graph.edge_index[:, edge_mask]
            tmp_g.edge_attr = graph.edge_attr[edge_mask]
            model(tmp_g)

            masked_pred = model.readout[0, y]
            granger_imp.append(float(orig_pred - masked_pred))

        granger_imp = torch.FloatTensor(granger_imp)
        scores = self.norm_imp(granger_imp).to(self.device)
        
        explainer = CX_Model(graph, h_dim=32).to(self.device) 
        optimizer = torch.optim.Adam(explainer.parameters(), lr=lr)

        for i in range(1, epoch + 1):

            optimizer.zero_grad()
            out = explainer()
            out = F.softmax(out)
            loss = F.kl_div(scores, out)
            loss.backward()
            optimizer.step()
            
        out = out.detach().cpu().numpy()

        if draw_graph:
            self.visualize(graph, out, self.name, vis_ratio=vis_ratio)
        self.last_result = (graph, out)

        return out


class CX_Model(torch.nn.Module):

    def __init__(self, graph, h_dim):
        super(CX_Model, self).__init__()
        self.x = torch.flatten(graph.x, start_dim=1, end_dim=-1)
        self.edge_index, self.edge_attr = graph.edge_index, graph.edge_attr
        # node encoder
        self.lin0 = Lin(self.x.size(-1), h_dim)
        self.relu0 = ReLU()
        self.edge_nn = Seq(Lin(in_features=graph.num_edge_features, out_features=h_dim),
                           ReLU(),
                           Lin(in_features=h_dim, out_features=h_dim * h_dim))
        self.conv = NNConv(in_channels = h_dim,
                           out_channels = h_dim,
                           nn = self.edge_nn)
        self.lin1 = torch.nn.Linear(h_dim, 8)
        self.relu1 = ReLU()
        self.lin2 = torch.nn.Linear(8, 1)

    def forward(self):

        x = torch.flatten(self.x, start_dim=1, end_dim=-1)
        x = self.relu0(self.lin0(x))
        x = self.conv(x, self.edge_index, self.edge_attr)
        edge_emb = x[self.edge_index[0,:]] * x[self.edge_index[1,:]]
        edge_emb = self.relu1(self.lin1(edge_emb))
        edge_score = self.lin2(edge_emb)
        edge_score = edge_score.view(-1)

        return edge_score
