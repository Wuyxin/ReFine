from explainers.base import Explainer
from explainers.meta_pgm_explainer import MetaPGMExplainer
import torch
import numpy as np


class PGMExplainer(Explainer):

    def __init__(self, device, gnn_model_path):
        super(PGMExplainer, self).__init__(device, gnn_model_path)
        

    def explain_graph(self, graph,
                      model=None,
                      epochs=200,
                      lr=1e-2,
                      draw_graph=0,
                      vis_ratio=0.2
                      ):

        if model == None:
            model = self.model
        dim = graph.x.size(1)

        model(graph)
        soft_pred = model.readout

        pred_threshold = 0.1 * torch.max(soft_pred)
        perturb_features_list = [i for i in range(dim)]
        explainer = MetaPGMExplainer(model, graph,
                                     snorm_n=None, snorm_e=None,
                                     perturb_feature_list=perturb_features_list,
                                     perturb_indicator="abs",
                                     perturb_mode="uniform")
        p_values= explainer.explain(num_samples=10, percentage=20,
                                    top_node=max(1, int(0.1*graph.num_nodes)), 
                                    p_threshold=0.05, pred_threshold=pred_threshold)
        p_values = np.array(p_values)
        row, col = graph.edge_index.detach().cpu()
        edge_imp = -p_values[row] * p_values[col]
        edge_imp -= np.min(edge_imp)

        if isinstance(edge_imp, float):
            edge_imp = edge_imp.reshape([1])
        edge_imp = self.norm_imp(edge_imp)

        if draw_graph:
            self.visualize(graph, edge_imp, self.name, vis_ratio=vis_ratio)
        self.last_result = (graph, edge_imp)

        return edge_imp


        # paras via grid search:
        # mutag  abs  uniform  num_samples=10, percentage=10, top_node=max(1, int(0.2*graph.num_nodes))
        # reddit abs  mean     num_samples=10, percentage=10, top_node=max(1, int(0.2*graph.num_nodes))
        # vg     abs  uniform  num_samples=10, percentage=10, top_node=max(1, int(0.2*graph.num_nodes))
        # ba3    abs  uniform  num_samples=10, percentage=20, top_node=max(1, int(0.1*graph.num_nodes))