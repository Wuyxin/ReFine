from explainers.base import Explainer
from explainers.meta_gnnexplainer import MetaGNNGExplainer


class GNNExplainer(Explainer):

    def __init__(self, device, gnn_model_path):
        super(GNNExplainer, self).__init__(device, gnn_model_path)
        
    def explain_graph(self, graph,
                      model=None,
                      epochs=100,
                      lr=1e-2,
                      draw_graph=0,
                      vis_ratio=0.2
                      ):

        if model == None:
            model = self.model

        explainer = MetaGNNGExplainer(model, epochs=epochs, lr=lr)
        edge_imp = explainer.explain_graph(graph)
        edge_imp = self.norm_imp(edge_imp.cpu().numpy())

        if draw_graph:
            self.visualize(graph, edge_imp, self.name, vis_ratio=vis_ratio)
        self.last_result = (graph, edge_imp)

        return edge_imp
