from torch.autograd import Variable
from explainers.base import Explainer


class SAExplainer(Explainer):


    def __init__(self, device, gnn_model_path):
        super(SAExplainer, self).__init__(device, gnn_model_path)
        
    def explain_graph(self, graph,
                      model=None,
                      draw_graph=0,
                      vis_ratio=0.2):

        if model == None:
            model = self.model
            
        tmp_graph = graph.clone()
        
        tmp_graph.edge_attr = Variable(tmp_graph.edge_attr, requires_grad=True)
        tmp_graph.x = Variable(tmp_graph.x, requires_grad=True)
        pred = model(tmp_graph)
        pred[0, tmp_graph.y].backward()
        
        edge_grads = pow(tmp_graph.edge_attr.grad, 2).sum(dim=1).cpu().numpy()
        edge_imp = self.norm_imp(edge_grads)
        
        if draw_graph:
            self.visualize(graph, edge_imp, self.name, vis_ratio=vis_ratio)
        self.last_result = (graph, edge_imp)
            
        return edge_imp