import random
import numpy as np
from explainers.base import Explainer


class RandomCaster(Explainer):
    
    def __init__(self, device, gnn_model_path):
        super(RandomCaster, self).__init__(device, gnn_model_path)
        
        
    def explain_graph(self, graph,
                      model = None,
                      ratio=0.1,
                      draw_graph=0,
                      vis_ratio=0.2):

        self.ratio = ratio
        topk = max(int(ratio * graph.num_edges), 1)

        random_edges = random.sample(range(graph.num_edges), topk)

        scores = np.zeros(graph.num_edges)
        scores[random_edges] = topk - np.array(range(topk))
        scores = self.norm_imp(scores)
        
        if draw_graph:
            self.visualize(graph, scores, self.name,vis_ratio=vis_ratio)
        self.last_result = (graph, scores)

        return scores
