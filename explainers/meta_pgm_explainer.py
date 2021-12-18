"""
Modified based on https://github.com/vunhatminh/PGMExplainer

Citation:
Vu et al. PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks
"""

import time
import torch
import copy
import numpy as np
import pandas as pd
from scipy.special import softmax
from pgmpy.estimators import ConstraintBasedEstimator
from pgmpy.estimators.CITests import chi_square


def n_hops_A(A, n_hops):
    # Compute the n-hops adjacency matrix
    adj = torch.tensor(A, dtype=torch.float)
    hop_adj = power_adj = adj
    for i in range(n_hops - 1):
        power_adj = power_adj @ adj
        prev_hop_adj = hop_adj
        hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()
    return hop_adj.numpy().astype(int)


class MetaPGMExplainer:
    def __init__(
        self,
        model,
        graph,
        num_layers = None,
        perturb_feature_list = None,
        perturb_mode = "mean", # mean, zero, max or uniform
        perturb_indicator = "diff", # diff or abs
        print_result = 1,
        snorm_n = None, 
        snorm_e = None
    ):
        self.model = model
        self.model.eval()
        self.graph = graph
        self.snorm_n = snorm_n
        self.snorm_e = snorm_e
        self.num_layers = num_layers
        self.perturb_feature_list = perturb_feature_list
        self.perturb_mode = perturb_mode
        self.perturb_indicator = perturb_indicator
        self.print_result = print_result
        self.X_feat = graph.x
        self.E_feat = graph.edge_attr
    
    def perturb_features_on_node(self, feature_matrix, node_idx, random = 0):
        
        X_perturb = copy.copy(feature_matrix)
        perturb_array = copy.copy(X_perturb[node_idx])
        epsilon = 0.05 * torch.max(self.X_feat, dim=0).values
        epsilon = epsilon.detach().cpu().numpy()

        seed = np.random.randint(2)
        
        if random == 1:
            if seed == 1:
                for i in range(perturb_array.size(0)):
                    if i in self.perturb_feature_list:
                        if self.perturb_mode == "mean":
                            perturb_array[i] = torch.mean(feature_matrix[:,i])
                        elif self.perturb_mode == "zero":
                            perturb_array[i] = 0
                        elif self.perturb_mode == "max":
                            perturb_array[i] = torch.max(feature_matrix[:,i])
                        elif self.perturb_mode == "uniform":
                            eps = epsilon[i].mean()
                            perturb_array[i] = perturb_array[i] + np.random.uniform(low=-eps, high=eps)
                            # if perturb_array[i] < 0:
                            #     perturb_array[i] = 0

                            # else:
                            #     _max = torch.max(self.X_feat, dim=0).values[i]
                            #     if perturb_array[i] > _max:
                            #         perturb_array[i] = _max

        
        X_perturb[node_idx] = perturb_array

        return X_perturb 
    
    def batch_perturb_features_on_node(self, num_samples, index_to_perturb,
                                            percentage, p_threshold, pred_threshold):
        self.model(self.graph)
        soft_pred = np.asarray(softmax(self.model.readout[0].detach().cpu().numpy()))
        pred_label = self.graph.y.detach().cpu().numpy()
        num_nodes = self.X_feat.size(0)
        Samples = []
        for iteration in range(num_samples):
            X_perturb = copy.copy(self.X_feat)
            sample = []
            for node in range(num_nodes):
                if node in index_to_perturb:
                    seed = np.random.randint(100)
                    if seed < percentage:
                        latent = 1
                        X_perturb = self.perturb_features_on_node(X_perturb, node, random = latent)
                    else:
                        latent = 0
                else:
                    latent = 0
                sample.append(latent)
            tmp_g = self.graph.clone()
            tmp_g.x = X_perturb
            self.model(tmp_g)
            soft_pred_perturb = np.asarray(softmax(self.model.readout[0].detach().cpu().numpy()))

            pred_change = np.max(soft_pred) - soft_pred_perturb[pred_label]
            
            sample.append(pred_change)
            Samples.append(sample)
        
        Samples = np.asarray(Samples)
        if self.perturb_indicator == "abs":
            Samples = np.abs(Samples)
        
        top = int(num_samples/8)
        top_idx = np.argsort(Samples[:,num_nodes])[-top:] 
        for i in range(num_samples):
            if i in top_idx:
                Samples[i,num_nodes] = 1
            else:
                Samples[i,num_nodes] = 0
            
        return Samples
    
    def explain(self, num_samples=1000, percentage=50, top_node=5, p_threshold=0.05, pred_threshold=0.1):

        num_nodes = self.X_feat.size(0)
        
#       Round 1
        Samples = self.batch_perturb_features_on_node(int(num_samples/2), range(num_nodes), percentage,
                                                            p_threshold, pred_threshold)         
        
        data = pd.DataFrame(Samples)
        p_values = []
        
        target = num_nodes # The entry for the graph classification data is at "num_nodes"
        for node in range(num_nodes):
            chi2, p = chi_square(node, target, [], data)
            p_values.append(p)
        
        number_candidates = min(int(top_node*4), num_nodes-1)
        candidate_nodes = np.argpartition(p_values, number_candidates)[0:number_candidates]
        
#         Round 2
        Samples = self.batch_perturb_features_on_node(num_samples, candidate_nodes, percentage, 
                                                            p_threshold, pred_threshold)          
        data = pd.DataFrame(Samples)
        
        p_values = []
        dependent_nodes = []
        
        target = num_nodes
        for node in range(num_nodes):
            chi2, p = chi_square(node, target, [], data)
            p_values.append(p)
            if p < p_threshold:
                dependent_nodes.append(node)
        
        return p_values