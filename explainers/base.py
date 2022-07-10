import os
import math
import numpy as np
import torch
from .visual import *

EPS = 1e-6


class Explainer(object):

    def __init__(self, device, gnn_model_path):
        self.device = device
        self.model = torch.load(gnn_model_path).to(self.device)
        self.model.eval()
        self.model_name = self.model.__class__.__name__
        self.name = self.__class__.__name__

        self.path = gnn_model_path
        self.last_result = None
        self.vis_dict = None

    def explain_graph(self, graph, **kwargs):
        """
        Main part for different graph attribution methods
        :param graph: target graph instance to be explained
        :param kwargs:
        :return: edge_imp, i.e., attributions for edges, which are derived from the attribution methods.
        """
        raise NotImplementedError

    @staticmethod
    def get_rank(lst, r=1):

        topk_idx = list(np.argsort(-lst))
        top_pred = np.zeros_like(lst)
        n = len(lst)
        k = int(r * n)
        for i in range(k):
            top_pred[topk_idx[i]] = n - i
        return top_pred

    @staticmethod
    def norm_imp(imp):
        imp[imp < 0] = 0
        imp += 1e-16
        return imp / imp.sum()

    def _relabel(self, g, edge_index):

        sub_nodes = torch.unique(edge_index)
        x = g.x[sub_nodes]
        batch = g.batch[sub_nodes]
        row, col = edge_index
        pos = None
        try:
            pos = g.pos[sub_nodes]
        except:
            pass

        # remapping the nodes in the explanatory subgraph to new ids.
        node_idx = row.new_full((g.num_nodes,), -1)
        node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
        edge_index = node_idx[edge_index]
        return x, edge_index, batch, pos

    def _reparameterize(self, log_alpha, beta=0.1, training=True):

        if training:
            random_noise = torch.rand(log_alpha.size()).to(self.device)
            gate_inputs = torch.log2(random_noise) - torch.log2(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta + EPS
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def pack_explanatory_subgraph(self, top_ratio=0.2, 
                                  graph=None, imp=None, relabel=True):

        if graph is None:
            graph, imp = self.last_result
        assert len(imp) == graph.num_edges, 'length mismatch'
        
        top_idx = torch.LongTensor([])
        graph_map = graph.batch[graph.edge_index[0, :]]
        exp_subgraph = graph.clone()
        exp_subgraph.y = graph.y
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
            Gi_n_edge = len(edge_indicator)
            topk = min(max(math.ceil(top_ratio * Gi_n_edge), 1), Gi_n_edge)
            Gi_pos_edge_idx = np.argsort(-imp[edge_indicator])[:topk]
            top_idx = torch.cat([top_idx, edge_indicator[Gi_pos_edge_idx]])
        # retrieval properties of the explanatory subgraph
        # .... the edge_attr.
        exp_subgraph.edge_attr = graph.edge_attr[top_idx]
        # .... the edge_index.
        exp_subgraph.edge_index = graph.edge_index[:, top_idx]
        # .... the nodes.
        # exp_subgraph.x = graph.x
        if relabel:
            exp_subgraph.x, exp_subgraph.edge_index, exp_subgraph.batch, exp_subgraph.pos = self._relabel(exp_subgraph, exp_subgraph.edge_index)
        
        return exp_subgraph

    def evaluate_recall(self, topk=10):

        graph, imp = self.last_result
        E = graph.num_edges
        if isinstance(graph.ground_truth_mask, list):
            graph.ground_truth_mask = graph.ground_truth_mask[0]
        index = np.argsort(-imp)[:topk]
        values = graph.ground_truth_mask[index]
        return float(values.sum()) / float(graph.ground_truth_mask.sum())
        
    
    def evaluate_acc(self, top_ratio_list, graph=None, imp=None):
        
        if graph is None:
            assert self.last_result is not None
            graph, imp = self.last_result
        acc = np.array([[]])
        prob = np.array([[]])
        y = graph.y
        for idx, top_ratio in enumerate(top_ratio_list):
            
            if top_ratio == 1.0:
                self.model(graph)
            else:
                exp_subgraph = self.pack_explanatory_subgraph(top_ratio, 
                                                            graph=graph, imp=imp)
                self.model(exp_subgraph)
            res_acc = (y == self.model.readout.argmax(dim=1)).detach().cpu().float().view(-1, 1).numpy()
            res_prob = self.model.readout[0, y].detach().cpu().float().view(-1, 1).numpy()
            acc = np.concatenate([acc, res_acc], axis=1)
            prob = np.concatenate([prob, res_prob], axis=1)
        return acc, prob

    def visualize(self, graph=None, edge_imp=None, 
                  counter_edge_index=None ,vis_ratio=0.2, 
                  save=False, layout=False, name=None):
        
        if graph is None:
            assert self.last_result is not None
            graph, edge_imp = self.last_result
        
        topk = max(int(vis_ratio * graph.num_edges), 1)
        idx = np.argsort(-edge_imp)[:topk]
        G = nx.DiGraph()
        G.add_nodes_from(range(graph.num_nodes))
        G.add_edges_from(list(graph.edge_index.cpu().numpy().T))
        
        if not counter_edge_index==None:
            G.add_edges_from(list(counter_edge_index.cpu().numpy().T))
        if self.vis_dict is None:
            self.vis_dict = vis_dict[self.model_name] if self.model_name in vis_dict.keys() else vis_dict['defult']
        
        folder = Path(r'image/%s' % (self.model_name))
        if save and not os.path.exists(folder):
            os.makedirs(folder)

        edge_pos_mask = np.zeros(graph.num_edges, dtype=np.bool_)
        edge_pos_mask[idx] = True
        vmax = sum(edge_pos_mask)
        node_pos_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
        node_neg_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
        node_pos_idx = np.unique(graph.edge_index[:, edge_pos_mask].cpu().numpy()).tolist()
        node_neg_idx = list(set([i for i in range(graph.num_nodes)]) - set(node_pos_idx))
        node_pos_mask[node_pos_idx] = True
        node_neg_mask[node_neg_idx] = True
        
        if self.model_name == "GraphSST2Net":
            plt.figure(figsize=(10, 4), dpi=100)
            ax = plt.gca()
            node_imp = np.zeros(graph.num_nodes)
            row, col = graph.edge_index[:, edge_pos_mask].cpu().numpy()
            node_imp[row] += edge_imp[edge_pos_mask]
            node_imp[col] += edge_imp[edge_pos_mask]
            node_alpha = node_imp / max(node_imp)
            pos, width, height = sentence_layout(graph.sentence_tokens[0], length=2)
            
            nx.draw_networkx_edges(G, pos=pos,
                                   edgelist=list(graph.edge_index.cpu().numpy().T),
                                   edge_color='whitesmoke',
                                   width=self.vis_dict['width'], arrows=True,
                                   connectionstyle="arc3,rad=0.2"  # <-- THIS IS IT
                                   )
            nx.draw_networkx_edges(G, pos=pos,
                                   edgelist=list(graph.edge_index[:, edge_pos_mask].cpu().numpy().T),
                                   edge_color=self.get_rank(edge_imp[edge_pos_mask]),
                                   width=self.vis_dict['width'],
                                   edge_cmap=cm.get_cmap('Greys'),
                                   edge_vmin=-vmax, edge_vmax=vmax,
                                   arrows=True, connectionstyle="arc3,rad=0.2" 
                                   )
            
            for i in node_pos_idx:
                patch = Rectangle(
                    xy=(pos[i][0]-width[i]/2, pos[i][1]-height[i]/2) ,width=width[i], height=height[i],
                    linewidth=1, color='orchid', alpha=node_alpha[i], fill=True, label=graph.sentence_tokens[0][i])
                ax.add_patch(patch)
                
            nx.draw_networkx_labels(G, pos=pos,
                                    labels={i: graph.sentence_tokens[0][i] for i in range(graph.num_nodes)},
                                    font_size=self.vis_dict['font_size'],
                                    font_weight='bold', font_color='k'
                                    )
            if not counter_edge_index==None:
                nx.draw_networkx_edges(G, pos=pos,
                                   edgelist=list(counter_edge_index.cpu().numpy().T),
                                   edge_color='mediumturquoise',
                                   width=self.vis_dict['width']/2.0,
                                   arrows=True, connectionstyle="arc3,rad=0.2" 
                                   )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
        if 'Motif' in self.model_name:
            plt.figure(figsize=(8, 6), dpi=100)
            ax = plt.gca()
            pos = graph.pos[0]
            nx.draw_networkx_nodes(G, pos={i: pos[i] for i in node_pos_idx},
                                   nodelist=node_pos_idx,
                                   node_size=self.vis_dict['node_size'],
                                   node_color=graph.z[0][node_pos_idx],
                                   alpha=1, cmap='winter',
                                   linewidths=self.vis_dict['linewidths'],
                                   edgecolors='red',
                                   vmin=-max(graph.z[0]), vmax=max(graph.z[0])
                                   )
            nx.draw_networkx_nodes(G, pos={i: pos[i] for i in node_neg_idx},
                                   nodelist=node_neg_idx,
                                   node_size=self.vis_dict['node_size'],
                                   node_color=graph.z[0][node_neg_idx],
                                   alpha=0.2, cmap='winter',
                                   linewidths=self.vis_dict['linewidths'],
                                   edgecolors='whitesmoke',
                                   vmin=-max(graph.z[0]), vmax=max(graph.z[0])
                                   )
            nx.draw_networkx_edges(G, pos=pos,
                                       edgelist=list(graph.edge_index.cpu().numpy().T),
                                       edge_color='whitesmoke',
                                       width=self.vis_dict['width'],
                                       arrows=False
                                       )
            nx.draw_networkx_edges(G, pos=pos,
                                   edgelist=list(graph.edge_index[:, edge_pos_mask].cpu().numpy().T),
                                   edge_color=self.get_rank(edge_imp[edge_pos_mask]),
                                   # np.ones(len(edge_imp[edge_pos_mask])),
                                   width=self.vis_dict['width'],
                                   edge_cmap=cm.get_cmap('bwr'),
                                   edge_vmin=-vmax, edge_vmax=vmax,
                                   arrows=False
                                   )
            if not counter_edge_index==None:
                nx.draw_networkx_edges(G, pos=pos,
                                   edgelist=list(counter_edge_index.cpu().numpy().T),
                                   edge_color='mediumturquoise',
                                   width=self.vis_dict['width']/3.0,
                                   arrows=False
                                   )
            
        if 'Mutag' in self.model_name:
            from rdkit.Chem.Draw import rdMolDraw2D
            idx = [int(i/2) for i in idx]
            x = graph.x.detach().cpu().tolist()
            edge_index = graph.edge_index.T.detach().cpu().tolist()
            edge_attr = graph.edge_attr.detach().cpu().tolist()
            mol = graph_to_mol(x, edge_index, edge_attr)
            d = rdMolDraw2D.MolDraw2DCairo(500, 500)
            hit_at = np.unique(graph.edge_index[:,idx].detach().cpu().numpy()).tolist()
            def add_atom_index(mol):
                atoms = mol.GetNumAtoms()
                for i in range( atoms ):
                    mol.GetAtomWithIdx(i).SetProp(
                        'molAtomMapNumber', str(mol.GetAtomWithIdx(i).GetIdx()))
                return mol

            hit_bonds=[]
            for (u, v) in graph.edge_index.T[idx]:
                hit_bonds.append(mol.GetBondBetweenAtoms(int(u), int(v)).GetIdx())
            rdMolDraw2D.PrepareAndDrawMolecule(
                d, mol, highlightAtoms=hit_at, highlightBonds=hit_bonds,
                highlightAtomColors={i:(0, 1, 0) for i in hit_at},
                highlightBondColors={i:(0, 1, 0) for i in hit_bonds})
            d.FinishDrawing()
            bindata = d.GetDrawingText()
            iobuf = io.BytesIO(bindata)
            image = Image.open(iobuf)
            image.show()
            if save:
                if name:
                    d.WriteDrawingText('image/%s/%s-%d-%s.png' % (self.model_name, name, int(graph.y[0]), self.name)) 
                else:
                    d.WriteDrawingText('image/%s/%s-%d-%s.png' % (self.model_name, str(graph.name[0]), int(graph.y[0]), self.name)) 
            return 
            
            
        if 'MNIST' in self.model_name:
            plt.figure(figsize=(6, 6), dpi=100)
            ax = plt.gca()
            pos = graph.pos.detach().cpu().numpy()
            row, col = graph.edge_index
            z = np.zeros(graph.num_nodes)
            for i in idx:
                z[row[i]] += edge_imp[i]
                z[col[i]] += edge_imp[i]
            z = z / max(z)

            row, col = graph.edge_index
            pos = graph.pos.detach().cpu().numpy()
            z = graph.x.detach().cpu().numpy()
            edge_mask = torch.tensor(graph.x[row].view(-1) * graph.x[col].view(-1), dtype=torch.bool).view(-1)

            nx.draw_networkx_edges(
                    G, pos=pos,
                    edgelist=list(graph.edge_index.cpu().numpy().T),
                    edge_color='whitesmoke',
                    width=self.vis_dict['width'],
                    arrows=False
                )
            nx.draw_networkx_edges(
                    G, pos=pos,
                    edgelist=list(graph.edge_index[:,edge_mask].cpu().numpy().T),
                    edge_color='black',
                    width=self.vis_dict['width'],
                    arrows=False
                )
            nx.draw_networkx_nodes(G, pos=pos,
                                   node_size=self.vis_dict['node_size'],
                                   node_color='black', alpha=graph.x, 
                                   linewidths=self.vis_dict['linewidths'],
                                   edgecolors='black'
                                   )
            nx.draw_networkx_edges(G, pos=pos,
                                   edgelist=list(graph.edge_index[:, edge_pos_mask].cpu().numpy().T),
                                   edge_color=self.get_rank(edge_imp[edge_pos_mask]),
                                   width=self.vis_dict['width'],
                                   edge_cmap=cm.get_cmap('YlOrRd'),
                                   edge_vmin=-vmax, edge_vmax=vmax,
                                   arrows=False
                                   )
            nx.draw_networkx_nodes(G, pos={i: pos[i] for i in node_pos_idx},
                                   nodelist=node_pos_idx,
                                   node_size=self.vis_dict['node_size'],
                                   node_color='brown', alpha=z[node_pos_idx], 
                                   linewidths=self.vis_dict['linewidths'],
                                   edgecolors='black'
                                   )
            if not counter_edge_index==None:
                nx.draw_networkx_edges(G, pos=pos,
                                   edgelist=list(counter_edge_index.cpu().numpy().T),
                                   edge_color='mediumturquoise',
                                   width=self.vis_dict['width']/3.0,
                                   arrows=False
                                   )
        if self.model_name == "VGNet":
            from visual_genome import local as vgl
            idx = np.argsort(-edge_imp)[:topk]
            top_edges = graph.edge_index[:, idx]

            scene_graph = vgl.get_scene_graph(image_id=int(graph.name),
                                              images='visual_genome/raw',
                                              image_data_dir='visual_genome/raw/by-id/',
                                              synset_file='visual_genome/raw/synsets.json')
            # scene_graph = api.get_scene_graph_of_image(id=int(graph.id))
            r = 0.95  # transparency
            img = Image.open("data/VG/raw/%d-%d.jpg" % (graph.name, graph.y))
            data = list(img.getdata())
            ndata = list(
                [(int((255 - p[0]) * r + p[0]), int((255 - p[1]) * r + p[1]), int((255 - p[2]) * r + p[2])) for p in
                 data])
            mode = img.mode
            width, height = img.size
            edges = list(top_edges.T)
            for i, (u, v) in enumerate(edges[::-1]):
                r = 1.0 - 1.0 / len(edges) * (i + 1)
                obj1 = scene_graph.objects[u]
                obj2 = scene_graph.objects[v]
                for obj in [obj1, obj2]:
                    for x in range(obj.x, obj.width + obj.x):
                        for y in range(obj.y, obj.y + obj.height):
                            ndata[y * width + x] = (int((255 - data[y * width + x][0]) * r + data[y * width + x][0]),
                                                    int((255 - data[y * width + x][1]) * r + data[y * width + x][1]),
                                                    int((255 - data[y * width + x][2]) * r + data[y * width + x][2]))

            img = Image.new(mode, (width, height))
            img.putdata(ndata)

            plt.imshow(img)
            ax = plt.gca()
            for i, (u, v) in enumerate(edges):
                obj1 = scene_graph.objects[u]
                obj2 = scene_graph.objects[v]
                ax.annotate("", xy=(obj2.x, obj2.y), xytext=(obj1.x, obj1.y),
                            arrowprops=dict(width=topk - i, color='wheat', headwidth=5))
                for obj in [obj1, obj2]:
                    ax.text(obj.x, obj.y - 8, str(obj), style='italic',
                            fontsize=13,
                            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 3,
                                  'edgecolor': rec_color[i % len(rec_color)]}
                            )
                    ax.add_patch(Rectangle((obj.x, obj.y),
                                           obj.width,
                                           obj.height,
                                           fill=False,
                                           edgecolor=rec_color[i % len(rec_color)],
                                           linewidth=1.5))
            plt.tick_params(labelbottom='off', labelleft='off')
            plt.axis('off')
        if save:
            if name:
                plt.savefig(folder / Path(r'%s-%d-%s.png' % (name, int(graph.y[0]), self.name)), dpi=500,
                                bbox_inches='tight')
            else:
                if isinstance(graph.name[0], str):
                    plt.savefig(folder / Path(r'%s-%d-%s.png' % (str(graph.name[0]), int(graph.y[0]), self.name)), dpi=500,
                                bbox_inches='tight')
                else:
                    plt.savefig(folder / Path(r'%d-%d-%s.png' % (int(graph.name[0]), int(graph.y[0]), self.name)), dpi=500,
                                bbox_inches='tight')
        
        plt.show()
