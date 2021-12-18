import torch
import numpy as np

import os
from gnns import *
from datasets import *
if os.path.exists("visual_genome"):
    from datasets.vg_dataset import Visual_Genome
from torch_geometric.datasets import MNISTSuperpixels

class MNISTTransform(object):
    
    def __init__(self, norm=True, max_value=None, cat=True):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def __call__(self, data):
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        cart = pos[col] - pos[row]
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart

        if self.norm and cart.numel() > 0:
            max_value = cart.abs().max() if self.max is None else self.max
            cart = cart / (2 * max_value) + 0.5

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart
            
        row, col = data.edge_index
        data.ground_truth_mask = (data.x[row] > 0).view(-1).bool() * (data.x[col] > 0).view(-1).bool()
        return data

    def __repr__(self):
        return '{}(norm={}, max_value={})'.format(self.__class__.__name__,
                                                  self.norm, self.max)


def get_datasets(name, root='data/'):
    if name == "mutag":
        folder = os.path.join(root, 'MUTAG')
        train_dataset = Mutagenicity(folder, mode='training')
        test_dataset = Mutagenicity(folder, mode='testing')
        val_dataset = Mutagenicity(folder, mode='evaluation')
    elif name == "ba3":
        folder = os.path.join(root, 'BA3')
        train_dataset = BA3Motif(folder, mode='training')
        test_dataset = BA3Motif(folder, mode='testing')
        val_dataset = BA3Motif(folder, mode='evaluation')
    elif name == "mnist":
        folder = os.path.join(root, 'MNIST')
        transform = MNISTTransform(cat=False, max_value=9)
        train_dataset = MNISTSuperpixels(folder, True, transform=transform)
        test_dataset = MNISTSuperpixels(folder, False, transform=transform)
        # Reduced dataset
        train_dataset = train_dataset[:6000]
        val_dataset = test_dataset[1000:2000]
        test_dataset = test_dataset[:1000]
    elif name == "vg":
        folder = os.path.join(root, 'VG')
        test_dataset = Visual_Genome(folder, mode='testing')
        val_dataset = Visual_Genome(folder, mode='evaluation')
        train_dataset = Visual_Genome(folder, mode='training')
    else:
        raise ValueError

    return train_dataset, val_dataset, test_dataset
