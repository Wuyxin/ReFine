from enum import Enum

import io
import pickle
import PIL.Image 
import numpy as np
import networkx as nx
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

n_class_dict = {
    'MutagNet': 2, 'Tox21Net': 2, 
    'Reddit5kNet': 5, 'VGNet': 5, 
    'BA2MotifNet': 2, 'BA3MotifNet': 3, 
    'TR3MotifNet': 3, 'MNISTNet':10}
vis_dict = {
    'MutagNet': {'node_size': 400, 'linewidths': 1, 'font_size': 10, 'width': 3},
    'Tox21Net': {'node_size': 400, 'linewidths': 1, 'font_size': 10, 'width': 3},
    'BA3MotifNet': {'node_size': 300, 'linewidths': 1, 'font_size': 10, 'width': 3},
    'TR3MotifNet': {'node_size': 300, 'linewidths': 1, 'font_size': 10, 'width': 5},
    'GraphSST2Net': {'node_size': 400, 'linewidths': 1, 'font_size': 12, 'width': 3},
    'MNISTNet': {'node_size': 100, 'linewidths': 1, 'font_size': 10, 'width': 2},
    'defult': {'node_size': 200, 'linewidths': 1, 'font_size': 10, 'width': 2}
}
chem_graph_label_dict = {'MutagNet': {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S',
                                      8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'},
                         'Tox21Net': {0: 'O', 1: 'C', 2: 'N', 3: 'F', 4: 'Cl', 5: 'S', 6: 'Br', 7: 'Si',
                                      8: 'Na', 9: 'I', 10: 'Hg', 11: 'B', 12: 'K', 13: 'P', 14: 'Au',
                                      15: 'Cr', 16: 'Sn', 17: 'Ca', 18: 'Cd', 19: 'Zn', 20: 'V', 21: 'As',
                                      22: 'Li', 23: 'Cu', 24: 'Co', 25: 'Ag', 26: 'Se', 27: 'Pt', 28: 'Al',
                                      29: 'Bi', 30: 'Sb', 31: 'Ba', 32: 'Fe', 33: 'H', 34: 'Ti', 35: 'Tl',
                                      36: 'Sr', 37: 'In', 38: 'Dy', 39: 'Ni', 40: 'Be', 41: 'Mg', 42: 'Nd',
                                      43: 'Pd', 44: 'Mn', 45: 'Zr', 46: 'Pb', 47: 'Yb', 48: 'Mo', 49: 'Ge',
                                      50: 'Ru', 51: 'Eu', 52: 'Sc'}
                         }
rec_color = ['cyan', 'mediumblue', 'deeppink', 'darkorange', 'gold', 'chartreuse', 'lightcoral', 'darkviolet', 'teal',
             'lightgrey', ]


def sentence_layout(sentence, length, margin=0.2):
    num_token = len(sentence)
    pos = {}; height = []; width = []
    
    right_margin = len(sentence[-1]) * 0.05
    gap = (length-right_margin) / (num_token-1)
    start = 0
    for i in range(num_token):
        pos[i] = np.array([start + gap*i, gap/5 * pow(-1, i)])
        width.append(len(sentence[i]) * 0.04)
        height.append(gap/3)
    return pos, np.array(width), np.array(height)


def e_map_mutag(bond_type, reverse=False):

    from rdkit import Chem
    if not reverse:
        if bond_type == Chem.BondType.SINGLE:
            return 0
        elif bond_type == Chem.BondType.DOUBLE:
            return 1
        elif bond_type == Chem.BondType.AROMATIC:
            return 2
        elif bond_type == Chem.BondType.TRIPLE:
            return 3
        else:
            raise Exception("No bond type found")

    if bond_type == 0:
        return Chem.BondType.SINGLE
    elif bond_type == 1:
        return Chem.BondType.DOUBLE
    elif bond_type == 2:
        return Chem.BondType.AROMATIC
    elif bond_type == 3:
        return Chem.BondType.TRIPLE
    else:
        raise Exception("No bond type found")

class x_map_mutag(Enum):
    C=0
    O=1
    Cl=2
    H=3
    N=4
    F=5
    Br=6
    S=7
    P=8
    I=9
    Na=10
    K=11
    Li=12
    Ca=13

def graph_to_mol(X, edge_index,edge_attr):
    mol = Chem.RWMol()
    X = [
        Chem.Atom(x_map_mutag(x.index(1)).name)
        for x in X
    ]

    E = edge_index
    for x in X:
        mol.AddAtom(x)
    for (u, v), attr in zip(E, edge_attr):
        attr = e_map_mutag(attr.index(1), reverse=True)

        if mol.GetBondBetweenAtoms(u, v):
            continue
        mol.AddBond(u, v, attr)
    return mol