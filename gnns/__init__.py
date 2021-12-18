
from .mnist_gnn import MNISTNet
from .ba3motif_gnn import BA3MotifNet
from .mutag_gnn import MutagNet
import os
if os.path.exists("../visual_genome"):
    from .vg_gnn import VGNet