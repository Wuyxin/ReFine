from .mutag_dataset import Mutagenicity
from .ba3motif_dataset import BA3Motif
import os
if os.path.exists("../visual_genome"):
    from .vg_dataset import Visual_Genome