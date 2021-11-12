# ReFine: Multi-Grained Explainability for GNNs
We are trying hard to update the code, but it may take a while to complete due to our tight schedule recently. Thank you for your waiting!

## Installation
**Requirements**
- CPU or NVIDIA GPU, Linux, Python 3.7
- PyTorch, various Python packages

**Main Packages**
1. Pytorch Geometric. [Official Download](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
```
# We use TORCH version 1.6.0
CUDA=cu101
TORCH=1.6.0 
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```
2. Visual Genome. [Google Drive Download](https://drive.google.com/file/d/132ziPf2PKqjGoZkqh9194rT17qr3ywN8/view?usp=sharing).
  This is used for preprocessing the VG-5 dataset and visualizing the generated explanations. Manually download it to the same directory as `data/`. (Yes, this package can be installed using pip or API, but we find it slow to use).
  
## Datasets

1. The processed raw data for `BA-3motif` is available in the` data/` folder.
2. Datasets `MNIST`, `Mutagenicity` will be automatically downloaded when training models.
3. We select and label 4444 graphs from https://visualgenome.org/ to construct the **VG-5** dataset. The graphs are labeled with five classes: stadium, street, farm, surfing, forest. Each graph contains regions of the objects as the nodes, while edges indicate the relationships between object nodes. 

Download the dataset from [Google Drive](https://drive.google.com/file/d/1ONg9hFCynE3KynxakgFhqZxg0fWRXgv6/view?usp=shari). Arrange the dir as 
```
data ---BA3
 |------VG
        |---raw
``` 
Please remember to cite Visual Genome ([bibtex](https://dblp.uni-trier.de/rec/journals/ijcv/KrishnaZGJHKCKL17.html?view=bibtex)) if you use our VG-5 dataset.
## Training GNNs
```
cd gnns/
python ba3motif_gnn.py --epoch 100 --num_unit 2 --batch_size 128
```
The trained GNNs will be saved in `param/gnns`.

## Explaining the Predictions
code coming soon
## Evaluation \& Visualization
code coming soon

## Citation
Please cite our paper if you find the repository useful.
```
@inproceedings{2021refine,
  title={Towards Multi-Grained Explainability for Graph Neural Networks },
  author={Wang, Xiang and Wu, Ying-Xin and Zhang, An and He, Xiangnan and Chua, Tat-Seng},
  booktitle={Proceedings of the 35th Conference on Neural Information Processing Systems},
  year={2021} 
}
```
