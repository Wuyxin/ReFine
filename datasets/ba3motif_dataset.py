import random
import numpy as np
import os.path as osp
import pickle as pkl

import torch
from torch_geometric.data import InMemoryDataset, Data


class BA3Motif(InMemoryDataset):
    splits = ['training', 'evaluation', 'testing']

    def __init__(self, root, mode='testing', transform=None, pre_transform=None, pre_filter=None):

        assert mode in self.splits
        self.mode = mode
        super(BA3Motif, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return ['BA-3motif.npy']

    @property
    def processed_file_names(self):
        return ['training.pt', 'evaluation.pt', 'testing.pt']

    def download(self):
        if not osp.exists(osp.join(self.raw_dir, 'raw', 'BA-3motif.npy')):
            print("raw data of `BA-3motif.npy` doesn't exist, please redownload from our github.")
            raise FileNotFoundError

    def process(self):

        edge_index_list, label_list, ground_truth_list, role_id_list, pos = np.load(osp.join(self.raw_dir, self.raw_file_names[0]), allow_pickle=True)

        data_list = []
        alpha = 0.25
        for idx, (edge_index, y, ground_truth, z, p) in enumerate(zip(edge_index_list, label_list, ground_truth_list, role_id_list, pos)):
            edge_index = torch.from_numpy(edge_index)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            node_idx = torch.unique(edge_index)
            assert node_idx.max() == node_idx.size(0) - 1
            x = torch.zeros(node_idx.size(0), 4)
            index = [i for i in range(node_idx.size(0))]
            x[index, z] = 1
            x = alpha * x + (1 - alpha) * torch.rand((node_idx.size(0), 4))
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.tensor(y, dtype=torch.long).unsqueeze(dim=0)
            data = Data(x=x, y=y, z=z,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        pos = p,
                        ground_truth_mask=ground_truth,
                        name=f'TR-3motif{idx}', idx=idx)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        random.shuffle(data_list)
        torch.save(self.collate(data_list[800:]), self.processed_paths[0])
        torch.save(self.collate(data_list[400:800]), self.processed_paths[1])
        torch.save(self.collate(data_list[:400]), self.processed_paths[2])