import os
import os.path as osp

import torch
import random
import numpy as np
import sklearn.preprocessing as preprocessing
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data


class Mutagenicity(InMemoryDataset):

    url = ('https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/Mutagenicity.zip')

    splits = ['training', 'evaluation', 'testing']

    def __init__(self, root, mode='testing', transform=None, pre_transform=None, pre_filter=None):

        assert mode in self.splits
        self.mode = mode
        super(Mutagenicity, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])


    @property
    def raw_file_names(self):

        return ['Mutagenicity/' + i \
                for i in [
                    'Mutagenicity_A.txt',
                    'Mutagenicity_edge_labels.txt',
                    'Mutagenicity_graph_indicator.txt',
                    'Mutagenicity_graph_labels.txt',
                    'Mutagenicity_node_labels.txt',
                    ]
                ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'evaluation.pt', 'testing.pt']

    def download(self):

        if os.path.exists(osp.join(self.raw_dir, 'Mutagenicity')):
            print('Using existing data in folder Mutagenicity')
            return

        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):

        edge_index = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[0]), delimiter=',').T
        edge_index = torch.from_numpy(edge_index - 1.0).to(torch.long)  # node idx from 0

        edge_label = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[1]))
        encoder = preprocessing.OneHotEncoder().fit(np.unique(edge_label).reshape(-1, 1))
        edge_attr = encoder.transform(edge_label.reshape(-1, 1)).toarray()
        edge_attr = torch.Tensor(edge_attr)

        node_label = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[-1]))
        encoder = preprocessing.OneHotEncoder().fit(np.unique(node_label).reshape(-1, 1))
        x = encoder.transform(node_label.reshape(-1, 1)).toarray()
        x = torch.Tensor(x)

        z = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[2]), dtype=int)

        y = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[3]))
        y = torch.unsqueeze(torch.LongTensor(y), 1).long()
        num_graphs = len(y)
        total_edges = edge_index.size(1)
        begin = 0

        data_list = []
        for i in range(num_graphs):

            perm = np.where(z == i + 1)[0]
            bound = max(perm)
            end = begin
            for end in range(begin, total_edges):
                if int(edge_index[0, end]) > bound:
                    break

            data = Data(x=x[perm], y=y[i], z=node_label[perm],
                        edge_index=edge_index[:, begin:end] - int(min(perm)),
                        edge_attr=edge_attr[begin:end],
                        name="mutag_%d" % i, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            begin = end
            data_list.append(data)

        assert len(data_list) == 4337

        random.shuffle(data_list)
        torch.save(self.collate(data_list[1000:]), self.processed_paths[0])
        torch.save(self.collate(data_list[500:1000]), self.processed_paths[1])
        torch.save(self.collate(data_list[:500]), self.processed_paths[2])