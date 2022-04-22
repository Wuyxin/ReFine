import os
from tqdm import tqdm
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import sys
sys.path.append('..')
import visual_genome.local as vgl



class Visual_Genome(InMemoryDataset):

    splits = ['training', 'evaluation', 'testing']

    def __init__(self, root, mode='testing', transform=None, pre_transform=None, pre_filter=None):

        assert mode in self.splits
        self.mode = mode
        self.root = root
        super(Visual_Genome, self).__init__(root, transform, pre_transform, pre_filter)
        idx = self.processed_file_names.index('{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])


    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['training.pt', 'evaluation.pt', 'testing.pt']

    def download(self):
        return

    def process(self):
        images = {img.id: img for img in vgl.get_all_image_data('../visual_genome/raw')}
        data_list = []
        idx = 0
        print("It might take a while.")
        for img_name in tqdm(self.raw_file_names):
            idx += 1
            img_id = int(img_name[:-6])
            y = int(img_name[-5])
            scene_graph = vgl.get_scene_graph(img_id, images=images, image_data_dir='../visual_genome/raw/by-id/',
                                              synset_file='../visual_genome/raw/synsets.json')
            if len(scene_graph.relationships) < 1:
                os.remove(osp.join(self.root, f'raw/{img_name}'))
                print("filter out graphs with no relationship  ", img_id)
                continue
            
            img = Image.open(os.path.join(self.root, f'raw/{img_name}'))
            x = []
            for obj in scene_graph.objects:
                cropped = img.crop((obj.x, obj.y, obj.x + obj.width, obj.y + obj.height))  # (left, upper, right, lower)
                data = torch.Tensor(np.array(cropped)).transpose(0, -1)
                output = F.adaptive_avg_pool2d(data, (20, 20))
                x.append(output.numpy())
            edge_index = []
            objects = scene_graph.objects
            for rel in scene_graph.relationships:
                v = objects.index(rel.object)
                u = objects.index(rel.subject)
                if u == v: # delete self-loop edges
                    continue
                if (u, v) not in edge_index: # delete duplicate edges
                    edge_index.append((u, v))

            x = torch.FloatTensor(x)
            n_edge = len(edge_index)
            edge_index = torch.LongTensor(edge_index).T
            graph = Data(x=x, edge_index=edge_index, edge_attr=torch.ones(n_edge, 1), y=y, name=img_id)
            data_list.append(graph)
            

        random.shuffle(data_list)
        torch.save(self.collate(data_list[799:]), self.processed_paths[0])
        torch.save(self.collate(data_list[399:799]), self.processed_paths[1])
        torch.save(self.collate(data_list[:399]), self.processed_paths[-1])

