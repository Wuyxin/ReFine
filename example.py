import argparse

import torch
import numpy as np
from tqdm import tqdm
from utils.dataset import get_datasets
from explainers import *
from gnns import *
from torch_geometric.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain ReFine")
    parser.add_argument('--cuda', type=int, default=0,
                        help='GPU device.')
    parser.add_argument('--result_dir', type=str, default="results/",
                        help='Result directory.')
    parser.add_argument('--dataset', type=str, default='ba3',
                        choices=['mutag', 'ba3', 'graphsst2', 'mnist', 'vg', 'reddit5k'])
    parser.add_argument('--explainer', type=str, default="SAExplainer",
                        choices=['SAExplainer', 'GradCam', 'DeepLIFTExplainer', 
                        'IGExplainer', 'GNNExplainer', 'PGExplainer', 'PGMExplainer', 
                        'Screener', 'CXPlain'])
    parser.add_argument('--num_test', type=int, default=10)
    return parser.parse_args()


args = parse_args()
results = {}
if args.dataset == 'ba3':
    ground_truth = True
else:
    ground_truth = False
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset)
graph_mask = torch.load(f'param/filtered/{args.dataset}_idx_test.pt')
test_loader = DataLoader(test_dataset[graph_mask][:args.num_test], batch_size=1, shuffle=False, drop_last=False)
ratios = [0.1 *i for i in range(1,11)]

gnn_path = f'param/gnns/{args.dataset}_net.pt'
exec(f"e = {args.explainer}(device, gnn_path)")
acc_logger, recall = [], []
for g in tqdm(iter(test_loader), total=len(test_loader)):
    g.to(device)
    e.explain_graph(g)
    acc_logger.append(e.evaluate_acc([0.1*i for i in range(1,11)])[0])
    if ground_truth:
        recall.append(e.evaluate_recall(topk=5))
        
print(np.array(acc_logger).mean(axis=0))
print('AUC: %.3f' % np.array(acc_logger).mean(axis=0).mean())
if ground_truth:
    print('Recall %.3f' % np.array(recall).mean(axis=0))