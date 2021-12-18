import time
import random
import argparse
import numpy as np

import os
import os.path as osp
from tqdm import tqdm
from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader
from config import data_root, log_root, param_root

import sys
sys.path.append('..')
from gnns import *
from datasets.graphss2_dataset import get_dataloader  
from utils import set_seed
from explainers import ReFine
from utils.dataset import get_datasets
from utils.logger import Logger

np.set_printoptions(precision=2, suppress=True)
folder_dict = {'mutag': 'MUTAG', 'ba3': 'BA3', 
                'mnist': 'MNIST', 'vg': 'VG'}
n_classes_dict = {'mutag': 2, 'mnist': 10, 'ba3': 3, 'vg': 5}

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain ReFine")
    parser.add_argument('--cuda', type=int, default=0,
                        help='GPU device.')
    parser.add_argument('--epoch', type=int, default=20,
                        help='Number of loops to train the mask.')
    parser.add_argument('--dataset', type=str, default='mutag',
                        choices=['mutag', 'ba3', 'graphsst2', 'mnist', 'vg', 'reddit5k'])
    parser.add_argument('--ratio', type=float, default=0.4)
    parser.add_argument('--gamma', type=float, default=5)
    parser.add_argument('--hid', type=int, default=50, help='Hidden dim')
    parser.add_argument('--tau', type=float, default=0.1, help='Temperature.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--random_seed', type=int, default=2021)

    return parser.parse_args()

# get dataset
args = parse_args()
set_seed(args.random_seed)
folder = osp.join(data_root, folder_dict[args.dataset])
path = osp.join(param_root, 'gnns/%s_net.pt' % args.dataset)
train_dataset, val_dataset, test_dataset = get_datasets(args.dataset, root=data_root)
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
os.makedirs(log_root, exist_ok=True)
logger = Logger.init_logger(filename=log_root + f"/refine-{args.dataset}.log" )

if args.dataset == 'graphsst2':
    dataloader = get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        random_split_flag=True, 
        data_split_ratio=[0.8, 0.1, 0.1],   
        seed=2)
    train_loader, val_loader, test_loader = dataloader['train'], dataloader['eval'], dataloader['test']
else:
    # filter graphs with right prediction
    for label, dataset in zip(['train', 'val', 'test'], [train_dataset, val_dataset, test_dataset]):
        batch_size = 1 if label=='test' else args.batch_size
        dataset_mask = []
        flitered_path = osp.join(param_root, f"filtered/{args.dataset}_idx_{label}.pt")
        if osp.exists(flitered_path):
            graph_mask = torch.load(flitered_path)
        else:
            os.makedirs(osp.join(param_root, "filtered"), exist_ok=True)
            loader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False
                                )
            # filter graphs with right prediction
            model = torch.load(path).to(device)
            graph_mask = torch.zeros(len(loader.dataset), dtype=torch.bool)
            idx = 0
            for g in tqdm(iter(loader), total=len(loader)):

                g.to(device)
                model(g)
                if g.y == model.readout.argmax(dim=1):
                    graph_mask[idx] = True
                idx += 1

            torch.save(graph_mask, flitered_path)
            dataset_mask.append(graph_mask)

        logger.info("number of graphs(%s): %4d" % (label, graph_mask.nonzero().size(0)))
        exec("%s_loader = DataLoader(dataset[graph_mask], batch_size=%d, shuffle=False, drop_last=False)" % \
                                    (label, batch_size))

explainer = ReFine(device, path, gamma=args.gamma,
                  n_in_channels=torch.flatten(train_dataset[0].x, 1, -1).size(1),
                  e_in_channels=train_dataset[0].edge_attr.size(1),
                  n_label=n_classes_dict[args.dataset])

parameters = list()
for k, edge_mask in enumerate(explainer.edge_mask):
    edge_mask.train()
    parameters += list(explainer.edge_mask[k].parameters())

optimizer = torch.optim.Adam(parameters, lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer,
                                mode='min',
                                factor=0.2,
                                patience=3,
                                min_lr=1e-5
                                )
ratio = args.ratio
loss_all = 0
for epoch in range(args.epoch):
    for g in train_loader:
        g.to(device)
        optimizer.zero_grad()
        loss = explainer.pretrain(
            g,
            ratio=args.ratio,
            reperameter=True,
            temp=args.tau
            )
        loss.backward()
        optimizer.step()
        loss_all += loss

    test_G_acc_loger = []
    val_loss_all = 0
    with torch.no_grad():
        for g in val_loader:
            g.to(device)
            loss = explainer.pretrain(
                    g,
                    ratio=args.ratio,
                    reperameter=False,
                    temp=args.tau
                    )
            val_loss_all += loss

        scheduler.step(val_loss_all)
        lr = scheduler.optimizer.param_groups[0]['lr']

        imps = []
        for g in test_loader:
            g.to(device)
            imp = explainer.explain_graph(g, fine_tune=False, ratio=args.ratio)
            acc, _ = explainer.evaluate_acc(top_ratio_list=[0.1*i for i in range(1,11)])
            test_G_acc_loger.append(acc)
            imps.extend(imp)

    train_loss = loss_all / len(train_loader.dataset)
    val_loss = val_loss_all / len(val_loader.dataset)
    logger.info("Epoch: %d, LR: %.5f, Ratio: %.2f, Train Loss: %.3f, Val Loss: %.3f" % (epoch + 1, lr, ratio, train_loss, val_loss))
    if epoch % 2 == 0:
        logger.info(
            "ACC:" + str(np.array(test_G_acc_loger).mean(axis=0)) + 
            " ACC-AUC: %.3f" % np.array(test_G_acc_loger).mean() +  
            " Mean P: %.3f" % np.array(imps).mean()
            )

model_dir = osp.join(param_root, "refine/")
os.makedirs(model_dir, exist_ok=True)
torch.save(explainer, osp.join(model_dir, f'{args.dataset}.pt'))