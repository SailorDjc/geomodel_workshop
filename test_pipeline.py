import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from geo_dataset import GeoDataset
from gme_model_generate import GmeModelList
import tqdm
import argparse

from pyvistaqt import MultiPlotter
from gme_trainer import GmeTrainer, GmeTrainerConfig, GraphTransConfig
from models.model import GraphTransfomer
from retrieve_noddy_files import NoddyModelData

if __name__ == '__main__':
    # load and preprocess dataset
    print('Loading data')
    path_1 = os.path.abspath('..')
    root_path = os.path.join(path_1, 'geomodel_workshop')
    noddyData = NoddyModelData(root=r'F:\djc\NoddyDataset', max_model_num=30)
    noddy_models = noddyData.get_noddy_model_list_names(model_num=30, sample_random=False)
    pre_train_model_list = []
    for item in [16, 1, 22, 8, 7]:
        pre_train_model_list.append(noddy_models[item])
    # 只有第一次输入的noddy_model可以用到，之后代码会自动加载数据缓存
    gme_models = GmeModelList('gme_model', root=root_path, pre_train_model_list=pre_train_model_list[:2],
                              train_model_list=None, noddy_data=noddyData,
                              sample_operator=['axis_sections'], add_inverse_edge=True,
                              data_type='Noddy',  # 'Wells',  # 'Points',
                              dat_file_path=r"E:\Code\duanjc\PyCode\GeoScience\drill_dense_scatter.dat",
                              use_cols=[0, 1, 2, 3, 4], file_header=None, names=['id', 'x', 'y', 'z', 'label'],
                              file_sep=' ', has_air=False)
    dataset = GeoDataset(gme_models)

    # initialize a trainer instance and kick off training
    # 模型训练相关参数
    tconf = GmeTrainerConfig(max_epochs=10, num_workers=4,
                             ckpt_path=os.path.join(root_path, 'processed', 'latest_tran.pth'))

    model_idx = 0
    g = dataset[model_idx]
    # create GraphSAGE model
    in_size = g.ndata['feat'].shape[-1]
    print("in_size:", in_size)

    # 输出类别数
    # 要将 C:\Users\用户名\.dgl\gme_model-as-nodepred 删除更新
    out_size = dataset.num_classes['labels'][model_idx]

    vocab_size = 512
    # 模型结构相关参数
    mconf = GraphTransConfig(vocab_size, in_size, n_head=2, n_embd=256, out_size=out_size)
    # 构建预测模型
    model = GraphTransfomer(mconf)
    trainer = GmeTrainer(model, dataset, tconf)
    # model training
    print('Training...')

    trainer.train(data_split_idx=model_idx)
