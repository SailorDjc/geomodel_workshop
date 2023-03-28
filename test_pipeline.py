import copy
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
from dgl_geodataset import DglGeoDataset
from gme_model_generate import GmeModelList
import tqdm
import argparse

from pyvistaqt import MultiPlotter
from gme_trainer import GmeTrainer, GmeTrainerConfig, GraphTransConfig
from models.model import GraphTransfomer, GraphModel, SAGEModel, SAGETransfomer, GraphTransfomerNet
from retrieve_noddy_files import NoddyModelData
import model_visual_kit as mvk

if __name__ == '__main__':
    # load and preprocess dataset
    print('Loading data')
    path_1 = os.path.abspath('..')
    root_path = os.path.join(path_1, 'geomodel_workshop')
    noddyData = NoddyModelData(root=r'F:\djc\NoddyDataset', max_model_num=50)
    noddy_models = noddyData.get_noddy_model_list_names(model_num=50, sample_random=False)
    pre_train_model_list = []
    for item in [6, 42, 0, 1, 7, 9, 16]:  # [0, 1, 6, 7, 9, 15, 16, 21, 24, 33, 34, 39, 42, 44]:
        pre_train_model_list.append(noddy_models[item])
    model_idx = 0
    # 只有第一次输入的noddy_model可以用到，之后代码会自动加载数据缓存
    gme_models = GmeModelList('gme_model', root=root_path, pre_train_model_list=None, model_extern=[120, 120, 50],
                              noddy_data=noddyData,  # train_model_list=pre_train_model_list[3:4],
                              sample_operator=['rand_drills'],  # ['axis_sections'],
                              add_inverse_edge=True,
                              data_type='Noddy', drill_num=60)  # # 'Wells',  # 'Points'
    geodata = gme_models.geodata[model_idx]
    # train_x, train_y, test_x, test_y, bound = gme_models.predict_with_interpolate(model_idx=model_idx, method='idw')  # iso_list=[0, 1, 2, 3, 4, 5], save_path=rbf_path
    # geodata.export_drill_dict_dat_file(
    #     file_path=r'E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\train_data.dat')

    # gme_models.process_external_fields_model(
    #     intput_file_path=r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\Gridder.vtk",
    #     model_idx=0, iso_list=[0, 1, 2, 3, 4, 5])

    rf_path = os.path.join(root_path, 'processed', 'rf.vtk')
    # gme_models.predict_with_interpolate(model_idx=0, method='rbf', iso_list=[0, 1, 2, 3, 4, 5], save_path=rbf_path)
    gme_models.predict_with_machine_learning_method(model_idx=model_idx, method='rf', save_path=rf_path)
    # gme_models.set_dat_file_regular_grid(dat_file_path=os.path.join(root_path, 'processed', 'sample_drills.dat'),
    #                                      regular_grid_type='auto', is_save=True)
    # gme_models.set_dat_file_unregular_grid(dat_file_path=os.path.join(root_path, 'processed', 'sample_drills.dat'),
    #                                        edge_list_path=os.path.join(root_path, 'processed', 'sample_drills.edge'),
    #                                        grid_node_file_path=os.path.join(root_path, 'processed', 'sample_drills.node'),
    #                                        data_type='Wells', file_header=None, is_save=True)
    ##


    dataset = DglGeoDataset(gme_models)

    # initialize a trainer instance and kick off training
    # 模型训练相关参数
    save_path_1 = r'E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\vtk_model.vtk'
    trainer_config = GmeTrainerConfig(max_epochs=14000, batch_size=512, num_workers=4, learning_rate=1e-4,
                                      ckpt_path=os.path.join(root_path, 'processed', 'latest_tran.pth'),
                                      output_dir=os.path.join(root_path, 'output'),
                                      out_put_grid_file_name=save_path_1,
                                      sample_neigh=[10, 10, 15, 15])

    g = dataset[model_idx]
    ##
    # create GraphSAGE model
    in_size = g.ndata['feat'].shape[-1]
    print("in_size:", in_size)

    # 输出类别数
    # 要将 C:\Users\用户名\.dgl\gme_model-as-nodepred 删除更新
    out_size = dataset.num_classes['labels'][model_idx]

    # vocab_size = 512
    # 模型结构相关参数
    model_config = GraphTransConfig(in_size=in_size, n_head=4, n_embd=512, out_size=out_size, gnn_layer_num=4,
                                    gnn_n_head=3, n_layer=3,)
    # 构建预测模型
    model = GraphTransfomer(model_config)
    # model = GraphTransfomerNet(model_config)
    # model = GraphModel(model_config)
    # model = SAGEModel(model_config)
    # model = SAGETransfomer(model_config)
    trainer = GmeTrainer(model, dataset, trainer_config)
    # model training
    print('Training...')

    trainer.train(data_split_idx=model_idx, has_test_label=True)
