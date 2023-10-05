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
from geomodel_analysis import GmeModelGraphList
import tqdm
import argparse

from pyvistaqt import MultiPlotter
from gme_trainer import GmeTrainer, GmeTrainerConfig, GraphTransConfig
from models.model import GraphTransfomer, GraphModel, SAGEModel, SAGETransfomer, GraphTransfomerNet
from retrieve_noddy_files import NoddyModelData
from data_structure.grids import Grid
from geograph_parse import GeoMeshGraphParse
from data_structure.data_sampler import GeoGridDataSampler
from data_structure.boreholes import BoreholeSet, Borehole
from data_structure.sections import SectionSet, Section
import pyvista as pv
from utils.vtk_utils import CreateLUT
from utils.plot_utils import contorl_visibility_with_layer_label, control_threshold_with_scalars \
    , control_clip_with_plane, control_clip_with_spline

if __name__ == '__main__':
    # load and preprocess dataset
    print('Loading data')
    path_1 = os.path.abspath('..')
    root_path = os.path.join(path_1, 'geomodel_workshop')
    noddyData = NoddyModelData(root=r'F:\NoddyDataset', max_model_num=10)
    noddy_grid_list = noddyData.get_grid_model_by_idx(dataset='FOLD_FOLD_FOLD', idx=[0])  # 1 6
    grid_list = []
    for noddy_grid in noddy_grid_list:
        grid = Grid(grid_vtk=noddy_grid, name='GeoGrid')
        grid.resample_regular_grid(dim=np.array([150, 150, 120]))
        grid_list.append(grid)
    model_idx = 0

    geodata_drills = GeoGridDataSampler(grid=grid_list[0], sample_operator=['rand_drills'], drill_num=25
                                        , sample_data_names=['drills'])
    geodata_drills.execute()
    # 钻孔数据
    boreholes_data = geodata_drills.sample_data_list[0]
    # 将钻孔数据映射到空网格上
    geodata_sample = GeoGridDataSampler()
    geodata_sample.set_base_grid_by_boreholes(boreholes=boreholes_data, dims=np.array([150, 150, 120]))
    geodata_sample.execute()

    new_grid = geodata_sample.grid
    # plotter_1 = control_clip_with_plane(grid=grid_list[0], only_section=True)
    # plotter_1.show()
    plotter_2 = contorl_visibility_with_layer_label(geo_object_list=[boreholes_data, new_grid], grid_smooth=False
                                                    , show_edge=False)
    plotter_2.show()

    # 只有第一次输入的noddy_model可以用到，之后代码会自动加载数据缓存
    gme_models = GmeModelGraphList('gme_model', root=root_path,
                                   grid_data=grid_list,
                                   sample_operator=['rand_drills', 'axis_sections'],
                                   # ['axis_sections'],'axis_sections'
                                   sample_axis='x', section_num=1, scroll_pos=[0.5], resolution_xy=10, resolution_z=10,
                                   add_inverse_edge=True,
                                   drill_num=10)

    plot_data = gme_models.geodata[model_idx].sample_data.sample_data_list[0].get_sample_vtk_data()

    # xgboost_path = os.path.join(root_path, 'processed', 'xgboost_ori.vtk')
    # gme_models.predict_with_machine_learning_method(model_idx=0, method='xgboost', save_path=xgboost_path)

    dataset = DglGeoDataset(gme_models)

    # initialize a trainer instance and kick off training
    # 模型训练相关参数
    # save_path_1 = r'E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\vtk_model.vtk'
    trainer_config = GmeTrainerConfig(max_epochs=14000, batch_size=512, num_workers=4, learning_rate=1e-4,
                                      ckpt_path=os.path.join(root_path, 'processed', 'latest_tran.pth'),
                                      output_dir=os.path.join(root_path, 'output'),
                                      out_put_grid_file_name=os.path.join(gme_models.processed_dir, 'vtk_model.vtk'),
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
                                    gnn_n_head=3, n_layer=3, )
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

    #####
    gme_models.set_dat_file_regular_grid(
        dat_file_path=r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\drills.csv",
        is_create_graph=True, names=['x', 'y', 'z', 'label'], file_sep=',')
