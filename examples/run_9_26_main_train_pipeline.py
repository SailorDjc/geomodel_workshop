import os.path

import numpy as np
# import torch.nn.functional as F
# import torchmetrics.functional as MF
# import dgl.nn as dglnn
# from dgl.data import AsNodePredDataset
# from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from dgl_geodataset import DglGeoDataset
from geomodel_analysis import *
# import tqdm
# import argparse
# from pyvistaqt import MultiPlotter
from gme_trainer import GmeTrainer, GmeTrainerConfig, GraphTransConfig
from models.model import GraphTransfomer
from data.retrieve_noddy_files import NoddyModelData
from data_structure.reader import *
from utils.plot_utils import *

# from geograph_parse import GeoMeshGraphParse
# from data_structure.data_sampler import GeoGridDataSampler
# from data_structure.boreholes import BoreholeSet, Borehole
# from data_structure.sections import SectionSet, Section
# import pyvista as pv
# from utils.vtk_utils import CreateLUT

if __name__ == '__main__':
    # load and preprocess dataset
    print('Loading data')

    path_1 = os.path.abspath('../..')
    root_path = os.path.join(path_1, 'geomodel_workshop')

    noddyData = NoddyModelData(root=r'F:\djc\NoddyDataset', dataset_list=['FOLD_FOLD_FOLD'], max_model_num=10,
                               update_grid=False)
    noddy_grid_list = noddyData.get_grid_model_by_idx(dataset='FOLD_FOLD_FOLD', idx=[6])  # 1 6
    grid_list = []
    reader = ReadExportFile()
    # grid_data = reader.read_geodata(file_path=os.path.join(root_path, 'data', 'out_model', 'out_model.dat'))
    # grid_data.plot(activate_scalars='stratum')
    for noddy_grid in noddy_grid_list:
        # 数据重采样，三维模型的尺寸是[150, 150, 120]
        grid = Grid(grid_vtk=noddy_grid, name='GeoGrid')
        grid.resample_regular_grid(dim=np.array([50, 50, 50]))
        grid_list.append(grid)
    model_idx = 0

    # grid_list.append(grid_data.vtk_data)
    # visual_multiple_model(grid_list[0], grid_list[1])

    # 将三维模型规则网格数据构建为图网格数据，只有第一次输入的noddy_model可以用到，之后代码会自动加载数据缓存
    gme_models = GmeModelGraphList('gme_model', root=root_path,
                                   grid_data=grid_list,
                                   sample_operator=['rand_drills'],
                                   add_inverse_edge=True,
                                   split_ratio=DataSetSplit(0.6, 0.2),
                                   drill_num=50)
    # 节约存储空间，多余模型不加载
    gme_models.load_geograph(graph_id=0)

    # boreholes = gme_models.geograph[0].sample_data[0]
    # gg_dd = gme_models.geograph[0].data
    #
    # plotter_1 = control_visibility_with_layer_label(geo_object_list=[boreholes, gg_dd], grid_smooth=False
    #                                                 , show_edge=False)
    # plotter_1.show()
    dataset = DglGeoDataset(gme_models)

    # initialize a trainer instance and kick off training
    # 模型训练相关参数    初始训练参数的设置
    trainer_config = GmeTrainerConfig(max_epochs=2000, batch_size=512, num_workers=4, learning_rate=1e-4,
                                      ckpt_path=os.path.join(root_path, 'processed', 'latest_tran.pth'),
                                      output_dir=os.path.join(root_path, 'output'),
                                      out_put_grid_file_name=os.path.join(gme_models.processed_dir, 'output_model'),
                                      sample_neigh=[10, 10, 15, 15])
    # 从图数据集中取出一张图
    g = dataset[model_idx]
    ##
    # create GraphSAGE model
    in_size = g.ndata['feat'].shape[-1]
    print("in_size:", in_size)

    # 要将 C:\Users\用户名\.dgl\gme_model-as-nodepred 删除更新   dglDataset 也有自己的缓存文件，如果数据有修改要及时删除
    # 输出类别数
    out_size = dataset.num_classes['labels'][model_idx]

    # 模型结构相关参数
    model_config = GraphTransConfig(in_size=in_size, out_size=out_size, n_head=4, n_embd=512, gnn_layer_num=4,
                                    gnn_n_head=3, n_layer=3)
    # 构建预测模型
    model = GraphTransfomer(model_config)
    # model = GraphTransfomerNet(model_config)
    # model = GraphModel(model_config)
    # model = SAGEModel(model_config)
    # model = SAGETransfomer(model_config)
    trainer = GmeTrainer(model, dataset, trainer_config)
    # model training
    print('Training...')
    # 开始训练
    trainer.train(data_split_idx=model_idx, has_test_label=True, only_inference=True)

    # grid_model = Grid(grid_vtk_path=os.path.join(gme_models.processed_dir, 'vtk_model.vtk'))
    # boreholes_data = gme_models.sample_data[0]  # boreholes_data,
    # plotter_2 = control_visibility_with_layer_label(geo_object_list=[grid_model, boreholes_data], grid_smooth=False
    #                                                 , show_edge=False)
    # plotter_2.show()

    #####
    # gme_models.set_dat_file_regular_grid(
    #     dat_file_path=r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\drills.csv",
    #     is_create_graph=True, names=['x', 'y', 'z', 'label'], file_sep=',')
