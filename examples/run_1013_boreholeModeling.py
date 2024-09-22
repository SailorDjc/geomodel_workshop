import os.path

import numpy as np
from data_structure.reader import *
from dgl_geodataset import DglGeoDataset
from geomodel_analysis import *
from gme_trainer import GmeTrainer, GmeTrainerConfig, GraphTransConfig
from models.model import GraphTransfomer
from data_structure.reader import ReadExportFile
from data_structure.geodata import *
from utils.plot_utils import visual_multiple_model
from data_structure.terrain import TerrainData
from data_structure.grids import Grid
from data_structure.data_sampler import GeoGridDataSampler
from utils.plot_utils import control_visibility_with_layer_label
import random

# 本实例给出了一个完整的地质数据训练并最终生成地质模型的过程
# 实例使用从钻孔文件中读取的钻孔数据
if __name__ == '__main__':
    # load and preprocess dataset
    print('Loading data')
    root_path = os.path.abspath('..')
    xm_file_path = os.path.join(root_path, 'data', 'borehole_hill.dat')
    # # # 从外部数据文本中加载钻孔数据
    reader = ReadExportFile()
    boreholes_data = reader.read_boreholes_data_from_text_file(dat_file_path=xm_file_path)
    # select_ids = [bid for bid in list(np.arange(boreholes_data.borehole_num)) if bid not in delete_list]
    # boreholes_data = boreholes_data.get_boreholes(idx=select_ids)
    # 延展钻孔最底层
    print('xl:', boreholes_data.bounds[1] - boreholes_data.bounds[0])
    print('yl:', boreholes_data.bounds[3] - boreholes_data.bounds[2])
    print('zl:', boreholes_data.bounds[5] - boreholes_data.bounds[4])
    boreholes_data.extend_base_layer(base_label=36)
    a = boreholes_data.get_minimum_layer_interval()
    c = boreholes_data.get_thin_layers(threshold_dist=0.5)
    plot = control_visibility_with_layer_label([boreholes_data])
    plot.show()
    # boreholes_data.get_boreholes(idx=rcandom.sample(list(np.arange(len(boreholes_data))), 95))

    # 添加基底层
    # boreholes_data.add_base_layer_for_each_borehole()

    # boreholes_data.plot(borehole_radius=5)
    # boreholes_data_2 = reader.read_labels_map(
    #     map_file_path=os.path.join(root_path, 'data', 'sample_drills_0306.map'), encoding='ANSI')
    boreholes_data.set_boreholes_control_buffer_dist_xy(radius=5)
    gd = GeodataSet()
    gd.append(boreholes_data)

    model_idx = 0
    gd.standardize_labels()
    # 将三维模型规则网格数据构建为图网格数据
    # 这里采用真实钻孔，不采用虚拟钻孔采样
    terrain_data = TerrainData()
    gme_models = GmeModelGraphList('gme_model', root=root_path,
                                   split_ratio=DataSetSplit(1),
                                   input_sample_data=gd,
                                   dir_path=os.path.join(root_path, 'processed'),
                                   add_inverse_edge=True,
                                   grid_dims=[50, 50, 30],
                                   terrain_data=terrain_data)
    # gme_models.load_geograph(graph_id=model_idx, dir_path=os.path.join(root_path, 'processed'))

    # geodata = gme_models.geograph[model_idx]
    #
    # train_idx = geodata.train_data_indexes
    # val_idx = geodata.val_data_indexes
    # test_idx = geodata.test_data_indexes
    # grid_points = geodata.get_grid_points()
    # points_label = geodata.get_grid_points_labels()
    # gd.geodata_list[0].plot()
    # train_points = PointSet(points=grid_points[train_idx], point_labels=points_label[train_idx])
    # val_points = PointSet(points=grid_points[val_idx], point_labels=points_label[val_idx])
    # test_points = PointSet(points=grid_points[test_idx], point_labels=points_label[test_idx])
    #
    # plotter_2 = control_visibility_with_layer_label(geo_object_list=[train_points, val_points, test_points, gd.geodata_list[0]], grid_smooth=False
    #                                                 , show_edge=False)
    # plotter_2.show()

    dataset = DglGeoDataset(gme_models)

    # initialize a trainer instance and kick off training
    # 模型训练相关参数    初始训练参数的设置
    trainer_config = GmeTrainerConfig(max_epochs=1500, batch_size=512, num_workers=4, learning_rate=1e-5,
                                      ckpt_path=os.path.join(root_path, 'processed', 'latest_tran.pth'),
                                      output_dir=os.path.join(root_path, 'output'),
                                      out_put_grid_file_name=os.path.join(root_path, 'output', 'vtk_model'),
                                      sample_neigh=[10, 10, 15, 15], gpu_num=1)
    # 从图数据集中取出一张图
    g = dataset[model_idx]
    # create GraphSAGE model
    in_size = g.ndata['feat'].shape[-1]
    print("in_size:", in_size)
    # 要将 C:\Users\用户名\.dgl\gme_model-as-nodepred 删除更新   dglDataset 也有自己的缓存文件，如果数据有修改要及时删除
    # 输出类别数
    out_size = dataset.num_classes['labels'][model_idx]
    # 模型结构相关参数
    model_config = GraphTransConfig(in_size=in_size, out_size=out_size, n_head=4, n_embd=512, gnn_layer_num=3,
                                    gnn_n_head=4, n_layer=4)
    # 构建预测模型
    model = GraphTransfomer(model_config)
    trainer = GmeTrainer(model, dataset, trainer_config)
    # model training
    print('Training...')
    # 开始训练
    trainer.train(data_split_idx=model_idx, has_test_label=True, early_stop_patience=50)
