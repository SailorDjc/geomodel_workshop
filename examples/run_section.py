from data_structure.reader import ReadExportFile, BoreholeSetManager
from data_structure.geodata import GeodataSet, Grid, PointSet
from dgl_geodataset import DglGeoDataset
from geomodel_analysis import GmeModelGraphList, GeoDataMLClassifier
from gme_trainer import GmeTrainer, GmeTrainerConfig, GraphTransConfig
from models.model import GraphTransfomer
from data_structure.terrain import TerrainData
import numpy as np
import os
from utils.plot_utils import control_visibility_with_layer_label
from utils.plot_utils import visual_acc_picture


def plot_text_data(root_dir, boreholes=None):
    reader_0 = ReadExportFile()
    # log_path = r"E:\11-22-GeoSci\geomodel_workshop-main\processed\train_loss_log.txt"
    # a, b, c, d, e, f, h = reader_0.read_train_loss_log(txt_file_path=log_path, sep=',')
    # visual_acc_picture(train_acc=b, test_acc=e, title='Accuracy')
    model_path = r"E:\11-22-GeoSci\geomodel_workshop-main\output\vtk_sec_model.vtk"
    grid = reader_0.read_vtk_data(file_path=model_path)
    # grid.plot()
    sec_grid = Grid()
    sec_grid.set_vtk_grid(grid_vtk=grid, labels_standardize=False)
    geo_list = []
    geo_list.append(sec_grid)
    if boreholes is not None:
        geo_list.append(boreholes)
    plot = control_visibility_with_layer_label(geo_object_list=geo_list)
    plot.show()

    # import torch
    # y = torch.empty(512, dtype=torch.long).random_(2, 5)
    # y[0:100] = torch.randint(8, 12, size=(100,))


if __name__ == "__main__":

    model_idx = 0
    root_path = os.path.abspath('..')

    # plot_text_data(root_dir=root_path)  # , boreholes=geod.geodata_list[1]

    reader = ReadExportFile()
    # 这是一个剖面的钻孔密采样
    file_sec_path = os.path.join(root_path, 'data', '20m_VirtualDrill.xlsx')
    # 真实钻孔
    file_drill_path = os.path.join(root_path, 'data', '测试样本数据_钻孔.xlsx')
    boreholes = reader.tmp_read_boreholes(excel_path=file_drill_path)
    sec_boreholes = reader.tmp_read_virtual_boreholes(dat_file_path=file_sec_path)
    print('sec_bounds:', sec_boreholes.bounds)
    bounds = [507200, 508500, 4299149.58, 4303055.14, 306.088,
              960.074]  # [506000, 508500, 4299149.58, 4303055.14, 306.088, 960.074]
    # 钻孔按范围筛选
    sec_boreholes = sec_boreholes.search_by_rect2d(rect2d=bounds)
    boreholes = boreholes.search_by_rect2d(rect2d=bounds)
    # sec_boreholes.show()
    geod = GeodataSet()
    geod.append(boreholes)
    geod.append(sec_boreholes)
    # 设置统一的地层标签映射
    label_dict = {'1-1': 0, '1-2': 1, '1-3': 2, '2-41': 3, '2-122': 4, '2-153': 5, '4-41': 6, '4-133': 7, '5-41': 8,
                  '5-122': 9, '5-161': 10, '10-113': 11, '23-21': 12, '23-12': 13, '23-13': 14, '87-12': 15,
                  '87-13': 16, '118-11': 17, '118-12': 18, '118-13': 19, '118-22': 20, '118-23': 21, '118-32': 22,
                  '118-33': 23, '119-11': 24, '119-12': 25, '119-13': 26, '119-22': 27, '119-23': 28, '119-32': 29,
                  '119-33': 30, '119-52': 31, '119-53': 32, '123-12': 33, '123-13': 34, '123-23': 35, '124-11': 36,
                  '124-12': 37, '124-13': 38, '124-23': 39}
    # 标签统一处理
    geod.standardize_labels(label_dict=label_dict)
    # plot = control_visibility_with_layer_label(geo_object_list=[geod.geodata_list[1]])
    # plot.show()
    #
    # 由钻孔顶部点生成地形
    top_points = geod.geodata_list[1].get_top_points()
    terr = TerrainData()
    bot_points = geod.geodata_list[1].get_bottom_points()
    # 范围约束，按照剖面线缓冲，生成线状建模区域
    terr.set_boundary_from_line_buffer(trajectory_line_xy=top_points, buffer_dist=30)
    terr.set_control_points(top_points)
    terr.execute()
    # 为钻孔添加基底层
    geod.geodata_list[1].add_base_layer_for_each_borehole()
    # 设置钻孔控制半径范围
    geod.geodata_list[1].set_boreholes_control_buffer_dist_xy(radius=3)
    geod.geodata_list[0].set_boreholes_control_buffer_dist_xy(radius=3)
    # 注意网格尺寸参数 grid_cell_density，若设置过小，计算机可能无法处理
    gme_models = GmeModelGraphList('gme_model', root=root_path,
                                   input_sample_data=geod,  # ,  geod
                                   add_inverse_edge=True,
                                   terrain_data=terr,  # terr
                                   grid_cell_density=[2, 2, 0.5],
                                   val_ratio=0.2)  #
    dataset = DglGeoDataset(gme_models)
    import gc

    del gme_models
    gc.collect()
    # initialize a trainer instance and kick off training
    # 模型训练相关参数    初始训练参数的设置
    trainer_config = GmeTrainerConfig(max_epochs=120, batch_size=512, num_workers=4, learning_rate=1e-4,
                                      ckpt_path=os.path.join(root_path, 'processed', 'latest_tran.pth'),
                                      output_dir=os.path.join(root_path, 'output'),
                                      out_put_grid_file_name='vtk_sec_model.vtk',
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
    model_config = GraphTransConfig(in_size=in_size, out_size=out_size, n_head=4, n_embd=512, gnn_layer_num=4,
                                    gnn_n_head=3, n_layer=3)
    # 构建预测模型
    model = GraphTransfomer(model_config)
    trainer = GmeTrainer(model, dataset, trainer_config)
    # model training
    print('Training...')
    # 开始训练
    trainer.train(data_split_idx=model_idx, has_test_label=True)
