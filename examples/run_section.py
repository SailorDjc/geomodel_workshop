from data_structure.reader import ReadExportFile
from data_structure.geodata import GeodataSet, Grid, BoreholeSet, Borehole
from dgl_geodataset import DglGeoDataset
from geomodel_analysis import GmeModelGraphList, GeoDataMLClassifier
from gme_trainer import GmeTrainer, GmeTrainerConfig, GraphTransConfig
from models.model import GraphTransfomer
from data_structure.terrain import TerrainData
import numpy as np
import os
from utils.plot_utils import control_visibility_with_layer_label, visual_multiple_model
from utils.plot_utils import visual_edge_list


# 本实例提供了一个完整的地质钻孔数据的训练建模流程
# 本实例的钻孔数据是从地质剖面图上密集采样的虚拟钻孔，采样间隔为20米，
# 将剖面转换为钻孔数据进行建模是一种剖面数据参与建模的方式。

# 建模结果与建模钻孔联合可视化
def plot_text_data(boreholes=None):
    reader_0 = ReadExportFile()
    # log_path = r"E:\11-22-GeoSci\geomodel_workshop-main\processed\train_loss_log.txt"
    # a, b, c, d, e, f, h = reader_0.read_train_loss_log(txt_file_path=log_path, sep=',')
    # visual_acc_picture(train_acc=b, test_acc=e, title='Accuracy')
    model_path = r"E:\11-22-GeoSci\geomodel_workshop-main\output\vtk_sec_model.vtk"
    grid = reader_0.read_vtk_data(file_path=model_path)
    sec_grid = Grid()
    sec_grid.set_vtk_grid(grid_vtk=grid, labels_standardize=False)
    geo_list = []
    geo_list.append(sec_grid)
    if boreholes is not None:
        geo_list.append(boreholes)
    plot = control_visibility_with_layer_label(geo_object_list=geo_list)
    plot.show()


root_path = os.path.abspath('..')
reader = ReadExportFile()
# bounds_seg_0 = [507200, 508500, 4299149.58, 4303055.14, 306.088, 960.074]  #
bounds_seg_1 = [506250, 507210, 4299149.58, 4303055.14, 306.088, 960.074]
# 这是一个剖面的钻孔密采样
file_sec_path = os.path.join(root_path, 'data', '20m_VirtualDrill.xlsx')
# 真实钻孔
file_drill_path = os.path.join(root_path, 'data', '测试样本数据_钻孔.xlsx')
# 获取钻孔数据
boreholes = reader.tmp_read_boreholes(excel_path=file_drill_path)
# 获取虚拟钻孔数据
sec_boreholes = reader.tmp_read_virtual_boreholes(dat_file_path=file_sec_path)
boreholes_select = sec_boreholes.search_by_rect2d(rect2d=bounds_seg_1)
print('sec_bounds:', sec_boreholes.bounds)
geodataset = GeodataSet()
# geod.append(boreholes)
geodataset.append(boreholes_select)
# 设置钻孔控制半径范围
geodataset.geodata_list[0].set_boreholes_control_buffer_dist_xy(radius=3)
# 设置统一的地层标签映射
label_dict = {'1-1': 0, '1-2': 1, '1-3': 2, '2-41': 3, '2-122': 4, '2-153': 5, '4-41': 6, '4-133': 7, '5-41': 8,
              '5-122': 9, '5-161': 10, '10-113': 11, '23-21': 12, '23-12': 13, '23-13': 14, '87-12': 15,
              '87-13': 16, '118-11': 17, '118-12': 18, '118-13': 19, '118-22': 20, '118-23': 21, '118-32': 22,
              '118-33': 23, '119-11': 24, '119-12': 25, '119-13': 26, '119-22': 27, '119-23': 28, '119-32': 29,
              '119-33': 30, '119-52': 31, '119-53': 32, '123-12': 33, '123-13': 34, '123-23': 35, '124-11': 36,
              '124-12': 37, '124-13': 38, '124-23': 39}
geodataset.standardize_labels(label_dict=label_dict)

# geod_seg = geodataset.search_by_rect2d(rect2d=bounds_seg_1)
hh = geodataset.geodata_list[0]
# hh.show()
zz = np.array([h.holelayer_list[-1].top_pos for h in hh])
zz = zz[:, 2]
zm = np.min(zz)
# 标签统一处理

if __name__ == "__main__":
    model_idx = 0
    # 由钻孔顶部点生成地形
    # top_points = geod.geodata_list[0].get_top_points()
    # terr = TerrainData()
    # # 范围约束，按照剖面线缓冲，生成线状建模区域
    # terr.set_boundary_from_line_buffer(trajectory_line_xy=top_points, buffer_dist=20)
    # terr.set_control_points(top_points)

    top_points_seg = geodataset.geodata_list[0].get_top_points()
    terr_seg = TerrainData()
    # 范围约束，按照剖面线缓冲，生成线状建模区域
    terr_seg.set_boundary_from_line_buffer(trajectory_line_xy=top_points_seg, buffer_dist=20)
    terr_seg.set_control_points(top_points_seg)

    # epoch, train_loss, train_acc, train_rmse, val_loss, val_acc, val_rmse = reader.read_train_loss_log(
    #     txt_file_path=r"E:\11-22-GeoSci\geomodel_workshop-main\output\model_backup\seg_1\new 1.txt", sep=',')
    # from utils.plot_utils import visual_acc_picture, visual_loss_picture
    # visual_acc_picture(train_acc=train_acc, test_acc=val_acc, title='Accuracy')
    # visual_loss_picture(train_loss=train_loss, test_loss=val_loss, title='Loss')

    # grid_seg_0 = reader.read_geodata(file_path=r"E:Pycode\2024\geoscience\geomodel_workshop-main\output\vtk_sec_model_0")
    # grid_seg_0.label_dict = {'1-2': 0, '5-41': 1, '5-122': 2, '118-12': 3, '118-13': 4, '119-11': 5, '119-12': 6,
    #                          '119-13': 7, '119-22': 8, '119-23': 9, '119-53': 10, '124-13': 11}

    # grid_seg_0.uniform_labels(label_dict=geodataset.label_dict)

    # 注意网格尺寸参数 grid_cell_density，若设置过小，计算机可能无法处理
    gme_models = GmeModelGraphList('gme_model', root=root_path,
                                   input_sample_data=geodataset,  #
                                   add_inverse_edge=True,
                                   terrain_data=terr_seg,
                                   grid_cell_density=[10, 10, 10],
                                   val_ratio=0.2,
                                   is_regular=True,
                                   update_graph=False)  # 不规则网格
    gme_models.load_geograph(graph_id=1)
    # grid_2 = gme_models.geograph[1].data
    # gme_models.append_rigid_ristriction(points_data=grid_seg_0.get_points_data(), g_idx=model_idx)

    dataset = DglGeoDataset(gme_models, graph_id=model_idx)

    # initialize a trainer instance and kick off training
    # 模型训练相关参数    初始训练参数的设置
    trainer_config = GmeTrainerConfig(max_epochs=96, batch_size=512, num_workers=4, learning_rate=1e-4,
                                      ckpt_path=os.path.join(root_path, 'processed', 'latest_tran.pth'),
                                      output_dir=os.path.join(root_path, 'output'),
                                      out_put_grid_file_name='vtk_sec_model_{}'.format(model_idx),
                                      sample_neigh=[10, 10, 15, 15], gpu_num=1)
    # 从图数据集中取出一张图
    g = dataset[0]
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
    trainer.train(data_split_idx=model_idx, has_test_label=False)
