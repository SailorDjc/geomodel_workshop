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

root_path = os.path.abspath('..')
reader = ReadExportFile()
file_drill_path = os.path.join(root_path, 'data', 'all_sampling_results.dat')
# 获取钻孔数据
boreholes = reader.read_boreholes_data_from_text_file(dat_file_path=file_drill_path)
print('sec_bounds:', boreholes.bounds)

geodataset = GeodataSet()
geodataset.append(boreholes)
# 坐标缩放
geodataset.points_trans_scale(scale=[0.005, 0.005, 0.01])

# 设置钻孔控制半径范围
geodataset.geodata_list[0].set_boreholes_control_buffer_dist_xy(radius=10)
label_dict = {"1_Base_Tommy": 0, "2_Base_Isa": 1, "3_Base_Soldiers_Cap": 2, "4_Base_Calvert": 3, "5_Base_Quilalar": 4
              , "7_Base_Bulonga": 5, "8_Base_Leichhardt": 6, "9_Base_L_Volcs": 7, "Williams_Naraku_Granites": 8}
geodataset.standardize_labels(label_dict=label_dict)

pl = control_visibility_with_layer_label(geo_object_list=[geodataset.geodata_list[0]])
pl.show()
geodataset.geodata_list[0].show()
if __name__ == "__main__":
    model_idx = 0

    top_points_seg = geodataset.geodata_list[0].get_top_points()
    terr_seg = TerrainData()
    # 范围约束，按照剖面线缓冲，生成线状建模区域
    terr_seg.set_control_points(top_points_seg)

    # 注意网格尺寸参数 grid_cell_density，若设置过小，计算机可能无法处理
    gme_models = GmeModelGraphList('gme_model', root=root_path,
                                   input_sample_data=geodataset,  #
                                   add_inverse_edge=True,
                                   terrain_data=terr_seg,
                                   grid_cell_density=[1, 1, 2],
                                   val_ratio=0.2,
                                   is_regular=True,
                                   update_graph=False)  # 不规则网格
    gme_models.load_geograph(graph_id=0)

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




