import os.path
from dgl_geodataset import DglGeoDataset
from geomodel_analysis import GmeModelGraphList
from gme_trainer import GmeTrainer, GmeTrainerConfig, GraphTransConfig
from models.model import GraphTransfomer
from data_structure.reader import ReadExportFile
from data_structure.geodata import GeodataSet
from data_structure.grids import Grid
import pandas as pd

if __name__ == '__main__':

    # load and preprocess dataset
    print('Loading data')
    root_path = os.path.abspath('..')
    file_path = os.path.join(root_path, 'data', 'origin_borehole_data.dat')
    # 从外部数据文本中加载钻孔数据
    reader = ReadExportFile()
    boreholes_data_0 = reader.read_boreholes_data_from_text_file(dat_file_path=file_path)
    # 导出顶点坐标
    top_points = boreholes_data_0.get_top_points()
    export_data = pd.DataFrame(top_points)
    export_data.to_csv(os.path.join(root_path, 'output', 'top_points.dat'), index=False, header=False, sep='\t')

    # boreholes_data_0.show()
    # 地质数据的容器, 使用钻孔或散点或剖面建模，将数据都加入到容器中，容器中的数据会联合约束地质模型的构建
    gd = GeodataSet()
    gd.append(boreholes_data_0)
    model_idx = 0
    gd.standardize_labels()
    # 将三维模型规则网格数据构建为图网格数据
    # 这里采用真实钻孔，不采用虚拟钻孔采样
    gme_models = GmeModelGraphList('gme_model', root=root_path,
                                   input_sample_data=gd,
                                   add_inverse_edge=True)

    # 训练已经结束，结果保存到  os.path.join(gme_models.processed_dir, 'vtk_model.vtk') 路径下
    from utils.plot_utils import control_visibility_with_layer_label

    # label_map = False 不进行标签标准化处理
    grid_model = Grid(grid_vtk_path=os.path.join(gme_models.processed_dir, 'vtk_model.vtk'), label_map=False)
    boreholes_data_1 = gd.geodata_list[0]  # boreholes_data,
    plotter_2 = control_visibility_with_layer_label(geo_object_list=[grid_model, boreholes_data_1], grid_smooth=False
                                                    , show_edge=False)
    plotter_2.show()

    dataset = DglGeoDataset(gme_models)

    # initialize a trainer instance and kick off training
    # 模型训练相关参数    初始训练参数的设置
    trainer_config = GmeTrainerConfig(max_epochs=2510, batch_size=512, num_workers=4, learning_rate=1e-4,
                                      ckpt_path=os.path.join(root_path, 'processed', 'latest_tran.pth'),
                                      output_dir=os.path.join(root_path, 'output'),
                                      out_put_grid_file_name=os.path.join(gme_models.processed_dir, 'vtk_model.vtk'),
                                      sample_neigh=[10, 10, 15, 15])
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

