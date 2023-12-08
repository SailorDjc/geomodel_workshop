import os.path
import numpy as np
from dgl_geodataset import DglGeoDataset
from geomodel_analysis import GmeModelGraphList
from gme_trainer import GmeTrainer, GmeTrainerConfig, GraphTransConfig
from models.model import GraphTransfomer, GraphModel, SAGEModel, SAGETransfomer, GraphTransfomerNet
from retrieve_noddy_files import NoddyModelData
from data_structure.reader import ReadExportFile
from data_structure.grids import Grid
from data_structure.data_sampler import GeoGridDataSampler

if __name__ == '__main__':
    # load and preprocess dataset
    print('Loading data')
    root_path = os.path.abspath('..')
    file_path = os.path.join(root_path, 'data', 'origin_borehole_data.dat')
    # 从外部数据文本中加载钻孔数据
    reader = ReadExportFile()
    boreholes_data = reader.read_boreholes_data_from_text_file(dat_file_path=file_path)

    model_idx = 0

    # 将三维模型规则网格数据构建为图网格数据
    # 这里采用真实钻孔，不采用虚拟钻孔采样
    gme_models = GmeModelGraphList('gme_model', root=root_path,
                                   sample_data=[boreholes_data],
                                   add_inverse_edge=True)

    dataset = DglGeoDataset(gme_models)

    # initialize a trainer instance and kick off training
    # 模型训练相关参数    初始训练参数的设置
    trainer_config = GmeTrainerConfig(max_epochs=5000, batch_size=512, num_workers=4, learning_rate=1e-4,
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
