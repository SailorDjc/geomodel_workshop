from dgl_geodataset import DglGeoDataset
from geomodel_analysis import *
from gme_trainer import GmeTrainer, GmeTrainerConfig, GraphTransConfig
from models.model import GraphTransfomer, SAGETransfomer
from data.retrieve_noddy_files import NoddyModelData
from utils.plot_utils import *
if __name__ == '__main__':
    print('Loading data')

    root_path = os.path.abspath('..')
    process_dir = os.path.join(root_path, '小论文实验', '虚拟数据2', '训练过程记录')
    gme_models = GmeModelGraphList('gme_model', root=root_path, processed_dir=process_dir, dir_path=process_dir)
    dataset = DglGeoDataset(gme_models)

    ckpt_path = os.path.join(root_path, '小论文实验','虚拟数据2','1正常损失-论文核函数', 'latest_tran.pth')
    trainer_config = GmeTrainerConfig(max_epochs=2500, batch_size=512, num_workers=4, learning_rate=1e-4,
                                      ckpt_path=ckpt_path,
                                      output_dir=os.path.join(root_path, 'output'),
                                      out_put_grid_file_name=os.path.join(gme_models.processed_dir, 'output_model_1'),
                                      sample_neigh=[10, 10, 15, 15])
    g = dataset[0]
    in_size = g.ndata['feat'].shape[-1]
    print("in_size:", in_size)
    # 输出类别数
    out_size = dataset.num_classes['labels'][0]

    # 模型结构相关参数
    model_config = GraphTransConfig(coors=3, in_size=in_size, out_size=out_size, n_head=4, n_embd=512, gnn_layer_num=4,
                                    gnn_n_head=3, n_layer=3)
    # 构建预测模型
    model = GraphTransfomer(model_config)

    # model = SAGETransfomer(model_config)
    trainer = GmeTrainer(model, dataset, trainer_config)
    trainer.train(data_split_idx=0, has_test_label=True, only_inference=True, early_stop_patience=100)