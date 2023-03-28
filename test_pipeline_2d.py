import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from dgl_geodataset import DglGeoDataset
from gme_model_generate import GmeModelList
import tqdm
import argparse
import pyvista as pv
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
    # for item in pre_train_model_list[0:4]:
    #     mesh = noddyData.get_grid_model(item)  # 1, 6
    #     mesh.plot()
    model_idx = 1
    # 只有第一次输入的noddy_model可以用到，之后代码会自动加载数据缓存
    gme_models = GmeModelList('gme_model', root=root_path, noddy_data=noddyData)
    # gme_models.generate_section_grid(model_idx=0, scroll_scale=0.3, is_save=True)

    # gme_models.predict_with_machine_learning_method(model_idx=model_idx, method='rf')
    # gme_models.predict_with_interpolate(model_idx=model_idx, method='rbf', iso_list=[0, 1, 2, 3, 4, 5])
    gme_models.process_external_fields_model(intput_file_path=r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\rbf.csv",
                                             model_idx=1, iso_list=[0, 1, 2, 3, 4, 5])
    # section_geodata = gme_models.geodata[model_idx]
    # # section_geodata.export_drill_dict_dat_file(file_path=r'E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\train_data.dat')
    #
    # section_geodata.extract_drills_layer_points()
    # train_idx = section_geodata.extract_drills_layer_points()
    # points = section_geodata.grid_points
    # labels = section_geodata.grid_point_label
    # train_points = points[train_idx]
    # train_labels = labels[train_idx]
    # test_idx = list(set(np.arange(len(points))) - set(train_idx))
    # test_points = points[test_idx]
    # test_labels = labels[test_idx]
    #
    # mvk.export_points_labels_dat_file(
    #     file_path=r'E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\train_data.dat',
    #     test_points=train_points, test_labels=train_labels)


    # edges_2d = mvk.visual_edge_list(edge_list=section_geodata.edge_list, edge_points=section_geodata.grid_points, is_show=False)
    # drills_points = section_geodata.grid_points[section_geodata.train_idx]
    # drills_points_label = section_geodata.grid_point_label[section_geodata.train_idx]
    #
    # dps = pv.PolyData(drills_points)
    # dps.point_data['stratum'] = drills_points_label
    #
    # sphere = pv.Sphere(radius=25, phi_resolution=10, theta_resolution=10)
    # pc = dps.glyph(scale=False, geom=sphere, orient=False)
    #
    # plotter = pv.Plotter()
    # plotter.add_mesh(pc)
    # plotter.add_mesh(edges_2d)
    # # plotter.add_mesh(section_geodata.sample_grid)
    # plotter.camera_position = [1, 0, 0]
    # plotter.show()

    dataset = DglGeoDataset(gme_models)

    save_path_1 = r'E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\vtk_model.vtk'
    trainer_config = GmeTrainerConfig(max_epochs=1900, batch_size=512, num_workers=4, learning_rate=1e-3,
                                      ckpt_path=os.path.join(root_path, 'processed', 'latest_tran.pth'),
                                      output_dir=os.path.join(root_path, 'output'),
                                      out_put_grid_file_name=save_path_1,
                                      sample_neigh=[10, 10, 15, 15])
    g = dataset[model_idx]
    # drills = mvk.visual_sample_data(section_geodata, is_show=False, drill_radius=25)
    # mvk.visual_multiple_model([section_geodata.sample_grid, drills[0]], [section_geodata.sample_grid, pc], camera=[1, 0, 0])

    geodata = dataset.dataset.geodata[model_idx]
    in_size = g.ndata['feat'].shape[-1]
    print("in_size:", in_size)
    out_size = dataset.num_classes['labels'][model_idx]

    model_config = GraphTransConfig(in_size=in_size, n_head=4, n_embd=512, out_size=out_size, gnn_layer_num=4,
                                    gnn_n_head=3, n_layer=3)
    model = GraphTransfomer(model_config)
    trainer = GmeTrainer(model, dataset, trainer_config)
    print('Training...')

    trainer.train(data_split_idx=model_idx)
