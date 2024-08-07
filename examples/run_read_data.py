from data_structure.reader import ReadExportFile
from data_structure.geodata import *
from dgl_geodataset import DglGeoDataset
from geomodel_analysis import *
from gme_trainer import *
from models.model import GraphTransfomer
import numpy as np
import os
from utils.plot_utils import control_visibility_with_layer_label, visual_multiple_model
from utils.math_libs import points_trans_scale
import pyvista as pv
import geopandas as gpd

# tar_files = os.listdir(self.raw_dir_path)
#     data_list = []
#     for tar_file in tar_files:
#         if tar_file.endswith('.tar'):
#             file = tar_file.replace('.tar', '')
#             data_list.append(file)
#     data_list = list(set(data_list))
#     return data_list

root_dir = os.path.abspath('..')
if __name__ == '__main__':
    # grid_data_path = os.path.join(root_dir, 'processed', 'tmp_graph1722932905', 'gme_base_grid', 'gme_base_grid.dat')
    # reader = ReadExportFile()
    # grid_data = reader.read_geodata(file_path=grid_data_path)
    # grid_data.points_transform(points_trans_scale, factor=[200, 200, 100])
    # grid_data.plot()

    plotter = pv.Plotter()

    interface_data_dir = os.path.join(root_dir, 'data', 'original_interface')
    interface_file_list = os.listdir(interface_data_dir)
    interface_data_list = []
    for interface_file in interface_file_list:
        if interface_file.endswith('.dxf'):
            interface_data_list.append(os.path.join(interface_data_dir, interface_file))

    for file_path in interface_data_list:
        gdf = gpd.read_file(file_path)
        a = gdf['geometry'][0].exterior
        b = gdf.loc[0].wkt
        # gdf_line =


    plotter.show()
