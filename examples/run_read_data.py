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
from utils.plot_utils import *
from utils.vtk_utils import *

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
    # reader = ReadExportFile()
    # train_loss_path = os.path.join(root_dir, 'output', 'train_loss_log.txt')
    # epoch, train_loss, train_acc, train_rmse, val_loss, val_acc, val_rmse = reader.read_train_loss_log(
    #     txt_file_path=train_loss_path, sep=',')
    # visual_loss_picture(train_loss=train_loss, test_loss=val_loss)
    interface_data_dir = os.path.join(root_dir, 'data', 'original_interface')
    grid_data_path = os.path.join(interface_data_dir, 'gme_base_grid', 'gme_base_grid.dat')
    reader = ReadExportFile()
    grid_data = reader.read_geodata(file_path=grid_data_path)
    # grid_data.points_transform(points_trans_scale, factor=[200, 200, 100])
    c = grid_data.vtk_data.extract_cells([0, 5])
    n = c.n_points
    b1 = c.n_cells
    surface_a_path = os.path.join(interface_data_dir, 'surface_a.vtp')
    surface_a = reader.read_vtk_data(surface_a_path)
    surface_a.points = points_trans_scale(points=surface_a.points, t_factor=[0.005, 0.005, 0.01], center=grid_data.center)

    cell_ids = poly_surf_intersect_with_grid(poly_surf=surface_a, grid=grid_data.vtk_data)
    # cells_series = np.full((len(grid_data.grid_points),), fill_value=-1)
    # cells_series[cell_ids] = 1
    # grid_data.vtk_data.cell_data['Scalar Field'] = cells_series
    pp = grid_data.vtk_data.extract_cells(ind=cell_ids)

    plotter = pv.Plotter()
    plotter.add_mesh(pp, opacity=0.5)
    plotter.add_mesh(surface_a)
    plotter.show()

    ff = surface_a.get_cell(index=0)
    f1 = surface_a.extract_surface()
    surface_a.plot()


    # interface_file_list = os.listdir(interface_data_dir)
    # interface_data_list = []
    # for interface_file in interface_file_list:
    #     if interface_file.endswith('.dxf'):
    #         interface_data_list.append(os.path.join(interface_data_dir, interface_file))
    # import shapely as sy
    # for file_path in interface_data_list:
    #     a = read_dxf_surface(geom_file_path=file_path)
    #     a.save(os.path.join(interface_data_dir, 'surface_a.vtp'))
        # a.save()
        # polygon = gdf['geometry'][0]
        # if isinstance(polygon, sy.Polygon):
        #     aa = type(polygon)
        #     print(aa)
        # exterior = polygon.exterior
        # interior = polygon.interiors.coords
        # gdf_line =
    plotter.show()
