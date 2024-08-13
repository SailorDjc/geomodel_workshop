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
from tqdm import tqdm

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
    reader = ReadExportFile()
    # original_interface文件夹里存放了一系列.dxf文件
    interface_data_dir = os.path.join(root_dir, 'data', 'original_interface')
    grid_data_path = os.path.join(interface_data_dir, 'gme_base_grid', 'gme_base_grid.dat')
    grid_data = reader.read_geodata(file_path=grid_data_path)
    grid_data.points_transform(points_trans_scale, factor=[200, 200, 100])
    # grid_data.vtk_data.plot()
    interface_file_list = os.listdir(interface_data_dir)  # 获取文件夹下所有文件
    interface_data_list = []
    # 获取所有面的文件路径
    for interface_file in interface_file_list:
        if interface_file.endswith('.dxf'):
            interface_data_list.append(os.path.join(interface_data_dir, interface_file))
    poly_surface_list = []
    # 把之前的属性数据清空
    grid_data.vtk_data.clear_cell_data()
    # 为网格cell赋值，初始值为-1
    cells_series = np.full((len(grid_data.grid_points),), fill_value=-1)
    grid_data.vtk_data.cell_data['Scalar Field'] = cells_series
    plotter = pv.Plotter()
    plotter.add_mesh(grid_data.vtk_data, opacity=0.5)
    print('正在读取.dxf层面数据')
    pbr = tqdm(enumerate(interface_data_list), total=len(interface_data_list))
    for it, file_path in pbr:
        surf = read_dxf_surface(geom_file_path=file_path)
        plotter.add_mesh(surf)
        poly_surface_list.append(surf)
    plotter.show()
    plotter_1 = pv.Plotter()
    pbr = tqdm(enumerate(poly_surface_list), total=len(poly_surface_list))
    for it, poly_surf in pbr:
        # 筛选出与surf面相交的cell的id号
        plotter_1.add_mesh(poly_surf)
        cell_ids = poly_surf_intersect_with_grid(poly_surf=poly_surf, grid=grid_data.vtk_data)
        cells_series[cell_ids] = it
        pp = grid_data.vtk_data.extract_cells(ind=cell_ids)
        plotter_1.add_mesh(pp, opacity=0.5)
        plotter_1.show()
    grid_data.vtk_data.cell_data['Scalar Field'] = cells_series
    # 保存grid文件
    grid_data.save(dir_path=interface_data_dir, out_name='new_model')
    # surface_a.points = points_trans_scale(points=surface_a.points, t_factor=[0.005, 0.005, 0.01], center=grid_data.center)


    #
    #
    # grid_data.vtk_data.cell_data['Scalar Field'] = cells_series

    # plotter = pv.Plotter()
    # plotter.add_mesh(pp, opacity=0.5)
    # plotter.add_mesh(surface_a)
    # plotter.show()
    #
    # ff = surface_a.get_cell(index=0)
    # f1 = surface_a.extract_surface()
    # surface_a.plot()



        # a.save()
        # polygon = gdf['geometry'][0]
        # if isinstance(polygon, sy.Polygon):
        #     aa = type(polygon)
        #     print(aa)
        # exterior = polygon.exterior
        # interior = polygon.interiors.coords
        # gdf_line =
    plotter.show()
