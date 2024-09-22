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

root_dir = os.path.abspath('..')
if __name__ == '__main__':
    reader = ReadExportFile()
    # original_interface文件夹里存放了一系列.dxf文件
    interface_data_dir = os.path.join(root_dir, 'data', 'original_interface')
    # plot = pv.Plotter()
    # 填充体素模型
    grid_path = os.path.join(interface_data_dir, 'result_grid', 'result_grid.dat')   # surface_list_model
    # 层面体素模型
    surf_grid = reader.read_geodata(file_path=os.path.join(interface_data_dir, 'out_grid_0', 'out_grid_0.dat'))
    out_grid = reader.read_geodata(file_path=grid_path)
    # out_grid_1 = reader.read_geodata(file_path=os.path.join(interface_data_dir, 'out_grid', 'out_grid.dat'))
    # result_grid = fill_cell_values_with_surface_grid(out_grid)
    # result_grid.save(dir_path=interface_data_dir, out_name='result_grid')
    # result_grid = reader.read_geodata(os.path.join(interface_data_dir, 'result_model', 'result_model.dat'))
    plot = control_visibility_with_layer_label([surf_grid, out_grid])  # , result_grid

    interface_file_list = os.listdir(interface_data_dir)  # 获取文件夹下所有文件
    interface_data_list = []
    # 获取所有面的文件路径
    for interface_file in interface_file_list:
        if interface_file.endswith('.dxf'):
            interface_data_list.append(os.path.join(interface_data_dir, interface_file))
    poly_surface_list = []
    pbr = tqdm(enumerate(interface_data_list), total=len(interface_data_list))
    for it, file_path in pbr:
        surf = read_dxf_surface(geom_file_path=interface_data_list[it])
        poly_surface_list.append(surf)
        plot.add_mesh(surf)
    plot.show()
    # result_grid.save(dir_path=interface_data_dir, out_name='result_model')
    # grid_data_path = os.path.join(interface_data_dir, 'gme_base_grid', 'gme_base_grid.dat')
    # grid_data = reader.read_geodata(file_path=grid_data_path)
    # grid_data.points_transform(points_trans_scale, factor=[200, 200, 100])

    # 把之前的属性数据清空
    # grid_data.vtk_data.clear_cell_data()
    # 为网格cell赋值，初始值为-1
    # cells_series = np.full((len(grid_data.grid_points),), fill_value=-1)
    # grid_data.vtk_data.cell_data['Scalar Field'] = cells_series
    print('正在读取.dxf层面数据')
    # ##
    cells_series = out_grid.vtk_data.cell_data['Scalar Field']
    pbr_1 = tqdm(enumerate(interface_data_list), total=len(interface_data_list))
    label_dict = {"1_Base_Tommy": 0, "2_Base_Isa": 1, "3_Base_Soldiers_Cap": 2, "4_Base_Calvert": 3,
                  "5_Base_Quilalar": 4, "7_Base_Bulonga": 5, "8_Base_Leichhardt": 6, "9_Base_L_Volcs": 7,
                  "Williams_Naraku_Granites": 8}
    for it, poly_surf in pbr_1:
        # 筛选出与surf面相交的cell的id号
        file_name = os.path.basename(interface_data_list[it])
        layer_name = os.path.splitext(file_name)[0]
        if label_dict[layer_name] == 0:
            cell_ids = poly_surf_intersect_with_grid(poly_surf=poly_surface_list[it], grid=out_grid.vtk_data,
                                                     check_level=0)
            # cell_ids = np.array(cell_ids, dtype=int)

            # origin_series = cells_series[cell_ids]
            # conflict_cells = np.argwhere((origin_series != -1) & (origin_series != 1)).flatten()
            # conflict_cells = cell_ids[conflict_cells]
            # cell_ids = list((set(cell_ids) - set(conflict_cells)))
            # 寻找这些冲突单元的上方的邻近单元格
            # rectify_cells = []
            # for cell_id in conflict_cells:
            #     neighbors = out_grid.vtk_data.cell_neighbors(cell_id, connections='faces')
            #     cp = out_grid.vtk_data.get_cell(cell_id).center
            #     z = cp[2]
            #     for n_c in neighbors:
            #         n_cp = out_grid.vtk_data.get_cell(n_c).center
            #         n_z = n_cp[2]
            #         if n_z > z and n_c not in conflict_cells:
            #             label = cells_series[n_c]
            #             if label == -1:
            #                 rectify_cells.append(n_c)

            # cells_0 = out_grid.vtk_data.extract_cells(conflict_cells)
            # cells_1 = out_grid.vtk_data.extract_cells(rectify_cells)
            # plotter = pv.Plotter()
            # plotter.add_mesh(cells_0, color='red')
            # plotter.add_mesh(cells_1, color='blue')
            # plotter.show()
            # cell_ids.extend(rectify_cells)
            cells_series[cell_ids] = label_dict[layer_name]
    out_grid.vtk_data.cell_data['Scalar Field'] = cells_series
    # 保存grid文件
    out_grid.save(dir_path=interface_data_dir, out_name='out_grid_0')   # surface_list_model
