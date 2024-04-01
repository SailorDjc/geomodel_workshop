from data_structure.terrain import TerrainData
from data_structure.reader import ReadExportFile
from pyvista import examples
import pyvista as pv
import numpy as np
from utils.vtk_utils import create_vtk_grid_by_rect_bounds
import os
import random
from data_structure.points import PointSet


def create_3d_surface_algo_samples(points_num=50):
    # 创建随机的10个坐标点
    points_list = []
    for i in np.arange(points_num):
        y = random.uniform(0, 200)
        z = random.uniform(-30, 10)
        x = random.uniform(0, 200)
        points_list.append(np.array([x, y, z]))
    points_list = np.array(points_list)
    return points_list


if __name__ == '__main__':
    control_points = create_3d_surface_algo_samples()
    terrain = TerrainData()
    terrain.set_control_points(control_points=PointSet(control_points))
    terrain.execute()
    grid_terrain = terrain.create_grid_from_terrain_surface(z_min=-80, cell_density=2, is_smooth=True)
    grid_terrain.plot()
    terrain.vtk_data.plot()

    # mask_bounds = [555000, 556000, 3501000, 3502000, -100, 579]
    # tiff_path = r"E:\MyDataset\ASTGTMV003_N31E117\ASTGTMV003_N31E117_dem.tif"
    # save_path = os.path.join(tiff_path, '../')
    # terrain = TerrainData()
    # terrain.set_input_tiff_file(file_path=tiff_path)
    # terrain.set_boundary(mask_bounds=mask_bounds)
    # terrain.execute()
    # grid_terrain = terrain.create_grid_from_terrain_surface(z_min=-80, cell_density=2, is_smooth=True)
    # grid_terrain.plot()
    # terrain.vtk_data.plot()
    # grid_bounds = [555000, 556000, 3501000, 3502000, -100, 559]
    # terrain_grid = create_vtk_grid_by_rect_bounds(dim=np.array([100, 100, 80]), bounds=np.array(grid_bounds))

    # reader = ReadExportFile()
    # file_path = r"E:\MyDataset\ASTGTMV003_N31E117\terrain"
    # terrain = reader.read_geodata(file_path=file_path)

