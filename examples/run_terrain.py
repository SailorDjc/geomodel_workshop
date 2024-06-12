from data_structure.terrain import TerrainData
from data_structure.reader import ReadExportFile
from pyvista import examples
import pyvista as pv
import numpy as np
from utils.vtk_utils import create_vtk_grid_by_rect_bounds
import os
import random
from data_structure.points import PointSet


# 本实例给出了地形对象的使用流程，1部分通过随机生成的高程点来生成地形曲面
# 2部分给出了通过tiff格式dem文件生成地形曲面的过程，读者可以换用自己的dem文件路径
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
    # 1 随机高程点生成地形
    control_points = create_3d_surface_algo_samples()
    terrain = TerrainData()
    # 设置高程控制点
    terrain.set_control_points(control_points=PointSet(control_points))
    # 执行地形曲面插值操作
    terrain.execute()
    # 地形曲面可视化
    terrain.vtk_data.plot()
    # 生成有一定深度的地形体素网格，注意cell_density设置过小会导致计算很慢，甚至计算失败。
    grid_terrain = terrain.create_grid_from_terrain_surface(z_min=-80, cell_density=2, is_smooth=True)
    # 可视化地形网格
    grid_terrain.plot()

    # 2 dem文件生成地形
    # 建模范围，会根据平面范围，对dem数据进行裁剪
    mask_bounds = [555000, 556000, 3501000, 3502000, -100, 579]
    tiff_path = r"E:\MyDataset\ASTGTMV003_N31E117\ASTGTMV003_N31E117_dem.tif"

    terrain = TerrainData()
    # 输入tiff文件
    terrain.set_input_tiff_file(file_path=tiff_path)
    # 设置裁剪范围
    terrain.set_boundary(mask_bounds=mask_bounds)
    # 执行地形曲面插值操作
    terrain.execute()
    terrain.vtk_data.plot()
    # 可以通过terrain.save()保存地形数据，下次使用可以直接加载。
    save_path = os.path.join(tiff_path, '../')

    # 生成有一定深度的地形体素网格
    grid_terrain = terrain.create_grid_from_terrain_surface(z_min=-80, cell_density=2, is_smooth=True)
    grid_terrain.plot()
