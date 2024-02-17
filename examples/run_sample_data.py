from data.retrieve_noddy_files import NoddyModelData
from data_structure.sections import Section, SectionSet, PointSet, BoreholeSet, Borehole
from data_structure.grids import Grid
import numpy as np
import random
import pandas as pd
import os


# 创建二维曲线插值的样本
def create_2d_curve_algo_samples(is_sort=False):
    x = 5
    # 创建随机的10个坐标点
    points_list = []
    for i in np.arange(10):
        y = random.uniform(0, 50)
        z = random.uniform(-30, 10)
        points_list.append(np.array([x, y, z]))
    points_list = np.array(points_list)
    # 以y为主轴进行排序
    if is_sort:
        p_ids = np.lexsort((points_list[:, 1], points_list[:, 2]))
        points_list = points_list[p_ids]
    return points_list


# 创建插值样本点
def create_3d_surface_algo_samples(points_num=50):
    # 创建随机的10个坐标点
    points_list = []
    for i in np.arange(points_num):
        y = random.uniform(0, 50)
        z = random.uniform(-30, 10)
        x = random.uniform(0, 50)
        points_list.append(np.array([x, y, z]))
    points_list = np.array(points_list)
    return points_list


if __name__ == "__main__":
    root_path = os.path.abspath('..')
    points_3d = create_3d_surface_algo_samples()
    export_data = pd.DataFrame(points_3d)
    export_data.to_csv(os.path.join(root_path, 'output', 'points_3d.dat'), index=False, header=False, sep='\t')

    noddyData = NoddyModelData(root=r'E:\NoddyDataset', dataset_list=['FOLD_FOLD_FOLD'], max_model_num=10,
                               update_grid=False)
    noddy_grid = noddyData.get_grid_model_by_idx(dataset='FOLD_FOLD_FOLD', idx=0)  # 1 6
    # 数据重采样，三维模型的尺寸是[150, 150, 120]
    grid = Grid(grid_vtk=noddy_grid, name='GeoGrid')
    grid.resample_regular_grid(dim=np.array([150, 150, 100]))
