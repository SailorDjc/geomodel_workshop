import numpy as np
from sklearn.neighbors import NearestNeighbors
import pyvista as pv
import scipy.spatial as spt
import copy
from sklearn.cluster import DBSCAN
import time
import os

from typing import List


def compute_nearest_neighbor_dist_from_pts(coords: np.ndarray):
    neighbors = NearestNeighbors(n_neighbors=2, algorithm='brute').fit(coords)
    neigh_dist, indices = neighbors.kneighbors(coords)
    neigh_dist = neigh_dist[:, 1]
    return neigh_dist


def merge_bounds(bounds_a: np.ndarray, bounds_b: np.ndarray):
    assert len(bounds_a) == 6, "bounds size should be 6"
    assert len(bounds_b) == 6, "bounds size should be 6"
    min_x = min(bounds_a[0], bounds_b[0])
    min_y = min(bounds_a[2], bounds_b[2])
    min_z = min(bounds_a[4], bounds_b[4])
    max_x = max(bounds_a[1], bounds_b[1])
    max_y = max(bounds_a[3], bounds_b[3])
    max_z = max(bounds_a[5], bounds_b[5])
    bounds = np.array([min_x, max_x, min_y, max_y, min_z, max_z])
    return bounds


# 获取点云数据集的包围盒
# Parameters: coords 输入是np.array 的二维数组，且坐标是3D坐标
# xy_buffer z_buffer 为大于0的浮点数，是包围盒的缓冲距离
def get_bounds_from_coords(coords: np.ndarray, xy_buffer=0, z_buffer=0):
    assert coords.ndim == 2, "input coords array is not 2D array"
    assert coords.shape[1] == 3, "input coords are not 3D"

    coord_min = coords.min(axis=0)
    coord_max = coords.max(axis=0)

    x_min = coord_min[0]
    x_max = coord_max[0]
    y_min = coord_min[1]
    y_max = coord_max[1]
    z_min = coord_min[2]
    z_max = coord_max[2]
    bounds = np.array([x_min, x_max, y_min, y_max, z_min, z_max])
    if xy_buffer != 0 or z_buffer != 0:
        if xy_buffer == 0:
            xy_buffer = z_buffer
        if z_buffer == 0:
            z_buffer = xy_buffer
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        dz = bounds[5] - bounds[4]
        bounds[0] = bounds[0] - xy_buffer * dx
        bounds[1] = bounds[1] + xy_buffer * dx
        bounds[2] = bounds[2] - xy_buffer * dy
        bounds[3] = bounds[3] + xy_buffer * dy
        bounds[4] = bounds[4] - z_buffer * dz
        bounds[5] = bounds[5] + z_buffer * dz
    return bounds


# 合并多个点云数据集
# 输入 *datasets 是一个列表： [coords_1, corrds_2, ...]
def concat_coords_from_datasets(*datasets):
    coords_list = []
    for dataset_i in datasets:
        assert type(dataset_i) == np.ndarray, "coord dataset is not a ndarray"
        assert dataset_i.ndim == 2, "input dataset is not 2D"
        coords_list.append(dataset_i)

    all_coords = np.vstack(coords_list)
    return all_coords


class PointSet(object):
    def __init__(self, points: np.ndarray = None, point_labels: np.ndarray = None, vectors: np.ndarray = None
                 , name=None):
        self.name = name
        self.points = points
        self.labels = point_labels
        self.points_num = 0
        self.bounds = None
        if self.points is not None:
            self.points_num = self.points.shape[0]
            self.bounds = get_bounds_from_coords(self.points)
        # 矢量
        self.vectors = vectors
        # 标量
        self.scalars = None  # dict
        self.scalars_grad = None  # dict
        self.scalars_grad_norm = None  # dict

        self.vtk_vector_data = None
        self.vtk_point_data = None

        self.color_map = {}  # 颜色字典，key为label  self.set_color_with_label=True
        self.scalar_color_map = {}  # 根据scalar的值范围确定颜色  self.set_color_with_label=False
        self.color_mode = 1  # set_color_with_label
        self.epsilon = 0.00001  # 足够小，作为距离阈值
        # vtk数据唯一性编码
        self.tmp_dump_str = 'tmp' + str(int(time.time()))
        self.buffer_dist = 5  # 点控制缓冲半径

    def is_empty(self):
        if self.points is not None and self.labels is not None:
            if self.points.shape[0] > 0:
                return False
        return True

    # 使用append方法，合并Pointset，会导致scalars和vectors丢失
    def append(self, item):
        if isinstance(item, PointSet):
            if not item.is_empty():
                new_item = self.points_data_merge([self, item])
                self.points_num += new_item.points_num
                self.bounds = get_bounds_from_coords(new_item.points)
                self.set_points(new_item.points)
                self.set_labels(new_item.labels)
                self.scalars = new_item.scalars
                self.scalars_grad = new_item.scalars_grad
                self.scalars_grad_norm = None
            else:
                raise ValueError('Neither of the two merged objects can be null.')

    @staticmethod
    def points_data_merge(points_data_list: list):
        # 要合并的PointSet对象要先进行数据检查，确保合并共有属性
        points_merge = []
        labels_merge = []
        vectors_merge = []
        scalras_merge = []
        scalars_grad_merge = []
        for data in points_data_list:
            if not isinstance(data, PointSet):  # and data.points_num == 0:
                raise ValueError('Input data should be PointSet type.')
            else:
                if not data.is_empty():
                    points_merge.append(data.points)
                    if data.labels is not None:
                        labels_merge.append(data.labels)
                    if data.vectors is not None:
                        vectors_merge.append(data.vectors)
                    if data.scalars is not None:
                        scalras_merge.append(data.scalars)
                    if data.scalars_grad is not None:
                        scalars_grad_merge.append(data.scalars_grad)
        if len(points_data_list) == 0:
            raise ValueError('Input list is empty.')
        else:
            if len(points_merge) == 0:
                raise ValueError('Input list is empty.')
            merge_num = len(points_merge)
            points_merge = np.vstack(points_merge)
            if len(labels_merge) == merge_num:
                labels_merge = np.hstack(labels_merge)
            else:
                labels_merge = None
            points_data_merge = PointSet(points=points_merge, point_labels=labels_merge)
            if len(vectors_merge) == merge_num:
                vectors_merge = np.vstack(vectors_merge)
                points_data_merge.set_vectors(vectors=vectors_merge)
            if len(scalras_merge) == merge_num:
                keys_map = scalras_merge[0].keys()
                scalars_common_merge = {}
                for key in keys_map:
                    scalars_common_merge[key] = []
                    for item in scalras_merge:
                        if key in item.keys():
                            scalars_common_merge[key].append(item[key])
                        else:
                            scalars_common_merge.pop(key)  # 只要有一个PointSet中没有相应属性，则该属性不保留
                            break
                for key in scalars_common_merge.keys():
                    scalars_value = np.vstack(scalars_common_merge[key])
                    points_data_merge.set_scalars(scalars=scalars_value, scalar_name=key)
            if len(scalars_grad_merge) == merge_num:
                keys_map = scalars_grad_merge[0].keys()
                scalars_grad_common_merge = {}
                for key in keys_map:
                    scalars_grad_common_merge[key] = []
                    for item in scalars_grad_merge:
                        if key in item.keys():
                            scalars_grad_common_merge[key].append(item[key])
                        else:
                            scalars_grad_common_merge.pop(key)  # 只要有一个PointSet中没有相应属性，则该属性不保留
                            break
                for key in scalars_grad_common_merge.keys():
                    scalars_grad_value = np.vstack(scalars_grad_common_merge[key])
                    points_data_merge.set_scalars_grad(scalars_grad=scalars_grad_value, scalar_name=key)
            return points_data_merge

    # 去除重复点， 距离小于阈值可以认为是重复点
    def remove_duplicate_points(self):
        clustering = DBSCAN(eps=self.epsilon, min_samples=1).fit(self.points)
        unique_indexes = np.unique(clustering.labels_, return_index=True)[1]
        self.points = self.points[unique_indexes]
        self.labels = self.labels[unique_indexes]
        if self.vectors is not None:
            self.vectors = self.vectors[unique_indexes]
        if self.scalars is not None and isinstance(self.scalars, dict):
            for k, v in self.scalars.keys():
                self.scalars[k] = self.scalars[k][unique_indexes]
        if self.scalars_grad is not None and isinstance(self.scalars_grad, dict):
            for k, v in self.scalars_grad.keys():
                self.scalars_grad[k] = self.scalars_grad[k][unique_indexes]
        if self.scalars_grad_norm is not None and isinstance(self.scalars_grad_norm, dict):
            for k, v in self.scalars_grad_norm.keys():
                self.scalars_grad_norm[k] = self.scalars_grad_norm[k][unique_indexes]

    def generate_vtk_data_for_points_as_sphere(self):
        if self.points is None:
            return None
        points_poly = pv.PolyData(self.points)
        if self.labels is not None and len(self.labels) == self.points.shape[0]:
            points_poly["stratum"] = self.labels
        self.vtk_point_data = points_poly
        return self.vtk_point_data

    def generate_vtk_data_for_vector_as_arrow(self):
        if self.vectors is None:
            return None
        vector_data = pv.vector_poly_data(self.points, self.vectors)
        # 'mag' 是矢量的大小， 'vectors' 表示矢量向量
        vector_plot = vector_data.glyph(orient='vectors', scale='mag')
        self.vtk_vector_data = vector_plot
        return self.vtk_vector_data

    def show(self, is_sphere=True, point_size=5, is_arrow=False):
        plotter = pv.Plotter()
        if is_sphere is True and self.points is not None:
            points_poly = self.generate_vtk_data_for_points_as_sphere()
            plotter.add_mesh(points_poly, render_points_as_spheres=True, point_size=point_size)
        if is_arrow is True and self.vectors is not None:
            vector_poly = self.generate_vtk_data_for_vector_as_arrow()
            plotter.add_mesh(vector_poly)
        plotter.show()

    def transform_points(self, scale):
        assert self.points is not None, "there are no points"
        self.points = scale.transform(self.points)

    def transform_vectors(self, scale):
        assert self.vectors is not None, "there are no vectors"
        self.vectors = scale.transform(self.vectors)

    # 缓冲半径
    def set_points_control_buffer_dist(self, radius: float):
        if radius > 0:
            self.buffer_dist = radius

    def set_points(self, points: np.ndarray):
        self.points = points
        self.points_num = points.shape[0]
        self.bounds = get_bounds_from_coords(self.points)

    def set_vectors(self, vectors: np.ndarray):
        if self.points is None:
            raise ValueError('Need to set points first.')
        if self.points.shape[0] != vectors.shape[0]:
            raise ValueError('Vectors array size not match to points.')
        self.vectors = vectors

    def set_labels(self, labels: np.ndarray):
        if self.points is None:
            raise ValueError('Need to set points first.')
        if labels is not None:
            if self.points.shape[0] != len(labels):
                raise ValueError('labels array size not match to points.')
        self.labels = labels

    def set_scalars(self, scalars: np.ndarray, scalar_name: str):
        if self.points is None:
            raise ValueError('Need to set points first.')
        if self.points.shape[0] != len(scalars):
            raise ValueError('Scalars array size not match to points.')
        if self.scalars is None:
            self.scalars = {}
        self.scalars[scalar_name] = scalars

    def set_scalars_grad(self, scalars_grad: np.ndarray, scalar_name: str):
        if self.points is None:
            raise ValueError('Need to set points first.')
        if self.points.shape[0] != len(scalars_grad):
            raise ValueError('Scalars_grad array size not match to points.')
        if self.scalars_grad is None:
            self.scalars_grad = {}
        self.scalars_grad[scalar_name] = scalars_grad

    def set_scalars_grad_norm(self, scalars_grad_norm: np.ndarray, scalar_name: str):
        if self.points is None:
            raise ValueError('Need to set points first.')
        if self.points.shape[0] != len(scalars_grad_norm):
            raise ValueError('Scalars_grad array size not match to points.')
        if self.scalars_grad_norm is None:
            self.scalars_grad_norm = {}
        self.scalars_grad_norm[scalar_name] = scalars_grad_norm

    # 获取点集合凸包，z值无效
    def get_convexhull_2d(self) -> np.ndarray:  # 3D points array
        if self.points is None:
            raise ValueError('Need to set points first.')
        points_2d = self.points[0: 2]
        hull = spt.ConvexHull(points_2d)
        simplex_idx = []
        for simplex in hull.simplices:
            simplex_idx.extend(list(simplex))
        unique_idx = list(np.unique(np.int64(simplex_idx)))
        # 对凸包线进行排序，点序列按邻接关系顺序排列
        convex_hull_dict = {}  # 点id为key，记录每个点的邻接点
        for simplex in hull.simplices:
            item_0, item_1 = simplex[0], simplex[1]
            if item_0 not in convex_hull_dict.keys():
                convex_hull_dict[item_0] = []
            if item_1 not in convex_hull_dict.keys():
                convex_hull_dict[item_1] = []
            convex_hull_dict[item_0].append(item_1)
            convex_hull_dict[item_1].append(item_0)
        # 随机选一个点作为起点， 构建线列表
        line_pnt_idx_front = unique_idx[0]  # 起始点
        line_pnt_idx = [line_pnt_idx_front]  # 索引列表 # 起始点先加入序列，从起始点开始往一个方向搜索
        for lit in np.arange(len(unique_idx)):
            strip_0 = convex_hull_dict[line_pnt_idx[lit]]  # 获取当前点的邻接点
            # 若是最后一个点，应为起始点，结束
            if lit == len(unique_idx) - 1:
                if line_pnt_idx_front not in strip_0:
                    raise ValueError('ConvexHull computation happens error.')
                break
            if strip_0[0] not in line_pnt_idx:
                line_pnt_idx.append(strip_0[0])
            else:
                line_pnt_idx.append(strip_0[1])
        # 传出的点序列没有重复点，每个点顺序连接成线
        return copy.deepcopy(self.points[line_pnt_idx])

    def __getitem__(self, idx):
        return self.points[idx]

    def save(self, dir_path: str):
        save_path = os.path.join(dir_path, self.tmp_dump_str)
        if self.vtk_point_data is not None and isinstance(self.vtk_point_data, pv.PolyData):
            self.vtk_point_data.save(filename=save_path + '_p.vtk')
            self.vtk_point_data = 'dumped'
        if self.vtk_vector_data is not None and isinstance(self.vtk_vector_data, pv.PolyData):
            self.vtk_vector_data.save(filename=save_path + '_v.vtk')
            self.vtk_vector_data = 'dumped'

    def load(self, dir_path: str):
        save_path = os.path.join(dir_path, self.tmp_dump_str)
        if self.vtk_point_data == 'dumped':
            if os.path.exists(save_path + '_p.vtk'):
                self.vtk_point_data = pv.read(filename=save_path + '_p.vtk')
            else:
                raise ValueError('vtk data file does not exist')
        if self.vtk_vector_data == 'dumped':
            if os.path.exists(save_path + '_v.vtk'):
                self.vtk_vector_data = pv.read(filename=save_path + '_v.vtk')
            else:
                raise ValueError('vtk data file does not exist')


if __name__ == "__main__":
    points = np.random.randint(10, 40, (10, 3))
    vectors = np.ones_like(points)
    points_data = PointSet(points=points, vectors=vectors)
    points_data.generate_vtk_data_for_vector_as_arrow()
