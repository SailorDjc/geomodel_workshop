import numpy as np
from sklearn.neighbors import NearestNeighbors
import pyvista as pv
import scipy.spatial as spt
import copy
from sklearn.cluster import DBSCAN
import time
import os


def compute_nearest_neighbor_dist_from_pts(coords: np.ndarray):
    neighbors = NearestNeighbors(n_neighbors=2, algorithm='brute').fit(coords)
    neigh_dist, indices = neighbors.kneighbors(coords)
    neigh_dist = neigh_dist[:, 1]
    return neigh_dist


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
    def __init__(self, points: np.ndarray = None, point_labels: np.ndarray = None, vectors: np.ndarray = None):
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
        self.scalars = None   # dict
        self.scalars_grad = None  # dict
        self.scalars_grad_norm = None # dict

        self.vtk_vector_data = None
        self.vtk_point_data = None

        self.color_map = {}  # 颜色字典，key为label  self.set_color_with_label=True
        self.scalar_color_map = {}  # 根据scalar的值范围确定颜色  self.set_color_with_label=False
        self.set_color_with_label = True
        self.epsilon = 0.00001  # 足够小，作为距离阈值

        self.tmp_dump_str = 'tmp' + str(int(time.time()))
        self.save_path = None

    def is_empty(self):
        if self.points is not None and self.labels is not None:
            return True
        else:
            return False

    # 使用append方法，合并Pointset，会导致scalars和vectors丢失
    def append(self, item):
        if isinstance(item, PointSet):
            if item.is_empty():
                np.append(self.points, values=item.points, axis=0)
                np.append(self.labels, values=item.labels, axis=0)
                self.points_num += item.points_num
                self.bounds = get_bounds_from_coords(self.points)

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
        points_data = pv.PolyData(self.points)
        self.vtk_point_data = points_data
        return self.vtk_point_data

    def generate_vtk_data_for_vector_as_arrow(self):
        vector_data = pv.vector_poly_data(self.points, self.vectors)
        # 'mag' 是矢量的大小， 'vectors' 表示矢量向量
        vector_plot = vector_data.glyph(orient='vectors', scale='mag')
        self.vtk_vector_data = vector_plot
        return self.vtk_vector_data

    def transform_points(self, scale):
        assert self.points is not None, "there are no points"
        self.points = scale.transform(self.points)

    def transform_vectors(self, scale):
        assert self.vectors is not None, "there are no vectors"
        self.vectors = scale.transform(self.vectors)

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
        if self.vtk_point_data is not None and isinstance(self.vtk_point_data, pv.PolyData):
            self.save_path = os.path.join(dir_path, self.tmp_dump_str)
            self.vtk_point_data.save(filename=self.save_path+'_p.vtk')
            self.vtk_point_data = 'dumped'
        if self.vtk_vector_data is not None and isinstance(self.vtk_vector_data, pv.PolyData):
            self.save_path = os.path.join(dir_path, self.tmp_dump_str)
            self.vtk_vector_data.save(filename=self.save_path+'_v.vtk')
            self.vtk_vector_data = 'dumped'

    def load(self):
        if self.save_path is not None:
            if self.vtk_point_data is not None and os.path.exists(self.save_path+'_p.vtk'):
                self.vtk_point_data = pv.read(filename=self.save_path+'_p.vtk')
            if self.vtk_vector_data is not None and os.path.exists(self.save_path+'_v.vtk'):
                self.vtk_vector_data = pv.read(filename=self.save_path+'_v.vtk')
        else:
            raise ValueError('vtk data file does not exist')


if __name__ == "__main__":
    points = np.random.randint(10, 40, (10, 3))
    vectors = np.ones_like(points)
    points_data = PointSet(points=points, vectors=vectors)
    points_data.generate_vtk_data_for_vector_as_arrow()
