import math

from data_structure.grids import Grid
from data_structure.sections import SectionSet, Section
from data_structure.boreholes import BoreholeSet, Borehole
from data_structure.points import PointSet
from data_structure.geodata import GeodataSet
import numpy as np
import random
import copy
import scipy.spatial as spt
import pyvista as pv
from utils.vtk_utils import vtk_polydata_to_vtk_unstructured_grid, create_vtk_grid_by_rect_bounds, \
    create_vtk_grid_by_boundary, get_bounds_from_coords, create_closed_cylinder_surface
from utils.math_libs import remove_repeated_elements_with_lists
from data_structure.terrain import TerrainData
from sklearn.model_selection import train_test_split

random.seed(1)


# 网格采样类基类
class GeoDataSampler(object):
    def __init__(self, grid=None, default_value=-1):
        self._base_grid = grid
        self._base_grid_points = None
        self._base_grid_labels = None
        self._bounds = None
        if grid is not None:
            if isinstance(grid, Grid):
                self._base_grid_points = grid.grid_points
                self._base_grid_labels = grid.grid_points_series
                self._bounds = grid.bounds
            elif isinstance(grid, Section):
                self._base_grid_points = grid.points
                self._base_grid_labels = grid.series
                self._bounds = grid.bounds
            else:
                raise ValueError('grid type is not supported.')
        self._sample_num = 0  # 采样次数
        self.sample_data_list = []  # 采样数据
        self.sample_operator = []
        self.map_flag = False
        self.default_value = default_value
        # 设置验证集和训练集
        self.val_ratio = None  # 为None则不设置验证集，值在0和1之间
        # 钻孔、剖面或散点  {sid: {'train':[], 'val': []}}  sid对应的是self.sample_data_list中的数据索引
        self.geo_sample_data_val_map = {}
        self.train_indexes = []
        self.val_indexes = []

    def update_train_val_split_state(self, sid=-1):
        if sid < 0 or sid >= len(self.sample_data_list):
            sid = len(self.sample_data_list) - 1
        if len(self.sample_data_list) >= 1:
            if self.sample_operator[sid] == 'None':
                val_idx = []
                if isinstance(self.sample_data_list[sid], (PointSet, SectionSet, Section)):
                    points_num = self.sample_data_list[sid].get_points_num()
                    sample_data_all_idx = range(points_num)
                    if self.val_ratio is not None:
                        val_sample_num = int(self.val_ratio * points_num)
                        val_idx = list(random.sample(sample_data_all_idx, val_sample_num))
                    train_idx = [x for x in sample_data_all_idx if x not in val_idx]
                elif isinstance(self.sample_data_list[sid], BoreholeSet):
                    borehole_num = self.sample_data_list[sid].borehole_num
                    sample_data_all_idx = range(borehole_num)
                    if self.val_ratio is not None:
                        val_sample_num = int(self.val_ratio * borehole_num)
                        val_idx = list(random.sample(sample_data_all_idx, val_sample_num))
                    train_idx = [x for x in sample_data_all_idx if x not in val_idx]
                else:
                    raise ValueError('Not support.')
                self.geo_sample_data_val_map[sid] = {}
                self.geo_sample_data_val_map[sid]['train'] = train_idx
                self.geo_sample_data_val_map[sid]['val'] = val_idx
            else:
                val_idx = []
                ckt = spt.cKDTree(self._base_grid_points)
                if isinstance(self.sample_data_list[sid], (PointSet, SectionSet, Section)):
                    sample_points = self.sample_data_list[sid].get_points_data().points
                    d, pid = ckt.query(sample_points)
                    # 点数据，直接按比例切分
                    sample_data_all_idx = list(sorted(np.unique(pid)))
                    train_idx = sample_data_all_idx
                    if self.val_ratio is not None:
                        train_idx, val_idx = train_test_split(
                            sample_data_all_idx, test_size=self.val_ratio, random_state=2)
                elif isinstance(self.sample_data_list[sid], BoreholeSet):
                    borehole_num = self.sample_data_list[sid].borehole_num
                    all_sample_points = self.sample_data_list[sid].get_points_data().points
                    da, all_pid = ckt.query(all_sample_points)
                    all_idx = list(sorted(set(all_pid)))
                    train_idx = all_pid
                    # 涉及到钻孔，数据集切分需要以钻孔为单位
                    if self.val_ratio is not None:
                        val_sample_num = int(self.val_ratio * borehole_num)
                        all_borehole_idx = np.arange(borehole_num)
                        val_borehole_idx = list(random.sample(all_borehole_idx, val_sample_num))
                        val_boreholes = self.sample_data_list[sid].get_boreholes(idx=val_borehole_idx)
                        val_sample_points = val_boreholes.get_points_data().points
                        dv, val_pid = ckt.query(val_sample_points)
                        val_idx = list(sorted(set(val_pid)))
                        train_idx = list(set(all_idx) - set(val_idx))
                else:
                    raise ValueError('Not support.')
                self.geo_sample_data_val_map[sid] = {}
                self.geo_sample_data_val_map[sid]['train'] = train_idx
                self.geo_sample_data_val_map[sid]['val'] = val_idx

    def set_val_boreholes_ratio(self, val_ratio):
        if val_ratio is None:
            self.val_ratio = None
            return
        if val_ratio < 0 or val_ratio >= 1:
            raise ValueError('ratio value must between 0 and 1.')
        self.val_ratio = val_ratio

    @property
    def sample_num(self):
        self._sample_num = len(self.sample_data_list)
        return self._sample_num

    def get_sample_data(self, idx):
        if idx < 0 or idx >= len(self.sample_data_list):
            raise ValueError('The input index is out of range.')
        return self.sample_data_list[idx]

    #
    def sample_grid_for_borehole(self, pos):
        pos_a = copy.deepcopy(pos)
        pos_b = copy.deepcopy(pos)
        pos_a[2] = self._base_grid.bounds[5]  # z_max
        pos_b[2] = self._base_grid.bounds[4]  # z_min
        # 沿直线采样
        pid = self._base_grid.vtk_data.find_cells_along_line(pointa=pos_a, pointb=pos_b)
        line_points = self._base_grid.grid_points[pid]
        line_series = self._base_grid.grid_points_series[pid]
        # line_points_z = line_points[::-1, 2]
        line_points_sort_ind = np.argsort(line_points[:, 2])
        line_points = line_points[line_points_sort_ind[::-1]]
        line_series = line_series[line_points_sort_ind[::-1]]
        one_borehole = Borehole(points=line_points, series=line_series, is_virtual=True)
        return one_borehole

    # 按比例随机采样
    def random_sample_grid_for_points(self, sample_ratio=0.1):
        if self._base_grid is None:
            raise ValueError('Need to input grid first.')
        sample_points_num = int(len(self._base_grid_points) * sample_ratio)
        sample_pids = random.sample(np.arange(len(self._base_grid_points)), sample_points_num)
        sample_points = self._base_grid_points[sample_pids]
        sample_labels = self._base_grid_labels[sample_pids]
        points_data = PointSet(points=sample_points, point_labels=sample_labels)
        return points_data

    # 对网格点索引进行间隔采样
    def uniformly_interval_sample_grid_for_points(self, interval=100):
        if self._base_grid is None:
            raise ValueError('Need to input grid first.')
        pid = np.arange(0, len(self._base_grid_points), interval)
        pid = list(set(sorted(pid)))
        point_labels = self._base_grid_labels[pid]
        sample_points = self._base_grid_points[pid]
        points_data = PointSet(points=sample_points, point_labels=point_labels)
        return points_data

    # 对规则网格点进行均匀采样
    def uniformly_sample_grid_for_points(self, n_points):
        if self._base_grid is None:
            raise ValueError('Need to input grid first.')
        assert self._base_grid.dims is not None, "grid dimensions are not set"
        # proportion of points along each dimension compared to the max number of points along a particular direction
        max_n_pts_dir = max(self._base_grid.dims)
        px = self._base_grid.dims[0] / max_n_pts_dir
        py = self._base_grid.dims[1] / max_n_pts_dir
        pz = self._base_grid.dims[2] / max_n_pts_dir
        # to uniformly sample the grid to get a total of n_points WHILE also respecting the ratio of points along two
        # directions the equation is
        # (x * px) ( x * py ) * ( x * pz ) = n_points, solve for x
        x = (((max_n_pts_dir ** 3) * n_points) / (self._base_grid.dims[0] * self._base_grid.dims[1]
                                                  * self._base_grid.dims[2])) ** (1.0 / 3.0)
        nx = round(x * px)
        ny = round(x * py)
        nz = round(x * pz)
        bounds = self._base_grid.vtk_data.GetBounds()
        sample_x = np.linspace(bounds[0], bounds[1], nx)
        sample_y = np.linspace(bounds[2], bounds[3], ny)
        sample_z = np.linspace(bounds[4], bounds[5], nz)
        zz, yy, xx = np.meshgrid(sample_z, sample_y, sample_x, indexing='ij')
        sample_points = np.vstack((xx, yy, zz)).reshape(3, -1).T
        pid = self._base_grid.vtk_data.find_containing_cell(sample_points)
        point_labels = self._base_grid.grid_points_series[pid]
        points_data = PointSet(points=sample_points, point_labels=point_labels)
        return points_data

    # 在网格范围内随机采样钻孔
    def random_sample_grid_for_boreholes(self, drill_pos=None, drill_num=None, sparse_dist=10):
        if drill_num is None:
            drill_num = 10
        if self._base_grid is None:
            raise ValueError('Need to input grid first.')
        # horizon_slice = self._base_grid.vtk_data.slice(normal='z')
        if drill_pos is not None:
            drill_num = len(drill_pos)
        else:
            # 获取grid水平切面，在切面上随机选取cell的中心点，打钻孔
            # points = horizon_slice.cell_centers().points.tolist()
            points = None
            x_min, x_max, y_min, y_max, _, _ = self._bounds
            xn = math.floor((x_max - x_min) // sparse_dist)
            yn = math.floor((y_max - y_min) // sparse_dist)
            xx = np.linspace(start=x_min, stop=x_max, num=xn)
            yy = np.linspace(start=y_min, stop=y_max, num=yn)
            zz = 0
            points_x, points_y, points_z = np.meshgrid(xx, yy, zz)
            points_x = points_x.flatten()
            points_y = points_y.flatten()
            points_z = points_z.flatten()
            points = np.concatenate((points_x[:, None], points_y[:, None], points_z[:, None]), axis=-1)
            drill_pos = random.sample(list(points), drill_num)
        borehole_list = BoreholeSet()
        for it, drill_id in enumerate(np.arange(drill_num)):
            pos = drill_pos[drill_id]
            one_borehole = self.sample_grid_for_borehole(pos=pos)
            one_borehole.borehole_id = it
            borehole_list.append(one_borehole=one_borehole)
        borehole_list.generate_vtk_data_as_tube(borehole_radius=5, is_tube=True)
        return borehole_list

    # 将散点采样标签映射到空白网格上
    def set_map_points_data_labels_to_base_grid(self, points_data: PointSet):
        self.sample_data_list.append(points_data)
        if self.map_flag:
            cells_series = self._base_grid_labels
        else:
            cells_series = np.full((len(self._base_grid_points),), fill_value=self.default_value)
        # 如果设置了验证钻孔比例，则
        self.update_train_val_split_state()
        log_id = len(self.sample_operator) - 1
        if log_id in self.geo_sample_data_val_map.keys():
            val_points_data = points_data.get_points_data_by_ids(ids=self.geo_sample_data_val_map[log_id]['val'])
            train_points_data = points_data.get_points_data_by_ids(ids=self.geo_sample_data_val_map[log_id]['train'])
            val_cell_indices, val_cell_labels = self.map_base_grid_points_by_sample_data(sample_data=val_points_data)
            train_cell_indices, train_cell_labels = self.map_base_grid_points_by_sample_data(
                sample_data=train_points_data)
            self.train_indexes.append(train_cell_indices)
            self.val_indexes.append(val_cell_indices)
            # 合并
            cell_indices = np.hstack((val_cell_indices, train_cell_indices))
            cell_labels = np.hstack((val_cell_labels, train_cell_labels))
            cell_indices, cell_labels = remove_repeated_elements_with_lists(
                list_item_1=cell_indices, list_item_2=cell_labels)
        else:
            cell_indices, cell_labels = self.map_base_grid_points_by_sample_data(sample_data=points_data)
            self.train_indexes.append(cell_indices)
        cells_series[cell_indices] = cell_labels
        self.map_flag = True
        self._base_grid.grid_points_series = cells_series
        self._base_grid_labels = cells_series
        self._base_grid.vtk_data.cell_data['Scalar Field'] = cells_series
        self._base_grid.classes = np.unique(cells_series)
        self._base_grid.classes_num = len(self._base_grid.classes)
        if self.default_value in np.unique(cells_series):
            self._base_grid.classes_num -= 1

    # 将钻孔采样标签映射到空白网格上
    # 会将钻孔存入 sample_data_list
    # z_buffer 会将每段钻孔柱上下缓冲一下，避免有数据漏选，但是值不能设置太大
    def set_map_boreholes_labels_to_base_grid(self, boreholes: BoreholeSet):
        self.sample_data_list.append(boreholes)
        # 遍历钻孔每一个分段，寻找与base_grid的交集
        # 默认 -1 为未知值
        if self.map_flag:
            cells_series = self._base_grid.grid_points_series
        else:
            cells_series = np.full((len(self._base_grid_points),), fill_value=self.default_value)
        # 如果设置了验证钻孔比例，则
        self.update_train_val_split_state()
        log_id = len(self.sample_operator) - 1
        if log_id in self.geo_sample_data_val_map.keys():
            val_boreholes = boreholes.get_boreholes(idx=self.geo_sample_data_val_map[log_id]['val'])
            train_boreholes = boreholes.get_boreholes(idx=self.geo_sample_data_val_map[log_id]['train'])
            val_cell_indices, val_cell_labels = self.map_base_grid_points_by_sample_data(sample_data=val_boreholes)
            train_cell_indices, train_cell_labels = self.map_base_grid_points_by_sample_data(
                sample_data=train_boreholes)
            self.train_indexes.append(train_cell_indices)
            self.val_indexes.append(val_cell_indices)
            # 合并
            cell_indices = np.hstack((val_cell_indices, train_cell_indices))
            cell_labels = np.hstack((val_cell_labels, train_cell_labels))
            cell_indices, cell_labels = remove_repeated_elements_with_lists(
                list_item_1=cell_indices, list_item_2=cell_labels)
        else:
            cell_indices, cell_labels = self.map_base_grid_points_by_sample_data(sample_data=boreholes)
            self.train_indexes.append(cell_indices)
        cells_series[cell_indices] = cell_labels
        self.map_flag = True
        self._base_grid.grid_points_series = cells_series
        self._base_grid_labels = cells_series
        self._base_grid.vtk_data.cell_data['Scalar Field'] = cells_series
        uniq_labels = np.unique(cells_series)
        self._base_grid.classes = uniq_labels
        self._base_grid.classes_num = len(self._base_grid.classes)
        if self.default_value in uniq_labels:
            self._base_grid.classes_num -= 1

    def map_base_grid_points_by_sample_data(self, sample_data):
        global_cell_indexes = []
        global_cell_labels = []
        if isinstance(sample_data, PointSet):
            sample_points = sample_data.points
            sample_labels = sample_data.labels
            buffer_dist = sample_data.buffer_dist
            grid_points = pv.PolyData(self._base_grid_points)
            for p_it, one_point in enumerate(sample_points):
                # 设置一个带有缓冲半径的球体
                sphere_surface = pv.Sphere(radius=buffer_dist, center=one_point)
                cell_indices = grid_points.select_enclosed_points(sphere_surface)
                cell_indices = cell_indices.point_data['SelectedPoints']
                cell_indices = np.argwhere(cell_indices > 0).flatten()
                if self._base_grid.vtk_data is not None:
                    pos_indices = self._base_grid.vtk_data.find_containing_cell(point=one_point)
                    cell_indices = np.array(np.unique(np.hstack((cell_indices, pos_indices))), dtype=int)
                cell_label = sample_labels[p_it]
                cells_labels = np.full_like(cell_indices, fill_value=cell_label, dtype=int)
                global_cell_indexes.append(cell_indices)
                global_cell_labels.append(cells_labels)
            global_cell_indexes = np.hstack(global_cell_indexes)
            global_cell_labels = np.hstack(global_cell_labels)
            global_cell_indexes, global_cell_labels = remove_repeated_elements_with_lists(
                list_item_1=global_cell_indexes,
                list_item_2=global_cell_labels)
            return global_cell_indexes, global_cell_labels
        elif isinstance(sample_data, BoreholeSet):
            z_buffer = 0.2
            if self._bounds is not None:
                if self._base_grid.dims is not None:
                    z_buffer = (self._bounds[5] - self._bounds[4]) / self._base_grid.dims[-1] / 2
            points_data = pv.PolyData(self._base_grid_points)
            for borehole_id in range(sample_data.borehole_num):
                one_borehole = sample_data[borehole_id]
                # 搜索范围上下缓冲一下，防止漏选
                hole_top_pos = copy.deepcopy(one_borehole.top_pos)
                hole_top_pos[2] += z_buffer
                hole_bottom_pos = copy.deepcopy(one_borehole.bottom_pos)
                hole_bottom_pos[2] -= z_buffer
                line_center = np.divide(np.add(hole_top_pos, hole_bottom_pos), 2)
                line_direction = np.subtract(hole_top_pos, hole_bottom_pos)
                if np.linalg.norm(line_direction) == 0:
                    raise ValueError('Borehole data occur except error')
                # line_direction_norm = line_direction / np.linalg.norm(line_direction)
                # height = np.linalg.norm(hole_top_pos - hole_bottom_pos)
                radius = one_borehole.buffer_dist_xy
                # 设置一个带有一定缓冲半径的椭圆柱
                cylinder_surface = create_closed_cylinder_surface(top_point=hole_top_pos, bottom_point=hole_bottom_pos,
                                                                  radius=radius, segment_num=20)
                # pyvista 构建的圆柱体不封闭
                # cylinder_surface = pv.Cylinder(center=line_center, direction=line_direction_norm
                #                                , height=height, radius=radius, capping=True, resolution=500)
                # cylinder_surface = cylinder_surface.extract_surface()
                # cylinder_surface = cylinder_surface.triangulate()

                # 求解 _base_grid_points在钻孔柱子控制范围内的点的索引 cell_indices
                cell_indices = points_data.select_enclosed_points(cylinder_surface, tolerance=0.000000001)
                cell_indices = cell_indices.point_data['SelectedPoints']
                cell_indices = np.argwhere(cell_indices > 0).flatten()
                if self._base_grid.vtk_data is not None:
                    # 为防止圆柱体半径过小，没有包含任何网格点，这里添加圆柱中心线经过的网格索引
                    line_indices = self._base_grid.vtk_data.find_cells_along_line(pointa=hole_top_pos,
                                                                                  pointb=hole_bottom_pos)
                    cell_indices = np.array(np.unique(np.hstack((cell_indices, line_indices))), dtype=int)
                boreholes_points = self._base_grid_points[cell_indices]
                for one_layer in one_borehole.holelayer_list:
                    top_pos_z = one_layer.top_pos[2]
                    bottom_pos_z = one_layer.bottom_pos[2]
                    label = one_layer.layer_label
                    # 根据钻孔每段分层的高程范围，设置分层类别标签
                    layer_indices = np.argwhere((boreholes_points[:, 2] <= top_pos_z) &
                                                (boreholes_points[:, 2] > bottom_pos_z)).flatten()
                    cells_labels = np.full_like(layer_indices, fill_value=label, dtype=int)
                    # 取基于grid_points的索引
                    indices = cell_indices[layer_indices]
                    global_cell_indexes.append(indices)
                    global_cell_labels.append(cells_labels)
            global_cell_indexes = np.hstack(global_cell_indexes)
            global_cell_labels = np.hstack(global_cell_labels)
            global_cell_indexes, global_cell_labels = remove_repeated_elements_with_lists(
                list_item_1=global_cell_indexes,
                list_item_2=global_cell_labels)
            return global_cell_indexes, global_cell_labels
        else:
            raise ValueError('Unknown input.')

    # 获取采样点相对于格网点的索引, idx=None 返回所有数据， idx=0则返回第一个采样数据
    # 获取所有采样点的索引 train + val （训练集+验证集）数据集
    def get_sample_points_indexes_for_grid_points(self) -> (list, list):  # 返回 index
        if self._base_grid_points is None:
            raise ValueError('Grid points should not be empty.')
        train_indexes = []
        val_indexes = []
        for sample_op_it in range(len(self.sample_operator)):
            if self.sample_operator[sample_op_it] != 'None':
                self.update_train_val_split_state(sid=sample_op_it)
                train_indexes_1 = self.geo_sample_data_val_map[sample_op_it]['train']
                val_indexes_1 = self.geo_sample_data_val_map[sample_op_it]['val']
                train_indexes.extend(train_indexes_1)
                val_indexes.extend(val_indexes_1)
        if 'None' in self.sample_operator:
            val_indexes_2 = self.val_indexes
            train_indexes_2 = self.train_indexes
            if len(self.val_indexes) > 0:
                val_indexes_2 = np.hstack(self.val_indexes)
            if len(self.train_indexes) > 0:
                train_indexes_2 = np.hstack(self.train_indexes)
            train_indexes.extend(train_indexes_2)
            val_indexes.extend(val_indexes_2)
        if len(train_indexes) > 0:
            train_indexes = np.hstack(train_indexes)
        if len(val_indexes) > 0:
            val_indexes = np.hstack(val_indexes)
        val_indexes = list(sorted(set(np.unique(val_indexes))))
        train_indexes = list(sorted(set(np.unique(train_indexes))))
        del_indexes = []
        for v_id in val_indexes:
            if v_id in train_indexes:
                del_indexes.append(train_indexes.index(v_id))
        if len(del_indexes) > 0:
            train_indexes = np.delete(train_indexes, del_indexes, axis=0)
        return train_indexes, val_indexes

    # 获取采样点数据
    def get_points_data(self, sample_id=0):
        if sample_id < 0 or sample_id > self.sample_num - 1:
            raise ValueError('Index out of range.')
        sample_data = self.sample_data_list[sample_id]
        if isinstance(sample_data, PointSet):
            return sample_data
        elif isinstance(sample_data, (BoreholeSet, SectionSet)):
            sample_data = sample_data.get_points_data()
            return sample_data

    def save(self, dir_path: str):
        for s_id in np.arange(len(self.sample_data_list)):
            self.sample_data_list[s_id].save(dir_path=dir_path)

    def load(self, dir_path: str):
        for s_id in np.arange(len(self.sample_data_list)):
            self.sample_data_list[s_id].load(dir_path=dir_path)


# Parameters:
# grid : data_structure.grids.Grid
#   外部输入网格，在该网格基础上进行采样操作
# sample_operator: list
# 列表中包含如下一个或多个字符串，指定不同的采样操作，根据列表中的操作指令字符串数量，来进行相应次数的采样
#   ['None', 'uniformly_points', 'eq_interval_points', 'rand_drills', 'axis_sections']
#   1. ‘None’:  无操作，不进行额外采样
#   2. ‘uniformly_points’: 均匀采样
#   3. 'eq_interval_points': 根据点索引等索引间距采样
#   4. ‘rand_drills’: 随机钻孔采样  若指定钻孔采样，需要添加额外的参数
#       drill_pos -np.ndarray 或 list 指定钻孔采样位置的3d points，坐标的z值可以随机
#       drill_num -int  指定采样钻孔数目，随机采样
#       drill_pos和drill_num二选其一，两者都有则以drill_pos为准
#   5. ‘axis_sections’: 轴向剖面采样， 需要添加额外参数
#       sample_axis -str 采样轴,指定 'x','y','z',
#       section_num -int 指定剖面数量
#       scroll_pos -list  0-1之间的小数值，个数与section_num对应，指定剖面切割位置，用于滑动切割剖面
#       resolution_xy -float  指定剖面xy水平方向的网格分辨率
#       resolution_z  -float  指定剖面z轴纵向上的网格分辨率
class GeoGridDataSampler(GeoDataSampler):
    def __init__(self, grid: Grid = None, sample_operator: list = None, sample_data_names: list = None, **kwargs):
        super().__init__(grid)
        if sample_operator is not None:
            self.sample_operator.extend(sample_operator)
        self.names = sample_data_names
        self.kwargs = kwargs

    @property
    def grid(self):
        return self._base_grid

    @grid.setter
    def grid(self, grid):
        self._base_grid = grid
        if grid is not None:
            self._base_grid_points = grid.grid_points
            self._base_grid_labels = grid.grid_points_series
            self._bounds = grid.bounds
        else:
            raise ValueError('Input grid is None')

    def execute(self, **kwargs):
        if self._base_grid is not None:
            self.kwargs.update(kwargs)
            sample_op_type = ['None', 'uniformly_points', 'eq_interval_points', 'rand_drills', 'axis_sections']
            idx_to_sample_op = {index: sample_op for index, sample_op in enumerate(sample_op_type)}
            for sit, sample_op in enumerate(self.sample_operator):
                if isinstance(sample_op, int):
                    sample_op = idx_to_sample_op[sample_op]
                if sample_op not in sample_op_type:
                    raise ValueError('Sample operate must be one of None, uniformly_points, eq_interval_points, '
                                     'rand_drills, axis_sections')
                else:
                    if sample_op == 'None':
                        continue
                    if sample_op == 'uniformly_points':
                        if 'points_num' in self.kwargs.keys():
                            points_num = self.kwargs['points_num']
                        else:
                            raise ValueError('Need to set value of parameter points_num.')
                        points_data = self.uniformly_sample_grid_for_points(n_points=points_num)
                        self.sample_data_list.append(points_data)
                    if sample_op == 'eq_interval_points':
                        if 'interval' in self.kwargs.keys():
                            interval = self.kwargs['interval']
                        else:
                            raise ValueError('Need to set value of parameter interval.')
                        points_data = self.uniformly_interval_sample_grid_for_points(interval=interval)
                        self.sample_data_list.append(points_data)
                    if sample_op == 'rand_drills':
                        drill_num = None
                        drill_pos = None
                        if 'drill_pos' in self.kwargs.keys():
                            drill_pos = self.kwargs['drill_pos']
                        if 'drill_num' in self.kwargs.keys():
                            drill_num = self.kwargs['drill_num']
                        if drill_pos is None and drill_num is None:
                            raise ValueError('Need to input parameter drill_pos or drill_num.')
                        sparse_dist = (self._base_grid.dims[0] + self._base_grid.dims[1]) / 2
                        borehole_list = self.random_sample_grid_for_boreholes(drill_pos=drill_pos, drill_num=drill_num
                                                                              , sparse_dist=sparse_dist)
                        self.sample_data_list.append(borehole_list)
                    if sample_op == 'axis_sections':
                        sample_axis = None
                        section_num = 1
                        scroll_pos = [0.5]
                        resolution_xy = None
                        resolution_z = None
                        if 'sample_axis' in self.kwargs.keys():
                            sample_axis = self.kwargs['sample_axis']
                        if 'section_num' in self.kwargs.keys():
                            section_num = self.kwargs['section_num']
                        if 'scroll_pos' in self.kwargs.keys():
                            scroll_pos = self.kwargs['scroll_pos']
                        if 'resolution_xy' in self.kwargs.keys():
                            resolution_xy = self.kwargs['resolution_xy']
                        if 'resolution_z' in self.kwargs.keys():
                            resolution_z = self.kwargs['resolution_z']
                        if sample_axis is None or resolution_xy is None or resolution_z is None:
                            raise ValueError(
                                'This method need to input 5 parameters, including sample_axis, section_num,'
                                'scroll_pos, resolution_xy and resolution_z.')
                        section_list = self.sample_with_sections_along_axis(sample_axis=sample_axis
                                                                            , section_num=section_num
                                                                            , scroll_pos=scroll_pos,
                                                                            resolution_xy=resolution_xy
                                                                            , resolution_z=resolution_z)
                        self.sample_data_list.append(section_list)
        else:
            raise ValueError('The grid data is empty.')

    # 混合数据集, 目前支持钻孔和散点
    # 当数据为钻孔时，支持地表约束，方式为将地表约束网格作为external_grid_vtk外部传入
    def set_base_grid_by_geodataset(self, geodataset: GeodataSet, dims: np.ndarray, bounds: np.ndarray = None
                                    , cell_resolution: np.ndarray = None, is_regular=True
                                    , external_grid_vtk=None, check_convexhell=False):
        if bounds is None:
            bounds = geodataset.bounds
        grid = Grid(name='gme_base_grid')
        if cell_resolution is not None:
            if not np.all(cell_resolution):
                raise ValueError("Cell resolution array can't exist 0.")
            if cell_resolution.shape[0] < 3:
                raise ValueError("Cell resolution array should be triple element data.")
            dims = np.divide(np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]),
                             cell_resolution)
        # 外部传入网格优先，若没有外部网格，则内部构建，先判断是否有范围和地形约束，然后判断是否是规则网格
        # 这里的规则网格是指规则体素栅格的中心点构成的网格
        # 不规则网格指的是有荻洛妮四面体生成算法内插出的顶点，这些顶点在空间中散布，以此构图，虽然最后三维可视化仍然是规则体素，但这些
        # 体素的属性是根据最近邻算法插值得到的。
        if external_grid_vtk is None:
            if dims is None:
                raise ValueError('Need to specify the size of the grid.')
            if not check_convexhell:
                sample_grid = create_vtk_grid_by_rect_bounds(dim=dims, bounds=bounds)
            else:
                convexhull_2d = geodataset.get_points_data().get_convexhull_2d()
                sample_grid = create_vtk_grid_by_boundary(dims=dims, bounds=bounds
                                                          , convexhull_2d=convexhull_2d
                                                          , cell_density=cell_resolution)
            external_grid_vtk = sample_grid
        grid.set_vtk_grid(grid_vtk=external_grid_vtk)
        self.grid = grid
        if self.sample_operator is None:
            self.sample_operator = []
        for sample_data in geodataset.geodata_list:
            if sample_data.__class__.__name__ == 'PointSet':
                self.sample_operator.append('None')
                self.set_map_points_data_labels_to_base_grid(points_data=sample_data)
            if sample_data.__class__.__name__ == 'BoreholeSet':
                self.sample_operator.append('None')
                self.set_map_boreholes_labels_to_base_grid(boreholes=sample_data)
            #

    # 已知部分钻孔，根据给定范围创建规则网格，钻孔控制范围内的网格点赋予标签，范围外的网格点无标签。
    def set_base_grid_by_boreholes(self, boreholes: BoreholeSet, dims: np.ndarray, bounds: np.ndarray = None
                                   , cell_resolution: np.ndarray = None, is_regular=True
                                   , check_convexhell=False, external_grid_vtk=None):
        if bounds is None:
            bounds = boreholes.bounds
        if dims.shape[0] != 3:
            raise ValueError('This method can only set three dimensions grid.')
        if cell_resolution is None:
            x_r = (bounds[1] - bounds[0]) / dims[0]
            y_r = (bounds[3] - bounds[2]) / dims[1]
            z_r = (bounds[5] - bounds[4]) / dims[2]
            cell_resolution = np.array([x_r, y_r, z_r])
        # 设置钻孔控制缓冲半径
        boreholes.set_boreholes_control_buffer_dist_xy(radius=float(1.5 * (cell_resolution[0] + cell_resolution[1])))
        grid = Grid(name='gme_base_grid')
        if external_grid_vtk is None:
            if not check_convexhell:
                sample_grid = create_vtk_grid_by_rect_bounds(dim=dims, bounds=bounds)
            else:
                convexhull_2d = boreholes.get_points_data().get_convexhull_2d()
                sample_grid, grid_outline = create_vtk_grid_by_boundary(dims=dims, bounds=bounds
                                                                        , convexhull_2d=convexhull_2d
                                                                        , cell_density=cell_resolution)
            external_grid_vtk = sample_grid
        # 将钻孔点标签映射到格网上，未知标签的格网点值默认为-1
        grid.set_vtk_grid(grid_vtk=external_grid_vtk)
        self.grid = grid
        if self.sample_operator is None:
            self.sample_operator = ['None']
        else:
            self.sample_operator.append('None')
        self.set_map_boreholes_labels_to_base_grid(boreholes=boreholes)

    #
    def set_base_grid_by_points_data(self, points_data: PointSet, dims: np.ndarray = None, bounds: np.ndarray = None
                                     , cell_resolution: np.ndarray = None, external_grid_vtk=None
                                     , is_regular=True, check_convexhell=False):
        if bounds is None:
            bounds = points_data.bounds
        grid = Grid(name='gme_base_grid')
        if cell_resolution is not None:
            if not np.all(cell_resolution):
                raise ValueError("Cell resolution array can't exist 0.")
            if cell_resolution.shape[0] < 3:
                raise ValueError("Cell resolution array should be triple element data.")
            dims = np.divide(np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]),
                             cell_resolution)
        if external_grid_vtk is None:
            if dims is None:
                raise ValueError('Need to specify the size of the grid.')
            if not check_convexhell:
                sample_grid = create_vtk_grid_by_rect_bounds(dim=dims, bounds=bounds)
            else:
                convexhull_2d = points_data.get_convexhull_2d()
                sample_grid = create_vtk_grid_by_boundary(dims=dims, bounds=bounds
                                                          , convexhull_2d=convexhull_2d
                                                          , cell_density=cell_resolution)
            external_grid_vtk = sample_grid
        grid.set_vtk_grid(grid_vtk=external_grid_vtk)
        self.grid = grid
        if self.sample_operator is None:
            self.sample_operator = ['None']
        else:
            self.sample_operator.append('None')
        self.set_map_points_data_labels_to_base_grid(points_data=points_data)

    # sample_axis 采样轴 ['x', 'y', 'z'],  is_random=Ture, 切割位置随机
    def sample_with_sections_along_axis(self, sample_axis=None, section_num=1, scroll_pos: list = None
                                        , resolution_xy=None, resolution_z=None):
        if self._base_grid is None:
            raise ValueError('Need to input grid first.')
        if resolution_xy is None:
            resolution_xy = (self._bounds[1] - self._bounds[0] + self._bounds[3] - self._bounds[2]) * 0.5 * 0.01
        if resolution_z is None:
            resolution_z = (self._bounds[5] - self._bounds[4]) * 0.01
        scroll_size = scroll_pos
        if scroll_pos is None or (isinstance(scroll_pos, (list, np.ndarray)) and len(scroll_pos) != section_num):
            scroll_size = [random.uniform(0, 1) for i in range(section_num)]
        if self.names is None:
            name = None
        else:
            name = self.names[len(self.sample_data_list)]
        section_list = SectionSet(name=name)
        for s_i in np.arange(section_num):
            section = Section()
            surface = section.prob_volume(grid=self._base_grid,
                                          surf=section.create_surface_along_axis(along_axis=sample_axis
                                                                                 , scroll_scale=
                                                                                 scroll_size[s_i],
                                                                                 resolution_xy=resolution_xy,
                                                                                 resolution_z=resolution_z,
                                                                                 grid_bounds=self._bounds))
            surface.plot()
            section.set_surface(surf=surface)
            section_list.append(section)
        return section_list


class GeoSectionDataSampler(GeoDataSampler):
    def __init__(self, section: Section = None, sample_operator: list = None, **kwargs):
        super().__init__()
        self._section = section
        if sample_operator is not None:
            self.sample_operator.extend(sample_operator)
        self.kwargs = kwargs

    @property
    def section(self):
        return self._base_grid

    @section.setter
    def section(self, section: Section):
        self._base_grid = section
        if section is not None:
            self._base_grid_points = section.points
            self._base_grid_labels = section.series
            self._bounds = section.bounds
        else:
            raise ValueError('Input grid is None')

    def execute(self, **kwargs):
        if self._base_grid is not None:
            self.kwargs.update(kwargs)
            sample_op_type = ['None', 'rand_points', 'rand_drills']
            idx_to_sample_op = {index: sample_op for index, sample_op in enumerate(sample_op_type)}
            for sit, sample_op in enumerate(self.sample_operator):
                if isinstance(sample_op, int):
                    sample_op = idx_to_sample_op[sample_op]
                if sample_op not in sample_op_type:
                    raise ValueError('Sample operate must be one of None, rand_points, rand_drill')
                else:
                    if sample_op == 'None':
                        continue
                    if sample_op == 'rand_points':
                        if 'sample_ratio' in self.kwargs.keys():
                            sample_ratio = self.kwargs['sample_ratio']
                        else:
                            sample_ratio = 0.1
                        points_data = self.random_sample_grid_for_points(sample_ratio=sample_ratio)
                        self.sample_data_list.append(points_data)
                    if sample_op == 'rand_drills':
                        drill_num = None
                        drill_pos = None
                        if 'drill_pos' in self.kwargs.keys():
                            drill_pos = self.kwargs['drill_pos']
                        if 'drill_num' in self.kwargs.keys():
                            drill_num = self.kwargs['drill_num']
                        if drill_pos is None and drill_num is None:
                            raise ValueError('Need to input parameter drill_pos or drill_num.')
                        borehole_list = self.random_sample_grid_for_boreholes(drill_pos=drill_pos, drill_num=drill_num)
                        self.sample_data_list.append(borehole_list)
        else:
            raise ValueError('The grid data is empty.')

    # 将钻孔连成一个空白剖面，将地质属性映射到剖面上，钻孔控制范围外的剖面网格点无标签属性。
    def set_base_section_by_boreholes(self, boreholes: BoreholeSet, dims: np.ndarray = None,
                                      bounds: np.ndarray = None, principal_axis: str = 'x',
                                      resolution_xy: float = None
                                      , resolution_z: float = None,
                                      radius_buffer_factor=3):
        # radius_buffer_factor 最小为1，表示网格分辨率的倍数
        if bounds is None:
            bounds = boreholes.bounds
        top_points = boreholes.get_top_points()
        # 将钻孔顶点坐标，沿着主方向轴进行排序，默认x轴，其次y轴
        axis_labels = ['x', 'y']
        label_to_index = {label: index for index, label in enumerate(axis_labels)}
        axis = principal_axis.lower()
        # 沿轴从左到右排序
        if axis in axis_labels:
            line_points_sort_ind = np.lexsort((top_points[:, 1 - label_to_index[axis]]
                                               , top_points[:, label_to_index[axis]]))
            line_points_sorted = top_points[line_points_sort_ind]
            section = Section()
            if resolution_xy is None or resolution_z is None:
                if dims is None:
                    raise ValueError('Need to input dims parameters or resolution parameters.')
                x_length = bounds[1] - bounds[0]
                y_length = bounds[3] - bounds[2]
                z_length = bounds[5] - bounds[4]
                resolution_xy = (x_length / dims[0] + y_length / dims[1]) * 0.5
                resolution_z = z_length / dims[2]
            grid_bounds = get_bounds_from_coords(boreholes.get_points_data().points)
            boreholes.set_boreholes_control_buffer_dist_xy(radius=radius_buffer_factor * resolution_xy)
            vtk_sec = section.create_surface_by_sweepline(trajectory_line_xy=line_points_sorted
                                                          , grid_bounds=grid_bounds
                                                          , resolution_xy=resolution_xy
                                                          , resolution_z=resolution_z)
            # 由 PolyData 转成 UnstructuredGrid
            u_sec = vtk_polydata_to_vtk_unstructured_grid(vtk_sec)

            sec = Section(name='gme_base_grid_2d')
            # 将钻孔点标签映射到格网上，未知标签的格网点值默认为-1
            sec.set_vtk_grid(grid_vtk=u_sec)
            self.section = sec
            if self.sample_operator is None:
                self.sample_operator = ['None']
            else:
                self.sample_operator.append('None')
            self.set_map_boreholes_labels_to_base_grid(boreholes=boreholes)

    # unlapped_indices = np.argwhere(boreholes_points_labels == self.default_value)
    # unlapped_indices = unlapped_indices.flatten()
    # # 删除未影响的点与标签, 受到控制的点标签不能为-1
    # if len(unlapped_indices) > 0:
    #     boreholes_points = np.delete(boreholes_points, unlapped_indices, axis=0)
    #     boreholes_points_labels = np.delete(boreholes_points_labels, unlapped_indices, axis=0)
    # new_point_data = PointSet(points=boreholes_points, point_labels=boreholes_points_labels)
    # new_point_data_list.append(new_point_data)
