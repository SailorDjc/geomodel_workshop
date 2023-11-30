from data_structure.grids import Grid
from data_structure.sections import SectionSet, Section
from data_structure.boreholes import BoreholeSet, Borehole
from data_structure.points import PointSet, get_bounds_from_coords
import numpy as np
import random
import copy
import scipy.spatial as spt
import pyvista as pv
from utils.vtk_utils import vtk_polydata_to_vtk_unstructured_grid, create_closed_cylinder_surface


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
class GeoGridDataSampler(object):
    def __init__(self, grid: Grid = None, sample_operator: list = None, sample_data_names: list = None, **kwargs):
        self._grid = grid
        self.sample_operator = sample_operator
        self.grid_points = None
        self.bounds = None
        self.names = sample_data_names

        if grid is not None:
            self.grid_points = grid.grid_points
            self.bounds = grid.bounds
        self.sample_num = 0  # 采样次数
        self.sample_data_list = []  # 采样数据
        self.kwargs = kwargs

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, grid):
        self._grid = grid
        if grid is not None:
            self.grid_points = grid.grid_points
            self.bounds = grid.bounds
        else:
            raise ValueError('Input grid is None')

    def execute(self, **kwargs):
        if self._grid is not None:
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
                        self.uniformly_sample_grid_for_points(n_points=points_num)
                    if sample_op == 'eq_interval_points':
                        if 'interval' in self.kwargs.keys():
                            interval = self.kwargs['interval']
                        else:
                            raise ValueError('Need to set value of parameter interval.')
                        self.uniformly_interval_sample_grid_for_points(interval=interval)
                    if sample_op == 'rand_drills':
                        drill_num = None
                        drill_pos = None
                        if 'drill_pos' in self.kwargs.keys():
                            drill_pos = self.kwargs['drill_pos']
                        if 'drill_num' in self.kwargs.keys():
                            drill_num = self.kwargs['drill_num']
                        if drill_pos is None and drill_num is None:
                            raise ValueError('Need to input parameter drill_pos or drill_num.')
                        self.random_sample_grid_for_boreholes(drill_pos=drill_pos, drill_num=drill_num)
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
                        self.sample_with_sections_along_axis(sample_axis=sample_axis, section_num=section_num
                                                             , scroll_pos=scroll_pos, resolution_xy=resolution_xy
                                                             , resolution_z=resolution_z)
        else:
            raise ValueError('The grid data is empty.')

    def get_sample_data(self, idx):
        if idx < 0 or idx >= len(self.sample_data_list):
            raise ValueError('The input index is out of range.')
        return self.sample_data_list[idx]

    # 已知部分钻孔，根据给定范围创建规则网格，钻孔控制范围内的网格点赋予标签，范围外的网格点无标签。
    def set_base_grid_by_boreholes(self, boreholes: BoreholeSet, dims: np.ndarray, bounds: np.ndarray = None
                                   , cell_density: np.ndarray = None, is_regular=True, external_grid=None):
        if bounds is None:
            bounds = boreholes.bounds
        if dims.shape[0] != 3:
            raise ValueError('This method can only set three dimensions grid.')
        if cell_density is None:
            x_r = (bounds[1] - bounds[0]) / dims[0]
            y_r = (bounds[3] - bounds[2]) / dims[1]
            z_r = (bounds[5] - bounds[4]) / dims[2]
            cell_density = np.array([x_r, y_r, z_r])
        boreholes.set_boreholes_control_buffer_dist_xy(radius=1.5 * (cell_density[0] + cell_density[1]))
        grid = Grid(name='gme_base_grid', grid_vtk=external_grid)
        if external_grid is None:
            if is_regular:
                sample_grid = grid.create_vtk_grid_by_rect_bounds(dim=dims, bounds=bounds)
            else:
                convexhull_2d = boreholes.get_points_data().get_convexhull_2d()
                sample_grid, grid_outline = grid.create_vtk_grid_by_unregular_bounds(dims=dims, bounds=bounds
                                                                                     , convexhull_2d=convexhull_2d
                                                                                     , cell_density=cell_density)
            # 将钻孔点标签映射到格网上，未知标签的格网点值默认为-1
            grid.set_vtk_grid(grid_vtk=sample_grid)
        self.grid = grid
        if self.sample_operator is None:
            self.sample_operator = ['None']
        else:
            self.sample_operator.append('None')
        self.set_map_boreholes_labels_to_base_grid(boreholes=boreholes)

    # 将钻孔连成一个空白剖面，将地质属性映射到剖面上，钻孔控制范围外的剖面网格点无标签属性。
    def set_base_grid_2d_by_boreholes(self, boreholes: BoreholeSet, dims: np.ndarray = None,
                                      bounds: np.ndarray = None, principal_axis: str = 'x', resolution_xy: float = None
                                      , resolution_z: float = None):
        if bounds is None:
            bounds = boreholes.bounds
        top_points = boreholes.get_top_points()
        # 将钻孔顶点坐标，沿着主方向轴进行排序，默认x轴
        axis_labels = ['x', 'y']
        label_to_index = {label: index for index, label in enumerate(axis_labels)}
        axis = principal_axis.lower()
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
            boreholes.set_boreholes_control_buffer_dist_xy(radius=3 * resolution_xy)
            grid_2d = section.create_surface_by_sweepline(trajectory_line_xy=line_points_sorted
                                                          , grid_bounds=grid_bounds
                                                          , resolution_xy=resolution_xy
                                                          , resolution_z=resolution_z)
            # 由 PolyData 转成 UnstructuredGrid
            ugrid = vtk_polydata_to_vtk_unstructured_grid(grid_2d)

            grid = Grid(name='gme_base_grid_2d')
            # 将钻孔点标签映射到格网上，未知标签的格网点值默认为-1
            grid.set_vtk_grid(grid_vtk=ugrid)
            self.grid = grid
            if self.sample_operator is None:
                self.sample_operator = ['None']
            else:
                self.sample_operator.append('None')
            self.set_map_boreholes_labels_to_base_grid(boreholes=boreholes)

    #
    def set_base_grid_by_points_data(self, points_data: PointSet, dims: np.ndarray = None, bounds: np.ndarray = None
                                     , cell_resolution: np.ndarray = None, external_grid=None, is_regular=True):
        if bounds is None:
            bounds = points_data.bounds
        grid = Grid(name='gme_base_grid', grid_vtk=external_grid)
        if cell_resolution is not None:
            if not np.all(cell_resolution):
                raise ValueError("Cell resolution array can't exist 0.")
            if cell_resolution.shape[0] < 3:
                raise ValueError("Cell resolution array should be triple element data.")
            dims = np.divide(np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]),
                             cell_resolution)
        if external_grid is None:
            if dims is None:
                raise ValueError('Need to specify the size of the grid.')
            if is_regular:
                sample_grid = grid.create_vtk_grid_by_rect_bounds(dim=dims, bounds=bounds)
            else:
                convexhull_2d = points_data.get_convexhull_2d()
                sample_grid = grid.create_vtk_grid_by_unregular_bounds(dims=dims, bounds=bounds
                                                                       , convexhull_2d=convexhull_2d
                                                                       , cell_density=cell_resolution)
            grid.set_vtk_grid(grid_vtk=sample_grid)
        self.grid = grid
        if self.sample_operator is None:
            self.sample_operator = ['None']
        else:
            self.sample_operator.append('None')
        self.set_map_points_data_labels_to_base_grid(points_data=pointsdata)

    # 将散点采样标签映射到空白网格上
    def set_map_points_data_labels_to_base_grid(self, points_data: PointSet):
        cells_series = np.full((len(self.grid.grid_points),), fill_value=-1)
        sample_points = points_data.points
        sample_labels = points_data.labels
        buffer_dist = points_data.buffer_dist
        grid_points = pv.PolyData(self.grid.grid_points)
        new_point_data_list = []
        for p_it, one_point in enumerate(sample_points):
            sphere_surface = pv.Sphere(radius=buffer_dist, center=one_point)
            cell_indices = grid_points.select_enclosed_points(sphere_surface)
            cell_indices = cell_indices.point_data['SelectedPoints']
            cell_indices = np.argwhere(cell_indices > 0).flatten()
            pos_indices = self.grid.vtk_data.find_containing_cell(point=one_point)
            cell_indices = np.array(np.unique(np.hstack((cell_indices, pos_indices))), dtype=int)
            cell_label = sample_labels[p_it]
            cells_labels = np.full_like(cell_indices, fill_value=cell_label, dtype=int)
            cells_series[cell_indices] = cells_labels
            # 创建 PointSet
            new_points = self.grid_points[cell_indices]
            new_data = PointSet(points=new_points, point_labels=cells_labels)
            new_point_data_list.append(new_data)
        new_point_data = PointSet()
        for data in new_point_data_list:
            new_point_data.append(data)
        self.sample_data_list.append(new_point_data)
        self.sample_num += 1
        self._grid.grid_points_series = cells_series
        self._grid.vtk_data.cell_data['Scalar Field'] = cells_series
        self._grid.classes = np.unique(cells_series)
        self._grid.classes_num = len(self.grid.classes)

    # 将钻孔采样标签映射到空白网格上
    def set_map_boreholes_labels_to_base_grid(self, boreholes: BoreholeSet):
        # 遍历钻孔每一个分段，寻找与base_grid的交集
        # 默认 -1 为未知值
        cells_series = np.full((len(self.grid.grid_points),), fill_value=-1)
        z_buffer = 2
        if self.bounds is not None and self.grid.dims is not None:
            z_buffer = (self.bounds[5] - self.bounds[4]) / self.grid.dims[2] / 2
        points_data = pv.PolyData(self.grid.grid_points)
        # 将钻孔控制区域的采样数据转换为PointSet类型
        new_point_data_list = []
        for one_borehole in boreholes:
            # 搜索范围上下缓冲一下，防止漏选
            hole_top_pos = copy.deepcopy(one_borehole.top_pos)
            hole_top_pos[2] += z_buffer
            hole_bottom_pos = copy.deepcopy(one_borehole.bottom_pos)
            hole_bottom_pos[2] -= z_buffer
            line_center = np.divide(np.add(hole_top_pos, hole_bottom_pos), 2)
            line_direction = np.subtract(hole_top_pos, hole_bottom_pos)
            if np.linalg.norm(line_direction) == 0:
                raise ValueError('Borehole data occur except error')
            line_direction_norm = line_direction / np.linalg.norm(line_direction)
            height = np.linalg.norm(hole_top_pos - hole_bottom_pos)
            radius = one_borehole.buffer_dist_xy
            cylinder_surface = pv.Cylinder(center=line_center, direction=line_direction_norm
                                           , height=height, radius=radius, capping=True)
            cylinder_surface = cylinder_surface.extract_surface()
            cell_indices = points_data.select_enclosed_points(cylinder_surface, check_surface=False)
            cell_indices = cell_indices.point_data['SelectedPoints']
            cell_indices = np.argwhere(cell_indices > 0).flatten()
            # 为防止圆柱体半径过小，没有包含任何网格点，这里添加圆柱中心线经过的网格索引
            line_indices = self._grid.vtk_data.find_cells_along_line(pointa=hole_top_pos, pointb=hole_bottom_pos)
            cell_indices = np.array(np.unique(np.hstack((cell_indices, line_indices))), dtype=int)
            boreholes_points = self.grid_points[cell_indices]
            boreholes_points_labels = np.full((len(boreholes_points),), fill_value=-1)
            for one_layer in one_borehole.holelayer_list:
                top_pos_z = one_layer.top_pos[2]
                bottom_pos_z = one_layer.bottom_pos[2]
                label = one_layer.layer_label
                layer_indices = np.argwhere((boreholes_points[:, 2] <= top_pos_z) &
                                            (boreholes_points[:, 2] > bottom_pos_z)).flatten()
                cells_labels = np.full_like(layer_indices, fill_value=label, dtype=int)
                #
                boreholes_points_labels[layer_indices] = cells_labels
                # 取基于grid_points的索引
                indices = cell_indices[layer_indices]
                cells_series[indices] = cells_labels
            new_point_data = PointSet(points=boreholes_points, point_labels=boreholes_points_labels)
            new_point_data_list.append(new_point_data)
        if len(new_point_data_list) > 0:
            self._grid.grid_points_series = cells_series
            self._grid.vtk_data.cell_data['Scalar Field'] = cells_series
            self._grid.classes = np.unique(cells_series)
            self._grid.classes_num = len(self.grid.classes)
            point_dataset = PointSet()
            for data in new_point_data_list:
                point_dataset.append(data)
            self.sample_data_list.append(point_dataset)
            self.sample_num += 1
        else:
            raise ValueError('Data is empty.')

    # 对网格点索引进行间隔采样
    def uniformly_interval_sample_grid_for_points(self, interval=100):
        if self.grid is None:
            raise ValueError('Need to input grid first.')
        assert self._grid.dims is not None, "grid dimensions are not set"
        pid = np.arange(0, len(self.grid_points), interval)
        pid = list(set(sorted(pid)))
        point_labels = self._grid.grid_points_series[pid]
        sample_points = self.grid_points[pid]
        if self.names is None:
            name = None
        else:
            name = self.names[len(self.sample_data_list)]
        points_data = PointSet(points=sample_points, point_labels=point_labels
                               , name=name)
        self.sample_data_list.append(points_data)
        self.sample_num += 1
        return points_data

    # 对网格点进行规则采样
    def uniformly_sample_grid_for_points(self, n_points):
        if self._grid is None:
            raise ValueError('Need to input grid first.')
        assert self._grid.dims is not None, "grid dimensions are not set"
        # proportion of points along each dimension compared to the max number of points along a particular direction
        max_n_pts_dir = max(self.grid.dims)
        px = self._grid.dims[0] / max_n_pts_dir
        py = self._grid.dims[1] / max_n_pts_dir
        pz = self._grid.dims[2] / max_n_pts_dir
        # to uniformly sample the grid to get a total of n_points WHILE also respecting the ratio of points along two
        # directions the equation is
        # (x * px) ( x * py ) * ( x * pz ) = n_points, solve for x
        x = (((max_n_pts_dir ** 3) * n_points) / (self._grid.dims[0] * self._grid.dims[1]
                                                  * self._grid.dims[2])) ** (1.0 / 3.0)
        nx = round(x * px)
        ny = round(x * py)
        nz = round(x * pz)
        bounds = self._grid.vtk_data.GetBounds()
        sample_x = np.linspace(bounds[0], bounds[1], nx)
        sample_y = np.linspace(bounds[2], bounds[3], ny)
        sample_z = np.linspace(bounds[4], bounds[5], nz)
        zz, yy, xx = np.meshgrid(sample_z, sample_y, sample_x, indexing='ij')
        sample_points = np.vstack((xx, yy, zz)).reshape(3, -1).T
        pid = self._grid.vtk_data.find_containing_cell(sample_points)
        point_labels = self._grid.grid_points_series[pid]
        if self.names is None:
            name = None
        else:
            name = self.names[len(self.sample_data_list)]
        points_data = PointSet(points=sample_points, point_labels=point_labels
                               , name=name)
        self.sample_data_list.append(points_data)
        self.sample_num += 1
        return points_data

    # 在网格范围内随机采样钻孔
    def random_sample_grid_for_boreholes(self, drill_pos=None, drill_num=None):
        if drill_num is None:
            drill_num = 10
        if self._grid is None:
            raise ValueError('Need to input grid first.')
        horizon_slice = self._grid.vtk_data.slice(normal='z')
        if drill_pos is not None:
            drill_num = len(drill_pos)
        else:
            # 获取grid水平切面，在切面上随机选取cell的中心点，打钻孔
            points = horizon_slice.cell_centers().points.tolist()
            drill_pos = random.sample(points, drill_num)
        if self.names is None:
            name = None
        else:
            name = self.names[len(self.sample_data_list)]
        borehole_list = BoreholeSet(name=name)
        for it, drill_id in enumerate(np.arange(drill_num)):
            pos = drill_pos[drill_id]
            one_borehole = self.sample_grid_for_borehole(pos=pos)
            one_borehole.borehole_id = it
            borehole_list.append(one_borehole=one_borehole)
        borehole_list.generate_vtk_data_as_tube(borehole_radius=5, is_tube=True)
        self.sample_data_list.append(borehole_list)
        self.sample_num += 1
        return borehole_list

    #
    def sample_grid_for_borehole(self, pos):
        pos_a = copy.deepcopy(pos)
        pos_b = copy.deepcopy(pos)
        pos_a[2] = self._grid.bounds[5]  # z_max
        pos_b[2] = self._grid.bounds[4]  # z_min
        # 沿直线采样
        pid = self._grid.vtk_data.find_cells_along_line(pointa=pos_a, pointb=pos_b)
        line_points = self._grid.grid_points[pid]
        line_series = self._grid.grid_points_series[pid]
        # line_points_z = line_points[::-1, 2]
        line_points_sort_ind = np.argsort(line_points[:, 2])
        line_points = line_points[line_points_sort_ind[::-1]]
        line_series = line_series[line_points_sort_ind[::-1]]
        one_borehole = Borehole(points=line_points, series=line_series, is_virtual=True)
        return one_borehole

    # sample_axis 采样轴 ['x', 'y', 'z'],  is_random=Ture, 切割位置随机
    def sample_with_sections_along_axis(self, sample_axis=None, section_num=1, scroll_pos: list = None
                                        , resolution_xy=None, resolution_z=None):
        if self._grid is None:
            raise ValueError('Need to input grid first.')
        if resolution_xy is None:
            resolution_xy = (self.bounds[1] - self.bounds[0] + self.bounds[3] - self.bounds[2]) * 0.5 * 0.01
        if resolution_z is None:
            resolution_z = (self.bounds[5] - self.bounds[4]) * 0.01
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
            surface = section.prob_volume(grid=self._grid, surf=section.create_surface_along_axis(along_axis=sample_axis
                                                                                       , scroll_scale=scroll_size[s_i]
                                                                                       , resolution_xy=resolution_xy
                                                                                       , resolution_z=resolution_z
                                                                                       , grid_bounds=self.bounds))
            section.set_surface(surf=surface)
            section_list.append(section)
        self.sample_data_list.append(section_list)
        self.sample_num += 1
        return section_list

    # 获取采样点相对于格网点的索引, idx=None 返回所有数据， idx=0则返回第一个采样数据
    def get_sample_points_indexex_for_grid_points(self, idx=None) -> list:  # 返回 index
        if self.grid_points is None:
            raise ValueError('Grid points should not be empty.')
        ckt = spt.cKDTree(self.grid_points)
        points_indexes = []
        for sample_id, sample_data in enumerate(self.sample_data_list):
            pid = []
            if idx is not None:
                if idx != sample_id:
                    continue
                return list(sorted(np.unique(pid)))
            if isinstance(sample_data, PointSet):
                sample_points = sample_data.points
                d, pid = ckt.query(sample_points)
                points_indexes.extend(pid)
            elif isinstance(sample_data, (BoreholeSet, SectionSet)):
                sample_points = sample_data.get_points_data().points
                d, pid = ckt.query(sample_points)
                points_indexes.extend(pid)
            points_indexes = list(sorted(np.unique(points_indexes)))
        return points_indexes

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

    def load(self):
        for s_id in np.arange(len(self.sample_data_list)):
            self.sample_data_list[s_id].load()
