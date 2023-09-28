from data_structure.grids import Grid
from data_structure.sections import SectionSet, Section
from data_structure.boreholes import BoreholeSet, Borehole
from data_structure.points import PointSet
import numpy as np
import random
import copy
import scipy.spatial as spt
import pyvista as pv


class GeoGridDataSampler(object):
    def __init__(self, grid: Grid = None, sample_operator: list = None, **kwargs):
        self.grid = grid
        self.sample_operator = sample_operator
        self.grid_points = None
        self.bounds = None
        self.sample_num = 0  # 采样次数
        self.sample_data_list = []  # 采样数据
        self.kwargs = kwargs

    def execute(self, **kwargs):
        if self.grid is not None:
            self.grid_points = self.grid.grid_points
            self.bounds = self.grid.bounds
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
                        sample_data = self.uniformly_sample_grid_for_points(n_points=points_num)
                        self.sample_data_list.append(sample_data)
                    if sample_op == 'eq_interval_points':
                        if 'interval' in self.kwargs.keys():
                            interval = self.kwargs['interval']
                        else:
                            raise ValueError('Need to set value of parameter interval.')
                        sample_data = self.uniformly_interval_sample_grid_for_points(interval=interval)
                        self.sample_data_list.append(sample_data)
                    if sample_op == 'rand_drills':
                        drill_num = None
                        drill_pos = None
                        if 'drill_pos' in self.kwargs.keys():
                            drill_pos = self.kwargs['drill_pos']
                        if 'drill_num' in self.kwargs.keys():
                            drill_num = self.kwargs['drill_num']
                        if drill_pos is None and drill_num is None:
                            raise ValueError('Need to input parameter drill_pos or drill_num.')
                        sample_data = self.random_sample_grid_for_boreholes(drill_pos=drill_pos, drill_num=drill_num)
                        self.sample_data_list.append(sample_data)
                    if sample_op == 'axis_sections':
                        sample_axis = None
                        section_num = 1
                        scroll_pos = 0.5
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
                        sample_data = self.sample_with_sections_along_axis(sample_axis=sample_axis
                                                                           , section_num=section_num
                                                                           , scroll_pos=scroll_pos
                                                                           , resolution_xy=resolution_xy
                                                                           , resolution_z=resolution_z)
                        self.sample_data_list.append(sample_data)
        else:
            raise ValueError('The grid data is empty.')

    def set_base_grid_by_boreholes(self, boreholes: BoreholeSet, dims: np.ndarray
                                   , bounds: np.ndarray = None, is_regular=True):
        if bounds is None:
            bounds = boreholes.bounds
            grid = Grid(model_name='gme_base_grid')
            if is_regular:
                sample_grid = grid.create_vtk_grid_by_rect_bounds(dim=dims, bounds=bounds)
            else:
                convexhull_2d = boreholes.get_points_data().get_convexhull_2d()
                sample_grid, grid_outline = grid.create_vtk_grid_by_unregular_bounds(dims=dims, bounds=bounds
                                                                                     , convexhull_2d=convexhull_2d)
            # 将钻孔点标签映射到格网上，未知标签的格网点值默认为-1
            grid.set_vtk_grid(grid_vtk=sample_grid)
            self.grid = grid
            self.sample_operator = ['None']
            self.sample_data_list.append(boreholes)
            self.grid_points = self.grid.grid_points
            self.sample_num += 1
            self.set_map_boreholes_labels_to_base_grid(boreholes=boreholes)
        else:
            raise ValueError('Please input bounds array.')

    # 将钻孔采样标签映射到空白网格上
    def set_map_boreholes_labels_to_base_grid(self, boreholes: BoreholeSet):
        # 遍历钻孔每一个分段，寻找与base_grid的交集
        # 默认 -1 为未知值
        cells_series = np.full_like(self.grid.grid_points, fill_value=-1)
        for one_borehole in boreholes:
            for one_layer in one_borehole.holelayer_list:
                top_pos = one_layer.top_pos
                bottom_pos = one_layer.bottom_pos
                label = one_layer.layer_label
                cells_ids = self.grid.vtk_data.find_cells_along_line(pointa=top_pos, pointb=bottom_pos)
                cells_ids = np.array(list(sorted(set(cells_ids))))
                cells_labels = np.full_like(cells_ids, fill_value=label, dtype=int)
                cells_series[cells_ids] = cells_labels
        self.grid.grid_points_series = cells_series

    def uniformly_interval_sample_grid_for_points(self, interval=100):
        if self.grid is None:
            raise ValueError('Need to input grid first.')
        assert self.grid.dims is not None, "grid dimensions are not set"
        pid = np.arange(0, len(self.grid_points), interval)
        pid = list(set(sorted(pid)))
        point_labels = self.grid.grid_points_series[pid]
        sample_points = self.grid_points[pid]
        points_data = PointSet(points=sample_points, point_labels=point_labels)
        self.sample_data_list.append(points_data)
        return points_data

    def uniformly_sample_grid_for_points(self, n_points):
        if self.grid is None:
            raise ValueError('Need to input grid first.')
        assert self.grid.dims is not None, "grid dimensions are not set"
        # proportion of points along each dimension compared to the max number of points along a particular direction
        max_n_pts_dir = max(self.grid.dims)
        px = self.grid.dims[0] / max_n_pts_dir
        py = self.grid.dims[1] / max_n_pts_dir
        pz = self.grid.dims[2] / max_n_pts_dir
        # to uniformly sample the grid to get a total of n_points WHILE also respecting the ratio of points along two
        # directions the equation is
        # (x * px) ( x * py ) * ( x * pz ) = n_points, solve for x
        x = (((max_n_pts_dir ** 3) * n_points) / (self.grid.dims[0] * self.grid.dims[1]
                                                  * self.grid.dims[2])) ** (1.0 / 3.0)
        nx = round(x * px)
        ny = round(x * py)
        nz = round(x * pz)
        bounds = self.grid.vtk_data.GetBounds()
        sample_x = np.linspace(bounds[0], bounds[1], nx)
        sample_y = np.linspace(bounds[2], bounds[3], ny)
        sample_z = np.linspace(bounds[4], bounds[5], nz)
        zz, yy, xx = np.meshgrid(sample_z, sample_y, sample_x, indexing='ij')
        sample_points = np.vstack((xx, yy, zz)).reshape(3, -1).T
        pid = self.grid.vtk_data.find_containing_cell(sample_points)
        point_labels = self.grid.grid_points_series[pid]
        points_data = PointSet(points=sample_points, point_labels=point_labels)
        self.sample_data_list.append(points_data)
        self.sample_num += 1
        return points_data

    def random_sample_grid_for_boreholes(self, drill_pos=None, drill_num=10):
        if self.grid is None:
            raise ValueError('Need to input grid first.')
        horizon_slice = self.grid.vtk_data.slice(normal='z')
        if drill_pos is not None:
            drill_num = len(drill_pos)
        else:
            # 获取grid水平切面，在切面上随机选取cell的中心点，打钻孔
            points = horizon_slice.cell_centers().points.tolist()
            drill_pos = random.sample(points, drill_num)
        borehole_list = BoreholeSet()
        for it, drill_id in enumerate(np.arange(drill_num)):
            pos = drill_pos[drill_id]
            one_borehole = self.sample_grid_for_borehole(pos=pos)
            one_borehole.borehole_id = it
            borehole_list.append(one_borehole=one_borehole)
        self.sample_data_list.append(borehole_list)
        self.sample_num += 1
        return borehole_list

    def sample_grid_for_borehole(self, pos):
        pos_a = copy.deepcopy(pos)
        pos_b = copy.deepcopy(pos)
        pos_a[2] = self.grid.bounds[5]  # z_max
        pos_b[2] = self.grid.bounds[4]  # z_min
        # 沿直线采样
        sample_line = self.grid.vtk_data.sample_over_line(pointa=pos_a, pointb=pos_b)
        line_points = sample_line.cell_centers().points
        pid = self.grid.vtk_data.find_cells_along_line(line_points)
        line_series = self.grid.grid_points_series[pid]
        one_borehole = Borehole(points=line_points, series=line_series, is_virtual=True)
        return one_borehole

    # sample_axis 采样轴 ['x', 'y', 'z'],  is_random=Ture, 切割位置随机
    def sample_with_sections_along_axis(self, sample_axis=None, section_num=1, scroll_pos: list = None
                                        , resolution_xy=None, resolution_z=None):
        if self.grid is None:
            raise ValueError('Need to input grid first.')
        section_list = SectionSet()
        for s_i in np.arange(section_num):
            if scroll_pos is None:  # 随机剖切
                scroll_size = scroll_pos[s_i]
            else:
                scroll_size = random.uniform(0, 1)
            section = Section()
            section.prob_volume(grid=self.grid, surf=section.create_surface_along_axis(along_axis=sample_axis
                                                                                       , scroll_scale=scroll_size
                                                                                       , resolution_xy=resolution_xy
                                                                                       , resolution_z=resolution_z
                                                                                       , grid_bounds=self.bounds))
            section_list.append(section)
        self.sample_num += 1
        return section_list

    # 获取采样点相对于格网点的索引(通过最近邻搜索计算得到) , idx=None 返回所有数据， idx=0则返回第一个采样数据
    def get_sample_points_indexex_for_grid_points(self, idx=None):
        ckt = spt.cKDTree(self.grid_points)
        points_indexes = []
        for sample_id, sample_data in enumerate(self.sample_data_list):
            pid = []
            if isinstance(sample_data, PointSet):
                sample_points = sample_data.points
                d, pid = ckt.query(sample_points)
                points_indexes.extend(pid)
            elif isinstance(sample_data, (BoreholeSet, SectionSet)):
                sample_points = sample_data.get_points_data().points
                d, pid = ckt.query(sample_points)
                points_indexes.extend(pid)
            if idx is not None and idx == sample_id:
                return list(set(sorted(pid)))
            points_indexes = list(set(sorted(points_indexes)))
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
