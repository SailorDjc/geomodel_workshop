import pyvista as pv
import numpy as np
from vtkmodules.all import vtkPoints, vtkCellArray, vtkCellData, vtkPolyData, vtkSurfaceReconstructionFilter \
    , vtkProbeFilter
from vtkmodules.util import numpy_support
from scipy.interpolate import interp1d, interp2d
import copy
import math
import scipy.spatial as spt
from data_structure.points import PointSet
from data_structure.boreholes import BoreholeSet, Borehole
import time
import os


class Section(object):
    def __init__(self, points: np.ndarray = None, series: np.ndarray = None, sample_spacing=None, sample_grid=None):
        if points is not None and series is not None:
            self.points = points
            self.points_num = points.shape[0]
            self.series = series
            self.scalars = None  # {}
            self.trajectory_line = None  # 轨迹线 numpy.ndarray  3D points
            self.dims = None
            self.bounds = self.get_points_data().bounds

            self.vtk_data = None

            self.tmp_dump_str = 'tmp' + str(int(time.time()))
            self.save_path = None

            if sample_spacing is not None:
                new_surf = self.create_implict_surface_reconstruct()
                self.vtk_data = new_surf
                self.prob_volume(grid=sample_grid, surf=new_surf)

    def get_points_data(self):
        points_data = PointSet(points=self.points, point_labels=self.series)
        if self.scalars is not None:
            for k, v in self.scalars.keys():
                points_data.set_scalars(scalars=v, scalar_name=k)
        return points_data

    # 设置剖面网格点 self.points
    def set_surface(self, surf: pv.PolyData):
        grid_points = surf.cell_centers().points
        self.vtk_data = surf
        self.points = grid_points
        self.points_num = grid_points.shape[0]

    def set_surface_series(self, series: np.ndarray):
        if self.points == 0:
            raise ValueError('Need to set points data first.')
        if series.shape[0] != self.points_num:
            raise ValueError('Series array size is not match to points data.')
        self.series = series
        self.vtk_data['stratum'] = series

    def set_scalars(self, scalars: np.ndarray, scalar_name: str):
        if scalars.shape[0] != self.points_num:
            raise ValueError('Scalars array size is not match to points data.')
        if scalars is None:
            scalars = {}
        scalars[scalar_name] = scalars
        self.vtk_data.cell_data['scalar_name'] = scalars

    # 沿轴向切割，形成一个平面
    def create_surface_along_axis(self, along_axis: str, scroll_scale, resolution_xy, resolution_z
                                  , grid_bounds: np.ndarray = None):
        if scroll_scale < 0 or scroll_scale > 1:
            raise ValueError('The value of parameter scroll_scale must be less than 1 and greater than 0.')
        axis_labels = ['x', 'y', 'z']
        label_to_index = {label: index for index, label in enumerate(axis_labels)}
        if along_axis.lower() in axis_labels:
            axis_index = label_to_index[along_axis.lower()]
            axis_scroll = (grid_bounds[2 * axis_index + 1] - grid_bounds[2 * axis_index]) * scroll_scale \
                          + grid_bounds[2 * axis_index]
            surf = None
            if axis_index == 0:
                line_points = np.array([[grid_bounds[2], axis_scroll, grid_bounds[5]],
                                        [grid_bounds[3], axis_scroll, grid_bounds[5]]])
                surf = self.create_surface_by_sweepline(trajectory_line_xy=line_points, resolution_xy=resolution_xy,
                                                        resolution_z=resolution_z, grid_bounds=grid_bounds)
                # py 与 px 互换
                proj_points = surf.GetPoints()
                proj_points = numpy_support.vtk_to_numpy(proj_points.GetData())  # np.ndarray
                proj_points = proj_points[:, [1, 0, 2]]  # 交换x和z的位置
                proj_points = numpy_support.numpy_to_vtk(proj_points)
                sample_points = vtkPoints()
                sample_points.SetData(proj_points)
                surf.SetPoints(sample_points)
            elif axis_index == 1:
                line_points = np.array([[grid_bounds[0], axis_scroll, grid_bounds[5]],
                                        [grid_bounds[1], axis_scroll, grid_bounds[5]]])
                surf = self.create_surface_by_sweepline(trajectory_line_xy=line_points, resolution_xy=resolution_xy,
                                                        resolution_z=resolution_z, grid_bounds=grid_bounds)
            elif axis_index == 2:
                line_points = np.array([[grid_bounds[0], axis_scroll, grid_bounds[3]],
                                        [grid_bounds[1], axis_scroll, grid_bounds[3]]])
                depth = grid_bounds[3] - grid_bounds[2]  # 沿y方向扫线
                # py - pz 互换
                surf = self.create_surface_by_sweepline(trajectory_line_xy=line_points, resolution_xy=resolution_xy,
                                                        resolution_z=resolution_z, grid_bounds=grid_bounds, depth=depth)
                proj_points = surf.GetPoints()
                proj_points = numpy_support.vtk_to_numpy(proj_points.GetData())  # np.ndarray
                proj_points = proj_points[:, [0, 2, 1]]  # 交换x和z的位置
                proj_points = numpy_support.numpy_to_vtk(proj_points)
                sample_points = vtkPoints()
                sample_points.SetData(proj_points)
                surf.SetPoints(sample_points)
            return surf
        else:
            raise ValueError('Parameter along_axis must be one of characters x, y, z')

    # 通过扫线形成一个平面, 输入线坐标点x序列 必须单增, 否则会出错
    def create_surface_by_sweepline(self, trajectory_line_xy: np.ndarray, resolution_xy, resolution_z
                                    , grid_bounds: np.ndarray
                                    , direction: np.ndarray = np.array([0, 0, -1])
                                    , depth=None, is_extent_xy=False):
        if trajectory_line_xy.shape[0] > 1 and trajectory_line_xy.ndim == 2:
            # 记录轨迹线
            if depth is None:
                depth = grid_bounds[5] - grid_bounds[4]
            if trajectory_line_xy.shape[1] != 2 and trajectory_line_xy.shape[1] != 3:
                raise ValueError('Trajector line ponts must be 2D or 3D')
            # 用包围盒裁剪或延展线段
            self.trajectory_line = self.clip_line_with_bounds(line_points=trajectory_line_xy, grid_bounds=grid_bounds)
            sample_points = vtkPoints()
            polys = vtkCellArray()
            surface = vtkPolyData()
            # 线性插值
            surface_line = self.densify_line_xy_points_with_interp(self.trajectory_line, resolution_xy=resolution_xy)
            cols = math.ceil(depth / resolution_z)
            # 扫线，获取平面每个格网点坐标
            s_id = 0
            for row_i in np.arange(surface_line.shape[0]):
                for col_j in np.arange(cols):
                    point = copy.deepcopy(surface_line[row_i])
                    point[0] = point[0] + direction[0] * col_j * resolution_z
                    point[1] = point[1] + direction[1] * col_j * resolution_z
                    point[2] = point[2] + direction[2] * col_j * resolution_z
                    sample_points.InsertPoint(s_id, point)
                    s_id += 1
            # 创建网格点索引
            for row_i in np.arange(surface_line.shape[0] - 1):
                for cow_j in np.arange(cols - 1):
                    a = cow_j + row_i * cols
                    b = a + 1
                    c = a + cols + 1
                    d = a + cols
                    face = np.array([a, b, c, d])
                    polys.InsertNextCell(4, face)
            surface.SetPoints(sample_points)
            surface.SetPolys(polys)
            vtk_surface = pv.wrap(surface)
            return vtk_surface
        else:
            raise ValueError('Trajectory line input is not supported.')

    # 隐式表面重建，根据一堆网格点来隐式地构建表面， 待修改
    def create_implict_surface_reconstruct(self, grid_points: np.ndarray, sample_spacing, neighbour_size=20):
        surface = vtkSurfaceReconstructionFilter()
        poly_data = vtkPolyData
        points = vtkPoints()
        points.SetData(grid_points)
        poly_data.SetPoints(points)
        surface.SetInputData(poly_data)
        surface.SetNeighborhoodSize(neighbour_size)
        surface.SetSampleSpacing(sample_spacing)
        surface.Update()
        surface = pv.wrap(surface)
        return surface

    # 输入的 line_points 的 x坐标必须是单调增的
    def densify_line_xy_points_with_interp(self, line_points: np.ndarray, resolution_xy, is_smooth=False
                                           , grid_bounds: np.ndarray = None, is_extent=False):
        control_line_xy = line_points[:, 0:2]
        x = control_line_xy[:, 0].ravel()
        y = control_line_xy[:, 1].ravel()
        if not is_smooth:
            interp_1d = interp1d(x, y, kind='linear', fill_value="extrapolate")
        else:
            interp_1d = interp1d(x, y, kind='cubic', fill_value="extrapolate")
        x_new = []
        for x_i in np.arange(x.shape[0]):
            item = x[x_i]
            x_new.append(item)
            if x_i != len(x) - 1:
                dist = spt.distance.euclidean(control_line_xy[x_i], control_line_xy[x_i + 1], w=None)
                p_num = math.floor(dist / resolution_xy) - 1
                while p_num > 0 and (x[x_i + 1] - item) > 0:
                    item = item + (x[x_i + 1] - item) / (p_num + 1)
                    x_new.append(item)
                    p_num -= 1
        if is_extent and grid_bounds is not None:
            if x[0] > grid_bounds[0]:  # x_min
                x_new.insert(0, grid_bounds[0])
            if x[len(x) - 1] < grid_bounds[1]:  # x_max
                x_new.append(grid_bounds[1])
        x_new = np.array(x_new)
        y_new = interp_1d(x_new)
        line_points_new = None
        if line_points.shape[1] == 2:
            line_points_new = np.array(list(zip(x_new, y_new)))
        elif line_points.shape[1] == 3:
            z = line_points[0, 2]
            z_new = np.full_like(x_new, z)
            line_points_new = np.array(list(zip(x_new, y_new, z_new)))
        return line_points_new

    # 地质探针，待修改
    def prob_volume(self, grid, surf: pv.PolyData):
        probe_volume = vtkProbeFilter()
        probe_volume.SetSourceData(grid)
        probe_volume.SetInputData(surf)
        probe_volume.Update()
        out = probe_volume.GetOutput()
        arr = out.GetPointData().GetArray("Scalar Field")
        out.GetPointData().SetScalars(arr)
        out = pv.wrap(out)
        out.plot()

    # 直线段的窗口裁剪 待修改
    def clip_line_with_bounds(self, line_points: np.ndarray, grid_bounds: np.ndarray):
        return line_points

    def save(self, dir_path: str):
        if self.vtk_data is not None and isinstance(self.vtk_data, pv.PolyData):
            self.save_path = os.path.join(dir_path, self.tmp_dump_str)
            self.vtk_data.save(filename=self.save_path+'.vtk')
            self.vtk_data = 'dumped'

    def load(self):
        if self.vtk_data is not None:
            if self.save_path is not None and os.path.exists(self.save_path+'.vtk'):
                self.vtk_data = pv.read(filename=self.save_path+'.vtk')
            else:
                raise ValueError('vtk data file does not exist')


class SectionSet(object):
    def __init__(self):
        self.sections = []
        self.sections_num = 0

    def append(self, section: Section):
        self.sections.append(section)
        self.sections_num += 1

    def __getitem__(self, idx):
        return self.sections[idx]

    def save(self, dir_path: str):
        for s_id in np.arange(len(self.sections)):
            self.sections[s_id].save(dir_path=dir_path)
            self.sections[s_id] = 'dumped'

    def load(self):
        for s_id in np.arange(len(self.sections)):
            self.sections[s_id].load()


if __name__ == "__main__":
    section = Section()
    # surface_x = section.create_surface_along_axis(along_axis='x', scroll_scale=0.5, resolution_xy=0.2,
    # resolution_z=0.2, grid_bounds=np.array([0.0, 4.0, -4.0, 4.0, -4.0, 4.0]))

    # surface.plot(show_edges=True)
    # section.set_surface(surf=surface_x)
