import pyvista as pv
import numpy as np
from vtkmodules.all import vtkCellArray, vtkPoints, vtkCellData, vtkPolyData \
    , vtkProbeFilter
from vtkmodules.util import numpy_support
import copy
import math
from data_structure.points import PointSet
from data_structure.boreholes import BoreholeSet, Borehole
from utils.vtk_utils import add_np_property_to_vtk_object
import time
import os
import pickle
from vtkmodules.all import vtkStructuredGrid, vtkImageData, vtkPoints, vtkCellCenters
from utils.math_libs import densify_line_xy_points_with_interp
from utils.vtk_utils import create_implict_surface_reconstruct


# 1. 外部输入vtk剖面数据
#   vtk_data:         外部输入vtk剖面数据
#   vtk_data_path:    外部vtk剖面文件路径，与vtk_data等价，两者只输入一个，若两者都有值，那么以vtk_data作为输入
# 2. 外部输入点云数据
#   points:           外部输入剖面网格点
#   series:           外部输入剖面网格点标签，与points对应，若存在外部输入vtk数据，则忽略
# 3. 通过剖切外部输入的采样网格，得到剖面
#   sample_spacing:   重建的剖面网格的网格间隔
#   sample_grid:      外部输入的采样三维网格模型，通过采样该网格为剖面附加地质类别属性，这里的网格是三维空间网格，1中的是二维剖面网格
#   name:             剖面名称，若存在外部输入vtk文件，且name未指定，则以vtk文件名作为name
class Section(object):
    def __init__(self, vtk_data=None, vtk_data_path=None, points: np.ndarray = None, series: np.ndarray = None
                 , sample_spacing=None, sample_grid=None, name=None, dir_path=None):
        self.name = name
        # if points is not None and series is not None:
        self.points = points
        if self.points is not None:
            self.points_num = points.shape[0]
        self.series = series
        # 标量
        self.scalars = None  # {}
        self.scalar_grad = None  # 梯度
        self.scalar_grad_norm = None

        self.trajectory_line = None  # 轨迹线 numpy.ndarray  3D points
        self.dims = None
        self.resolution = None
        self.vtk_data = None

        self._center = None
        self.classes = None
        self.classes_num = 0
        self.label_dict = None

        self.tmp_dump_str = 'tmp_sec' + str(int(time.time()))
        # 如果存在外部输入的vtk剖面数据
        if vtk_data is not None or (vtk_data_path is not None and os.path.exists(vtk_data_path)):
            self.vtk_data = vtk_data
            if self.vtk_data is None:
                self.vtk_data = pv.read(filename=vtk_data_path)
                file_name = os.path.basename(vtk_data_path)
                self.name = file_name.split('.')[0]
            self.points = None
            self.series = None
            # 以vtk数据更新points和labels
            if isinstance(vtk_data, pv.RectilinearGrid):
                self.dims = vtk_data.GetDimensions()
                self.points = vtk_data.cell_centers().points
                self.points_num = self.points.shape[0]
                # 从vtk数据中获取标签
                self.standardize_labels_from_vtk_data()
            if isinstance(vtk_data, (pv.PolyData, pv.UnstructuredGrid)):
                self.dims = None
                self.points = vtk_data.cell_centers().points
                self.points_num = self.points.shape[0]
                self.standardize_labels_from_vtk_data()
        # 当只有一片散点，可以创建隐式曲面，通过曲面来剖切采样网格获取地质类别属性
        if sample_spacing is not None and self.points is not None and sample_grid is not None and vtk_data is None \
                and vtk_data_path is None:
            self.points_num = points.shape[0]
            self.bounds = self.get_points_data().bounds
            new_surf = create_implict_surface_reconstruct(self.points, sample_spacing=sample_spacing)
            self.vtk_data = new_surf
            self.prob_volume(grid=sample_grid, surf=new_surf)
            # 从vtk数据中获取标签
            self.standardize_labels_from_vtk_data()

        self.dir_path = dir_path

    # pnt_file: png图片路径
    # corner_coordinates: 角点坐标,用来配准
    # pixel_label_dict: 像素值-标签字典
    def set_input_pngdata(self, png_file, corner_coordinates, pixel_label_dict):
        pass

    @property
    def grid_points_series(self):
        return self.series

    @grid_points_series.setter
    def grid_points_series(self, series):
        self.series = series

    def set_vtk_grid(self, grid_vtk):
        if isinstance(grid_vtk, (pv.RectilinearGrid, vtkImageData)):
            self.dims = grid_vtk.GetDimensions()
        else:
            self.dims = None
        self.bounds = np.array(grid_vtk.bounds)
        self.points = grid_vtk.cell_centers().points
        self.series = grid_vtk.active_scalars
        self.points_num = self.points.shape[0]
        self.vtk_data = grid_vtk
        self.scalars = None
        self.scalar_grad = None
        self.scalar_grad_norm = None

    # 根据范围筛选数据，返回PointSet对象
    def search_by_rect2d(self, rect2d):
        cur_points_data = self.get_points_data()
        return cur_points_data.search_by_rect2d(rect2d=rect2d)

    # 恢复初始标签
    def restore_labels(self):
        if self.label_dict is not None and self.series is not None:
            label_dict = {}
            for k, v in self.label_dict.items():
                label_dict[v] = k
            self.series = np.vectorize(label_dict.get)(np.array(self.series))

    def get_points_data(self):
        points_data = PointSet(points=self.points, point_labels=self.series)
        if self.scalars is not None:
            for k, v in self.scalars.keys():
                points_data.set_scalars(scalars=v, scalar_name=k)
        return points_data

    def get_points_num(self):
        points_data_list = self.get_points_data()
        return len(points_data_list)

    def get_classes(self):
        if self.classes is None:
            if self.series is None:
                raise ValueError('This grid lacks labels.')
            else:
                self.classes = sorted(np.unique(self.series))
                self.classes_num = len(self.classes)
        return self.classes

    # 将labels映射为连续自然数，从0开始，或按照传入的字典进行标签转换 , default_value 默认未知标签为-1
    # label_dict:  dict  标签映射字典，可以手动设置，例如{1:0, 2:1, 3:2, 4:3}
    def standardize_labels_from_vtk_data(self, label_dict: dict = None, default_value=-1):
        # 从vtk数据中获取属性数据，如果属性数据为空，则报错
        series_labels = self.vtk_data.active_scalars
        if series_labels is None:
            raise ValueError('The input data has not scalar values.')
        old_label = np.trunc(series_labels)
        unique_label = np.unique(old_label)
        sorted_label = sorted(unique_label)
        if label_dict is not None:
            # 判断label_dict 是否符合要求
            for idx, item in enumerate(sorted_label):
                if item not in label_dict.keys():
                    raise ValueError('The input label_dict is invalid.')
                if idx + 1 < len(sorted_label) and item + 1 != label_dict[idx + 1]:
                    raise ValueError('The input label_dict is invalid.')
            new_label = np.vectorize(label_dict.get)(np.array(old_label))
        else:
            label_dict = {}
            for idx, item in enumerate(sorted_label):
                if item == default_value:  # 对于默认未知值则不改变
                    label_dict[item] = item
                    continue
                label_dict[item] = idx
            new_label = np.vectorize(label_dict.get)(np.array(old_label))
        self.label_dict = label_dict
        self.classes_num = len(label_dict.values())
        self.classes = np.array(list(label_dict.values()))
        self.series = new_label
        self.__add_properties_to_vtk_object_if_present(grid=self)

    @property
    def center(self):
        if self.bounds is not None:
            center_x = (self.bounds[0] + self.bounds[1]) * 0.5
            center_y = (self.bounds[2] + self.bounds[3]) * 0.5
            center_z = (self.bounds[4] + self.bounds[5]) * 0.5
            self._center = np.array([center_x, center_y, center_z])
            return self._center
        else:
            raise ValueError('This section data is empty.')

    @staticmethod
    def __add_properties_to_vtk_object_if_present(grid):
        assert isinstance(grid, Section), "Input data should be of Section type"
        assert grid.vtk_data is not None, "there is no grid vtk object"
        if grid.series is not None:
            add_np_property_to_vtk_object(grid.vtk_data, "Scalar Field", grid.series, continuous=False)
        if grid.scalars is not None and isinstance(grid.scalars, dict):
            for scalar_name, scalars_values in grid.scalars.keys():  # series_name = "Scalar Field" + str(i)
                add_np_property_to_vtk_object(grid.vtk_data, scalar_name, scalars_values)
        if grid.scalar_grad is not None and isinstance(grid.scalar_grad, dict):
            for scalar_name, scalars_grad_values in grid.scalar_grad.keys():  # "Scalar Gradient" + str(i)
                add_np_property_to_vtk_object(grid.vtk_data, scalar_name, scalars_grad_values)
        if grid.scalar_grad_norm is not None and isinstance(grid.scalar_grad, dict):
            for scalar_name, scalars_grad_norm_values in grid.scalar_grad_norm.keys():  # "Scalar Gradient Norm"+str(i)
                add_np_property_to_vtk_object(grid.vtk_data, scalar_name, scalars_grad_norm_values)
        return grid

    # 设置剖面网格点 self.points
    def set_surface(self, surf: pv.PolyData):
        grid_points = surf.cell_centers().points
        self.vtk_data = surf
        self.points = grid_points
        self.points_num = grid_points.shape[0]
        # 从vtk数据中获取标签
        self.standardize_labels_from_vtk_data()

    def set_series(self, series: np.ndarray):
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
        self.resolution = np.array([resolution_xy, resolution_xy, resolution_z])
        if trajectory_line_xy.shape[0] > 1 and trajectory_line_xy.ndim == 2:
            # 记录轨迹线
            if depth is None:
                depth = grid_bounds[5] - grid_bounds[4]
            if trajectory_line_xy.shape[1] != 2 and trajectory_line_xy.shape[1] != 3:
                raise ValueError('trajectory line points must be 2D or 3D')
            # 用包围盒裁剪或延展线段
            # self.trajectory_line = self.clip_line_with_bounds(line_points=trajectory_line_xy, grid_bounds=grid_bounds)
            self.trajectory_line = trajectory_line_xy
            sample_points = vtkPoints()
            polys = vtkCellArray()
            surface = vtkPolyData()
            # 线性插值
            surface_line = densify_line_xy_points_with_interp(self.trajectory_line, resolution_xy=resolution_xy)
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

    # 地质探针，待修改
    def prob_volume(self, grid, surf: pv.PolyData):
        probe_volume = vtkProbeFilter()
        probe_volume.SetSourceData(grid.vtk_data)
        probe_volume.SetInputData(surf)
        probe_volume.Update()
        out = probe_volume.GetOutput()
        arr = out.GetPointData().GetArray("stratum")
        out.GetPointData().SetScalars(arr)
        out = pv.wrap(out)
        self.vtk_data = out
        return out

    def detach_vtk_component_with_label(self):
        if self.vtk_data is None:
            raise ValueError('The vtk_data of the section is empty.')
        classes = self.get_classes()
        vtk_dict = {}
        if isinstance(self.vtk_data, (pv.UnstructuredGrid, pv.PolyData, pv.StructuredGrid)):
            for item in classes:
                vtk_dict[item] = self.vtk_data.threshold(value=[item - 0.001, item + 0.001])
        return vtk_dict

    # 直线段的窗口裁剪 待修改
    def clip_line_with_bounds(self, line_points: np.ndarray, grid_bounds: np.ndarray):
        pass

    def __len__(self):
        return self.points_num

    def save(self, dir_path: str, out_name: str = None):
        self.tmp_dump_str = 'tmp_sec' + str(int(time.time()))
        self.dir_path = dir_path
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        save_dir = os.path.join(self.dir_path, self.tmp_dump_str)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.vtk_data is not None and isinstance(self.vtk_data, pv.PolyData):
            save_path = os.path.join(save_dir, self.tmp_dump_str)
            self.vtk_data.save(filename=save_path + '.vtk')
            self.vtk_data = 'dumped'
        file_name = self.tmp_dump_str
        if out_name is not None:
            file_name = out_name
        file_path = os.path.join(save_dir, file_name + '.dat')
        out_put = open(file_path, 'wb')
        out_str = pickle.dumps(self)
        out_put.write(out_str)
        out_put.close()
        return self.__class__.__name__, file_path

    def load(self, dir_path=None):
        if self.dir_path is not None:
            if dir_path is not None:
                self.dir_path = dir_path
            save_path = os.path.join(self.dir_path, self.tmp_dump_str)
            if self.vtk_data == 'dumped':
                if os.path.exists(save_path + '.vtk'):
                    self.vtk_data = pv.read(filename=save_path + '.vtk')
                else:
                    raise ValueError('vtk data file does not exist')


class SectionSet(object):
    def __init__(self, name=None, dir_path=None):
        self.sections = []
        self.sections_num = 0
        self.name = name
        self.dir_path = dir_path
        self.tmp_dump_str = 'tmp_secs' + str(int(time.time()))
        self.vtk_data = None
        self.label_dict = None
        self._center = None

    def append(self, section: Section):
        self.sections.append(section)
        self.sections_num += 1

    def get_points_data(self):
        points_data_list = []
        for sec in self.sections:
            points_data_list.append(sec.get_points_data())
        if len(points_data_list) > 0:
            points_data_list = PointSet.points_data_merge(points_data_list=points_data_list)
        return points_data_list

    # 恢复初始标签
    def restore_labels(self):
        for si in range(len(self.sections)):
            self.sections[si].restore_labels()

    # 根据范围筛选数据，返回PointSet对象
    def search_by_rect2d(self, rect2d):
        cur_points_data = self.get_points_data()
        return cur_points_data.search_by_rect2d(rect2d=rect2d)

    def get_points_num(self):
        points_data_list = self.get_points_data()
        return len(points_data_list)

    def plot(self):
        if len(self.sections) > 0:
            sec_list = []
            for sec in self.sections:
                if isinstance(sec, Section):
                    if sec.vtk_data is not None:
                        sec_list.append(sec)
            if len(sec_list) > 0:
                secs = pv.MultiBlock()
                for sec in sec_list:
                    secs.append(sec.vtk_data)
                secs.plot()

    def get_section(self, idx):
        if idx < 0 or idx >= len(self.sections):
            raise ValueError('The input index is out of range.')
        return self.sections[idx]

    @property
    def center(self):
        if self.sections_num > 0:
            for sec in self.sections:
                if self._center is None:
                    self._center = sec.center
                elif sec.center is not None:
                    self._center = (self._center + sec.center) * 0.5
            if self._center is not None:
                return self._center
            else:
                raise ValueError('This sections data is empty.')
        else:
            raise ValueError('This sections data is empty.')

    def __len__(self):
        return len(self.sections)

    def __getitem__(self, idx):
        return self.get_section(idx)

    def save(self, dir_path: str, out_name: str = None):
        self.tmp_dump_str = 'tmp_secs' + str(int(time.time()))
        self.dir_path = dir_path
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        save_dir = os.path.join(self.dir_path, self.tmp_dump_str)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.vtk_data is not None:
            for s_id in np.arange(len(self.sections)):
                self.sections[s_id].save(dir_path=save_dir)
                self.sections[s_id] = 'dumped'
        file_name = self.tmp_dump_str
        if out_name is not None:
            file_name = out_name
        file_path = os.path.join(save_dir, file_name + '.dat')
        out_put = open(file_path, 'wb')
        out_str = pickle.dumps(self)
        out_put.write(out_str)
        out_put.close()
        return self.__class__.__name__, file_path

    def load(self):
        for s_id in np.arange(len(self.sections)):
            self.sections[s_id].load()


if __name__ == "__main__":
    section = Section()
    # surface_x = section.create_surface_along_axis(along_axis='x', scroll_scale=0.5, resolution_xy=0.2,
    # resolution_z=0.2, grid_bounds=np.array([0.0, 4.0, -4.0, 4.0, -4.0, 4.0]))

    # surface.plot(show_edges=True)
    # section.set_surface(surf=surface_x)
