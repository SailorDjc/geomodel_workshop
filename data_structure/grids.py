import torch
from torch.utils.data import Dataset
import numpy as np
from vtkmodules.all import vtkStructuredGrid, vtkImageData, vtkPoints, vtkCellCenters
from vtkmodules.util import numpy_support
from utils.vtk_utils import add_np_property_to_vtk_object
import pyvista as pv
import scipy.spatial as spt
from data_structure.points import PointSet, get_bounds_from_coords
from data_structure.boreholes import Borehole, BoreholeSet
from data_structure.sections import Section, SectionSet
import time
import copy
import os


def generate_vtk_structure_grid_and_grid_points(bounds, xy_resolution, z_resolution):
    if bounds is None:
        raise ValueError('Bound array not support none value')
    else:
        if not isinstance(bounds, np.ndarray):
            raise TypeError('Bound array only support numpy.ndarray type')
        else:
            if bounds.size != 6:
                raise ValueError('Bound array is not the appropriate size')
            else:
                xmin = bounds[0]
                xmax = bounds[1]
                ymin = bounds[2]
                ymax = bounds[3]
                zmin = bounds[4]
                zmax = bounds[5]

            nx = int((xmax - xmin) / xy_resolution) + 1
            ny = int((ymax - ymin) / xy_resolution) + 1
            nz = int((zmax - zmin) / z_resolution) + 1
            dims = np.array([nx, ny, nz])
            sample_x = np.linspace(xmin, xmax, nx)
            sample_y = np.linspace(ymin, ymax, ny)
            sample_z = np.linspace(zmin, zmax, nz)
            zz, yy, xx = np.meshgrid(sample_z, sample_y, sample_x, indexing='ij')
            grid_pts = np.vstack((xx, yy, zz)).reshape(3, -1).T

            # Create VTK Structured Grid
            sgrid = vtkStructuredGrid()
            sgrid.SetDimensions(dims[0], dims[1], dims[2])
            points = vtkPoints()
            points.SetData(numpy_support.numpy_to_vtk(grid_pts, deep=True))
            sgrid.SetPoints(points)

            return grid_pts, sgrid


def generate_vtk_imagedata_grid_and_grid_points(bounds, xy_resolution, z_resolution):
    if not isinstance(bounds, np.ndarray):
        raise TypeError('Bound array must be numpy.ndarray')
    else:
        if bounds.size != 6:
            raise ValueError('Bound array is not the appropriate size')
        else:
            xmin = bounds[0]
            xmax = bounds[1]
            ymin = bounds[2]
            ymax = bounds[3]
            zmin = bounds[4]
            zmax = bounds[5]

        nx = int((xmax - xmin) / xy_resolution) + 1
        ny = int((ymax - ymin) / xy_resolution) + 1
        nz = int((zmax - zmin) / z_resolution) + 1
        origin = np.array([xmin, ymin, zmin])

        grd = vtkImageData()
        grd.SetOrigin(origin)
        grd.SetSpacing(xy_resolution, xy_resolution, z_resolution)
        grd.SetDimensions(nx, ny, nz)

        centers = vtkCellCenters()
        centers.SetInputData(grd)
        centers.Update()
        centers = numpy_support.vtk_to_numpy(centers.GetOutput().GetPoints().GetData())
        return centers, grd


class Grid(object):
    """Dataset manager for Grid points. Actual grid coordinates are computed prior and used as input to create this
     object. Grid should be rectilinear. Points are ordered in the following manner: x-axis fastest, y-axis next, z-axis
     slowest.
     Coordinate Scaling/Normalization occurs after object created
     Dataset includes:
     points: a matrix containing all 3D grid points [x, y, z]"""

    def __init__(self, model_name=None, grid_vtk=None, grid_vtk_path=None, grid_points: np.ndarray = None, series=None):
        self.bounds = None
        self._center = None
        self.model_name = model_name
        self.grid_points = None
        self.grid_points_num = 0
        self.grid_points_series = None  # 地层标签 np.ndarray([self.grid_points_num, ])
        # -1表示待预测值
        self.scalar_series = None  # dict 每个地层一个标量场, key: value  key-地层名， value-标量场值

        self.scalar_grad = None  # 梯度  dict
        self.scalar_grad_norm = None  # 正则化梯度  dict

        self.label_dict = None  # 标签映射字典
        self.classes_num = 0  #
        self.classes = None  # 类别

        self.tmp_dump_str = 'tmp' + str(int(time.time()))
        self.save_path = None

        # 外部输入的vtk规则网格
        self.vtk_data = grid_vtk
        if self.vtk_data is None and (grid_vtk_path is None or not os.path.exists(grid_vtk_path)):
            self.dims = None
            self.bounds = None
        else:
            # 传入的grid_vtk优先级更高，传入路径次之
            if self.vtk_data is None and grid_vtk_path is not None and os.path.exists(grid_vtk_path):
                self.vtk_data = pv.read(grid_vtk_path)
            if isinstance(self.vtk_data, pv.RectilinearGrid):
                self.dims = grid_vtk.GetDimensions()
                self.bounds = np.array(grid_vtk.bounds)
                self.grid_points = self.vtk_data.cell_centers().points
                self.standardize_labels_from_vtk_data()  # 处理标签
                self.grid_points_num = self.grid_points.shape[0]

    # 将labels映射为连续自然数，从0开始，或按照传入的字典进行标签转换 , default_value 默认未知标签为-1
    def standardize_labels_from_vtk_data(self, label_dict: dict = None, default_value=-1):
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
                if idx + 1 < len(sorted_label) and item + 1 != label_dict[idx+1]:
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
        self.grid_points_series = new_label

    # 可以自由创建 1维、2维、3维网格，如果是规则沿轴向，则只需要dim和bounds参数，也可以通过分割间断点序列xx,yy,zz来自定义网格
    def create_vtk_grid_by_rect_bounds(self, dim: np.ndarray = None, bounds: np.ndarray = None, grid_buffer_xy=0):
        if dim is None or bounds is None:
            raise ValueError('Bounds array can not be None')
        else:
            nx = dim[0]
            ny = dim[1]
            nz = dim[2]
            min_x = bounds[0] - grid_buffer_xy
            max_x = bounds[1] + grid_buffer_xy
            min_y = bounds[2] - grid_buffer_xy
            max_y = bounds[3] + grid_buffer_xy
            min_z = bounds[4]
            max_z = bounds[5]
            xrng = np.linspace(start=min_x, stop=max_x, num=nx)
            yrng = np.linspace(start=min_y, stop=max_y, num=ny)
            zrng = np.linspace(start=min_z, stop=max_z, num=nz)
            vtk_grid = pv.RectilinearGrid(xrng, yrng, zrng)
            return vtk_grid

    # 在规则格网的基础上，通过一个2d凸包范围切割格网
    def create_vtk_grid_by_unregular_bounds(self, dims: np.ndarray, bounds: np.ndarray
                                            , convexhull_2d: np.ndarray, cell_density=None):
        top_surface_points = copy.deepcopy(convexhull_2d)
        top_surface_points[:, 2] = bounds[5]  # z_max
        bottom_surface_points = copy.deepcopy(convexhull_2d)
        bottom_surface_points[:, 2] = bounds[4]  # z_min
        # 面三角化
        surface_points = np.concatenate((top_surface_points, bottom_surface_points), axis=0)
        # 顶面
        pro_point_2d = top_surface_points[:, 0:2]
        points_num = len(top_surface_points)
        tri = spt.Delaunay(pro_point_2d)
        tet_list = tri.simplices
        faces_top = []
        for it, tet in enumerate(tet_list):
            face = np.int64([3, tet[0], tet[1], tet[2]])
            faces_top.append(face)
        faces_top = np.int64(faces_top)
        # 底面的组织与顶面相同，face中的点号加一个points_num
        faces_bottom = []
        for it, face in enumerate(faces_top):
            face_new = copy.deepcopy(face)
            face_new[1:4] = np.add(face[1:4], points_num)
            faces_bottom.append(face_new)
        faces_bottom = np.int64(faces_bottom)
        faces_total = np.concatenate((faces_top, faces_bottom), axis=0)
        # 侧面
        # 需要先将三维度点投影到二维，上下面构成一个矩形，三角化
        surf_line_pnt_id = list(np.arange(points_num))
        surf_line_pnt_id.append(0)  # 环状线，首位相连
        surf_line_pnt_id_0 = copy.deepcopy(surf_line_pnt_id)  #
        surf_line_pnt_id_0 = np.add(surf_line_pnt_id_0, points_num)
        surf_line_pnt_id_total = np.concatenate((surf_line_pnt_id, surf_line_pnt_id_0), axis=0)
        top_line = []
        bottom_line = []
        for lit in np.arange(points_num + 1):
            xy_top = np.array([lit, self.bounds[5]])
            xy_bottom = np.array([lit, self.bounds[4]])
            top_line.append(xy_top)
            bottom_line.append(xy_bottom)
        top_line = np.array(top_line)
        bottom_line = np.array(bottom_line)
        line_pnt_total = np.concatenate((top_line, bottom_line), axis=0)
        # 矩形三角化
        tri = spt.Delaunay(line_pnt_total)
        tet_list = tri.simplices
        faces_side = []
        for it, tet in enumerate(tet_list):
            item_0 = tet[0]
            item_1 = tet[1]
            item_2 = tet[2]
            face = np.int64(
                [3, surf_line_pnt_id_total[item_0], surf_line_pnt_id_total[item_1], surf_line_pnt_id_total[item_2]])
            faces_side.append(face)
        faces_side = np.int64(faces_side)
        faces_total = np.concatenate((faces_total, faces_side), axis=0)
        convex_surface = pv.PolyData(surface_points, faces=faces_total)
        line_boundary = []
        line_top = [len(surf_line_pnt_id)]
        line_bottom = [len(surf_line_pnt_id)]
        for lid in np.arange(len(surf_line_pnt_id)):
            line_top.append(surf_line_pnt_id[lid])
            line_bottom.append(surf_line_pnt_id_0[lid])
            line_of_side = [2, surf_line_pnt_id[lid], surf_line_pnt_id_0[lid]]
            line_boundary.append(np.int64(line_of_side))
        line_top = np.int64(line_top)
        line_bottom = np.int64(line_bottom)
        line_boundary.append(line_top)
        line_boundary.append(line_bottom)
        line_boundary = np.concatenate(line_boundary, axis=0)
        grid_outline = pv.PolyData(surface_points, lines=line_boundary)
        if cell_density is None:
            if dims is None:
                raise ValueError('Need to input dims parameters.')
            x_length = bounds[1] - bounds[0]
            y_length = bounds[3] - bounds[2]
            z_length = bounds[5] - bounds[4]
            cell_density = np.ndarray([float(x_length / dims[0]), float(y_length / dims[1]), float(z_length / dims[2])])
        sample_grid = pv.voxelize(convex_surface, density=cell_density)
        return sample_grid, grid_outline

    def set_vtk_grid(self, grid_vtk):
        self.dims = grid_vtk.GetDimensions()
        self.bounds = np.array(grid_vtk.bounds)
        self.grid_points = grid_vtk.cell_centers().points
        self.grid_points_series = grid_vtk.active_scalars
        self.grid_points_num = self.grid_points.shape[0]
        self.vtk_data = grid_vtk
        self.scalar_series = None
        self.scalar_grad = None
        self.scalar_grad_norm = None

    def resample_grid(self, dim: np.ndarray, is_replace=True):
        if self.vtk_data is None:
            raise ValueError('')
        else:
            new_vtk_grid = self.create_vtk_grid_by_rect_bounds(dim=dim, bounds=self.bounds)
            new_vtk_grid_points = new_vtk_grid.cell_centers().points

            ckt = spt.cKDTree(self.grid_points)
            d, pid = ckt.query(new_vtk_grid_points)
            if is_replace:
                self.grid_points_series = self.grid_points_series[pid]
                self.grid_points = new_vtk_grid_points
                self.grid_points_num = new_vtk_grid_points.shape[0]
                # 属性数据
                if self.scalar_series is not None and isinstance(self.scalar_series, dict):
                    for key, value in self.scalar_series.keys():
                        self.scalar_series[key] = value[pid]
                if self.scalar_grad is not None and isinstance(self.scalar_grad, dict):
                    for key, value in self.scalar_grad.keys():
                        self.scalar_grad[key] = value[pid]
                if self.scalar_grad_norm is not None and isinstance(self.scalar_grad, dict):
                    for key, value in self.scalar_grad_norm.keys():
                        self.scalar_grad_norm[key] = value[pid]
                self.vtk_data = new_vtk_grid
                self.dims = new_vtk_grid.GetDimensions()
                self.bounds = np.array(new_vtk_grid.bounds)
                return self.__add_properties_to_vtk_object_if_present(grid=self)
            else:
                new_grid = Grid(model_name=self.model_name, grid_vtk=new_vtk_grid)
                if self.grid_points_series is not None:
                    new_grid.grid_points_series = self.grid_points_series[pid]
                if self.scalar_series is not None and isinstance(self.scalar_series, dict):
                    for scalar_name, scalars_values in self.scalar_series.keys():
                        new_grid.set_scalar_pred(scalar_pred=scalars_values[pid], series_name=scalar_name)
                if self.scalar_grad is not None and isinstance(self.scalar_grad, dict):
                    for scalar_name, scalars_grad_values in self.scalar_grad.keys():
                        new_grid.set_scalar_grad(scalar_grad_pred=scalars_grad_values[pid], series_name=scalar_name)
                if self.scalar_grad_norm is not None and isinstance(self.scalar_grad, dict):
                    for scalar_name, scalar_grad_norm_values in self.scalar_grad_norm.keys():
                        new_grid.set_scalar_grad_norm(scalar_grad_norm_pred=scalar_grad_norm_values[pid]
                                                      , series_name=scalar_name)
                return self.__add_properties_to_vtk_object_if_present(grid=new_grid)

    def __len__(self):
        return self.grid_points_num

    def __getitem__(self, idx):
        return self.grid_points[idx], self.grid_points_series[idx]

    def get_points_data(self):
        points_data = PointSet()
        if self.grid_points is not None:
            points_data.set_points(self.grid_points)
        if self.grid_points_series is not None:
            points_data.set_labels(self.grid_points_series)
        if self.scalar_series is not None:
            for scalar_name, scalar_value in self.scalar_series.keys():
                points_data.set_scalars(scalars=scalar_value, scalar_name=scalar_name)
        if self.scalar_grad is not None:
            for scalar_name, scalars_grad_value in self.scalar_grad.keys():
                points_data.set_scalars_grad(scalars_grad=scalars_grad_value, scalar_name=scalar_name)
        if self.scalar_grad_norm is not None:
            for scalar_name, scalars_grad_norm_value in self.scalar_grad_norm.keys():
                points_data.set_scalars_grad_norm(scalars_grad_norm=scalars_grad_norm_value, scalar_name=scalar_name)
        return points_data

    def transform(self, scalar):
        self.grid_points = scalar.transform(self.grid_points)

    @property
    def center(self):
        if self.bounds is not None:
            center_x = (self.bounds[0] + self.bounds[1]) * 0.5
            center_y = (self.bounds[2] + self.bounds[3]) * 0.5
            center_z = (self.bounds[4] + self.bounds[5]) * 0.5
            self._center = np.array([center_x, center_y, center_z])
            return self._center
        else:
            raise ValueError('This grid data is empty.')

    def send_grid_points_to_gpu(self, rank):
        self.grid_points = self.grid_points.to(rank)

    # 网格点标签
    def set_series_label(self, grid_point_series: np.ndarray):
        if self.grid_points is not None and self.grid_points.shape[0] == grid_point_series.shape[0]:
            self.grid_points_series = grid_point_series
        else:
            raise ValueError('Input data is invalid.')

    def set_scalar_pred(self, scalar_pred, series_id: int = None, series_name: str = None):
        scalar_name = series_name
        if series_id is not None:
            scalar_name = "Scalar Field" + str(series_id)
        if self.scalar_series is None:
            self.scalar_series = {}
        self.scalar_series[scalar_name] = scalar_pred

    def set_scalar_grad(self, scalar_grad_pred, series_id: int = None, series_name: str = None):
        scalar_name = series_name
        if series_id is not None:
            scalar_name = "Scalar Gradient" + str(series_id)
        if self.scalar_grad is None:
            self.scalar_grad = {}
        self.scalar_grad[scalar_name] = scalar_grad_pred

    def set_scalar_grad_norm(self, scalar_grad_norm_pred, series_id: int = None, series_name: str = None):
        scalar_name = series_name
        if series_id is not None:
            scalar_name = "Scalar Gradient Norm" + str(series_id)
        if self.scalar_grad_norm is None:
            self.scalar_grad_norm = {}
        self.scalar_grad_norm[scalar_name] = scalar_grad_norm_pred

    @staticmethod
    def __add_properties_to_vtk_object_if_present(grid):
        assert isinstance(grid, Grid), "Input data should be of Grid type"
        assert grid.vtk_data is not None, "there is no grid vtk object"
        if grid.grid_points_series is not None:
            add_np_property_to_vtk_object(grid.vtk_data, "Scalar Field", grid.grid_points_series, continuous=False)
        if grid.scalar_series is not None and isinstance(grid.scalar_series, dict):
            for scalar_name, scalars_values in grid.scalar_series.keys():  # series_name = "Scalar Field" + str(i)
                add_np_property_to_vtk_object(grid.vtk_data, scalar_name, scalars_values)
        if grid.scalar_grad is not None and isinstance(grid.scalar_grad, dict):
            for scalar_name, scalars_grad_values in grid.scalar_grad.keys():  # "Scalar Gradient" + str(i)
                add_np_property_to_vtk_object(grid.vtk_data, scalar_name, scalars_grad_values)
        if grid.scalar_grad_norm is not None and isinstance(grid.scalar_grad, dict):
            for scalar_name, scalars_grad_norm_values in grid.scalar_grad_norm.keys():  # "Scalar Gradient Norm"+str(i)
                add_np_property_to_vtk_object(grid.vtk_data, scalar_name, scalars_grad_norm_values)
        return grid

    def process_model_outputs(self, map_to_original_class_ids=None):
        # 1) remap unit predictions to original class ids
        if isinstance(self.grid_points_series, np.ndarray):
            if isinstance(map_to_original_class_ids, dict):
                self.grid_points_series = np.vectorize(map_to_original_class_ids.get)(self.grid_points_series)
        # 2) add model properties to vtk object
        self.vtk_data = self.__add_properties_to_vtk_object_if_present(grid=self).vtk_data

    def match_external_grid_to_this_grid(self, external_grid):
        if isinstance(external_grid, Grid):
            if self.grid_points is not None and self.vtk_data is not None:
                external_points_data = external_grid.get_points_data()
                ckt = spt.cKDTree(external_points_data.points)
                d, pid = ckt.query(self.grid_points)
                self.grid_points_series = external_points_data.labels[pid]
                self.scalar_series = external_points_data.scalars
                if self.scalar_series is not None:
                    for scalar_name, scalars_value in self.scalar_series.keys():
                        self.scalar_series[scalar_name] = scalars_value[pid]
                self.scalar_grad = external_points_data.scalars_grad
                if self.scalar_grad is not None:
                    for scalar_name, scalars_grad_value in self.scalar_grad.keys():
                        self.scalar_grad[scalar_name] = scalars_grad_value[pid]
                self.scalar_grad_norm = external_points_data.scalars_grad_norm
                if self.scalar_grad_norm is not None:
                    for scalar_name, scalars_grad_norm_value in self.scalar_grad_norm.keys():
                        self.scalar_grad_norm[scalar_name] = scalars_grad_norm_value[pid]
        else:
            raise ValueError('Please input an object of Grid class.')

    def save(self, dir_path: str):
        if self.vtk_data is not None and isinstance(self.vtk_data, pv.RectilinearGrid):
            self.save_path = os.path.join(dir_path, self.tmp_dump_str + '.vtk')
            self.vtk_data.save(filename=self.save_path)
            self.vtk_data = 'dumped'

    def load(self):
        if self.vtk_data is not None:
            if self.save_path is not None and os.path.exists(self.save_path):
                self.vtk_data = pv.read(filename=self.save_path)
            else:
                raise ValueError('vtk data file does not exist')


class GridPointDataDistributedSampler(object):
    def __init__(self, grid_dataset: Grid, ngpus):
        """ Evenly splits grids points into n = ngpus pieces. Each piece will be sent to different gpus for parallelized
        inference/prediction on the model after training.
        Assumed the coords/features are normalized/scaled already"""
        self.ngpus = ngpus
        indices = np.arange(grid_dataset.grid_points_num)
        split_indices = np.array_split(indices, ngpus)
        self.subset = {}
        for i in range(ngpus):
            grid_coord_subset = grid_dataset.grid_points[split_indices[i]]
            grid_series_subset = grid_dataset.grid_points_series[split_indices[i]]
            self.subset[i] = Grid(torch.from_numpy(grid_coord_subset).float(), grid_series_subset)

    def get_subset(self, i):
        return self.subset[i]


class GridData(Dataset):
    def __init__(self, coords):
        self.coords = coords
        self.n_grid_pts = self.coords.shape[0]

    def __len__(self):
        return self.n_grid_pts

    def __getitem__(self, idx):
        return self.coords[idx]
