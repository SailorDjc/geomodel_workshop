import copy
import pyvista as pv
import numpy as np
from vtkmodules.util import numpy_support
from vtkmodules.all import vtkXMLUnstructuredGridReader, vtkPoints, vtkCellArray, vtkTriangle, vtkUnstructuredGrid, \
    vtkImageData, vtkPlaneSource
from scipy import interpolate
from data_structure.grids import PointSet
from tqdm import tqdm
from utils.vtk_utils import create_vtk_grid_by_rect_bounds, vtk_unstructured_grid_to_vtk_polydata
from utils.vtk_utils import create_implict_surface_reconstruct, bounds_merge, get_bounds_from_coords, bounds_intersect
import os
import json
import time
import pickle
from scipy.interpolate import LinearNDInterpolator
import scipy.spatial as spt
from importlib import util
from matplotlib.path import Path
from vtkmodules.all import vtkPolyDataReader, vtkPolyDataMapper, vtkProperty, vtkRenderer, \
    vtkBooleanOperationPolyDataFilter
import matplotlib.pyplot as plt

has_shapely = util.find_spec("shapely")
has_rasterio = util.find_spec("rasterio")
has_geopandas = util.find_spec("geopandas")

if has_rasterio is None:
    has_rasterio = False
else:
    has_rasterio = True
    import rasterio
    import rasterio as rio
    from rasterio.warp import calculate_default_transform, reproject, transform_geom, Resampling
    from rasterio import crs
    import rasterio.features
    from rasterio.mask import mask
if has_shapely is None:
    has_shapely = False
else:
    has_shapely = True
    from shapely.geometry import box, Polygon
if has_geopandas is None:
    has_geopandas = False
else:
    has_geopandas = True
    import geopandas as gpd


# 计算投影分带的epsg编号
def longitude_to_proj_zone(longitude, zone_type='6'):
    if zone_type == '3':
        zone_no = np.floor(longitude / 3 + 0.5)
        if 25 <= zone_no <= 45:
            return 4534 + (zone_no - 25)
        else:
            raise ValueError('out of the expected range.')
    else:
        zone_no = np.floor(longitude / 6) + 1
        if 13 <= zone_no <= 23:
            return 4502 + (zone_no - 13)
        else:
            raise ValueError('out of the expected range.')


# 创建多边形几何对象
def create_polygon_from_boundary(boundary_2d):
    points_2d = []
    for item in boundary_2d:
        points_2d.append((item[0], item[1]))
    boundary = Polygon(points_2d)
    return boundary


# 构建mesh网格
def create_struct_mesh_from_bounds(bounds, resolution_xy):
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    dim_x = np.ceil((x_max - x_min) / resolution_xy)
    dim_y = np.ceil((y_max - y_min) / resolution_xy)
    x_d, y_d = dim_x + 1, dim_y + 1
    x_d = complex(0, x_d)
    y_d = complex(0, y_d)
    x, y = np.mgrid[x_min:x_max:x_d, y_min:y_max:y_d]
    z = np.zeros_like(x)
    surface = pv.StructuredGrid(x, y, z)
    return surface


# 包围盒转点
def bounds_to_corners_2d(bounds, inner_points=None):
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    z_value = (z_min + z_max) / 2
    p_a = [x_min, y_min, z_value]
    p_b = [x_min, y_max, z_value]
    p_c = [x_max, y_min, z_value]
    p_d = [x_max, y_max, z_value]
    corners_list = [p_a, p_b, p_c, p_d]
    if inner_points is not None:
        # 由于目前很多python插值算法只能插值凸包范围内的点，
        # 故通过内部点集计算bounds角点高程，取最近邻3个点的高程均值
        ckt = spt.cKDTree(inner_points)
        for p_i, pt in enumerate(corners_list):
            d, pid = ckt.query(pt, k=[1, 2, 3])
            search_pts = inner_points[pid]
            z_value = np.mean(search_pts[:, 2])
            corners_list[p_i][2] = z_value
    return np.array(corners_list)


# 从二维点集中获取二维包围盒
def get_bound_2d_from_points_2d(points_2d, buffer_dist=5):
    points = np.array(points_2d)
    min_x = np.min(points[:, 0]) - buffer_dist
    max_x = np.max(points[:, 0]) + buffer_dist
    min_y = np.min(points[:, 1]) - buffer_dist
    max_y = np.max(points[:, 1]) + buffer_dist
    return np.array([min_x, max_x, min_y, max_y, 0, 0])


# 比较扩大后的范围与数据源范围，如果超出数据源范围，则剪断
def compare_data_bounds(tmp_bounds, data_bounds):
    x_min_a, x_max_a, y_min_a, y_max_a, z_min_a, z_max_a = tmp_bounds
    x_min_b, x_max_b, y_min_b, y_max_b, z_min_b, z_max_b = data_bounds
    if x_min_a < x_min_b:
        x_min_a = x_min_b
    if x_max_a > x_max_b:
        x_max_a = x_max_b
    if y_min_a < y_min_b:
        y_min_a = y_min_b
    if y_max_a > y_max_b:
        y_max_a = y_max_b
    if z_min_a < z_min_b:
        z_min_a = z_min_b
    if z_max_a > z_max_b:
        z_max_a = z_max_b
    return np.array([x_min_a, x_max_a, y_min_a, y_max_a, z_min_a, z_max_a])


class TerrainData(object):
    def __init__(self, tiff_path=None, control_points=None):
        # 地形面数据
        self.grid_points = None  # 与dem对应栅格点坐标点坐标
        self._bounds = None
        # tiff 元数据
        self.tiff_path = tiff_path
        self._transform = None
        self._src_crs = None  # 原始投影
        self._tiff_dims = None  # 数字高程栅格维度
        self.src_tiff_bounds = None  # tiff数据范围
        # 高程控制点
        self.control_points = control_points
        self.dst_crs = None  # 目标投影
        self.vtk_data = None
        # 边界约束
        self.mask_bounds = None  # 规则矩形边界约束
        self.mask_shp_path = None  # shp矢量文件不规则边界约束
        self.boundary_2d = None  # 点集列表不规则边界约束

        self.boundary_points = None  # 边界点集，首尾不重复
        self.boundary_type = 3  # 边界类型 0为规则矩形掩膜边界范围，1为不规则多边形边界约束(输入点集)，2为shp多边形边界输入，
        # 当没有外界输入边界，默认为3或4，3为以点集的外包络矩形范围做约束，4为以点集凸包为边界约束
        #
        self.coord_type = 'xy'  # 'xy' 平面直角坐标系（单位米）   'dd' 经纬度
        # 保存参数
        self.dir_path = None
        self.tmp_dump_str = 'tmp_terrain' + str(int(time.time()))
        if self.tiff_path is not None:
            self.set_input_tiff_file(self.tiff_path)

    # 数据输入接口
    def set_control_points(self, control_points, buffer_dist=20):
        self.control_points = control_points
        if self.control_points is not None:
            self.modify_local_terrain_by_points(self.control_points, buffer_dist=buffer_dist)

    # boundary_2d是边界点按顺序排列的列表，首尾点不重复， mask_bounds 包围盒6元组， mask_shp_path shp文件
    def set_boundary(self, boundary_2d=None, mask_bounds=None, mask_shp_path=None, is_bound=False):
        self.boundary_2d = boundary_2d
        self.mask_bounds = mask_bounds
        self.mask_shp_path = mask_shp_path
        if self.mask_bounds is not None:
            min_x, max_x, min_y, max_y, _, _ = self.mask_bounds
            self.boundary_points = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
            self.boundary_type = 0
        elif self.mask_shp_path is not None:
            mask_gdf = gpd.read_file(self.mask_shp_path)
            self.boundary_points = np.array(mask_gdf['geometry'][0].exterior.coords)
            self.boundary_type = 1
        elif self.boundary_2d is not None:
            self.boundary_points = np.array(self.boundary_2d)
            self.boundary_type = 2
        elif is_bound is True:
            self.boundary_type = 3
        else:
            self.boundary_type = 4

    # 设置栅格影像输入
    def set_input_tiff_file(self, file_path, dst_crs_code=None):
        self.tiff_path = file_path
        if not os.path.exists(self.tiff_path):
            raise ValueError('Input path not exists.')
        # 预加载图像元数据
        with rasterio.open(self.tiff_path) as src:
            self._tiff_dims = (src.width, src.height)
            self._src_crs = src.crs
            self._transform = src.transform
        if dst_crs_code is not None:
            self.dst_crs = self.set_dst_crs_code(dst_crs_code=dst_crs_code)
        self.set_default_dst_crs()
        # 重投影
        if self.check_crs_change():
            self.reproject_tiff(self.tiff_path)

    def execute(self):
        if self.tiff_path is not None:
            self.read_tiff_data_from_file(file_path=self.tiff_path)
        if self.control_points is not None:
            self.set_control_points(self.control_points)
        self.create_terrain_surface_from_points(PointSet(points=self.grid_points))
        self.clip_terrain_surface_by_boundary_points()

    @property
    def bounds(self):
        cur_bounds = None
        if self.grid_points is not None:
            tmp_bounds = get_bounds_from_coords(self.grid_points)
            cur_bounds = bounds_merge(tmp_bounds, cur_bounds)
            self._bounds = bounds_merge(cur_bounds, self._bounds)
        elif self.boundary_points is not None:
            tmp_bounds = get_bounds_from_coords(self.boundary_points)
            cur_bounds = bounds_merge(tmp_bounds, cur_bounds)
            self._bounds = bounds_merge(cur_bounds, self._bounds)
        elif self.control_points is not None:
            tmp_bounds = get_bounds_from_coords(self.control_points)
            cur_bounds = bounds_merge(tmp_bounds, cur_bounds)
            self._bounds = bounds_merge(cur_bounds, self._bounds)
        else:
            self._bounds = None
        return self._bounds

    # 重投影栅格影像
    def reproject_tiff(self, file_path):
        with rio.open(file_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, self.dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': self.dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            tiff_dir_path, tiff_name = os.path.split(self.tiff_path)
            tiff_name = 'reproj_' + tiff_name
            write_path = os.path.join(tiff_dir_path, tiff_name)
            with rasterio.open(write_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=self.dst_crs,
                        resampling=Resampling.nearest)
            self.tiff_path = write_path

    # mask_bounds [min_x, max_x, min_y, max_y, min_z, max_z]

    # def create_terrain_surface(self):
    #     if self.grid_points is not None:
    #         surface = pv.StructuredGrid()
    #         surface.points = self.grid_points
    #         surface.dimensions = [self._tiff_dims[0], self._tiff_dims[1], 1]
    #         surface['Elevation'] = self.grid_points[:, 2]
    #         return surface

    def read_tiff_data_from_file(self, file_path):
        with rio.open(file_path) as src:
            self.grid_points = []
            self._tiff_dims = (src.width, src.height)
            self._src_crs = src.crs
            self.src_tiff_bounds = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top, 0, 0]
            self._transform = src.transform
            z_matrix = src.read(1)
            if self.boundary_points is not None:
                # 在已有掩膜范围的基础上，扩大范围，搜索更多范围外的点
                bounds_2d = get_bound_2d_from_points_2d(self.boundary_points, buffer_dist=200)
                # 与数据源范围作比较，超出范围部分剪断
                bounds_2d = compare_data_bounds(bounds_2d, self.src_tiff_bounds)
                bounds_points_2d = bounds_to_corners_2d(bounds_2d)
                polygon_2d = create_polygon_from_boundary(boundary_2d=bounds_points_2d)
                out_image, out_transform = mask(dataset=src, shapes=[polygon_2d], crop=True, nodata=-32768)
                self._transform = out_transform
                z_matrix = out_image[0]
                self._tiff_dims = [z_matrix.shape[0], z_matrix.shape[1]]
            print('Processing the tiff matrix...')
            pbar = tqdm(range(z_matrix.shape[0]), total=z_matrix.shape[0], desc='Reading coordinates...')
            for i in pbar:
                for j in range(z_matrix.shape[1]):
                    x, y = (i + 0.5, j + 0.5) * self._transform
                    # # 排除无效值
                    if z_matrix[i, j] == -32768:
                        continue
                    self.grid_points.append(np.array([x, y, z_matrix[i, j]]))
            self.grid_points = np.array(self.grid_points)

    # 设置目标投影
    def set_dst_crs_code(self, dst_crs_code):
        if self.dst_crs is not None:
            if self.dst_crs.data['init'] == 'epsg:{}'.format(dst_crs_code):
                return False
            self.dst_crs = crs.CRS().from_epsg(code=dst_crs_code)
            return True
        else:
            self.dst_crs = crs.CRS().from_epsg(code=dst_crs_code)
            return True

    # 检查原来投影与目标投影间是否发生变化
    def check_crs_change(self):
        if self.dst_crs is None or self._src_crs is None:
            return False
        if self._src_crs.data['init'] == self.dst_crs.data['init']:
            return False
        else:
            return True

    # 设置默认的投影坐标系，平面坐标系默认为大地2000, 经纬度默认EPSG:4326
    def set_default_dst_crs(self):
        if self.dst_crs is None:
            if self.coord_type == 'xy':
                if not self._src_crs.is_projected:
                    epsg_code = longitude_to_proj_zone(longitude=self._transform.c)
                    self.dst_crs = crs.CRS().from_epsg(code=epsg_code)
            if self.coord_type == 'dd':
                if self._src_crs.is_projected:
                    self.dst_crs = crs.CRS().from_epsg(code=4326)

    def create_terrain_surface_from_points(self, points_data: PointSet, resolution_xy=5):
        tmp_bounds = points_data.bounds
        result_bounds = bounds_merge(self.bounds, tmp_bounds)
        self._bounds = result_bounds
        if self.boundary_type == 0:
            result_bounds = get_bound_2d_from_points_2d(self.boundary_points, buffer_dist=0)
        if self.boundary_type == 3:
            result_bounds = get_bound_2d_from_points_2d(points_data.points, buffer_dist=0)
        if self.boundary_type == 4:  # 凸包
            self.boundary_points = points_data.get_convexhull_2d()
        terrain_surface = create_struct_mesh_from_bounds(bounds=result_bounds, resolution_xy=resolution_xy)
        known_points = points_data.points
        if points_data.nidm == 2:
            if isinstance(points_data.scalars, dict):
                z_values = points_data.scalars.get('elevation')
                if z_values is not None:
                    known_points = np.concatenate((known_points, z_values), axis=1)
                else:
                    raise ValueError('Points have no z_value data.')
        # 插值
        b_points = bounds_to_corners_2d(result_bounds, inner_points=known_points)
        # 将角点加入已知点集，角点高程是均值填充的
        known_points = np.concatenate((known_points, b_points), axis=0)
        x = known_points[:, 0]
        y = known_points[:, 1]
        z = known_points[:, 2]
        interp = LinearNDInterpolator(list(zip(x, y)), z)
        cell_points = terrain_surface.points
        pred_x = cell_points[:, 0]
        pred_y = cell_points[:, 1]
        pred_z = interp(pred_x, pred_y)
        terrain_surface['Elevation'] = pred_z
        terrain_surface = terrain_surface.warp_by_scalar()
        terrain_surface = vtk_unstructured_grid_to_vtk_polydata(terrain_surface)
        self.grid_points = terrain_surface.cell_centers().points
        self.vtk_data = terrain_surface

    def clip_terrain_surface_by_boundary_points(self):
        if self.vtk_data is None:
            raise ValueError('The vtk surface is None.')
        max_z = self.bounds[5]
        min_z = self.bounds[4]
        z_max = max_z + 5
        z_min = min_z - 5
        if self.boundary_type == 0 or self.boundary_type == 3:
            return self.vtk_data
        else:   # 4 or 1 or 2:
            boundary_points = copy.deepcopy(self.boundary_points)
        # 范围约束没有高程设置，这里使用最大高程和最小高程，创建一个范围盒子，筛选出在盒子范围内的网格点，范围外的网格删除
        if boundary_points.shape[1] == 2:
            boundary_points = np.pad(array=boundary_points, pad_width=((0, 0), (0, 1))
                                          , constant_values=((z_max, z_max), (z_max, z_max)))
        else:
            boundary_points[:, 2] = z_max
        points_3d = copy.deepcopy(boundary_points)
        N = len(points_3d)
        face = [N + 1] + list(range(N)) + [0]
        polygon = pv.PolyData(points_3d, faces=face)
        polygon = polygon.triangulate()
        polygon = polygon.subdivide_adaptive(max_n_passes=5)  # max_edge_len=resolytion_xy
        side_surf = polygon.extrude((0, 0, z_min - z_max), capping=True)
        # 筛选出在边界范围内的网格点索引
        grid_points = pv.PolyData(self.grid_points)
        selected = grid_points.select_enclosed_points(side_surf, tolerance=0.000001)
        cell_indices = selected.point_data['SelectedPoints']
        delete_cell_indices = np.argwhere(cell_indices <= 0).flatten()
        new_terrain_surface = self.vtk_data.remove_cells(delete_cell_indices)
        return new_terrain_surface


    def create_grid_from_terrain_surface(self, z_min=-100, boundary_2d=None, bounds=None, cell_density=2
                                         , is_smooth=False):
        if boundary_2d is not None:
            self.set_boundary(boundary_2d=boundary_2d, mask_bounds=bounds)
        if self.vtk_data is None:
            raise ValueError('The vtk surface is None.')
        size_x = self.bounds[1] - self.bounds[0] + 5
        size_y = self.bounds[3] - self.bounds[2] + 5
        dim_x = size_x / cell_density
        max_dim = 400
        if dim_x > max_dim:
            cell_density = size_x / max_dim
        plane = pv.Plane(center=(self.vtk_data.center[0], self.vtk_data.center[1], z_min), direction=(0, 0, -1)
                         , i_size=size_x, j_size=size_y)
        grid_extrude_trim = self.vtk_data.extrude_trim((0, 0, z_min), trim_surface=plane)
        grid_extrude_trim = pv.voxelize(grid_extrude_trim, density=cell_density)
        if is_smooth:
            grid_extrude_trim = vtk_unstructured_grid_to_vtk_polydata(grid_extrude_trim)
            grid_extrude_trim = grid_extrude_trim.smooth(n_iter=1000)
        return grid_extrude_trim

    # 通过高程点修改局部地形， 注意points_data.buffer_dist，这个缓冲距离表示高程点的控制范围
    def modify_local_terrain_by_points(self, control_points_data: PointSet, buffer_dist):
        if self.grid_points is None:
            self.grid_points = control_points_data.get_points()
            if self.boundary_points is None:
                self.boundary_points = control_points_data.get_convexhull_2d()
            return
        tree = spt.cKDTree(self.grid_points)
        erase_points_list = []
        # 球状影响范围搜索，搜索临近点，做替换
        control_points = control_points_data.get_points()
        for con_pnt in control_points:
            ll = tree.query_ball_point(x=con_pnt, r=buffer_dist)
            erase_points_list.extend(ll)
        # 删除影响范围内的点集
        erase_points_list = list(set(erase_points_list))
        if len(erase_points_list) > 0:
            self.grid_points = np.delete(self.grid_points, obj=erase_points_list, axis=0)
        if len(self.grid_points) > 0:
            self.grid_points = np.stack((self.grid_points, control_points), axis=0)
        else:
            self.grid_points = control_points

    # @brief 坐标范围整体偏移
    # @param new_rect_2d和new_center二选一，优先考虑new_center，new_rect_2d除平移外，还涉及坐标伸缩变换
    def translation(self, new_center=None, new_rect_2d=None):
        if self.grid_points is None:
            raise ValueError('Lacking coordinate information.')
        if new_center is not None:
            pass
        elif new_rect_2d is not None:
            pass
        else:
            raise ValueError('Need to input transform parameters.')

    def save(self, dir_path: str, out_name: str = None):
        self.dir_path = dir_path
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        if self.vtk_data is not None and isinstance(self.vtk_data, (pv.RectilinearGrid, pv.UnstructuredGrid
                                                                    , pv.StructuredGrid)):
            save_path = os.path.join(dir_path, self.tmp_dump_str + '.vtk')
            self.vtk_data.save(filename=save_path)
            self.vtk_data = 'dumped'
        file_name = self.tmp_dump_str
        if out_name is not None:
            file_name = out_name
        file_path = os.path.join(self.dir_path, file_name)
        with open(file_path, 'wb') as out_put:
            out_str = pickle.dumps(self)
            out_put.write(out_str)
            out_put.close()
        print('save terrain file into {}'.format(file_path))
        return self.__class__.__name__, file_path

    # 加载该类附属的vtk模型
    def load(self):
        if self.dir_path is not None:
            if self.vtk_data == 'dumped':
                save_path = os.path.join(self.dir_path, self.tmp_dump_str + '.vtk')
                if os.path.exists(save_path):
                    self.vtk_data = pv.read(filename=save_path)
                else:
                    raise ValueError('vtk data file does not exist')

# if self.mask_bounds is not None:
#     min_x, max_x, min_y, max_y, _, _ = self.mask_bounds
#     bbox = box(min_x, min_y, max_x, max_y)
#     geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=src.crs.data)
#     coords = [json.loads(geo.to_json())['features'][0]['geometry']]
#     out_img, out_transform = mask(dataset=src, shapes=coords, crop=True)
#     self._transform = out_transform
#     z_matrix = out_img[0]
#     self._tiff_dims = [z_matrix.shape[0], z_matrix.shape[1]]
# elif self.mask_shp_path is not None:
#     if os.path.exists(self.mask_shp_path):
#         mask_gdf = gpd.read_file(self.mask_shp_path)
#         out_image, out_transform = mask(dataset=src, shapes=mask_gdf.geometry, crop=True)
#         self._transform = out_transform
#         z_matrix = out_image[0]
# def extend_mesh_from_surface_by_bounds(self, bounds):
#     top = self.vtk_data.points.copy()
#     bottom = self.vtk_data.points.copy()
#     z_min = bounds[4]
#     bottom[:, -1] = z_min
#     mesh = pv.StructuredGrid()
#     mesh.points = np.vstack((top, bottom))
#     mesh.dimensions = [*self.vtk_data.dimensions[0:2], 2]
#     return mesh
#     @property
#     def bounds_2d(self):
#         return self._bounds_2d
