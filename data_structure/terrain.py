import copy
import pyvista as pv
import numpy as np
from vtkmodules.util import numpy_support
from vtkmodules.all import vtkXMLUnstructuredGridReader, vtkPoints, vtkCellArray, vtkTriangle, vtkUnstructuredGrid, \
    vtkImageData
from scipy import interpolate
from data_structure.grids import PointSet, get_bounds_from_coords, Grid
from tqdm import tqdm
from utils.vtk_utils import create_vtk_grid_by_rect_bounds
from utils.vtk_utils import create_implict_surface_reconstruct
import os
import json
import time
import pickle
from scipy.interpolate import LinearNDInterpolator
import scipy.spatial as spt
from importlib import util

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


def create_polygon_from_boundary(boundary_2d):
    points_2d = []
    for item in boundary_2d:
        points_2d.append((item[0], item[1]))
    boundary = Polygon(points_2d)
    return boundary


class TerrainData(object):
    def __init__(self, tiff_path=None, control_points=None):
        # 地形面数据
        self.grid_points = None  # 与dem对应栅格点坐标点坐标
        self._bounds = None

        # tiff 元数据
        self.tiff_path = tiff_path
        self._transform = None
        self._bounds_2d = None
        self._src_crs = None  # 原始投影
        self._tiff_dims = None  # 数字高程栅格维度
        # 高程控制点
        self.control_points = control_points
        self.dst_crs = None  # 目标投影
        self.vtk_data = None
        # 边界约束
        self.mask_bounds = None  # 规则矩形边界约束
        self.mask_shp_path = None  # shp矢量文件不规则边界约束
        self.boundary_2d = None   # 点集列表不规则边界约束
        self.boundary_points = None  # 边界点集，首尾不重复
        #
        self.coord_type = 'xy'  # 'xy' 平面直角坐标系（单位米）   'dd' 经纬度
        # 保存参数
        self.dir_path = None
        self.tmp_dump_str = 'tmp_terrain' + str(int(time.time()))
        if self.tiff_path is not None:
            self.set_input_tiff_file(self.tiff_path)

    def execute(self, ):
        if self.tiff_path is not None:
            self.read_tiff_data_from_file(file_path=self.tiff_path)
            # self.create_terrain_surface()
            # self.create_terrain_surface_from_points(points_data=PointSet(points=self.grid_points)
            #                                         , bounds=self.bounds)
        if self.control_points is not None:
            self.modify_local_terrain_by_points(self.control_points)
        self.create_terrain_surface_from_points(PointSet(points=self.grid_points))

    @property
    def bounds(self):
        if self.grid_points is None:
            self._bounds = None
        else:
            self._bounds = get_bounds_from_coords(coords=self.grid_points)
        return self._bounds

    @property
    def bounds_2d(self):
        return self._bounds_2d

    def set_control_points(self, control_points):
        self.control_points = control_points

    # boundary_2d是边界点按顺序排列的列表，首尾点不重复
    def set_boundary(self, boundary_2d=None, mask_bounds=None, mask_shp_path=None):
        self.boundary_2d = boundary_2d
        self.mask_bounds = mask_bounds
        self.mask_shp_path = mask_shp_path
        if self.mask_bounds is not None:
            min_x, max_x, min_y, max_y, _, _ = self.mask_bounds
            self.boundary_points = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, min_y]])
        elif self.mask_shp_path is not None:
            mask_gdf = gpd.read_file(self.mask_shp_path)
            self.boundary_points = mask_gdf['geometry'][0].exterior.coords
        elif self.boundary_2d is not None:
            self.boundary_points = self.boundary_2d
        else:
            self.boundary_points = None

    # 设置栅格影像输入
    def set_input_tiff_file(self, file_path, dst_crs_code=None):
        self.tiff_path = file_path
        if not os.path.exists(self.tiff_path):
            raise ValueError('Input path not exists.')
        # 预加载图像元数据
        with rasterio.open(self.tiff_path) as src:
            self._tiff_dims = (src.width, src.height)
            self._src_crs = src.crs
            # self._meta = src.meta
            self._bounds_2d = src.bounds
            self._transform = src.transform
        if dst_crs_code is not None:
            self.dst_crs = self.set_dst_crs_code(dst_crs_code=dst_crs_code)
        self.set_default_dst_crs()
        # 重投影
        if self.check_crs_change():
            self.reproject_tiff(self.tiff_path)

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
    def read_tiff_data_from_file(self, file_path):
        with rio.open(file_path) as src:
            self.grid_points = []
            self._tiff_dims = (src.width, src.height)
            self._src_crs = src.crs
            # self._meta = src.meta
            self._bounds_2d = src.bounds
            self._transform = src.transform
            z_matrix = src.read(1)
            if self.boundary_points is not None:
                boundary_2d = create_polygon_from_boundary(boundary_2d=self.boundary_points)
                out_image, out_transform = mask(dataset=src, shapes=boundary_2d, crop=True)
                self._transform = out_transform
                z_matrix = out_image[0]
                self._tiff_dims = [z_matrix.shape[0], z_matrix.shape[1]]
            print('Processing the tiff matrix...')
            pbar = tqdm(range(z_matrix.shape[0]), total=z_matrix.shape[0], desc='Reading coordinates...')
            for i in pbar:
                for j in range(z_matrix.shape[1]):
                    x, y = (i + 0.5, j + 0.5) * self._transform
                    # # 排除无效值
                    # if z_matrix[i, j] == -32768:
                    #     continue
                    self.grid_points.append(np.array([x, y, z_matrix[i, j]]))
            self.grid_points = np.array(self.grid_points)
            if self.boundary_points is None:
                bounds = get_bounds_from_coords(self.grid_points)
                min_x, max_x, min_y, max_y, _, _ = bounds
                self.boundary_points = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, min_y]])

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

    def create_terrain_surface(self):
        if self.grid_points is not None:
            surface = pv.StructuredGrid()
            surface.points = self.grid_points
            surface.dimensions = [self._tiff_dims[0], self._tiff_dims[1], 1]
            surface['Elevation'] = self.grid_points[:, 2]
            self.vtk_data = surface

    def extend_mesh_from_surface_by_bounds(self, bounds):
        top = self.vtk_data.points.copy()
        bottom = self.vtk_data.points.copy()
        z_min = bounds[4]
        bottom[:, -1] = z_min
        mesh = pv.StructuredGrid()
        mesh.points = np.vstack((top, bottom))
        mesh.dimensions = [*self.vtk_data.dimensions[0:2], 2]
        return mesh

    def create_terrain_surface_from_points(self, points_data: PointSet, resolytion_xy=30):
        known_points = points_data.points
        if points_data.nidm == 2:
            if isinstance(points_data.scalars, dict):
                z_values = points_data.scalars.get('elevation')
                if z_values is not None:
                    known_points = np.concatenate((known_points, z_values), axis=1)
                else:
                    raise ValueError('Points have no z_value data.')
        # 插值
        x = known_points[:, 0]
        y = known_points[:, 1]
        z = known_points[:, 2]
        interp = LinearNDInterpolator(list(zip(x, y)), z)
        if self.boundary_points is not None:
            if self.boundary_points.shape[1] == 2:
                points_3d = np.pad(self.boundary_points, [(0, 0), (0, 1)])
            else:
                points_3d = self.boundary_points
            N = len(points_3d)
            face = [N + 1] + list(range(N)) + [0]
            polygon = pv.PolyData(points_3d, faces=face)
            polygon = polygon.triangulate()
            terrain_surface = polygon.subdivide_adaptive(max_edge_len=resolytion_xy)
            terrain_surface.plot(show_edges=True)
            grid_points = terrain_surface.points
            # grid_xy = create_vtk_grid_by_rect_bounds(dim=np.array([dim_x, dim_y, 1]), bounds=np.array(
            #     [min_x, max_x, min_y, max_y, 0, 0]))
            # grid_points = grid_xy.cell_centers().points
            pred_x = grid_points[:, 0]
            pred_y = grid_points[:, 1]
            pred_z = interp(pred_x, pred_y)
            self.grid_points = np.array(list(zip(pred_x, pred_y, pred_z)))
            # terrain_surface.points = self.grid_points
            terrain_surface['Elevation'] = self.grid_points[:, 2]
            terrain_surface = terrain_surface.warp_by_scalar()
            terrain_surface.plot(show_edges=True)
            self.vtk_data = terrain_surface

    # 通过高程点修改局部地形， 注意points_data.buffer_dist，这个缓冲距离表示高程点的控制范围
    def modify_local_terrain_by_points(self, control_points_data: PointSet, buffer_dist=20):
        if self.grid_points is None:
            self.grid_points = control_points_data.points
            if self.boundary_points is None:
                self.boundary_points = control_points_data.get_convexhull_2d()
            return
        tree = spt.cKDTree(self.grid_points)
        erase_points_list = []
        for con_pnt in control_points_data.points:
            ll = tree.query_ball_point(con_pnt, buffer_dist)
            erase_points_list.extend(ll)
        erase_points_list = list(set(erase_points_list))
        if len(erase_points_list) > 0:
            self.grid_points = np.delete(self.grid_points, obj=erase_points_list, axis=0)
        self.grid_points = np.stack((self.grid_points, control_points_data.points), axis=0)

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
