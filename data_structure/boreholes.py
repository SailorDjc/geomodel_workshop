import numpy as np
import pyvista as pv
import copy
import scipy.spatial as spt
from data_structure.points import get_bounds_from_coords, PointSet
import pandas as pd
import time
import os
import pickle


# 检查list中元素类型是否一致
def check_list_item_instance(x: list) -> bool:
    if len(x) > 0:
        for item in x:
            if isinstance(x, type(x[0])):
                continue
            else:
                return False
    return True


# 钻孔结构
class Borehole(object):
    def __init__(self, points: np.ndarray = None, series: np.ndarray = None, is_vertical=True, is_virtual=False
                 , radius=None, buffer_dist_xy=5, default_base=None, borehole_id=None):
        self.points = points
        self.points_num = 0
        self.series = series  # 地层标签
        self.is_vertical = is_vertical  # 是否为垂直钻孔
        self.is_virtual = is_virtual  # 是否为虚拟钻孔
        self.classes = None
        if self.points is not None and self.series is not None:
            self.points_num = points.shape[0]
            self.classes = sorted(np.unique(series))
        self.scalar = {}  # {scalar_name: value} 与self.coords一一对应
        self.sub_att_scalar_pnt = {}  # 附加属性点，在钻孔勘测范围内，不一定是分层点
        self.radius = radius  # 孔径， 用于可视化
        self.buffer_dist_xy = buffer_dist_xy  # 缓冲大小，距离钻孔中心一定缓冲范围内，属于该钻孔控制范围
        self.borehole_id = borehole_id  # 钻孔唯一标识

        self.vtk_data = None

        self.holelayer_list = []  # 地层分层
        self.holelayer_num = 0
        self.default_base = default_base  # 基底，若为None，则尚未指定基底
        # 去重, 钻孔分层表中只保存界面点
        self.top_pos = None  # 钻孔顶点
        self.bottom_pos = None  # 钻孔底点
        self.update_holelayer_list()

    class Holelayer(object):
        def __init__(self, coord_top, coord_bottom, layer_label, azimuth=0, dip=90, is_virtual=False
                     , buffer_dist_xy=5, borehole_id=None):
            self.borehole_id = borehole_id  # 所属钻孔标识
            self.top_pos = coord_top  # 顶部点
            self.bottom_pos = coord_bottom  # 底部点
            self.azimuth = azimuth  # 测斜数据(垂直孔不需要)
            self.dip = dip  # 倾角
            self.layer_label = layer_label  # 地层编号(编码)
            self.layer_label_old = layer_label  # 作为老标签的备份，在标签标准化后，值不变
            self.is_virtual = is_virtual  # 是否是虚拟地层
            self.buffer_dist_xy = buffer_dist_xy  # 钻孔控制范围

    def update_holelayer_list(self):
        self.holelayer_list = []  # 清空
        if self.points is not None or self.series is not None:
            if self.points.shape[0] == self.series.shape[0] and self.points.ndim == 2:
                points, series = self.remove_duplicates_series()
                num = points.shape[0]
                if num < 2:
                    raise ValueError('Borehole data is invalid because of lack of points')
                self.top_pos = points[0]
                self.bottom_pos = points[num - 1]
                for i in range(num - 1):
                    one_holelayer = self.Holelayer(coord_top=points[i], coord_bottom=points[i + 1],
                                                   layer_label=series[i], borehole_id=self.borehole_id)
                    self.holelayer_list.append(one_holelayer)
                self.holelayer_num = num - 1

    # 遍历钻孔地层序列点，获取界面点 is_delete=False, 中间点不删除，True则删除
    def remove_duplicates_series(self, is_delete=False):
        if self.series is None:
            raise ValueError('Series array can not be None')
        extract_labels = []
        extract_points = []
        # 从上至下遍历钻孔点
        front_label = self.series[0]
        extract_labels.append(self.series[0])
        extract_points.append(self.points[0])
        points_num = self.series.shape[0]
        # 提取分界点的索引，先把第一个点放进来
        for l_idx in range(0, points_num):
            next_label = self.series[l_idx]
            if next_label != front_label:
                extract_labels.append(self.series[l_idx])
                extract_points.append(self.points[l_idx])
                front_label = self.series[l_idx]
            else:
                if l_idx == points_num - 1:
                    extract_labels.append(self.series[l_idx])
                    extract_points.append(self.points[l_idx])
        if not is_delete:
            return np.array(extract_points), np.array(extract_labels)
        self.points = np.array(extract_points)
        self.series = np.array(extract_labels)
        self.points_num = self.points.shape[0]

    def get_points_data(self, only_interface=False):
        if only_interface:
            points = []
            labels = []
            for l_it, one_layer in enumerate(self.holelayer_list):
                points.append(one_layer.top_pos)
                labels.append(one_layer.layer_label)
                if l_it == self.holelayer_num - 1:
                    points.append(one_layer.bottom_pos)
                    labels.append(one_layer.layer_label)
            points_data = PointSet(points=np.array(points), point_labels=np.array(labels))
        else:
            points_data = PointSet(points=self.points, point_labels=self.series)
        return points_data

    def set_att_scalar_with_layer_pnt(self, att_name: str, att_values: np.ndarray):
        if att_name not in self.scalar.keys():
            self.scalar[att_name] = att_values

    # 添加带有坐标的属性点，可以不是钻孔分层点
    def set_sub_att_scalar_with_pnt(self, coords: np.ndarray, att_name: str, att_values):
        if att_name not in self.sub_att_scalar_pnt.keys():
            self.sub_att_scalar_pnt[att_name] = []
        else:
            if isinstance(coords, np.ndarray) and isinstance(att_values, np.ndarray):
                if coords.ndim == 2 and coords.shape[0] == att_values.shape[0]:
                    for coord, value in zip(coords, att_values):
                        self.sub_att_scalar_pnt[att_name].append((coord, value))
            if isinstance(att_values, float) and isinstance(coords, np.ndarray):
                if coords.shape[0] == 1:
                    self.sub_att_scalar_pnt[att_name].append((coords, att_values))

    def has_att_scalar(self, att_name):
        if att_name in self.scalar.keys():
            return True
        else:
            return False

    def get_top_point(self):
        if self.points is not None:
            return copy.deepcopy(self.points[0])

    # 判断一个点属于钻孔的哪一个分层，若返回-1则在钻孔控制深度之下，若为-2则在地表之上
    def get_layer_label_with_point_z(self, point_value):
        pnt_z = None
        if isinstance(point_value, np.ndarray):
            if point_value.ndim == 2 and point_value.shape[1] == 3:
                pnt_z = point_value[2]
        elif isinstance(point_value, int):
            pnt_z = point_value
        else:
            raise ValueError('Input data must be a 3D point or Integer z value.')
        if pnt_z is None:
            raise ValueError('Input data is not supported.')
        layer_label = -2
        for l_id, layer in enumerate(self.holelayer_list):
            # 判断输入点xy坐标是否在钻孔中心点一定范围内
            if layer.bottom_pos[2] < pnt_z <= layer.top_pos[2]:
                layer_label = layer.layer_label
            elif pnt_z <= layer.bottom_pos[2] and l_id == len(self.holelayer_list) - 1:
                layer_label = -1
        return layer_label

    # 钻孔加密，有两种方式，一种是设定采样间隔，按等距离加密，一种是指定每层加密点数
    def boreholes_points_densify(self, dist: float = None, pnt_num_per_layer: int = None):
        new_points = []
        new_labels = []
        for l_id, layer in enumerate(self.holelayer_list):
            top = self.holelayer_list[l_id].top_pos
            bottom = self.holelayer_list[l_id].bottom_pos
            label = self.holelayer_list[l_id].layer_label
            # 根据间距采样
            if dist is not None:
                pass
            # 根据每层采样点数目采样
            elif pnt_num_per_layer is not None:
                pass

    # 判断一个点是否在钻孔的控制范围内
    def check_point_belong_borehole(self, point: np.ndarray):
        if point.ndim == 2 and point.shape[1] == 3:
            borehole_xy = self.get_top_point()[0:2]
            input_xy = point[0:2]
            dist = spt.distance.euclidean(borehole_xy, input_xy, w=None)
            if self.buffer_dist_xy is None:
                raise ValueError('Parameter buffer_dist_xy values need to be specified')
            else:
                if dist > self.buffer_dist_xy:
                    return False
                else:
                    top_z = self.top_pos[2]
                    bottom_z = self.bottom_pos[2]
                    point_z = point[2]
                    if bottom_z < point_z < top_z:
                        return True
                    else:
                        return False
        else:
            raise ValueError('Input data must be a 3D point.')


# 钻孔集
class BoreholeSet(object):
    def __init__(self, points: np.ndarray = None, series: np.ndarray = None, borehole_idx: np.ndarray = None
                 , radius: float = 10, name=None, dir_path=None):
        self.name = name
        self.points = points  # 二维
        self.points_num = 0
        self.series = series
        self.boreholes_index = borehole_idx  # 钻孔点集索引, 记录每个钻孔首个点在点集中的索引, 必须是递增序列
        self.borehole_num = 0
        self.boreholes_list = []  # 存储Borehole的列表
        self.bounds = None
        self.radius = radius

        self.classes = None
        self.classes_num = 0

        self.vtk_data = None

        self.tmp_dump_str = 'tmp_hole' + str(int(time.time()))

        # 通过坐标数组和标签数组，初始化钻孔
        if self.points is not None and self.series is not None and self.boreholes_index is not None:
            # 遍历钻孔集合，需要遍历索引集合self.boreholes_index，从而将顺序存储的点集匹配到每个钻孔
            self.points_num = self.points.shape[0]
            start_iter = 0
            for hole_id in np.arange(len(self.boreholes_index)):
                if self.boreholes_index[hole_id] > self.points_num:
                    cur_borehole_pn = self.boreholes_index[hole_id]  # 当前钻孔点数
                    borehole_points = []
                    borehole_series = []
                    for p_j in np.arange(start=start_iter, stop=start_iter + cur_borehole_pn):
                        borehole_points.append(np.array(self.points[p_j]))
                        borehole_series.append(np.array(self.series[p_j]))
                    one_borehole = Borehole(points=np.array(borehole_points), series=np.array(borehole_series)
                                            , radius=self.radius)
                    self.boreholes_list.append(one_borehole)
                    self.borehole_num += 1
                    start_iter += cur_borehole_pn  # 更新索引

            # 对象拷贝
        self.dir_path = dir_path

    def search_by_rect2d(self, rect2d):
        x_min, x_max, y_min, y_max = rect2d[0], rect2d[1], rect2d[2], rect2d[3]
        search_boreholes_list = BoreholeSet()
        for one_borehole in self.boreholes_list:
            pos_2d = one_borehole.top_pos[0:2]
            if x_min <= pos_2d[0] <= x_max and y_min <= pos_2d[1] <= y_max:
                search_boreholes_list.append(one_borehole=one_borehole)
        return search_boreholes_list

    def get_classes(self):
        if self.classes is None:
            if self.series is None:
                raise ValueError('This borehole dataset lacks labels.')
            else:
                self.classes = sorted(np.unique(self.series))
                self.classes_num = len(self.classes)
        return self.classes

    def select_virtual_bolehole(self, is_virtual=True):
        select_boreholes = []
        select_boreholes_id = []
        for borehole_id, one_borehole in enumerate(self.boreholes_list):
            if (is_virtual and one_borehole.is_virtual) or (not is_virtual and not one_borehole.is_virtual):
                select_boreholes.append(one_borehole)
                select_boreholes_id.append(borehole_id)
        return select_boreholes, select_boreholes_id

    def append(self, one_borehole: Borehole):
        if not isinstance(one_borehole, Borehole):
            raise TypeError('The input data should be Borehole type.')
        else:
            self.boreholes_list.append(one_borehole)
            if one_borehole.points is None or one_borehole.points_num < 2:
                raise ValueError('Input borehole data is invalid.')
            # 记录每个入库钻孔点数
            if self.boreholes_index is None:
                self.boreholes_index = np.array([one_borehole.points.shape[0]])
            else:
                self.boreholes_index = np.append(arr=self.boreholes_index,
                                                 values=np.array([one_borehole.points.shape[0]]), axis=0)
            if self.points is None:
                self.points = one_borehole.points
            else:
                self.points = np.append(arr=self.points, values=one_borehole.points, axis=0)
            if self.series is None:
                self.series = one_borehole.series
            else:
                self.series = np.append(arr=self.series, values=one_borehole.series, axis=0)
            self.borehole_num += 1
            self.bounds = get_bounds_from_coords(self.points)
            self.points_num = self.points.shape[0]
            _ = self.get_classes()

    def set_boreholes_control_buffer_dist_xy(self, radius: float):
        for hole_id in range(len(self.boreholes_list)):
            self.boreholes_list[hole_id].buffer_dist_xy = radius
            for layer_id in range(self.boreholes_list[hole_id].holelayer_num):
                self.boreholes_list[hole_id].holelayer_list[layer_id].buffer_dist_xy = radius

    def generate_vtk_data_as_tube(self, borehole_radius=1.0, is_tube=True):
        # 遍历钻孔
        borehole_list = pv.MultiBlock()  # 钻孔序列
        for one_borehole in self.boreholes_list:
            layer_points = []
            layer_labels = []
            layer_lines = []
            for one_layer in one_borehole.holelayer_list:
                one_line = np.array([2, len(layer_points), len(layer_points) + 1], dtype=int)
                layer_label = one_layer.layer_label
                point_a = one_layer.top_pos
                point_b = one_layer.bottom_pos
                layer_points.append(point_a)
                layer_points.append(point_b)
                layer_labels.append(layer_label)
                layer_lines.append(one_line)
            one_borehole_vtk_data = pv.PolyData(np.array(layer_points, dtype=float),
                                                lines=np.array(layer_lines, dtype=int))
            one_borehole_vtk_data.cell_data['Scalar Field'] = np.array(layer_labels)
            if is_tube:
                one_borehole_vtk_data = one_borehole_vtk_data.tube(radius=borehole_radius)
            borehole_list.append(one_borehole_vtk_data)
        self.vtk_data = borehole_list
        return borehole_list

    def generate_vtk_data_as_line(self):
        borehole_list = self.generate_vtk_data_as_tube(is_tube=False)
        return borehole_list

    def show(self, is_tube=True, borehole_radius=1.0):
        self.generate_vtk_data_as_tube(is_tube=is_tube, borehole_radius=borehole_radius)
        if isinstance(self.vtk_data, pv.MultiBlock):
            self.vtk_data.plot()

    def get_sample_vtk_data(self, block_id=None):
        if block_id is None:
            return self.vtk_data
        if isinstance(block_id, list):
            return [self.vtk_data.GetBlock(id) for id in block_id]
        elif isinstance(block_id, (slice, int)):
            return self.vtk_data.__getitem__(block_id)
        else:
            raise ValueError('Input Index is invalid.')

    def get_boreholes(self, idx):
        if isinstance(idx, int):
            return self.__getitem__(idx=idx)
        elif isinstance(idx, (list, np.ndarray)):
            borehole_list = BoreholeSet()
            for id in idx:
                borehole_list.append(self.__getitem__(idx=id))
            return borehole_list

    def get_boreholes_by_id(self, borehole_id):
        for one_borehole in self.boreholes_list:
            if one_borehole.borehole_id == borehole_id:
                return one_borehole
        return None

    def get_top_points(self):
        top_points = []
        for one_borehole in self.boreholes_list:
            top_point = one_borehole.get_top_point()
            top_points.append(top_point)
        top_points = np.array(top_points)
        return top_points

    def get_top_points_data(self):
        top_points = []
        top_labels = []
        for one_borehole in self.boreholes_list:
            top_point = one_borehole.get_top_point()
            top_points.append(top_point)
            top_labels.append(one_borehole.holelayer_list[0].layer_label)
        top_points = np.array(top_points)
        top_points_data = PointSet(points=top_points, point_labels=np.array(top_labels))
        return top_points_data

    def get_points_data(self, only_interface=False):
        points = []
        labels = []
        for one_hole in self.boreholes_list:
            one_hole_points_data = one_hole.get_points_data(only_interface=only_interface)
            points.append(one_hole_points_data.points)
            labels.append(one_hole_points_data.labels)
        point_arr = np.concatenate(points, axis=0)
        label_arr = np.concatenate(labels, axis=0)
        points_data = PointSet(points=point_arr, point_labels=label_arr)
        return points_data

    # 按标签拆分，返回每一类标签所属的cell
    def detach_vtk_component_with_label(self):
        vtk_dict = {}
        for one_hole in self.boreholes_list:
            for one_layer in one_hole.holelayer_list:
                label = one_layer.layer_label
                points = np.vstack((one_layer.top_pos, one_layer.bottom_pos))
                vtk_data = pv.PolyData(points, lines=np.array([2, 0, 1]))
                vtk_data.cell_data['Scalar Field'] = np.array(label)
                vtk_data = vtk_data.tube(radius=one_layer.buffer_dist_xy)
                if label not in vtk_dict.keys():
                    vtk_dict[label] = []
                vtk_dict[label].append(vtk_data)
        for key in vtk_dict.keys():
            vtk_dict[key] = self.merge_vtk_lines(vtk_dict[key])
        return vtk_dict

    def merge_vtk_lines(self, poly_lines: list):
        if len(poly_lines) == 0:
            raise ValueError('Input data is empty.')
        else:
            vtk_data = None
            for one_line in poly_lines:
                if not isinstance(one_line, pv.PolyData):
                    raise ValueError('Only support pv.PolyData type input.')
                if vtk_data is None:
                    vtk_data = one_line
                else:
                    vtk_data = vtk_data.merge(one_line)
            return vtk_data

    # 创建钻孔集的凸包，获取凸包面体与外框
    def get_boreholes_convexhull_bouding_surface_and_outline(self):
        if self.boreholes_list is None or len(self.boreholes_list):
            raise ValueError('Boreholes list is empty.')
        points_data = PointSet(self.get_top_points())
        top_ring = points_data.get_convexhull_2d()

        top_surface_points = copy.deepcopy(top_ring)
        top_surface_points[:, 2] = self.bounds[5]  # z_max
        bottom_surface_points = copy.deepcopy(top_ring)
        bottom_surface_points[:, 2] = self.bounds[4]  # z_min
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
        return convex_surface, grid_outline

    # 获取钻孔数据字典 {borehole_id: np.array(x, y, z, label)}
    def get_boreholes_id_points_labels_map(self, only_interface=False):
        borehole_id_points_labels_map = {}
        for one_borehole in self.boreholes_list:
            if one_borehole.borehole_id is None:
                raise ValueError('Need to set borehole_id first.')
            if one_borehole.borehole_id in borehole_id_points_labels_map.keys():
                raise ValueError('Exists boreholes with duplicate borehole_id.')
            points_data = one_borehole.get_points_data(only_interface=only_interface)
            borehole_points = points_data.points
            borehole_series = points_data.labels
            borehole_series = borehole_series.reshape(-1, 1)
            points_labels = np.concatenate((borehole_points, borehole_series), axis=1)
            borehole_id_points_labels_map[one_borehole.borehole_id] = points_labels
        return borehole_id_points_labels_map

    # 用来导出钻孔信息， dict{drill_key: [drill_points, drill_points_labels]} 将所有依次导出到一个文件
    def export_boreholes_dict_dat_file(self, file_path, only_interface=True):
        boreholes_dict_map = self.get_boreholes_id_points_labels_map(only_interface=only_interface)
        out_path = file_path
        pd_list = []
        key_list = []
        for k, v in boreholes_dict_map.items():
            key_n = [k for k_id in range(v.shape[0])]
            pd_n = pd.DataFrame(v)
            pd_n.insert(loc=0, column='', value=key_n)
            pd_list.append(pd_n)
        pd_file = pd.concat(pd_list, axis=0)
        pd_file.dropna(axis=0, how='any')
        pd_file.to_csv(out_path, index=False, header=False, sep='\t')

    def __len__(self):
        return self.borehole_num

    def __getitem__(self, idx):
        return self.boreholes_list[idx]

    def save(self, dir_path: str, out_name: str = None):
        self.dir_path = dir_path
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        if self.vtk_data is not None and isinstance(self.vtk_data, pv.MultiBlock):
            num = self.vtk_data.n_blocks
            if num > 0:
                save_path = os.path.join(dir_path, self.tmp_dump_str)
                self.vtk_data.save(filename=save_path + '.vtm')
                self.vtk_data = 'dumped'
        file_name = self.tmp_dump_str
        if out_name is not None:
            file_name = out_name
        file_path = os.path.join(dir_path, file_name)
        out_put = open(file_path, 'wb')
        out_str = pickle.dumps(self)
        out_put.write(out_str)
        out_put.close()
        return self.__class__.__name__, file_path

    def load(self):
        if self.dir_path is not None:
            if self.vtk_data == 'dumped':
                save_path = os.path.join(self.dir_path, self.tmp_dump_str)
                if os.path.exists(save_path + '.vtm'):
                    self.vtk_data = pv.read(filename=save_path + '.vtm')
                else:
                    raise ValueError('vtk data file does not exist')
