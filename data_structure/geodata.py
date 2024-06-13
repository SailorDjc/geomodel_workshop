from data_structure.grids import Grid
from data_structure.points import PointSet, get_bounds_from_coords
from data_structure.boreholes import BoreholeSet, Borehole
from data_structure.sections import SectionSet, Section
from data_structure.terrain import TerrainData
import time
import pickle
import os
import numpy as np
from utils.vtk_utils import bounds_merge, compute_bounds_center
import copy
from utils.math_libs import points_trans_scale


def load_object(file_path, gtype=None):
    if not os.path.exists(file_path):
        raise ValueError('file is not exists.')
    with open(file_path, 'rb') as file:
        object = pickle.loads(file.read())
        if isinstance(object, (PointSet, BoreholeSet, Grid, Section, SectionSet, GeodataSet
                               , TerrainData)):
            if gtype is not None:
                if gtype != object.__class__.__name__:
                    raise ValueError('The data type is inconsistent.')
            object.load()
        else:
            raise ValueError("Unsupported data type.")
        return object


# 地质数据的容器， 支持钻孔、剖面、网格、散点的导入和一体化处理
class GeodataSet(object):
    def __init__(self, name=None, dir_path=None):
        self.name = name
        self.geodata_list = []
        self.tmp_dump_str = 'tmp_geo' + str(int(time.time()))
        # 对象拷贝
        self.dir_path = dir_path
        self._bounds = None
        self.classes_num = 0
        self.classes = None
        self.label_dict = None
        self._center = None
        self._bounds = None

    def append(self, data):
        if not isinstance(data, (PointSet, Borehole, BoreholeSet, SectionSet, Section, Grid)):
            raise ValueError("Input data type is not supported.")
        self.geodata_list.append(copy.deepcopy(data))

    # 根据水平范围筛选数据，返回一个GeodataSet对象
    def search_by_rect2d(self, rect2d):
        selected_list = []
        selected_geodata = GeodataSet()
        selected_geodata.label_dict = self.label_dict
        for g_i in range(len(self.geodata_list)):
            selected_list.append(self.geodata_list[g_i].search_by_rect2d(rect2d=rect2d))
        selected_geodata.geodata_list = selected_list
        return selected_geodata

    # 处理均匀数据，有的非常稀疏，不均匀分布的数据如此均匀分段可能会造成分段无数据的问题。
    # 对大范围数据进行分段，可以按x轴或y轴进行分段，seg_num是分割段数，seg_axis是切割轴向，overlap_ratio是分段间的重叠率
    def get_geodata_segment(self, seg_num, seg_axis='x', overlap_ratio=0):
        if overlap_ratio < 0 or overlap_ratio >= 1:
            raise ValueError('overlay_ratio is out of range.')
        geo_data_list = []
        data_bounds = self.bounds
        axis_labels = ['x', 'y']
        label_to_index = {label: index for index, label in enumerate(axis_labels)}
        if seg_axis.lower() in axis_labels:
            axis_index = label_to_index[seg_axis.lower()]
            seg_length = (data_bounds[2 * axis_index + 1] - data_bounds[2 * axis_index]) / seg_num
            cur_bounds = copy.deepcopy(data_bounds)
            for i in range(seg_num):
                a = data_bounds[2 * axis_index] + i * seg_length
                b = data_bounds[2 * axis_index] + (i + 1) * seg_length + overlap_ratio * seg_length
                cur_bounds[2 * axis_index] = a
                cur_bounds[2 * axis_index + 1] = b
                if i == seg_num - 1:
                    cur_bounds[2 * axis_index + 1] += 1  # 增加一定的缓冲，防止漏选
                geo_data_list.append(self.search_by_rect2d(rect2d=cur_bounds))
        return geo_data_list

    @property
    def center(self):
        if self.bounds is not None:
            center_x = (self.bounds[0] + self.bounds[1]) * 0.5
            center_y = (self.bounds[2] + self.bounds[3]) * 0.5
            center_z = (self.bounds[4] + self.bounds[5]) * 0.5
            self._center = np.array([center_x, center_y, center_z])
            return self._center
        else:
            raise ValueError('This geodata data is empty.')

    @property
    def bounds(self):
        for g_data in self.geodata_list:
            points_data = g_data.get_points_data()
            if self._bounds is None:
                self._bounds = get_bounds_from_coords(points_data.points)
            else:
                new_bounds = get_bounds_from_coords(points_data.points)
                self._bounds = bounds_merge(self._bounds, new_bounds)
        return self._bounds

    def set_class_dict(self, label_dict=None):
        pass

    # 三维空间坐标按比例缩放
    def points_trans_scale(self, scale, center=None):
        if center is None:
            center = self.center
        for g_id, g_data in enumerate(self.geodata_list):
            points_data = g_data.get_points_data()
            self.geodata_list[g_id].points = points_trans_scale(points=points_data.points, center=center, sx=scale[0]
                                                                , sy=scale[1], sz=scale[2])

    # 获取钻孔顶部点坐标
    def get_terrain_points(self):
        terrain_points = []
        for gd in self.geodata_list:
            if isinstance(gd, BoreholeSet):
                top_points = gd.get_top_points()
                terrain_points.append(top_points)
            if isinstance(gd, PointSet):
                pass
        if len(terrain_points) > 0:
            terrain_points = np.concatenate(terrain_points, axis=0)
            return terrain_points
        else:
            return None

    # 计算相对坐标
    def compute_relative_coords(self):
        # 求中心点坐标
        bb_list = []
        for geo_data in self.geodata_list:
            cur_bounds = geo_data.bounds
            bb_list.append(cur_bounds)
        if len(bb_list) >= 1:
            result_bounds = bb_list[0]
            for id in range(len(bb_list) - 1):
                result_bounds = bounds_merge(bounds_a=result_bounds, bounds_b=bb_list[id + 1])
        else:
            raise ValueError('Data is empty.')
        center = compute_bounds_center(result_bounds)
        for geo_data in self.geodata_list:
            geo_data.compute_relative_points(center=center)

    # 通过字典更新标签
    def update_labels_from_label_dict(self, label_dict):
        # 更新标签
        # 只记录初始键到目标值之间的映射，中间的键值对变化不记录
        n_label_dict = copy.deepcopy(label_dict)
        if self.label_dict is not None:
            for ki, vi in n_label_dict.items():
                for kj, vj in self.label_dict.items():
                    if ki == vj:
                        self.label_dict[kj] = vi
        else:
            self.label_dict = n_label_dict
        for sample_data in self.geodata_list:
            sample_data.classes = sorted(n_label_dict.values())
            sample_data.label_dict = self.label_dict
            if isinstance(sample_data, PointSet):
                sample_data.labels = np.vectorize(n_label_dict.get)(np.array(sample_data.labels))
            if isinstance(sample_data, BoreholeSet):
                sample_data.series = np.vectorize(n_label_dict.get)(np.array(sample_data.series))
                # 遍历钻孔
                for idx in np.arange(len(sample_data.boreholes_list)):
                    sample_data.boreholes_list[idx].series = (
                        np.vectorize(n_label_dict.get)(np.array(sample_data.boreholes_list[idx].series)))
                    for l_id in np.arange(len(sample_data.boreholes_list[idx].holelayer_list)):
                        sample_data.boreholes_list[idx].holelayer_list[l_id].layer_label = (
                            n_label_dict)[sample_data.boreholes_list[idx].holelayer_list[l_id].layer_label]
            if isinstance(sample_data, Grid):
                sample_data.grid_points_series = np.vectorize(n_label_dict.get)(
                    np.array(sample_data.grid_points_series))
            if isinstance(sample_data, Section):
                sample_data.series = np.vectorize(n_label_dict.get)(np.array(sample_data.series))
            if isinstance(sample_data, SectionSet):
                for idx in np.arange(len(sample_data.sections)):
                    sample_data.sections[idx].series = (
                        np.vectorize(n_label_dict.get)(np.array(sample_data.sections[idx].series)))
        self.classes_num = len(n_label_dict.values())
        self.classes = np.array(list(n_label_dict.values()))

    # is_continuous=True, 设置的标签必须连续，若不连续
    def standardize_labels(self, label_dict: dict = None, default_value=-1, is_continuous=True):
        n_label_dict = copy.deepcopy(label_dict)
        series_labels = self.get_points_data().labels
        # 原始数据必须有标签值
        if series_labels is None:
            raise ValueError('The input data has not scalar values.')
        arr_type = series_labels.dtype
        continuous_flag = False  # 连续化操作
        if n_label_dict is not None:
            unique_label = np.unique(series_labels)
            # 字符型数组，若标签是字符，需要通过输入的映射表，将字符标签映射为数值标签
            # 判断label_dict 是否符合要求
            if default_value in n_label_dict.keys():  # 默认值不映射
                raise ValueError('The input label_dict conflicts with the default value.')
            for idx, item in enumerate(unique_label):
                # 所有标签都覆盖到
                if item == default_value:
                    continue
                if item not in n_label_dict.keys():
                    raise ValueError('The input label_dict is invalid.')
            # 删除多余映射，确保每一个映射唯一
            del_list = []
            for item in n_label_dict.keys():
                if item not in unique_label:
                    del_list.append(item)
            for item in del_list:
                n_label_dict.pop(item)
            # 标签映射
            self.update_labels_from_label_dict(label_dict=n_label_dict)
            if self.classes.dtype.kind in 'SU':
                raise ValueError('The labels must be numeric.')
            series_labels = self.get_points_data().labels
            sorted_label = sorted(np.unique(np.trunc(series_labels)))
            # 判断映射后标签的连续性
            if is_continuous:
                # 不连续
                for idx, item in enumerate(sorted_label):
                    if idx + 1 < len(sorted_label) and item + 1 != sorted_label[idx + 1]:
                        # 连续化
                        continuous_flag = True  # 标签不连续
                        break
        else:
            if arr_type.kind in 'SUO':
                raise ValueError('The labels must be numeric, should input label_dict.')
        # 数值型标签 或 输入标签映射表不连续
        if (n_label_dict is None and arr_type.kind in 'biuf') or continuous_flag is True:
            # 没有映射表，则按数值大小自动排序
            # 标签自动连续化
            series_labels = self.get_points_data().labels
            trunc_label = np.trunc(series_labels)  # 将浮点型化为整型
            sorted_label = sorted(np.unique(trunc_label))
            new_label_dict = {}
            for idx, item in enumerate(sorted_label):
                if item == default_value:  # 对于默认未知值则不改变
                    continue
                new_label_dict[item] = idx
            self.update_labels_from_label_dict(label_dict=new_label_dict)

    def get_points_data(self):
        points_data_list = []
        for gd in self.geodata_list:
            points_data_list.append(gd.get_points_data())
        if len(points_data_list) > 0:
            points_data_list = PointSet.points_data_merge(points_data_list=points_data_list)
            return copy.deepcopy(points_data_list)
        else:
            return PointSet()

    @property
    def bounds(self):
        points_data = self.get_points_data()
        if points_data.is_empty():
            raise ValueError('The dataset is empty.')
        self._bounds = get_bounds_from_coords(points_data.points)
        return self._bounds

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.geodata_list):
            raise ValueError("Out of list range.")
        return self.geodata_list[idx]

    def __len__(self):
        return len(self.geodata_list)

    def save(self, dir_path: str, out_name: str = None):
        self.dir_path = dir_path
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        for s_id in np.arange(len(self.geodata_list)):
            self.geodata_list[s_id] = self.geodata_list[s_id].save(dir_path=dir_path)
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
        for s_id in np.arange(len(self.geodata_list)):
            self.geodata_list[s_id] = load_object(file_path=self.geodata_list[s_id][1]
                                                  , gtype=self.geodata_list[s_id][0])
