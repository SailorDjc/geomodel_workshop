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


def load_object(file_path, gtype=None):
    if not os.path.exists(file_path):
        raise ValueError('file is not exists.')
    with open(file_path, 'rb') as file:
        object = pickle.loads(file.read())
        if isinstance(object, (PointSet, BoreholeSet, Grid, Section, SectionSet, GeodataSet, TerrainData)):
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

    def append(self, data):
        if not isinstance(data, (PointSet, Borehole, BoreholeSet, SectionSet, Section, Grid)):
            raise ValueError("Input data type is not supported.")
        self.geodata_list.append(data)

    def set_class_dict(self, label_dict=None):
        pass

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

    def update_labels_from_label_dict(self, label_dict):
        # 更新标签
        for sample_data in self.geodata_list:
            sample_data.classes = sorted(label_dict.values())
            if isinstance(sample_data, PointSet):
                sample_data.labels = np.vectorize(label_dict.get)(np.array(sample_data.labels))
            if isinstance(sample_data, BoreholeSet):
                sample_data.series = np.vectorize(label_dict.get)(np.array(sample_data.series))
                # 遍历钻孔
                for idx in np.arange(len(sample_data.boreholes_list)):
                    sample_data.boreholes_list[idx].series = (
                        np.vectorize(label_dict.get)(np.array(sample_data.boreholes_list[idx].series)))
                    for l_id in np.arange(len(sample_data.boreholes_list[idx].holelayer_list)):
                        sample_data.boreholes_list[idx].holelayer_list[l_id].layer_label = (
                            label_dict)[sample_data.boreholes_list[idx].holelayer_list[l_id].layer_label]
            if isinstance(sample_data, Grid):
                sample_data.grid_points_series = np.vectorize(label_dict.get)(np.array(sample_data.grid_points_series))
            if isinstance(sample_data, Section):
                sample_data.series = np.vectorize(label_dict.get)(np.array(sample_data.series))
            if isinstance(sample_data, SectionSet):
                for idx in np.arange(len(sample_data.sections)):
                    sample_data.sections[idx].series = (
                        np.vectorize(label_dict.get)(np.array(sample_data.sections[idx].series)))
        self.label_dict = label_dict
        self.classes_num = len(label_dict.values())
        self.classes = np.array(list(label_dict.values()))

    def standardize_labels(self, label_dict: dict = None, default_value=-1, is_continuous=True):
        series_labels = self.get_points_data().labels
        # 原始数据必须有标签值
        if series_labels is None:
            raise ValueError('The input data has not scalar values.')
        arr_type = series_labels.dtype
        continuous_flag = False  # 连续化操作
        if label_dict is not None:
            unique_label = np.unique(series_labels)
            # 字符型数组，若标签是字符，需要通过输入的映射表，将字符标签映射为数值标签
            # 判断label_dict 是否符合要求
            if default_value in label_dict.keys():  # 默认值不映射
                raise ValueError('The input label_dict conflicts with the default value.')
            for idx, item in enumerate(unique_label):
                # 所有标签都覆盖到
                if item == default_value:
                    continue
                if item not in label_dict.keys():
                    raise ValueError('The input label_dict is invalid.')
                    # 标签映射
            self.update_labels_from_label_dict(label_dict=label_dict)
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
                        continuous_flag = True
        else:
            if arr_type.kind in 'SU':
                raise ValueError('The labels must be numeric, should input label_dict.')
        if (label_dict is None and arr_type.kind in 'biuf') or continuous_flag is True:
            # 数值型标签
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
            if continuous_flag is True and label_dict is not None:
                self.label_dict = {}
                for n_key, n_value in new_label_dict.items():
                    for key, value in label_dict.items():
                        if n_key == value:
                            self.label_dict[key] = n_value
                            break

    def get_points_data(self):
        points_data_list = []
        for gd in self.geodata_list:
            points_data_list.append(gd.get_points_data())
        if len(points_data_list) > 0:
            points_data_list = PointSet.points_data_merge(points_data_list=points_data_list)
            return points_data_list
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
