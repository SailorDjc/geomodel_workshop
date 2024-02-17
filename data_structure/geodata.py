from data_structure.grids import Grid
from data_structure.points import PointSet, get_bounds_from_coords
from data_structure.boreholes import BoreholeSet, Borehole
from data_structure.sections import SectionSet, Section
from data_structure.terrain import TerrainData
import time
import pickle
import os
import numpy as np


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

    def standardize_labels(self, label_dict: dict = None, default_value=-1):
        series_labels = self.get_points_data().labels
        if series_labels is None:
            raise ValueError('The input data has not scalar values.')
        old_label = np.trunc(series_labels)
        unique_label = np.unique(old_label)
        sorted_label = sorted(unique_label)
        if label_dict is not None:
            # 判断label_dict 是否符合要求
            # 默认值不映射
            if default_value in label_dict.keys():
                raise ValueError('The input label_dict is invalid.')
            for idx, item in enumerate(sorted_label):
                # 所有标签都包含
                if item == default_value:
                    continue
                if item not in label_dict.keys():
                    raise ValueError('The input label_dict is invalid.')
                # 不连续
                if idx + 1 < len(sorted_label) and item + 1 != label_dict[idx + 1]:
                    raise ValueError('The input label_dict is invalid.')
        else:
            label_dict = {}
            for idx, item in enumerate(sorted_label):
                if item == default_value:  # 对于默认未知值则不改变
                    continue
                label_dict[item] = idx
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
