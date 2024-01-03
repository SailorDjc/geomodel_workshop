from data_structure.boreholes import BoreholeSet, Borehole, PointSet
import numpy as np


class BoreholeDataPrepare(object):
    def __init__(self, boreholes: BoreholeSet, label_dict: dict = None, label_map=False, default_value=-1):
        self.borehole_set = boreholes
        self.label_dict = label_dict
        self.default_value = default_value
        self.classes = None
        self.classes_num = 0
        if label_dict is not None or label_map:
            self.standardize_labels(label_dict=label_dict, default_value=self.default_value)
        else:
            self.classes = self.borehole_set.classes
            self.classes_num = self.borehole_set.classes_num
        self.labels_sequences = []
        self.align_label_flag = False

    def standardize_labels(self, label_dict: dict, default_value=-1):
        series_labels = self.borehole_set.get_points_data().labels
        if series_labels is None:
            raise ValueError('The input data has not scalar values.')
        old_label = np.trunc(series_labels)
        unique_label = np.unique(old_label)
        sorted_label = sorted(unique_label)
        if label_dict is None:
            label_dict = {}
            for idx, item in enumerate(sorted_label):
                if item == default_value:  # 对于默认未知值则不改变
                    continue
                label_dict[item] = idx
        # 更新标签
        if isinstance(self.borehole_set, BoreholeSet):
            self.borehole_set.series = np.vectorize(label_dict.get)(np.array(self.borehole_set.series))
            # 遍历钻孔
            for idx in np.arange(len(self.borehole_set.boreholes_list)):
                self.borehole_set.boreholes_list[idx].series = (
                    np.vectorize(label_dict.get)(np.array(self.borehole_set.boreholes_list[idx].series)))
                for l_id in np.arange(len(self.borehole_set.boreholes_list[idx].holelayer_list)):
                    self.borehole_set.boreholes_list[idx].holelayer_list[l_id].layer_label = (
                        label_dict)[self.borehole_set.boreholes_list[idx].holelayer_list[l_id].layer_label]
        self.label_dict = label_dict
        self.classes_num = len(label_dict.values())
        self.classes = np.array(list(label_dict.values()))

    # 对齐标签序列, 默认-2为空气，-1为地下待求值
    def align_labels_sequence(self, top_z=None, bottom_z=None, dim_z=50, default_up_value=-2, default_down_value=-1):
        #
        points_data = self.borehole_set.get_points_data()
        points = points_data.points
        # 若没有指定上下界，则自适应
        if top_z is None:
            top_z = np.max(points[:, 2])
        if bottom_z is None:
            bottom_z = np.min(points[:, 2])
        zrng = np.linspace(start=bottom_z, stop=top_z, num=dim_z)
        for h_id in np.arange(len(self.borehole_set)):
            layers_sequence = []
            top_pnt = self.borehole_set[h_id].top_pos
            for pnt_z in zrng:
                layer_label = (self.borehole_set[h_id].get_layer_label_with_point_z
                               (np.array([top_pnt[0], top_pnt[1], pnt_z])))
                layers_sequence.append(layer_label)

