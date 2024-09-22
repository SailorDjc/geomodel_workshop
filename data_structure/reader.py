import pandas as pd
import os
import numpy as np
from data_structure.geodata import GeodataSet, load_object, Section, Borehole, BoreholeSet, Grid, PointSet, TerrainData
import pickle
from utils.vtk_utils import reader_xml_polydata_file, reader_unstructured_mesh_file
import pyvista as pv
import torch


class ReadExportFile(object):
    def __init__(self):
        pass

    # 获取钻孔集 BoreholeSet
    @staticmethod
    def read_boreholes_data_from_text_file(dat_file_path: str = None, **kwargs):
        if not os.path.exists(dat_file_path):
            raise ValueError('The file path does not exist.')
        comment = "#"
        encoding = 'utf-8'
        header = None
        sep = '\s+'
        if 'comment' in kwargs.keys():
            comment = kwargs['comment']
        if 'encoding' in kwargs.keys():
            encoding = kwargs['encoding']
        if 'header' in kwargs.keys():
            header = kwargs['header']
        if 'sep' in kwargs.keys():
            sep = kwargs['sep']
        df = pd.read_table(dat_file_path, header=header, comment=comment, sep=sep, encoding=encoding)
        columns_num = df.columns.size
        names = ['borehole_id', 'x', 'y', 'z', 'label']
        if columns_num == 5:
            df.columns = names
        else:
            raise ValueError('File data should be [id x y z label].')
        # 清理空值
        df.dropna()
        points_data = np.float32(df.loc[:, ['x', 'y', 'z']])  # , 'label'
        points_label = np.array(df.loc[:, 'label'])

        borehole_ids = list(df.loc[:, 'borehole_id'])
        borehole_map = {}
        for id, borehole_id in enumerate(borehole_ids):
            if borehole_id not in borehole_map.keys():
                borehole_map[borehole_id] = {}
                borehole_map[borehole_id]['points'] = []
                borehole_map[borehole_id]['label'] = []
            borehole_map[borehole_id]['points'].append(points_data[id])
            borehole_map[borehole_id]['label'].append(points_label[id])
        borehole_list = BoreholeSet()
        for k in borehole_map.keys():
            borehole_points = np.vstack(borehole_map[k]['points'])
            borehole_series = np.array(borehole_map[k]['label'])
            one_borehole = Borehole(points=borehole_points, series=borehole_series, borehole_id=k)
            borehole_list.append(one_borehole=one_borehole)
        return borehole_list

    # node [x, y, z, label]
    # 获取点集 PointSet
    @staticmethod
    def read_points_data_from_text_file(dat_file_path: str = None, **kwargs):
        if not os.path.exists(dat_file_path):
            raise ValueError('The file path does not exist.')
        comment = "#"
        encoding = 'utf-8'
        header = None
        sep = ' '
        use_cols = [0, 1, 2, 3]
        if 'comment' in kwargs.keys():
            comment = kwargs['comment']
        if 'encoding' in kwargs.keys():
            encoding = kwargs['encoding']
        if 'header' in kwargs.keys():
            header = kwargs['header']
        if 'sep' in kwargs.keys():
            sep = kwargs['sep']
        if 'use_cols' in kwargs.keys():
            use_cols = kwargs['use_cols']
        df_points = pd.read_table(dat_file_path, header=header, comment=comment, sep=sep, encoding=encoding)
        columns_num = df_points.columns.size
        if columns_num >= max(use_cols):
            if len(use_cols) == 3:
                df_points = df_points.iloc[:, use_cols]
                df_points.columns = ['x', 'y', 'z']
                df_points.dropna()
                points = np.float32(df_points.loc[:, ['x', 'y', 'z']])
                return points
            if len(use_cols) == 4:
                df_points = df_points.iloc[:, use_cols]
                df_points.columns = ['x', 'y', 'z', 'label']
                df_points.dropna()
                points_labels = np.array(df_points.loc[:, ['x', 'y', 'z', 'label']])
                points = np.float32(points_labels[:, 0:3])
                labels = np.array(points_labels[:, 3])
                points_data = PointSet(points=points, point_labels=labels)
                return points_data
        else:
            raise ValueError('Input data file is not support for expected format.')

    # 获取 边集 数据
    # edge [src, dst]
    @staticmethod
    def read_graph_data_from_edge_files(edge_file_path, is_edge_node_begin_from_zero=False, **kwargs):
        if edge_file_path is None:
            raise ValueError('Edge file is necessary.')
        if not os.path.exists(edge_file_path):
            raise ValueError('The file path does not exist.')
        header = None
        sep = '\s+'
        comment = "#"
        encoding = 'utf-8'
        use_cols_edge = [0, 1]
        if 'comment' in kwargs.keys():
            comment = kwargs['comment']
        if 'encoding' in kwargs.keys():
            encoding = kwargs['encoding']
        if 'header' in kwargs.keys():
            header = kwargs['header']
        if 'sep' in kwargs.keys():
            sep = kwargs['sep']
        if 'use_cols_edge' in kwargs.keys():
            use_cols_edge = kwargs['use_cols_edge']
        globals().update(kwargs)
        df_edge = pd.read_table(edge_file_path, header=header, comment=comment, sep=sep, encoding=encoding)
        cols_edge_num = df_edge.columns.size
        if cols_edge_num >= max(use_cols_edge):
            df_edge = df_edge.iloc[:, use_cols_edge]
            if len(use_cols_edge) == 2:
                df_edge.columns = ['src', 'dst']
            if len(use_cols_edge) == 3:
                df_edge.columns = ['src', 'dst', 'other']
        else:
            raise ValueError('Edge file is not supported for expected format.')
        edge_list = df_edge.loc[:, ['src', 'dst']].values.tolist()
        # 有向图， 去除重复边
        new_edge_list = list(set(tuple(sorted(sub)) for sub in edge_list))
        for e_i, edge in enumerate(new_edge_list):
            edge = list(edge)
            # 若从1开始，需要减1
            if not is_edge_node_begin_from_zero:
                edge[0] -= 1
                edge[1] -= 1
            new_edge_list[e_i] = np.array(edge)
        return np.array(new_edge_list)

    # 读取自定义的地质数据结构(Grid, BoreholeSet,SectionSet, Section等)
    @staticmethod
    def read_geodata(file_path: str):
        geo_object = load_object(file_path=file_path)
        return geo_object

    # 读取vtk模型文件
    @staticmethod
    def read_vtk_data(file_path: str):
        path, filename = os.path.split(file_path)
        model_name, suffix = os.path.splitext(filename)
        if suffix == '.vtp':
            model = reader_xml_polydata_file(pd_filename=file_path)
            model = pv.wrap(model)
        elif suffix == '.vtk':
            # model = reader_unstructured_mesh_file(mesh_filename=file_path)
            # model = pv.wrap(model)
            model = pv.read(filename=file_path)
        elif suffix == '.vtm':
            model = pv.read(filename=file_path)
        else:
            model = reader_unstructured_mesh_file(mesh_filename=file_path)
            model = pv.wrap(model)
        return model

    # col_names = ['label', 'code', 'name']
    @staticmethod
    def read_labels_map(map_file_path, col_names=None, **kwargs):
        if col_names is None:
            col_names = ['label', 'code', 'name']
        if not os.path.exists(map_file_path):
            raise ValueError('The file path does not exist.')
        comment = "#"
        encoding = 'unicode_escape'
        header = None
        sep = '\s+'
        if 'comment' in kwargs.keys():
            comment = kwargs['comment']
        if 'encoding' in kwargs.keys():
            encoding = kwargs['encoding']
        if 'header' in kwargs.keys():
            header = kwargs['header']
        if 'sep' in kwargs.keys():
            sep = kwargs['sep']
        df = pd.read_table(map_file_path, header=header, comment=comment, sep=sep, encoding=encoding)
        df.columns = col_names
        # 清理空值
        df.dropna(how='all')
        map_data = df.loc[:, col_names]
        map_dict = []
        for index, row in map_data.iterrows():
            for r_i in range(len(col_names)):
                map_record = {col_names[r_i]: row[col_names[r_i]]}
                map_dict.append(map_record)
        return map_dict

    @staticmethod
    def tmp_read_virtual_boreholes(dat_file_path):
        df_0 = pd.read_excel(dat_file_path, sheet_name=0)
        df_1 = pd.read_excel(dat_file_path, sheet_name=1)
        layer_info = df_1.groupby(df_1.columns[0])
        borehole_list = BoreholeSet()
        for borehole_id, group in layer_info:
            borehole_info = df_0[df_0[df_0.columns[0]] == borehole_id]
            b_id = borehole_info.iat[0, 0]
            xx = borehole_info.iat[0, 1]
            yy = borehole_info.iat[0, 2]
            top = borehole_info.iat[0, 3]
            depth = borehole_info.iat[0, 4]
            one_borehole = Borehole(borehole_id=b_id)
            one_borehole.top_pos = np.array([xx, yy, top])
            one_borehole.bottom_pos = np.array([xx, yy, top - depth])
            for index, row in group.iterrows():
                layer_code = row[df_1.columns[1]]
                layer_top = row[df_1.columns[2]]
                layer_bottom = row[df_1.columns[3]]
                one_layer = Borehole.Holelayer(coord_top=np.array([xx, yy, top - layer_top]),
                                               coord_bottom=np.array([xx, yy, top - layer_bottom]),
                                               layer_label=layer_code)
                one_borehole.holelayer_list.append(one_layer)
            borehole_list.append(one_borehole)
        return borehole_list

    @staticmethod
    def tmp_read_boreholes(excel_path):
        df_0 = pd.read_excel(excel_path, sheet_name=1)
        df_1 = pd.read_excel(excel_path, sheet_name=2)
        layer_info = df_1.groupby(df_1.columns[1])
        borehole_list = BoreholeSet()
        for borehole_id, group in layer_info:
            borehole_info = df_0[df_0[df_0.columns[1]] == borehole_id]
            b_id = borehole_info.iat[0, 1]
            xx = borehole_info.iat[0, 2]
            yy = borehole_info.iat[0, 3]
            top = borehole_info.iat[0, 4]
            depth = borehole_info.iat[0, 5]
            one_borehole = Borehole(borehole_id=b_id)
            one_borehole.top_pos = np.array([xx, yy, top])
            one_borehole.bottom_pos = np.array([xx, yy, top - depth])
            for index, row in group.iterrows():
                layer_code = row[df_1.columns[2]]
                layer_top = row[df_1.columns[3]]
                layer_bottom = row[df_1.columns[4]]
                one_layer = Borehole.Holelayer(coord_top=np.array([xx, yy, top - layer_top]),
                                               coord_bottom=np.array([xx, yy, top - layer_bottom]),
                                               layer_label=layer_code)
                one_borehole.holelayer_list.append(one_layer)
            borehole_list.append(one_borehole)
        return borehole_list

    @staticmethod
    def read_train_loss_log(txt_file_path, **kwargs):
        if not os.path.exists(txt_file_path):
            raise ValueError('The file path does not exist.')
        comment = "#"
        encoding = 'utf-8'
        header = None
        sep = ' '
        use_cols = [0, 1, 2, 3]
        if 'comment' in kwargs.keys():
            comment = kwargs['comment']
        if 'encoding' in kwargs.keys():
            encoding = kwargs['encoding']
        if 'header' in kwargs.keys():
            header = kwargs['header']
        if 'sep' in kwargs.keys():
            sep = kwargs['sep']
        if 'use_cols' in kwargs.keys():
            use_cols = kwargs['use_cols']
        df_logs = pd.read_table(txt_file_path, header=header, comment=comment, sep=sep, encoding=encoding)
        columns_num = df_logs.columns.size
        names = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
        if columns_num == len(names):
            df_logs.columns = names
        else:
            raise ValueError()
        df_logs.dropna()
        epoch = np.array(df_logs.loc[:, ['epoch']])
        train_loss = np.array(df_logs.loc[:, ['train_loss']])
        train_acc = np.array(df_logs.loc[:, ['train_acc']])
        val_loss = np.array(df_logs.loc[:, ['val_loss']])
        val_acc = np.array(df_logs.loc[:, ['val_acc']])
        result_log = {
            'epochs': epoch.flatten(),
            'train_loss': train_loss.flatten(),
            'train_accuracy': train_acc.flatten(),
            'valid_loss': val_loss.flatten(),
            'valid_accuracy': val_acc.flatten(),
        }
        return result_log

    @staticmethod
    def read_array_from_txt(txt_file_path) -> torch.Tensor:
        loaded_tensor = torch.from_numpy(np.loadtxt(txt_file_path))
        return loaded_tensor


class WriteExportFile(object):
    def __init__(self):
        pass

    @staticmethod
    def write_nodes(points_data: PointSet, file_path, out_label=False, add_index=True, index_start_number=1):
        points_num = points_data.points_num
        coords = points_data.get_points()
        indexes = None
        out_data = coords
        if add_index:
            indexes = np.arange(index_start_number, points_num+index_start_number)
        if indexes is not None:
            out_data = np.column_stack((indexes, coords))
        out_data = pd.DataFrame(out_data)
        if indexes is not None:
            out_data.columns = ['a', 'b', 'c', 'd']
            out_data = out_data.astype({'a': int})
        out_data.to_csv(file_path, index=False, header=False, sep='\t')

# class BoreholeSetManager(object):
#     def __init__(self):
#         self.reader = ReadExportFile()
#         self.borehole_data = []
#         self.labels_map = None
#         self.col_names = ['label', 'code', 'name']
#
#     def get_labels(self, ind=0):
#         # if self.labels_map is None:
#         #     # labels
#         #     for borehole_data in self.borehole_data:
#         #         labels = self.borehole_data.get_classes()
#         #     return labels
#         # labels = [l_item.get(self.col_names[ind]) for l_item in self.labels_map]
#         # return np.trunc(labels)
#         pass
#
#     def append_boreholes_dataset_from_txt_file(self, dat_file_path, **kwargs):
#         borehole_data = self.reader.read_boreholes_data_from_text_file(dat_file_path=dat_file_path, **kwargs)
#         self.borehole_data.append(borehole_data)
#
#     def read_labels_map(self, map_file_path, col_names=None, **kwargs):
#         if col_names is not None:
#             self.col_names = col_names
#         self.labels_map = self.reader.read_labels_map(map_file_path=map_file_path, col_names=col_names, **kwargs)
#
#     # 标准化标签，标签是从0开始的连续自然数
#     def standardize_labels(self, label_dict: dict = None, default_value=-1):
#         series_labels = self.get_labels()
#         if series_labels is None:
#             raise ValueError('The input data has not scalar values.')
#         sorted_label = sorted(series_labels)
#         if label_dict is not None:
#             # 判断label_dict 是否符合要求
#             # 默认值不映射
#             if default_value in label_dict.keys():
#                 raise ValueError('The input label_dict is invalid.')
#             for idx, item in enumerate(sorted_label):
#                 # 所有标签都包含
#                 if item == default_value:
#                     continue
#                 if item not in label_dict.keys():
#                     raise ValueError('The input label_dict is invalid.')
#                 # 不连续
#                 if idx + 1 < len(sorted_label) and item + 1 != label_dict[idx + 1]:
#                     raise ValueError('The input label_dict is invalid.')
#         else:
#             label_dict = {}
#             for idx, item in enumerate(sorted_label):
#                 if item == default_value:  # 对于默认未知值则不改变
#                     continue
#                 label_dict[item] = idx
#         # 更新标签
#         for sample_data in self.borehole_data:
#             sample_data.classes = sorted(label_dict.values())
#             if isinstance(sample_data, BoreholeSet):
#                 sample_data.series = np.vectorize(label_dict.get)(np.array(sample_data.series))
#                 # 遍历钻孔
#                 for idx in np.arange(len(sample_data.boreholes_list)):
#                     sample_data.boreholes_list[idx].series = (
#                         np.vectorize(label_dict.get)(np.array(sample_data.boreholes_list[idx].series)))
#                     for l_id in np.arange(len(sample_data.boreholes_list[idx].holelayer_list)):
#                         sample_data.boreholes_list[idx].holelayer_list[l_id].layer_label = (
#                             label_dict)[sample_data.boreholes_list[idx].holelayer_list[l_id].layer_label]
#             else:
#                 raise ValueError('The Input data is not BoreholeSet.')
#         if self.labels_map is not None:
#             for r_i in range(len(self.labels_map)):
#                 self.labels_map[r_i][self.col_names[0]] = label_dict[self.labels_map[r_i][self.col_names[0]]]
#
#     # 去除特殊地质体
#     def delete_special_geologic_body(self, del_labels_list):
#         pass
#
#     # 获取钻孔地层序列
#     def get_boreholes_labels_sequence(self, is_align_bott=False, is_align_top=False):
#         label_sequence = []
#         for one_borehole in self.borehole_data:
#             pass
