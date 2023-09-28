import pandas as pd
import os
import numpy as np
from data_structure.grids import Grid
from data_structure.boreholes import Borehole, BoreholeSet
from data_structure.points import PointSet
from data_structure.sections import Section, SectionSet


class ReadExportFile(object):
    def __init__(self):
        pass

    # 获取钻孔集 BoreholeSet
    def read_boreholes_data_from_text_file(self, dat_file_path: str = None, **kwargs):
        if not os.path.exists(dat_file_path):
            raise ValueError('The file path does not exist.')
        comment = "#"
        encoding = 'utf-8'
        header = None
        sep = ' '
        globals().update(kwargs)
        df = pd.read_table(dat_file_path, header=header, comment=comment, sep=sep, encoding=encoding)
        columns_num = df.columns.size()
        names = ['borehole_id', 'x', 'y', 'z', 'label']
        if columns_num == 5:
            df.columns = names
        else:
            raise ValueError('File data should be [id x y z label].')
        # 清理空值
        df.dropna()
        points_data = np.array(df.loc[:, ['x', 'y', 'z', 'label']])
        borehole_ids = df.loc[:, ['borehole_id']]
        borehole_map = {}
        for id, borehole_id in enumerate(borehole_ids):
            if borehole_id not in borehole_map.keys():
                borehole_map[borehole_id] = []
            borehole_map[borehole_id].append(points_data[id])
        borehole_list = BoreholeSet()
        for k, v in borehole_map.keys():
            borehole_points = np.array(borehole_map[k])[:, 0:3]
            borehole_series = np.array(borehole_map[k])[:, 3]
            one_borehole = Borehole(points=borehole_points, series=borehole_series, borehole_id=k)
            borehole_list.append(one_borehole=one_borehole)
        return borehole_list

    # node [x, y, z, label]
    # 获取点集 PointSet
    def read_points_data_from_text_file(self, dat_file_path: str = None, **kwargs):
        if not os.path.exists(dat_file_path):
            raise ValueError('The file path does not exist.')
        comment = "#"
        encoding = 'utf-8'
        header = None
        sep = ' '
        use_cols = [0, 1, 2, 3]
        globals().update(kwargs)
        df_points = pd.read_table(dat_file_path, header=header, comment=comment, sep=sep, encoding=encoding)
        columns_num = df_points.columns.size()
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
    def read_graph_data_from_edge_files(self, edge_file_path, is_edge_node_begin_from_zero=False, **kwargs):
        if edge_file_path is None:
            raise ValueError('Edge file is necessary.')
        if not os.path.exists(edge_file_path):
            raise ValueError('The file path does not exist.')
        header = None
        sep = '\s+'
        comment = "#"
        encoding = 'utf-8'
        use_cols_edge = [0, 1]
        use_cols_node = [0, 1, 2]
        globals().update(kwargs)
        df_edge = pd.read_table(edge_file_path, header=header, comment=comment, sep=sep, encoding=encoding)
        cols_edge_num = df_edge.columns.size()
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
