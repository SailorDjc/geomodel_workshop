import copy
import pickle
import numpy as np

import os
import dgl
import dgl.backend as F
import torch
from dgl.data.utils import load_graphs, save_graphs
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate
from geograph_parse import GeoMeshGraphParse
import time
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

from data_structure.grids import Grid
from data_structure.boreholes import BoreholeSet, Borehole
from data_structure.sections import Section, SectionSet
from data_structure.points import PointSet


# from data_structure.data_sampler import GeoGridDataSampler
# from scipy.interpolate import RBFInterpolator, NearestNDInterpolator, LinearNDInterpolator
# from scipy.stats.qmc import Halton
# from sklearn.preprocessing import StandardScaler  # 标准化
# from sklearn.preprocessing import MinMaxScaler  # 归一化
# from sklearn.metrics import classification_report
# import model_visual_kit as mvk
# import torchmetrics.functional as MF
# import pyvista as pv
# import pandas as pd
# from sklearn import preprocessing
# import copy
# from scipy.interpolate import griddata

def create_dgl_graph(edge_list, node_feat=None, edge_feat=None, node_label=None, add_inverse_edge=False):
    num_node = len(node_feat)
    num_edge = edge_list.shape[1]

    if np.isnan(node_label).any():
        node_label = torch.from_numpy(node_label).to(torch.float32)
    else:
        node_label = torch.from_numpy(node_label).to(torch.long)

    print('Processing graphs...')
    graph = dict()
    # handling edge
    if add_inverse_edge:
        # duplicate edge
        duplicated_edge = np.repeat(edge_list[:, 0:num_edge], 2, axis=1)
        duplicated_edge[0, 1::2] = duplicated_edge[1, 0::2]
        duplicated_edge[1, 1::2] = duplicated_edge[0, 0::2]

        graph['edge_index'] = duplicated_edge

        if edge_feat is not None:
            graph['edge_feat'] = np.repeat(edge_feat[0:num_edge], 2, axis=0)
        else:
            graph['edge_feat'] = None

    else:
        graph['edge_index'] = edge_list[:, 0:num_edge]

        if edge_feat is not None:
            graph['edge_feat'] = edge_feat[0:num_edge]
        else:
            graph['edge_feat'] = None

    # handling node
    if node_feat is not None:
        graph['node_feat'] = node_feat[0:num_node]
    else:
        graph['node_feat'] = None

    graph['num_nodes'] = num_node

    g = dgl.graph((graph['edge_index'][0], graph['edge_index'][1]), num_nodes=graph['num_nodes'])

    if graph['edge_feat'] is not None:
        g.edata['feat'] = torch.from_numpy(graph['edge_feat'])

    if graph['node_feat'] is not None:
        g.ndata['feat'] = torch.from_numpy(graph['node_feat'])
    if node_label is not None:
        g.ndata['label'] = F.reshape(node_label, (g.num_nodes(),))
    return g


class GmeModelGraphList(object):
    def __init__(self, name, root, grid_data: list = None, input_sample_data=None, val_ratio=None,
                 sample_operator=None, self_loop=False, add_inverse_edge=True,
                 dgl_graph_param=None, update_graph=False, grid_dims=None, terrain_data=None,
                 grid_cell_density=None, **kwargs):
        # 注：.dat文件格式与Voxler软件一致
        if dgl_graph_param is None:
            dgl_graph_param = [['position'], None]  # [[node_feat], [edge_feat]]
        # 外部传入参数
        self.grid_data = grid_data  # list[Grid]
        self.input_sample_data = input_sample_data
        self.grid_dims = grid_dims
        if grid_dims is None:
            self.grid_dims = np.array([80, 80, 50])
        self.sample_data = []
        self.dgl_graph_param = dgl_graph_param  # 图节点特征、边特征类型参数
        self.sample_operator = sample_operator  # ['rand_drills', 'axis_sections'] 设置已知数据采样方式
        self.val_ratio = val_ratio
        self.self_loop = self_loop  # 添加自环
        self.add_inverse_edge = add_inverse_edge  # 无向图
        self.root = root  # 代码工作空间根目录， 会默认将处理后数据存放在 root/processed目录下
        self.name = name  # 数据集名称
        self.update_graph = update_graph
        # 数据存储参数
        self.geograph = []
        self.result_model = {}
        self.graph_log = {}  # 待删除
        self.num_classes = None  # 数据集分类数目
        self.terrain_data = terrain_data  # 地形数据
        self.grid_cell_density = grid_cell_density
        # 文件存储路径，在processed文件夹下共存储3个数据文件，两个模型训练checkpoint存储文件
        processed_dir = os.path.join(self.root, 'processed')  # 存储

        self.processed_dir = processed_dir
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        self.processed_file_path = os.path.join(processed_dir, 'gme_data_processed')  # 存储与训练的dgl_graph图数据
        self.processed_geodata_path = os.path.join(processed_dir, 'geograph.pkl')  # 存储geodata模型数据
        # 待删除
        # 记录 dgl_graph, 以及图生成参数，包括node_feat类型, edge_feat类型,类别数, 节点数, 边数.
        self.graph_log_data_path = os.path.join(processed_dir, 'graph_log.pkl')  # 存储dgl_graph的日志信息
        ##
        self.graph = None  # 图数据
        super(GmeModelGraphList, self).__init__()
        self.kwargs = kwargs
        self.terrain_data = terrain_data
        self.access_model_data()

    # 访问模型数据
    def access_model_data(self):
        # 如果存在处理后数据，则直接从文件加载
        # 加载图日志数据
        if os.path.exists(self.graph_log_data_path):
            self.graph_log = self.load_graph_log()
        # 加载dgl_graph和geodata, 其中dgl_graph是图神经网络训练的数据源, geodata主要存储地质数据源用来可视化分析
        if os.path.exists(self.processed_geodata_path):  # geodata
            self.load_geograph()
            geodata = self.geograph[0]
            label_num = geodata.data.classes_num
            print('Loading {} GeoModel Data ...'.format(len(self.geograph)))
        # 加载图数据
        if os.path.exists(self.processed_file_path):
            self.graph, self.num_classes = load_graphs(self.processed_file_path)  # dgl_graph
            print('Loading {} Saved Graphs Files Data ...'.format(len(self.graph)))
        # 容器初始化
        if self.graph is None:
            self.graph = []
        if self.num_classes is None:
            self.num_classes = {'labels': []}
        if len(self.graph) == 0 or self.update_graph:
            self.process_grid_data_to_graph_data()

    # 将网格数据处理为图结构数据
    def process_grid_data_to_graph_data(self):
        # 批量处理每一个地质模型数据文件
        dgl_graph_list = []
        labels_num_list = []
        save_flag = False
        if self.grid_data is not None:
            for model_index in np.arange(len(self.grid_data)):
                mesh = self.grid_data[model_index]
                geodata = GeoMeshGraphParse(mesh, name=mesh.name)
                dgl_graph = geodata.execute(sample_operator=self.sample_operator,
                                            edge_feat=self.dgl_graph_param[1], node_feat=self.dgl_graph_param[0],
                                            feat_normalize=True, val_ratio=self.val_ratio, **self.kwargs)
                # 对标签进行处理
                label_num = geodata.classes_num
                labels_num_list.append(label_num)
                # 已存储数据集
                pre_save_graph_num = len(self.graph)
                is_connected = geodata.is_connected_graph()
                print('is_connected:', is_connected)
                self.geograph.append(geodata)
                dgl_graph_list.append(dgl_graph)
                self.update_graph_log(model_name=mesh.name,
                                      save_idx=model_index + pre_save_graph_num,
                                      node_feat=self.dgl_graph_param[0], edge_feat=self.dgl_graph_param[1],
                                      edge_num=dgl_graph.num_edges(), node_num=dgl_graph.num_nodes())
                save_flag = True
            self.graph.extend(dgl_graph_list)
            if torch.is_tensor(self.num_classes['labels']):
                self.num_classes['labels'] = self.num_classes['labels'].numpy().tolist()
            self.num_classes['labels'].extend(labels_num_list)
            self.num_classes = {'labels': torch.tensor(self.num_classes['labels']).to(torch.long)}
        # 只有采样数据，没有输入的网格数据的情况下，进行建模
        elif self.input_sample_data is not None and len(self.input_sample_data) > 0:
            # 只支持单图构建
            geodata = GeoMeshGraphParse(input_sample_data=self.input_sample_data, name='boreholes_model'
                                        , grid_dims=self.grid_dims)
            external_grid = None
            # 以带地形的体素网格作为建模框架
            if self.terrain_data is not None:
                if self.terrain_data.vtk_data is None:
                    top_points = self.input_sample_data.get_terrain_points()
                    self.terrain_data.set_control_points(control_points=top_points)
                    self.terrain_data.execute()
                build_bounds = self.input_sample_data.bounds
                external_grid = self.terrain_data.create_grid_from_terrain_surface(z_min=build_bounds[4]
                                                                                   , cell_density=self.grid_cell_density
                                                                                   , is_smooth=False)
            dgl_graph = geodata.execute(edge_feat=self.dgl_graph_param[1], node_feat=self.dgl_graph_param[0],
                                        feat_normalize=True, ext_grid=external_grid, val_ratio=self.val_ratio
                                        , **self.kwargs)
            print("Hmmmm")
            geodata.data.vtk_data.plot()
            # 对标签进行处理
            label_num = geodata.classes_num
            labels = geodata.data.classes
            labels_num_list.append(label_num)
            # 已存储数据集
            pre_save_graph_num = len(self.graph)
            # is_connected = geodata.is_connected_graph()
            # print('is_connected:', is_connected)
            self.geograph.append(geodata)
            dgl_graph_list.append(dgl_graph)
            self.update_graph_log(model_name='boreholes',
                                  save_idx=pre_save_graph_num,
                                  node_feat=self.dgl_graph_param[0], edge_feat=self.dgl_graph_param[1],
                                  edge_num=dgl_graph.num_edges(), node_num=dgl_graph.num_nodes())
            save_flag = True

            self.graph.extend(dgl_graph_list)
            if torch.is_tensor(self.num_classes['labels']):
                self.num_classes['labels'] = self.num_classes['labels'].numpy().tolist()
            self.num_classes['labels'].extend(labels_num_list)
            self.num_classes = {'labels': torch.tensor(self.num_classes['labels']).to(torch.long)}
        # 数据存储
        if save_flag is True:
            print('Saving...')
            self.save_graph_log()
            self.save_geograph()
            self.load_geograph()
            self.graph_log = self.load_graph_log()
            print('Saving...')
            save_graphs(self.processed_file_path, self.graph, self.num_classes)
            self.graph, self.num_classes = load_graphs(self.processed_file_path)

    # 如果数据集已经进行了验证集分割，则这里的参数val_ratio无效
    def get_split_idx(self, idx, val_ratio=None, test_ratio=None):
        if idx >= len(self.graph) or idx < 0:
            raise ValueError('Input idx is out of graph array range.')
        else:
            node_num = self.graph[idx].num_nodes()
            x = np.arange(node_num)
            # 已经预先划分了训练数据，即已知标签数据
            default_val_ratio = 0.1
            if val_ratio is None:
                val_ratio = default_val_ratio
            if test_ratio is None or test_ratio + val_ratio >= 1:
                test_ratio = (1 - val_ratio) * 0.7
            if self.geograph[idx].train_data_indexes is None:
                # random_state 确保每次切分是确定的
                val_train, test = train_test_split(x, test_size=test_ratio, random_state=2)
                train, val = train_test_split(val_train, test_size=val_ratio, random_state=2)
            else:
                print('get split Dataset')
                train = np.array(self.geograph[idx].train_data_indexes)
                if self.geograph[idx].val_data_indexes is not None and len(self.geograph[idx].val_data_indexes) > 0:
                    val = np.array(self.geograph[idx].val_data_indexes)
                else:
                    train, val = train_test_split(train, test_size=val_ratio, random_state=2)
                test = np.array(list(set(x) - set(train) - set(val)))
            train_idx = torch.from_numpy(train)
            valid_idx = torch.from_numpy(val)
            test_idx = torch.from_numpy(test)
            return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    # 更新self.graph_log 添加dgl_graph数据信息
    # {'graph': {save_idx:{graph_name: , dims: , node_feat:, edge_feat:}}
    # save_idx 与 goedata 中的索引 geo_idx 对应，也与 graph中的索引对应
    def update_graph_log(self, model_name=None, save_idx=0, **kwargs):
        if model_name is None:
            model_name = ""
        graph_name = model_name + '_' + str(int(time.time()))
        if 'graph' not in self.graph_log.keys():
            self.graph_log['graph'] = {}
        if save_idx not in self.graph_log['graph']:
            self.graph_log['graph'][save_idx] = {}
        self.graph_log['graph'][save_idx]['graph_name'] = graph_name
        for key in kwargs.keys():
            self.graph_log['graph'][save_idx][key] = kwargs[key]
        # 'edge_num' 'node_num'

    # 存储和加载函数
    def save_geograph(self):
        out_put = open(self.processed_geodata_path, 'wb')
        for g_id in np.arange(len(self.geograph)):
            self.geograph[g_id].save(dir_path=self.processed_dir)
        out_str = pickle.dumps(self.geograph)
        out_put.write(out_str)
        out_put.close()

    def load_geograph(self):
        with open(self.processed_geodata_path, 'rb') as file:
            self.geograph = pickle.loads(file.read())
            for g_id in np.arange(len(self.geograph)):
                self.geograph[g_id].load()
            self.sample_data = []
            for geo_idx in np.arange(len(self.geograph)):
                self.sample_data.append(self.geograph[geo_idx].sample_data)
            return self.geograph

    def save_graph_log(self):
        out_put = open(self.graph_log_data_path, 'wb')
        out_str = pickle.dumps(self.graph_log)
        out_put.write(out_str)
        out_put.close()

    def load_graph_log(self):
        with open(self.graph_log_data_path, 'rb') as file:
            out_put = pickle.loads(file.read())
            return out_put

    def __getitem__(self, idx):
        if idx < len(self.graph):
            return self.graph[idx]

    def __len__(self):
        if self.graph is None:
            self.graph = []
        return len(self.graph)

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))


# GeoGridMLClassifier
# grid: optional  Grid类型，建模区域网格
# method: str, 'svm', 'rf', 'xgboost'


# if self.grid is not None:
#     data_sampler = GeoGridDataSampler()
#     data_sampler.set_base_grid_by_boreholes(boreholes=known_data, external_grid=self.grid)
#     sample_points_data = data_sampler.get_sample_data(idx=0)
#     points_data.append(sample_points_data)
class GeoDataMLClassifier(object):
    def __init__(self, method: str = None, is_grid_search=False, **kwargs):
        self.params = None
        self.data_list = []  # 已知数据  支持分批加入
        self.known_data = None
        self.unknown_data = None  # 未知数据

        # K折交叉验证，如果 k_fold > 1， 则使用
        self.k_fold = 1
        # 参数网格搜索
        self.grid_search = is_grid_search
        self.method = method
        if self.method is not None:
            self.method = self.method.lower()
        else:
            self.method = 'svm'
        self.support_methods = ['svm', 'rf', 'xgboost']
        self.estimator = None
        self.classes = None
        self.classes_num = 0

        self.train_ratio = 0.5  # 训练集比例
        self.valid_ratio = None
        self.test_ratio = None
        self.kwargs = kwargs
        self.best_model = None

    # 添加已知数据
    def append_data(self, data: PointSet):
        if isinstance(data, PointSet):
            self.data_list.append(data)
        else:
            raise ValueError('Input data is not supported.')

    # index=None 则数据全部获取，若index=0则只获取第一份数据
    # predict_label 需要预测的数据标签，默认为-1
    def extract_known_data_from_data_list(self, index=None, predict_label=-1):
        known_points_data = PointSet()
        unknown_points_data = PointSet()
        for k_it, data in enumerate(self.data_list):
            if index is not None and 0 <= index < len(self.data_list):
                if k_it != index:
                    continue
            if isinstance(data, PointSet):
                labels = data.get_labels()
                if predict_label in labels:
                    train_idx = np.argwhere(labels != -1).flatten()
                    known_data = data.get_points_data_by_ids(ids=train_idx)
                    known_points_data.append(known_data)
                    pred_idx = np.argwhere(labels == -1).flatten()
                    unknown_data = data.get_points_data_by_ids(ids=pred_idx)
                    unknown_points_data.append(unknown_data)
        self.known_data = known_points_data
        return self.known_data, self.unknown_data

    def execute_train(self, index=None):
        known_points_data, unknown_points_data = self.extract_known_data_from_data_list(index=index)
        # 已知数据
        train_x = known_points_data.points
        train_y = known_points_data.labels
        if self.method is not None and self.method in self.support_methods:
            method_to_index = {label: index for index, label in enumerate(self.support_methods)}
            if method_to_index[self.method] == 0:
                self.estimator = svm.SVC()
                self.params = [{'kernel': ['rbf'], 'C': [1, 10, 100, 200, 500]}]
            elif method_to_index[self.method] == 1:
                self.estimator = RandomForestClassifier()
                self.params = [{'n_estimators': [50, 120, 160, 200, 250, 280, 300, 350, 400]
                                   , 'max_depth': [2, 4, 6, 8, 10, 12, 14]}]
            elif method_to_index[self.method] == 2:
                self.estimator = XGBClassifier()
                self.params = [{'n_estimators': [50, 120, 160, 200, 250], 'max_depth': [2, 4, 6, 8, 10]
                                   , 'learning_rate': [0.001, 0.01, 0.03]}]
            else:
                raise ValueError('The method type is not supported.')
        if self.grid_search:
            self.estimator = GridSearchCV(estimator=self.estimator, param_grid=self.params)
        print('Classifier computing ...')
        self.estimator.fit(train_x, train_y)
        clf_best = self.estimator.best_estimator_
        self.best_model = clf_best

    def predict(self, grid):
        if isinstance(grid, (Grid, Section, PointSet)):
            # 获得预测出的模型类别值集合，可用于可视化
            new_grid = copy.deepcopy(grid)
            predict_points = new_grid.points
            if self.best_model is not None:
                predict_test_y = self.best_model.predict(predict_points)
                new_grid.labels = predict_test_y
                if new_grid.vtk_data is not None:
                    new_grid.vtk_data['label'] = predict_test_y
                return new_grid
        return None

    # 添加一些分类参数
    def set_estimator_param(self, **kwargs):
        self.kwargs.update(kwargs)

    # 设置训练集，验证集和测试集的比例
    def set_train_test_valid_ratio(self, train_ratio: float = 0.8, valid_ratio: float = None, set_test=False):
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = None
        # 设置测试集
        if not set_test:
            if valid_ratio is None:
                self.valid_ratio = 1 - train_ratio
        else:
            if valid_ratio is None or self.train_ratio + self.valid_ratio > 1:
                raise ValueError('Need to set data_ratio again.')
            self.test_ratio = 1 - self.train_ratio - self.valid_ratio

    def set_standard_scaler(self):
        pass

    # 设置K折交叉验证
    def set_k_fold_valid(self, k_ford_num):
        self.k_fold = k_ford_num


class GeoGridInterpolator(object):
    def __init__(self, grid, method):
        self.grid = grid
        self.grid_points = None
        self.bounds = None
        # 这里的已知数据与未知数据的空间范围是一致的，在预测未知数据时会用到所有的已知数据
        self.known_data_list = []  # 已知数据
        self.unkown_data_list = None  # 未知数据
        if grid is not None:
            self.grid_points = grid.grid_points
            self.bounds = grid.bounds
        self.k_fold = 1
        self.method = method
        if self.method is not None:
            self.method = self.method.lower()
        self.support_methods = ['rbf', 'idw']

# 支持向量机

#     graph_num = len(self.graph) + len(self.predict_graph)
#     if model_idx < graph_num:
#         if param is None and method is 'svm':
#             param = [{'kernel': ['rbf'], 'C': [1, 10, 100, 200, 500]}]  #
#         if param is None and method is 'rf':
#             param = [{'n_estimators': [50, 120, 160, 200, 250, 280, 300, 350, 400],
#                       'max_depth': [2, 4, 6, 8, 10, 12, 14]}]
#         if param is None and method is 'xgboost':
#             param = [{'n_estimators': [50, 120, 160, 200, 250], 'max_depth': [2, 4, 6, 8, 10],
#                       'learning_rate': [0.001, 0.01, 0.03]}]
#         geodata = self.geodata[model_idx]
#         label = np.int64(geodata.grid_point_label)
#         label = np.squeeze(label)
#         prediction = copy.deepcopy(label)
#
#         split = self.get_split_idx(model_idx)
#         train_idx = split['train']
#
#         train_x = geodata.grid_points[train_idx]
#         train_y = label[train_idx]
#         test_idx = list(set(np.arange(len(geodata.grid_points))) - set(geodata.train_idx))
#
#         # test_x = geodata.grid_points[test_idx]
#         test_x = geodata.grid_points
#         # 测试集真实标签
#         # test_y = label[test_idx]
#         test_y = label
#         clf = None
#         if method == 'svm':
#             clf = GridSearchCV(estimator=svm.SVC(), param_grid=param, cv=cv)
#         elif method == 'rf':
#             clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param)
#         elif method == 'xgboost':
#             clf = GridSearchCV(estimator=XGBClassifier(), param_grid=param)
#         print('Classifier computing ...')
#         clf.fit(train_x, train_y)  # 输出测试集的预测结果
#         clf_best = clf.best_estimator_
#         predict_test_y = clf_best.predict(test_x)
#         # 获得预测出的模型类别值集合，可用于可视化
#         # prediction[test_idx] = predict_test_y
#         prediction = predict_test_y
#         print(classification_report(test_y, predict_test_y))
#         # accuracy = MF.accuracy(torch.tensor(predict_test_y[test_idx]), torch.tensor(test_y[test_idx]))
#         # print('================Test Accuracy {:.4f}================'.format(accuracy.item()))
#
#         gen_mesh = mvk.generate_model_on_base_grid(geodata, prediction, save_path=save_path)
#         mvk.visual_multiple_model(geodata.sample_grid, gen_mesh, camera=[1, 0, 0])
#     else:
#         raise ValueError

# method = {'rbf', 'nearest', 'idw'}
# def predict_with_interpolate(self, model_idx, method='rbf', iso_list=None, stratum_match=None,
#                              save_path=None, **kwargs):
#     graph_num = len(self.graph) + len(self.predict_graph)
#     if model_idx < graph_num:
#         geodata = self.geodata[model_idx]
#         label = np.int64(geodata.grid_point_label)
#         label = np.squeeze(label)
#         prediction = np.float32(copy.deepcopy(label))
#         # train_idx = geodata.extract_drills_layer_points()
#         train_idx = geodata.train_idx
#         train_x = geodata.grid_points[train_idx]
#         train_y = label[train_idx]
#         test_idx = list(set(np.arange(len(geodata.grid_points))) - set(train_idx))
#         test_x = geodata.grid_points[test_idx]
#         test_y = label[test_idx]
#         predict_test_y = None
#         if method == 'rbf':
#             neighbors = None
#             smoothing = 0
#             degree = None
#             if 'neighbors' in kwargs.keys():
#                 neighbors = kwargs['neighbors']
#                 smoothing = kwargs['neighbors']
#                 degree = kwargs['degree']
#                 # , neighbors=neighbors, smoothing=smoothing, degree=degree
#             yflat = RBFInterpolator(train_x, train_y)(test_x)
#             predict_test_y = torch.tensor(np.array(yflat))
#         elif method == 'idw':
#             # 暂时还不支持三维空间散点的IDW插值，将数据以文件形式到处，使用voxler软件处理
#             return train_x, train_y, test_x, test_y, geodata.bound
#         elif method == 'nearest':
#             interp = NearestNDInterpolator(train_x, train_y)
#             predict_value = interp(test_x)
#             predict_test_y = torch.tensor(np.array(predict_value))
#         if predict_test_y is None:
#             raise ValueError
#         prediction[test_idx] = predict_test_y
#         sample_grid, contour, cell_mesh, cell_stratum = mvk.process_point_field_model(geodata.sample_grid,
#                                                                                       cell_labels=prediction,
#                                                                                       iso_list=iso_list,
#                                                                                       stratum_match=stratum_match,
#                                                                                       save_path=save_path)
#         predict_labels = np.int64(cell_stratum)
#         predict_labels_test = torch.tensor(predict_labels[test_idx])
#         accuracy = MF.accuracy(predict_labels_test, torch.tensor(test_y))
#         print('================Test Accuracy {:.4f}================'.format(accuracy.item()))
#
#         tmp_drill_param = geodata.train_plot_data[0]
#         drill_pos, drill_num = tmp_drill_param
#         drills, _, _, _ = geodata.sample_with_drills(drill_pos=drill_pos)
#         drills = mvk.drill_construct_tube(drills, drill_radius=25)
#         mvk.visual_multiple_model([drills, contour], geodata.sample_grid, cell_mesh, sample_grid)
#
#         ori_path = os.path.join(self.root, 'processed', 'ori.vtk')
#         geodata.sample_grid.save(filename=ori_path)
#
#         slice_x = mvk.clip_section_along_axis(cell_mesh, sample_axis='x')
#         slice_y = mvk.clip_section_along_axis(cell_mesh, sample_axis='y')
#         mvk.visual_multiple_model([drills, slice_x, slice_y])
#
#     else:
#         raise ValueError

# 生成二维剖面网格，
# def generate_section_grid(self, model_idx, sample_axis='x', scroll_scale=0.5, drill_num=6, is_save=False):
#     graph_num = len(self.graph) + len(self.predict_graph)
#     if model_idx < graph_num:
#         geodata = self.geodata[model_idx]
#         section, section_point, section_point_label, drills, drill_pid, train_plot_data_type, train_plot_data = \
#             geodata.generate_section2d_with_drills_test(sample_axis=sample_axis, scroll_scale=scroll_scale,
#                                                         drill_num=drill_num)
#         for sec in section:
#             axis_labels = ['x', 'y', 'z']
#             label_to_index = {label: index for index, label in enumerate(axis_labels)}
#             ax_index = label_to_index[sample_axis.lower()]
#             section_geodata = GeoMeshParse(sec, name=geodata.name + '_sec2d')
#             section_geodata.sample_grid_extent = copy.deepcopy(geodata.sample_grid_extent)
#             section_geodata.sample_grid_extent[ax_index] = 1
#
#             section_geodata.sample_grid = sec
#             section_geodata.grid_points = section_point
#             section_geodata.grid_point_label = section_point_label
#             section_geodata.train_idx = drill_pid
#             # 获取二维剖面三角网格
#             section_geodata.get_triangulate_edges_2d(axis_label=sample_axis)
#             # 获取节点特征
#             section_geodata.get_node_feat(node_feat='position')
#             # 将采样数据参数记录下来
#             section_geodata.train_plot_data_type = []
#             section_geodata.train_plot_data = []
#             section_geodata.train_plot_data_type.append(train_plot_data_type)
#             section_geodata.train_plot_data.append(train_plot_data)
#             # 构建dgl图结构网格
#             sectiona_graph = section_geodata.create_dgl_graph(
#                 edge_list=np.int64(section_geodata.edge_list).transpose(),
#                 node_feat=section_geodata.node_feat,
#                 edge_feat=section_geodata.edge_feat,
#                 node_label=np.int64(section_geodata.grid_point_label),
#                 self_loop=False, add_inverse_edge=True, normalize=True)
#             # 将数据存入对象
#             self.geodata.append(section_geodata)
#             self.predict_graph.append(sectiona_graph)
#             save_idx = len(self.predict_graph)
#             self.update_graph_log(model_name=section_geodata.name,
#                                   save_idx=save_idx, extern=section_geodata.sample_grid_extent,
#                                   node_feat=self.dgl_graph_param[0], edge_feat=self.dgl_graph_param[1],
#                                   is_pre_train=False)
#             # 处理好图
#             if torch.is_tensor(self.predict_num_classes['labels']):
#                 self.predict_num_classes['labels'] = self.predict_num_classes['labels'].numpy().tolist()
#             if model_idx < len(self.graph):
#                 if torch.is_tensor(self.num_classes['labels']):
#                     self.num_classes['labels'] = self.num_classes['labels'].numpy().tolist()
#                 label_num = self.num_classes['labels'][model_idx]
#             else:
#                 label_num = self.predict_num_classes['labels'][model_idx - len(self.graph)]
#             self.predict_num_classes['labels'].append(label_num)
#             self.predict_num_classes = {'labels': torch.tensor(self.predict_num_classes['labels']).to(torch.long)}
#             if is_save:
#                 print('Saving...')
#                 save_graphs(self.processed_predict_graph_path, self.predict_graph, self.predict_num_classes)
#                 self.predict_graph, self.predict_num_classes = load_graphs(self.processed_predict_graph_path)
#                 self.save_graph_log()
#                 self.save_geodata()
#                 self.geodata = self.load_geodata()
#                 self.graph_log = self.load_graph_log()

# 修改训练集比例
# def change_train_idx_pro(self, model_index, sample_operator, replace=False, **kwargs):
#     graph_num = len(self.graph)
#     if model_index < graph_num:
#         x = np.arange(len(self.geodata[model_index].grid_points))
#         known_proportion = self.geodata[model_index].train_data_proportion
#         print('Changing Graph Data train_data proportion ...')
#         print('The previous train data proportion is {}.'.format(known_proportion))
#         self.geodata[model_index].set_virtual_geo_sample(sample_operator=sample_operator, **kwargs)
#         # 替换并存储
#         if replace:
#             self.save_geodata()
#             self.geodata = self.load_geodata()
