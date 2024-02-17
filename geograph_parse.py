from sklearn import preprocessing
import numpy as np
import pyvista as pv
import os
import pickle
import scipy.spatial as spt

import networkx as nx
import random
import dgl
import torch
import copy
import dgl.backend as F
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from data_structure.grids import Grid
from data_structure.boreholes import Borehole, BoreholeSet
from data_structure.points import PointSet
from data_structure.sections import Section, SectionSet
from data_structure.geodata import load_object, GeodataSet
from vtkmodules.all import vtkProbeFilter
from data_structure.data_sampler import GeoGridDataSampler
import time
import pytetgen
import tetgen


class GeoMeshGraphParse(object):
    def __init__(self, mesh: Grid = None, input_sample_data=None, grid_dims=None, name=None
                 , is_normalize=False, dir_path=None):  # , pre_train=True
        self.name = name
        self.is_normalize = is_normalize  # 坐标是否归一化
        self.data = mesh
        if mesh is not None:
            self.center = mesh.center  # 输入mesh的中心点坐标
            # ori 即输入的原始地质格网数据, 原始模型数据均不做更改，坐标、标签变换在 sample数据中进行
            self.grid_points = mesh.grid_points
            self.grid_points_series = mesh.grid_points_series
            self.classes_num = mesh.classes_num  # np.array
        # 图参数
        self.is_create_graph = False  # 为False是外部传入图数据，内部无需构建，为True则是内部构建图
        self.edge_list = None  # list 边集（采样数据，包括训练数据与测试数据）
        # 训练数据样本构建，随机散点、钻孔、剖面，作为带标签数据输入模型进行训练
        self.train_data_indexes = None  # list 训练数据的idx索引, 对应self.grid_points
        self.train_data_proportion = 0  # 已知标签数据比例
        # 可以外部输入，或从网格中进行采样
        self.input_sample_data = input_sample_data  # 采样数据(类型为散点、钻孔、剖面)
        self.grid_dims = grid_dims
        self.sample_data = []
        # 图特征  下面两个变量用来装数据
        self.node_feat = None  # 图节点特征    np.float32
        self.edge_feat = None  # 图边特征      np.float32

        self.dir_path = dir_path
        self.tmp_dump_str = 'tmp' + str(int(time.time()))

    def execute(self, sample_operator=None, edge_feat=None, node_feat=None, feat_normalize=False,
                is_create_graph=True, **kwargs):
        self.is_create_graph = is_create_graph
        if node_feat is None:
            node_feat = ['position']
        if edge_feat is None:
            edge_feat = []
        # 对标签标准化处理
        # self.map_grid_vertex_labels()
        # 选择测试模型的样本形式
        if self.data is not None:
            self.set_virtual_geo_sample(grid=self.data, sample_operator=sample_operator, **kwargs)
        elif self.input_sample_data is not None and len(self.input_sample_data) > 0:
            self.set_geo_sample_data(input_sample_data=self.input_sample_data, grid_dims=self.grid_dims)
        # 生成三角网剖分
        self.get_triangulate_edges()
        # 生成边权重，以距离作为边权
        for node_feat_type in node_feat:
            self.get_node_feat(node_feat=node_feat_type)
        for edge_feat_type in edge_feat:
            self.get_edge_weight_feat(edge_feat=edge_feat_type, normalize=feat_normalize)

        return self.create_dgl_graph(edge_list=np.int64(self.edge_list).transpose(), node_feat=self.node_feat,
                                     edge_feat=self.edge_feat, node_label=np.int64(self.grid_points_series),
                                     self_loop=False, add_inverse_edge=True, normalize=feat_normalize)

    # 设置虚拟地质采样切分以获取训练集
    def set_virtual_geo_sample(self, grid: Grid, sample_operator=None, **kwargs):
        if sample_operator is None:
            sample_operator = ['None']
        geo_grid_sampler = GeoGridDataSampler(grid=grid, sample_operator=sample_operator, **kwargs)
        geo_grid_sampler.execute(**kwargs)
        self.sample_data = geo_grid_sampler.sample_data_list
        self.train_data_indexes = geo_grid_sampler.get_sample_points_indexex_for_grid_points()
        if self.train_data_indexes is not None and len(self.train_data_indexes) > 0:
            self.train_data_proportion = len(self.train_data_indexes) / len(self.grid_points)
            print('Set train_data_proportion is {} ...'.format(self.train_data_proportion))

    def set_geo_sample_data(self, input_sample_data: GeodataSet, grid_dims=None, **kwargs):
        # 将钻孔数据映射到空网格上
        geo_borehole_sample = GeoGridDataSampler(**kwargs)
        geo_borehole_sample.set_base_grid_by_geodataset(geodataset=input_sample_data, dims=grid_dims)
        geo_borehole_sample.execute()
        self.sample_data = self.input_sample_data.geodata_list
        self.data = geo_borehole_sample.grid
        self.grid_points = self.data.grid_points
        self.grid_points_series = self.data.grid_points_series
        self.classes_num = self.data.classes_num  # np.array
        self.train_data_indexes = geo_borehole_sample.get_sample_points_indexex_for_grid_points()
        if self.train_data_indexes is not None and len(self.train_data_indexes) > 0:
            self.train_data_proportion = len(self.train_data_indexes) / len(self.grid_points)
            print('Set train_data_proportion is {} ...'.format(self.train_data_proportion))

    # 要保证 edge_list 图中没有自环
    def create_dgl_graph(self, edge_list=None, node_feat=None, edge_feat=None, node_label=None,
                         self_loop=False, add_inverse_edge=False, normalize=False, is_regular_grid=True):

        # 为每一个节点添加一个固定属性-坐标 coord
        # edge_list  node_feat edge_feat node_label add_inverse_edge
        print('Creating Dgl Graph Data')
        num_node = len(node_feat)
        num_edge = edge_list.shape[1]
        # num_edge = len(edge_list)
        # 标签
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
        # 记录每一个图节点的空间坐标
        if normalize is True:
            sample_points = np.float32(self.grid_points_normalize())
        else:
            sample_points = np.float32(self.grid_points)
        graph['position'] = sample_points[0:num_node]
        graph['num_nodes'] = num_node
        g = dgl.graph((graph['edge_index'][0], graph['edge_index'][1]), num_nodes=graph['num_nodes'])
        if graph['edge_feat'] is not None:
            g.edata['feat'] = torch.from_numpy(graph['edge_feat'])
        if graph['node_feat'] is not None:
            g.ndata['feat'] = torch.from_numpy(graph['node_feat'])
        if graph['position'] is not None:
            g.ndata['position'] = torch.from_numpy(graph['position'])
        if node_label is not None:
            g.ndata['label'] = F.reshape(node_label, (g.num_nodes(),))
        return g

    def is_connected_graph(self):
        if self.grid_points is None:
            raise ValueError('Graph Data is empty.')
        elif self.edge_list is None:
            print('Graph Data is empty.')
        else:
            graph = nx.Graph(self.edge_list)
            is_connected = nx.is_connected(graph)
            return is_connected

    # 如果normalize为False， 临时输出归一化坐标，但是对self.grid_matrix_point不做更新
    def grid_points_normalize(self, normalize=False):
        if self.grid_points is None:
            raise ValueError
        # 判断如果已经对网格坐标点进行了normalize操作，则直接返回网格坐标点
        if self.is_normalize is True:
            return self.grid_points
        minmax_pnt = preprocessing.MinMaxScaler().fit_transform(self.grid_points)
        if normalize:
            self.grid_points = minmax_pnt
            self.is_normalize = normalize
        return minmax_pnt

    # 构建图结构边集合，采用delaunay三角剖分
    def get_triangulate_edges(self, tetgen_mode=True):
        print('Building Delaunay Tetgen of {}'.format(self.name))
        edge_list = []
        if self.grid_points is not None:
            vertex = self.grid_points
        else:
            raise ValueError
        if tetgen_mode:
            tri = pytetgen.Delaunay(vertex)
            tet_list = tri.simplices
        else:
            tri = spt.Delaunay(vertex)
            tet_list = tri.simplices
        pbar = tqdm(enumerate(tet_list), total=len(tet_list))
        # 将四面体处理成三角网的边集
        for it, tet in pbar:
            for n_i in np.arange(len(tet)):
                for n_j in np.arange(n_i + 1, len(tet)):
                    edge = [tet[n_i], tet[n_j]]
                    edge_list.append(edge)
        # 去重，除了重复，[0,1] [1,0]只保留[0,1]
        new_edge_list = list(set(tuple(sorted(sub)) for sub in edge_list))
        # mvk.visual_edge_list(edge_list=new_edge_list, edge_points=self.grid_points)
        self.edge_list = new_edge_list
        return new_edge_list

    # 构建的是二维规则网格，网格是根据现有三维模型网格确定的，格网点坐标不变，是从三维中剖切出来的
    def get_triangulate_edges_2d(self, axis_label='x', tetgen_mode=True):
        axis_labels = ['x', 'y', 'z']
        label_to_index = {label: index for index, label in enumerate(axis_labels)}
        ax_index = label_to_index[axis_label]
        grid_point = self.grid_points
        # 投影面，目前只能投影到坐标轴平面上，如 'x', 'y'
        pro_ind = [ind for ind in np.arange(3) if ind != ax_index]
        # 转二维坐标，然后构建三角网
        pro_point_2d = grid_point[:, pro_ind]
        print('Building delaunay tetgen of {}'.format(self.name))
        if tetgen_mode:
            tri = pytetgen.Delaunay(pro_point_2d)
            tet_list = tri.simplices

        else:
            tri = spt.Delaunay(pro_point_2d)
            tet_list = tri.simplices
        edge_list = []
        pbar = tqdm(enumerate(tet_list), total=len(tet_list))
        # 将四面体处理成三角网的边集
        for it, tet in pbar:
            for n_i in np.arange(len(tet)):
                for n_j in np.arange(n_i + 1, len(tet)):
                    edge = [tet[n_i], tet[n_j]]
                    edge_list.append(edge)
        # 去重，除了重复，[0,1] [1,0]只保留[0,1]
        new_edge_list = list(set(tuple(sorted(sub)) for sub in edge_list))
        self.edge_list = new_edge_list
        return new_edge_list

    # 分配边权重
    def get_edge_weight_feat(self, edge_feat, normalize=False, is_replace=True):
        if self.edge_list is None:
            raise ValueError('call get_triangulate_edges function')
        if edge_feat is None:
            return None
        edge_weight_dist = []
        # 归一化坐标
        if normalize is True:
            sample_points = self.grid_points_normalize()
        else:
            sample_points = self.grid_points
        for item in self.edge_list:
            if edge_feat == 'euclidean':
                coord_i = sample_points[item[0]]
                coord_j = sample_points[item[1]]
                dist = np.sqrt(np.sum(np.square(coord_i - coord_j)))
                edge_weight_dist.append(dist)  # 边权重
        if is_replace:
            if self.edge_feat is not None:
                self.edge_feat = np.vstack((self.edge_feat, np.float32(edge_weight_dist)))
            else:
                self.edge_feat = np.float32(edge_weight_dist)
        return np.float32(edge_weight_dist)

    # 获取节点特征
    def get_node_feat(self, node_feat='position', is_regular_grid=True,
                      is_replace=True):  # , has_train=True , default_value=0, rand_mask_pro=0.5
        node_feat_data = None
        if node_feat == 'position':
            node_feat_data = self.grid_points_normalize()
            node_feat_data = np.float32(node_feat_data)
        if is_replace:
            if self.node_feat is not None:
                self.node_feat = np.hstack((self.node_feat, node_feat_data))
            else:
                self.node_feat = node_feat_data
        return node_feat_data

    def save(self, dir_path):
        # save self.sample_data
        # save self.data
        if self.data is not None and isinstance(self.data, Grid):
            self.data = self.data.save_object(dir_path=dir_path)
        if self.sample_data is not None:
            for s_i in np.arange(len(self.sample_data)):
                if isinstance(self.sample_data[s_i], (Grid, BoreholeSet, SectionSet, Section, PointSet)):
                    self.sample_data[s_i] = self.sample_data[s_i].save_object(dir_path=dir_path)
                else:
                    raise ValueError("The data type is not supported.")

    def load(self, dir_path=None):
        self.dir_path = dir_path
        if self.data is not None:
            self.data = load_object(gtype=self.data[0], file_path=self.data[1])
        if self.sample_data is not None:
            for s_i in np.arange(len(self.sample_data)):
                self.sample_data[s_i] = load_object(gtype=self.sample_data[s_i][0], file_path=self.sample_data[s_i][1])

    # 钻孔数据增强
    def drill_data_augmentation(self):
        pass

    # # 待转移
    # # match_type: None 最邻近搜索， svm 支持向量机
    # def match_unregular_grid_to_regular_grid(self, cell_density=1, predict_point_label=None, match_type='rf'):
    #     if self.edge_list is not None and self.unregular_grid_points is not None:
    #         # 计算不规则格网节点的凸包
    #         hull_surface, grid_outline = self.get_unregular_grid_points_convexhull_surface()
    #         sample_grid = pv.voxelize(hull_surface, density=cell_density)
    #         # 不规则网格没有尺寸
    #         # self.sample_grid_extent = None
    #         if predict_point_label is not None:
    #             grid_points = sample_grid.cell_centers().points
    #             if match_type is 'nearest':
    #                 ckt = spt.cKDTree(self.unregular_grid_points)
    #                 d, pid = ckt.query(grid_points)
    #                 # grid_point_label = np.array(self.unregular_grid_point_label)[pid]
    #                 grid_point_idx = pid
    #                 scalar = predict_point_label[grid_point_idx]
    #                 sample_grid.cell_data['stratum'] = scalar
    #             if match_type is 'rf' and predict_point_label is not None:
    #                 train_x = self.unregular_grid_points
    #                 train_y = np.int64(predict_point_label)
    #                 test_x = grid_points
    #                 # 测试集真实标签
    #                 clf = RandomForestClassifier(n_estimators=200, max_depth=8)
    #                 clf.fit(train_x, train_y)
    #                 # clf_best = clf.best_estimator_
    #                 # 输出测试集的预测结果
    #                 predict_test_y = clf.predict(test_x)
    #                 # 获得预测出的模型类别值集合，可用于可视化
    #                 sample_grid.cell_data['stratum'] = predict_test_y
    #         return sample_grid, grid_outline
    #
