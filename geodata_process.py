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
from vtkmodules.all import vtkProbeFilter
from data_structure.data_sampler import GeoGridDataSampler


class GeoMeshGraphParse(object):
    def __init__(self, mesh: Grid = None, name=None, is_regular_grid=True):  # , pre_train=True
        self.data = mesh
        self.name = name
        self.is_normalize = False  # 坐标是否归一化
        #
        self.center = mesh.center  # 输入mesh的中心点坐标
        # ori 即输入的原始地质格网数据, 原始模型数据均不做更改，坐标、标签变换在 sample数据中进行
        self.grid_points = mesh.grid_points
        self.grid_scalar = mesh.scalar_series
        # self.ori_label, self.ori_label_num = self.convert_noddy_scalar_to_ori_label()  # 获取 ori_label
        # self.is_regular_grid = True

        self.grid_point_label = None  # np.array
        self.grid_point_label_num = None  # int

        #
        self.is_create_graph = False  # 为False是外部传入图数据，内部无需构建，为True则是内部构建图
        self.edge_list = None  # list 边集（采样数据，包括训练数据与测试数据）

        # 训练数据样本构建，随机散点、钻孔、剖面，作为带标签数据输入模型进行训练
        self.train_data_indexes = None  # list 训练数据的idx索引, 对应self.grid_points
        self.train_data_proportion = 0  # 已知标签数据比例
        self.sample_data = None  # 采样数据(类型为散点、钻孔、剖面)

        # 图特征  下面两个变量用来装数据
        self.node_feat = None  # 图节点特征    np.float32
        self.edge_feat = None  # 图边特征      np.float32

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
        self.set_virtual_geo_sample(grid=self.data, sample_operator=sample_operator, **kwargs)
        # 生成三角网剖分
        self.get_triangulate_edges()
        # 生成边权重，以距离作为边权
        for node_feat_type in node_feat:
            self.get_node_feat(node_feat=node_feat_type)
        for edge_feat_type in edge_feat:
            self.get_edge_weight_feat(edge_feat=edge_feat_type, normalize=feat_normalize)

        return self.create_dgl_graph(edge_list=np.int64(self.edge_list).transpose(), node_feat=self.node_feat,
                                     edge_feat=self.edge_feat, node_label=np.int64(self.grid_point_label),
                                     self_loop=False, add_inverse_edge=True, normalize=feat_normalize)

    # 设置虚拟地质采样来切分train_idx   is_update则更新，覆盖之前的采样，否则是追加，保留之前的采样
    def set_virtual_geo_sample(self, grid: Grid, sample_operator=None, **kwargs):
        geo_grid_sampler = GeoGridDataSampler(grid=grid, sample_operator=sample_operator, **kwargs)
        self.sample_data = geo_grid_sampler
        self.train_data_indexes = geo_grid_sampler.get_sample_points_indexex_for_grid_points()
        if self.train_idx is not None and len(self.train_idx) > 0:
            self.train_data_proportion = len(self.train_data_indexes) / len(self.grid_points)
            print('Set train_data_proportion is {} ...'.format(self.train_data_proportion))

    # 要保证 edge_list 图中没有自环
    def create_dgl_graph(self, edge_list=None, node_feat=None, edge_feat=None, node_label=None,
                         self_loop=False, add_inverse_edge=False, normalize=False, is_regular_grid=True):

        # 为每一个节点添加一个固定属性-坐标 coord
        # edge_list  node_feat edge_feat node_label add_inverse_edge
        print('Creating Dgl Graph Data')
        num_node = len(node_feat)
        # num_edge = edge_list.shape[1]
        num_edge = len(edge_list)
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
            sample_points = np.float32(self.grid_points_normalize(is_regular_grid=is_regular_grid))
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
        if self.grid_points is None and self.unregular_grid_points is None:
            raise ValueError('Graph Data is empty.')
        elif self.edge_list is None:
            print('Graph Data is empty.')
        else:
            graph = nx.Graph(self.edge_list)
            is_connected = nx.is_connected(graph)
            return is_connected

    # 如果normalize为False， 临时输出归一化坐标，但是对self.grid_matrix_point不做更新
    def grid_points_normalize(self, normalize=False, is_regular_grid=True):
        if is_regular_grid:
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
        else:
            if self.unregular_grid_points is None:
                raise ValueError
            if self.is_normalize is True:
                return self.unregular_grid_points
            minmax_pnt = preprocessing.MinMaxScaler().fit_transform(self.unregular_grid_points)
            if normalize:
                self.unregular_grid_points = minmax_pnt
                self.is_normalize = normalize
            return minmax_pnt

    # 将scalar转换为label, 并将其处理为从0开始的连续自然数
    def convert_noddy_scalar_to_ori_label(self):
        if self.ori_scalar is None:
            raise ValueError
        label = []
        for item in self.ori_scalar:
            label.append(int(item))
        unique_label = np.unique(label)
        sorted_label = sorted(unique_label)
        label_dict = {}
        for idx, item in enumerate(sorted_label):
            label_dict[item] = idx
        for idx, item in enumerate(label):
            label[idx] = [label_dict[item]]
        # label = np.array(label)
        label_num = len(unique_label)
        return label, label_num

    # 获取采样节点对应标签，标签形如[0,1,2,3,……]自然数列表，从0开始，依次递增
    # 处理 待转移
    def map_grid_vertex_labels(self, fill_well_data=False):
        if self.ori_label is None:
            raise ValueError
        if self.is_noddy is True:
            if self.grid_point_idx is not None:
                label = np.array(self.ori_label)[self.grid_point_idx]
                self.grid_point_label = label
        elif self.is_noddy is False:
            # 有规则网格和非规则网格
            # grid_point_label = []
            grid_point_label = np.full(len(self.grid_points), -1)
            # 规则格网
            grid_points_num = 0
            if self.grid_points is not None and self.grid_points.size != 0:
                # 遍历所有网格节点，如果网格节点不属于采样数据，则赋予标签-1，否则赋予相应采样标签
                for lid in np.arange(len(self.grid_points)):
                    if lid in self.grid_cell_idx:
                        # 获取相应采样点的标签
                        o_id = self.grid_cell_idx.index(lid)
                        o_label = self.ori_label[o_id]
                        grid_point_label[lid] = o_label
                self.train_idx = list(set(sorted(self.grid_cell_idx)))
                self.grid_point_label = np.array(grid_point_label)
                grid_points_num = len(self.grid_points)
            if self.unregular_grid_points is not None and self.unregular_grid_points.size != 0:
                for lid in np.arange(len(self.unregular_grid_points)):
                    if lid not in self.unregular_grid_idx:
                        grid_point_label.append([-1])
                    else:
                        o_id = self.unregular_grid_idx.index(lid)
                        o_label = self.ori_label[o_id]
                        grid_point_label.append([o_label])
                self.train_idx = list(set(sorted(self.unregular_grid_idx)))
                self.unregular_grid_point_label = np.array(grid_point_label)
                grid_points_num = len(self.unregular_grid_points)
            if fill_well_data:
                # 是填充钻孔，对于只有地层分界点的钻孔给数据，可以进行钻孔格网纵向加密
                # 有采样钻孔数据， 有规则格网
                if self.train_plot_data_type is not None:
                    # self.fill_dat_well_grid_point()
                    pass
            # 输出已知数据比例
            if self.train_idx is not None and len(self.train_idx) > 0 and grid_points_num > 0:
                self.known_pro = len(self.train_idx) / grid_points_num
                print('Set train_pro is {} ...'.format(self.known_pro))
        if self.grid_point_label is not None:
            unique_label = np.unique(self.grid_point_label)
            label_num = len(unique_label)
            if -1 in unique_label:
                label_num -= 1
            self.grid_point_label_num = label_num
            return self.grid_point_label, self.grid_point_label_num
        elif self.unregular_grid_point_label is not None:
            unique_label = np.unique(self.unregular_grid_point_label)
            label_num = len(unique_label)
            if -1 in unique_label:
                label_num -= 1
            self.grid_point_label_num = label_num
            return self.unregular_grid_point_label, self.grid_point_label_num
        else:
            raise ValueError

    # 待迁移
    # 构建图结构边集合，采用delaunay三角剖分
    def get_triangulate_edges(self):
        print('Building Delaunay Tetgen of {}'.format(self.name))
        edge_list = []
        if self.grid_points is not None:
            vertex = self.grid_points
        else:
            raise ValueError
        tri = spt.Delaunay(vertex)
        tet_list = tri.simplices
        pbar = tqdm(enumerate(tet_list), total=len(tet_list))
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

    # 待迁移
    # 构建的是二维规则网格，网格是根据现有三维模型网格确定的，格网点坐标不变，是从三维中剖切出来的
    def get_triangulate_edges_2d(self, axis_label='x'):
        axis_labels = ['x', 'y', 'z']
        label_to_index = {label: index for index, label in enumerate(axis_labels)}
        ax_index = label_to_index[axis_label]
        grid_point = self.grid_points
        # 投影面，目前只能投影到坐标轴平面上，如 'x', 'y'
        pro_ind = [ind for ind in np.arange(3) if ind != ax_index]
        # 转二维坐标，然后构建三角网
        pro_point_2d = grid_point[:, pro_ind]
        print('Building delaunay tetgen of {}'.format(self.name))

        tri = spt.Delaunay(pro_point_2d)
        tet_list = tri.simplices
        edge_list = []
        pbar = tqdm(enumerate(tet_list), total=len(tet_list))
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
    def get_edge_weight_feat(self, edge_feat, normalize=False, is_regular_grid=True, is_replace=True):
        if self.edge_list is None:
            raise ValueError('call get_triangulate_edges function')
        if edge_feat is None:
            return None
        edge_weight_dist = []
        # 归一化坐标
        if normalize is True:
            sample_points = self.grid_points_normalize(is_regular_grid=is_regular_grid)
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
            node_feat_data = self.grid_points_normalize(is_regular_grid=is_regular_grid)
            node_feat_data = np.float32(node_feat_data)
        if is_replace:
            if self.node_feat is not None:
                self.node_feat = np.hstack((self.node_feat, node_feat_data))
            else:
                self.node_feat = node_feat_data
        return node_feat_data




















 # 待迁移
    def extract_drills_interface_points(self, drills_map):
        extract_labels = []
        extract_points = []
        # 遍历钻孔map
        for drill_key in drills_map.keys():
            drill_points_labels = drills_map[drill_key]
            drill_points = drill_points_labels[:, 0:3]
            drill_label = drill_points_labels[:, 3]
            # 从上至下遍历钻孔点
            front_label = drill_label[0]
            extract_labels.append(drill_label[0])
            extract_points.append(drill_points[0])
            # 提取分界点的索引，先把第一个点放进来
            for l_idx in range(0, len(drill_label)):
                next_label = drill_label[l_idx]
                if next_label != front_label:
                    extract_labels.append(drill_label[l_idx])
                    extract_points.append(drill_points[l_idx])
                    front_label = drill_label[l_idx]
        return extract_points, extract_labels

    # 待迁移
    # 获取钻孔的 {drill_key: points—labels}属性字典， sample_data_it=0是第1个采样数据， 支持多重采样
    def get_drill_points_labels_map(self, sample_data_it=0):
        tmp_drill_param = self.train_plot_data[sample_data_it]
        drill_map = {}
        if isinstance(tmp_drill_param, tuple):
            drill_pos, drill_num = tmp_drill_param
            for it, sample_point in enumerate(drill_pos):
                # 采样虚拟钻孔
                pos_a = copy.deepcopy(sample_point)
                pos_a[2] = self.bound[5]  # z_max
                pos_b = copy.deepcopy(sample_point)
                pos_b[2] = self.bound[4]  # z_min
                drill_cell_idx = self.sample_grid.find_cells_along_line(pointa=pos_a, pointb=pos_b)
                if len(drill_cell_idx) == 0:
                    raise ValueError('drill cannot find.')
                drill_cell_label = self.grid_point_label[drill_cell_idx]
                drill_points = self.grid_points[drill_cell_idx]
                drill_map[it] = np.concatenate((drill_points, drill_cell_label), axis=1)
        elif isinstance(tmp_drill_param, dict):
            for drill_key in tmp_drill_param.keys():
                drill_points_labels = tmp_drill_param[drill_key]
                drill_map[drill_key] = drill_points_labels
        else:
            raise ValueError
        return drill_map

    def extract_drills_layer_points(self):
        train_idx = []
        for it, plot_data_type in enumerate(self.train_plot_data_type):
            if plot_data_type == 'drill':
                drill_map = self.get_drill_points_labels_map(sample_data_it=0)
                ckt = spt.cKDTree(self.grid_points)
                for drill_key in drill_map.keys():
                    drill_points_labels = drill_map[drill_key]
                    drill_points = drill_points_labels[:, 0:3]
                    drill_label = drill_points_labels[:, 3]
                    d, pid = ckt.query(drill_points)
                    drill_cell_idx = pid
                    # 从上至下遍历钻孔点
                    front_label = drill_label[0]
                    # 提取分界点的索引，先把第一个点放进来
                    extract_idx = [drill_cell_idx[0]]
                    for l_idx in range(0, len(drill_label)):
                        next_label = drill_label[l_idx]
                        if next_label != front_label:
                            extract_idx.append(drill_cell_idx[l_idx])
                            front_label = drill_label[l_idx]
                    if drill_cell_idx[-1] not in extract_idx:
                        extract_idx.append(drill_cell_idx[-1])
                    train_idx.extend(extract_idx)
        if len(train_idx) == 0:
            raise ValueError
        train_idx = list(set(sorted(train_idx)))
        return train_idx

    # 数据检查
    def check_drills_collide(self):
        min_dist = 0
        train_plot_data_type = self.train_plot_data_type
        for it, plot_data_type in enumerate(train_plot_data_type):
            if plot_data_type == 'drill':
                tmp_drill_param = self.train_plot_data[it]
                if isinstance(tmp_drill_param, tuple):
                    drill_pos, drill_num = tmp_drill_param
                    drill_pos = np.float32(drill_pos)
                    min_dist = self.calculate_mini_dist_points(drill_pos)
                elif isinstance(tmp_drill_param, dict):
                    drill_pos = []
                    for drill_key in tmp_drill_param:
                        drill_points_labels = tmp_drill_param[drill_key]
                        pos = drill_points_labels[0, 0:2]
                        drill_pos.append(pos)
                    drill_pos = np.float32(drill_pos)
                    min_dist = self.calculate_mini_dist_points(drill_pos)
        return min_dist

    # 待删除
    def check_unregular_grid_points_collide(self):
        # 计算每一条边的长
        dist_list = []
        sample_points = self.unregular_grid_points
        new_edge_list = list(set(tuple(sorted(sub)) for sub in self.edge_list))
        for item in new_edge_list:
            coord_i = sample_points[item[0]]
            coord_j = sample_points[item[1]]
            dist = np.sqrt(np.sum(np.square(coord_i - coord_j)))
            if dist == 0:
                continue
            dist_list.append(dist)  # 边权重
        min_dist = min(dist_list)
        max_dist = max(dist_list)
        aver_dist = np.average(dist_list)
        return min_dist, max_dist, aver_dist

    # 待转移 待删除
    def calculate_mini_dist_points(self, points):
        dist_list = []
        pos_num = np.size(points, 0)
        for i in np.arange(0, pos_num - 1):
            tmp_pos = points[i + 1:pos_num]
            pnt = points[i:i + 1]
            dists = spt.distance.cdist(tmp_pos, pnt, metric='euclidean')
            dists = dists.flatten()
            dist_list.extend(list(dists))
        min_dist = min(dist_list)
        return min_dist

    # Read
    # 待转移
    def map_external_csv_data_to_base_grid(self, csv_path, extent=None, names=None):
        raster_df = pd.read_csv(csv_path)
        if names is None:
            names = ['x', 'y', 'z', 'label']
        raster_df.columns = names
        coords = raster_df[['x', 'y', 'z']].values
        labels = np.subtract(raster_df['label'].values, 1)
        output_mesh, mesh_extent = self.create_base_grid(only_out_put=True, extent=extent)
        sample_points = output_mesh.cell_centers().points
        ckt = spt.cKDTree(coords)
        d, pid = ckt.query(sample_points)
        output_scalars = labels[pid]
        output_mesh.cell_data['scalars'] = output_scalars
        return output_mesh

    # 待转移 Read
    # 将外部输入的场模型映射到geodata预设尺寸的规则网格上， 场模型带有顶点属性，渲染类型是point_scalar，cell是没有属性值的
    def map_external_model_to_base_grid(self, external_model):
        output_mesh, mesh_extent = self.create_base_grid(only_out_put=True)
        sample_points = output_mesh.cell_centers().points
        external_scalars = external_model.active_scalars  # points_active_data
        external_points = external_model.points
        ckt = spt.cKDTree(external_points)
        d, pid = ckt.query(sample_points)
        output_scalars = external_scalars[pid]
        output_mesh.cell_data['scalars'] = output_scalars
        return output_mesh

    # 待迁移
    def export_points_data_to_xml_polydata_file(self, points, labels, save_path):
        points_data = pv.PolyData(np.array(points))
        points_data['level'] = np.array(labels)
        points_data.save(filename=save_path)

    # 待转移
    # match_type: None 最邻近搜索， svm 支持向量机
    def match_unregular_grid_to_regular_grid(self, cell_density=1, predict_point_label=None, match_type='rf'):
        if self.edge_list is not None and self.unregular_grid_points is not None:
            # 计算不规则格网节点的凸包
            hull_surface, grid_outline = self.get_unregular_grid_points_convexhull_surface()
            sample_grid = pv.voxelize(hull_surface, density=cell_density)
            # 不规则网格没有尺寸
            # self.sample_grid_extent = None
            if predict_point_label is not None:
                grid_points = sample_grid.cell_centers().points
                if match_type is 'nearest':
                    ckt = spt.cKDTree(self.unregular_grid_points)
                    d, pid = ckt.query(grid_points)
                    # grid_point_label = np.array(self.unregular_grid_point_label)[pid]
                    grid_point_idx = pid
                    scalar = predict_point_label[grid_point_idx]
                    sample_grid.cell_data['stratum'] = scalar
                if match_type is 'rf' and predict_point_label is not None:
                    train_x = self.unregular_grid_points
                    train_y = np.int64(predict_point_label)
                    test_x = grid_points
                    # 测试集真实标签
                    clf = RandomForestClassifier(n_estimators=200, max_depth=8)
                    clf.fit(train_x, train_y)
                    # clf_best = clf.best_estimator_
                    # 输出测试集的预测结果
                    predict_test_y = clf.predict(test_x)
                    # 获得预测出的模型类别值集合，可用于可视化
                    sample_grid.cell_data['stratum'] = predict_test_y
            return sample_grid, grid_outline

    def get_unregular_grid_points_convexhull_surface(self, points_data=None):
        if points_data is not None:
            pass
        elif self.unregular_grid_points is not None:
            points_data = self.unregular_grid_points
        else:
            raise ValueError('Points data is empty.')
        grid_points_2d = points_data[:, 0:2]
        hull = spt.ConvexHull(grid_points_2d)
        simplex_idx = []
        for simplex in hull.simplices:
            simplex_idx.extend(list(simplex))
        unique_idx = list(np.unique(np.int64(simplex_idx)))
        top_surface_points = copy.deepcopy(points_data[unique_idx])
        top_surface_points[:, 2] = self.bound[5]  # z_max
        bottom_surface_points = copy.deepcopy(points_data[unique_idx])
        bottom_surface_points[:, 2] = self.bound[4]  # z_min
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
        # 先对凸包线排序，随机指定一个点作为起始点
        convex_hull_dict = {}
        for simplex in hull.simplices:
            item_0, item_1 = simplex[0], simplex[1]
            if item_0 not in convex_hull_dict.keys():
                convex_hull_dict[item_0] = []
            if item_1 not in convex_hull_dict.keys():
                convex_hull_dict[item_1] = []
            convex_hull_dict[item_0].append(item_1)
            convex_hull_dict[item_1].append(item_0)
        # 随机选一个点作为起点
        line_pnt_idx_front = unique_idx[0]
        line_pnt_idx = [line_pnt_idx_front]
        surf_line_pnt_id = [0]
        for lit in np.arange(points_num):
            strip_0 = convex_hull_dict[line_pnt_idx[lit]]
            if lit == points_num - 1:
                line_pnt_idx.append(line_pnt_idx_front)
                surf_line_pnt_id.append(0)
                break
            if strip_0[0] not in line_pnt_idx:
                line_pnt_idx.append(strip_0[0])
                surf_line_pnt_id.append(unique_idx.index(strip_0[0]))
            else:
                line_pnt_idx.append(strip_0[1])
                surf_line_pnt_id.append(unique_idx.index(strip_0[1]))
        surf_line_pnt_id_0 = copy.deepcopy(surf_line_pnt_id)  #
        surf_line_pnt_id_0 = np.add(surf_line_pnt_id_0, points_num)
        surf_line_pnt_id_total = np.concatenate((surf_line_pnt_id, surf_line_pnt_id_0), axis=0)

        top_line = []
        bottom_line = []
        for lit in np.arange(points_num + 1):
            xy_top = np.array([lit, self.bound[5]])
            xy_bottom = np.array([lit, self.bound[4]])
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

    # 钻孔数据增强
    def drill_data_augmentation(self):
        pass