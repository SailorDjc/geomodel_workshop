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
import model_visual_kit as mvk
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


class GeoMeshParse(object):
    def __init__(self, mesh=None, name=None, is_regular_grid=True):  # , pre_train=True
        self.data = mesh
        self.name = name
        self.is_normalize = False  # 坐标是否归一化
        if mesh is not None:
            self.bound = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
            self.center = mesh.center  # 输入mesh的中心点坐标
            # ori 即输入的原始地质格网数据, 原始模型数据均不做更改，坐标、标签变换在 sample数据中进行
            self.ori_points = mesh.cell_centers().points
            self.ori_scalar = mesh.active_scalars
            self.ori_label, self.ori_label_num = self.convert_noddy_scalar_to_ori_label()  # 获取 ori_label
            self.is_noddy = True
            self.is_regular_grid = True
        else:
            self.bound = None
            self.center = None
            self.ori_points = None
            self.ori_scalar = None
            self.ori_label, self.ori_label_num = None, None
            self.is_noddy = False
            self.is_regular_grid = is_regular_grid
        self.known_pro = 0  # 已知标签数据比例
        self.ori_data_param = None  # 原始数据的维度 (nx, ny, nz)
        # sample数据，是一个规则grid格网，方便训练数据的可视化
        self.grid_point_label = None  # np.array
        self.grid_point_label_num = None  # int
        self.grid_points = None  # list 格网点阵
        self.unregular_grid_points = None  # 不规则格网节点，与self_edge_list中的点号对应，若想可视化则需要映射到相应的规则网格上
        self.unregular_grid_point_label = None
        self.unregular_grid_idx = None  # list

        # self.grid_point_idx 是与 self.sample_grid 的每一个单元格一一对应的，是对ori_points的规则栅格采样，允许有重复
        # 当is_nodddy=True时，是从ori_points场数据中进行稀疏采样，当is_noddy=False时，则是将不规则离散点映射到建模网格上，
        # 前者是多采少，后者是少采多。对于前者，grid_point_idx的主要作用是作为格网数据与原始数据的映射索引，ori_points[grid_point_idx]是网格点
        # 阵列；而对于后者，ori_points相对于格网点阵是少量的，是唯一知道标签的点数据，需要将其与一一映射到各网点上，网格点阵没有映射到的点的
        # 标签是未知的
        # train_idx，其余为None
        self.grid_point_idx = None  # list
        self.grid_cell_idx = None  # idx
        self.sample_grid = None  # mesh
        self.sample_grid_outline = None
        #
        self.edge_list = None  # list 边集（采样数据，包括训练数据与测试数据）
        self.sample_grid_extent = None  # 输出grid的extern[3]: nx,ny,nz

        # 训练数据样本构建，随机散点、钻孔、剖面，作为带标签数据输入模型进行训练
        # self.pre_train = pre_train  # 是否为预训练，True则不进行钻孔、剖面采样，False则采样设置train_idx
        self.train_idx = None  # list 训练数据的idx索引
        self.train_plot_data = None  # 用于可视化采样数据，这里只存储采样参数，不存模型数据
        self.train_plot_data_type = None  # 采样数据类型(散点、钻孔、剖面)
        self.train_sample_operator = None  # 采样操作类型

        # 图特征  下面两个变量用来装数据
        self.node_feat = None  # 图节点特征    np.float32
        self.edge_feat = None  # 图边特征      np.float32

    def execute(self, sample_operator=None, extent=None, edge_feat=None, node_feat=None, normalize=False,
                regular_grid_type=None, **kwargs):
        if node_feat is None:
            node_feat = ['position']
        if edge_feat is None:
            edge_feat = []
        if extent is None:
            extent = [120, 120, 60]
        # 创建栅格格网格架
        if self.ori_scalar is not None:
            self.sample_grid, self.sample_grid_extent = self.create_base_grid(extent=extent, set_label=True)
            self.sample_grid_outline = self.sample_grid.outline()
        else:  # dat_file
            if regular_grid_type is None:
                self.sample_grid, self.sample_grid_extent = self.create_base_grid(extent=extent, set_label=False)
                self.sample_grid_outline = self.sample_grid.outline()
            else:
                convex_surface, grid_outline = self.get_unregular_grid_points_convexhull_surface(self.ori_points)
                self.sample_grid = pv.voxelize(convex_surface, density=2)
                self.sample_grid_outline = grid_outline
                self.grid_points = self.sample_grid.cell_centers().points
                ckt = spt.cKDTree(self.grid_points)
                d, pid = ckt.query(self.ori_points)
                self.grid_cell_idx = pid.tolist()
                self.sample_grid_extent = None
        # 对标签标准化处理
        self.map_grid_vertex_labels()
        # 选择测试模型的样本形式
        self.set_visual_geo_sample(extent=extent, sample_operator=sample_operator, is_noddy=self.is_noddy, **kwargs)
        # 生成三角网剖分
        self.get_triangulate_edges()
        # 生成边权重，以距离作为边权
        for node_feat_type in node_feat:
            self.get_node_feat(node_feat=node_feat_type)
        for edge_feat_type in edge_feat:
            self.get_edge_weight_feat(edge_feat=edge_feat_type, normalize=normalize)
        return self.create_dgl_graph(edge_list=np.int64(self.edge_list).transpose(), node_feat=self.node_feat,
                                     edge_feat=self.edge_feat, node_label=np.int64(self.grid_point_label),
                                     self_loop=False, add_inverse_edge=True, normalize=normalize)

    # 设置虚拟地质采样来切分train_idx   is_update则更新，覆盖之前的采样，否则是追加，保留之前的采样
    def set_visual_geo_sample(self, extent=None, sample_operator=None, is_noddy=True, is_update=False, **kwargs):
        if extent is None:
            if self.sample_grid_extent is None:
                raise ValueError
            extent = self.sample_grid_extent
        sample_op_type = ['rand_pro', 'eq_interval', 'rand_drills', 'axis_sections']
        sample_op_to_idx = {sample_op: index for index, sample_op in enumerate(sample_op_type)}
        if is_noddy:
            for sit, sample_op in enumerate(sample_operator):
                sample_data, pids, train_plot_data_type, train_plot_data = None, None, None, None
                if sample_op not in sample_op_type:
                    break
                if sample_op == 'rand_pro':
                    train_pro = 0.7
                    if 'train_pro' in kwargs.keys():
                        train_pro = kwargs['train_pro']
                        if train_pro <= 0 or train_pro >= 1:
                            raise ValueError
                    x = np.arange(len(self.grid_points))
                    test_size = 1 - train_pro
                    self.train_idx, _ = train_test_split(x, test_size=test_size)
                # if sample_op == 'eq_interval':
                #     self.sample_eq_interval()
                if sample_op == 'rand_drills':
                    drill_pos = None
                    drill_num = 10
                    if 'drill_pos' in kwargs.keys():
                        drill_pos = kwargs['drill_pos']
                    if 'drill_num' in kwargs.keys():
                        drill_num = kwargs['drill_num']
                    sample_data, pids, train_plot_data_type, train_plot_data = \
                        self.sample_with_drills(drill_pos=drill_pos, drill_num=drill_num, extent=extent)
                if sample_op == 'axis_sections':
                    sample_type = None
                    center_random = False
                    if 'sample_type' in kwargs.keys():
                        sample_type = kwargs['sample_type']
                    if 'center_random' in kwargs.keys():
                        center_random = kwargs['center_random']
                    sample_data, pids, train_plot_data_type, train_plot_data = \
                        self.sample_with_sections_along_axis(sample_type=sample_type, center_random=center_random,
                                                             extent=extent)
                if sample_data is not None and pids is not None and train_plot_data_type is not None and train_plot_data is not None:
                    pids = list(set(sorted(pids)))
                    if self.train_plot_data_type is None:
                        self.train_plot_data_type = []
                    if self.train_plot_data is None:
                        self.train_plot_data = []
                    if self.train_idx is not None:
                        self.train_idx.extend(pids)
                        self.train_idx = list(set(sorted(self.train_idx)))
                    else:
                        self.train_idx = pids
                    if is_update and sit == 0:
                        self.train_plot_data_type = []
                        self.train_plot_data = []
                        self.train_plot_data_type.append(train_plot_data_type)
                        self.train_plot_data.append(train_plot_data)
                        self.train_idx = pids
                    else:
                        self.train_plot_data_type.append(train_plot_data_type)
                        self.train_plot_data.append(train_plot_data)

                if self.train_idx is not None and len(self.train_idx) > 0:
                    self.known_pro = len(self.train_idx) / len(self.grid_points)
                    print('Set train_pro is {} ...'.format(self.known_pro))

    # 要保证 edge_list 图中没有自环
    def create_dgl_graph(self, edge_list=None, node_feat=None, edge_feat=None, node_label=None,
                         self_loop=False, add_inverse_edge=False, normalize=False, is_regular_grid=True):

        # 为每一个节点添加一个固定属性-坐标 coord
        # edge_list  node_feat edge_feat node_label add_inverse_edge
        print('Creating Dgl Graph Data')
        num_node = len(node_feat)
        num_edge = edge_list.shape[1]
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

    # 可以自由创建 1维、2维、3维网格，如果是规则沿轴向，则只需要extent和bounds参数，可以通过xx,yy,zz来自定义网格
    def create_grid(self, extent=None, bounds=None, xx=None, yy=None, zz=None):
        xrng, yrng, zrng = None, None, None
        if extent is not None and bounds is not None:
            nx = extent[0]
            ny = extent[1]
            nz = extent[2]
            xrng = np.linspace(start=bounds[0], stop=bounds[1], num=nx)
            yrng = np.linspace(start=bounds[2], stop=bounds[3], num=ny)
            zrng = np.linspace(start=bounds[4], stop=bounds[5], num=nz)
        if xx is not None:
            xrng = xx
        if yy is not None:
            yrng = yy
        if zz is not None:
            zrng = zz
        if xrng is None or yrng is None or zrng is None:
            raise ValueError
        grid = pv.RectilinearGrid(xrng, yrng, zrng)
        return grid

    # 这个函数必须要先调用
    # 在原始grid中进行规则三维矩阵点采样，构建训练数据的格架，所有训练均在这个格架上进行
    # only_out_put=True 只输出一个空模型，不修改geodata任何数据
    # grid_buffer, 向建模边界外扩展一点距离
    def create_base_grid(self, extent=None, set_label=False, only_out_put=False, grid_buffer=0, bounds=None):
        if extent is not None:
            model_extent = extent
        elif self.sample_grid_extent is not None:
            model_extent = self.sample_grid_extent
        else:
            model_extent = [120, 120, 60]
        nx = model_extent[0]
        ny = model_extent[1]
        nz = model_extent[2]
        # 创建二维规则网格
        self.grid_buffer = grid_buffer
        if bounds is None:
            bounds = self.bound
        min_x = bounds[0] + grid_buffer
        max_x = bounds[1] + grid_buffer
        min_y = bounds[2] + grid_buffer
        max_y = bounds[3] + grid_buffer
        min_z = bounds[4]
        max_z = bounds[5]
        xrng = np.linspace(start=min_x, stop=max_x, num=nx)
        yrng = np.linspace(start=min_y, stop=max_y, num=ny)
        zrng = np.linspace(start=min_z, stop=max_z, num=nz)
        grid = pv.RectilinearGrid(xrng, yrng, zrng)
        if only_out_put:
            return grid, model_extent
        if set_label is True and self.is_noddy is True:
            sample_points = grid.cell_centers().points
            # grid_point_idx将ori_points映射到了grid节点上
            ckt = spt.cKDTree(self.ori_points)
            d, pid = ckt.query(sample_points)
            self.grid_point_idx = pid
            self.grid_points = self.ori_points[self.grid_point_idx]
            grid.cell_data['scalars'] = np.array(self.ori_label)[self.grid_point_idx]
        elif self.is_noddy is False:
            # 处理 .dat 数据
            self.grid_points = grid.cell_centers().points
            # 此处的ori_points是采样点
            pid = grid.find_containing_cell(self.ori_points)  # 将带标签节点映射到网格上，设置为训练数据
            # grid_cell_idx 记录采样点对应的规则网格id
            self.grid_cell_idx = pid.tolist()
        self.sample_grid_extent = model_extent
        return grid, self.sample_grid_extent

    def is_connected_graph(self):
        if self.grid_points is None and self.unregular_grid_points is None:
            raise ValueError('Graph Data is empty.')
        elif self.edge_list is None:
            raise ValueError('Graph Data is empty.')
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
    # 处理
    def map_grid_vertex_labels(self, fill_well_data=False):
        if self.ori_label is None:
            raise ValueError
        if self.is_noddy is True:
            if self.grid_point_idx is not None:
                label = np.array(self.ori_label)[self.grid_point_idx]
                self.grid_point_label = label
        elif self.is_noddy is False:
            # 有规则网格和非规则网格
            grid_point_label = []
            # 规则格网
            grid_points_num = 0
            if self.grid_points is not None and self.grid_points.size != 0:
                # 遍历所有网格节点，如果网格节点不属于采样数据，则赋予标签-1，否则赋予相应采样标签
                for lid in np.arange(len(self.grid_points)):
                    if lid not in self.grid_cell_idx:
                        grid_point_label.append([-1])
                    else:
                        # 获取相应采样点的标签
                        o_id = self.grid_cell_idx.index(lid)
                        o_label = self.ori_label[o_id]
                        grid_point_label.append([o_label])
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
        print('Building Delaunay Tetgen of {}'.format(self.name))

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
            raise 'call get_triangulate_edges function'
        if edge_feat is None:
            return None
        edge_weight_dist = []
        if normalize is True:
            sample_points = self.grid_points_normalize(is_regular_grid=is_regular_grid)
        else:
            if is_regular_grid:
                sample_points = self.grid_points
            else:
                sample_points = self.unregular_grid_points
        for item in self.edge_list:
            if edge_feat == 'euclidean':
                coord_i = sample_points[item[0]]
                coord_j = sample_points[item[1]]
                dist = np.sqrt(np.sum(np.square(coord_i - coord_j)))
                edge_weight_dist.append(dist)  # 边权重
        if is_replace:
            if self.edge_feat is not None:
                self.edge_feat = np.vstack(self.edge_feat, np.float32(edge_weight_dist))
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

    # 提取地层分界点
    def extract_stratum_layer_points(self):
        if self.grid_points is not None and self.grid_point_label is not None:
            rand_point = self.grid_points[0]
            rand_point_z = rand_point[2]
            plane_points_idx_list = []
            self.train_idx = []
            # 获取一个随机深度的所有点，即一个水平面的采样点
            for point_idx, point in enumerate(self.grid_points):
                if point[2] == rand_point_z:
                    plane_points_idx_list.append(point_idx)
            # 打虚拟钻孔，平面上所有位置
            train_idx = []
            for point_idx in plane_points_idx_list:
                sample_point = self.grid_points[point_idx]
                # 采样虚拟钻孔
                pos_a = copy.deepcopy(sample_point)
                pos_a[2] = self.bound[5]  # z_max
                pos_b = copy.deepcopy(sample_point)
                pos_b[2] = self.bound[4]  # z_min
                drill_cell_idx = self.sample_grid.find_cells_along_line(pointa=pos_a, pointb=pos_b)
                drill_cell_label = self.grid_point_label[drill_cell_idx]
                # 从上至下遍历钻孔点
                front_label = drill_cell_label[0]
                extract_idx = [drill_cell_idx[0]]
                for l_idx in range(0, len(drill_cell_label)):
                    next_label = drill_cell_label[l_idx]
                    if next_label != front_label:
                        extract_idx.append(drill_cell_idx[l_idx])
                        front_label = drill_cell_label[l_idx]
                if drill_cell_idx[-1] not in extract_idx:
                    extract_idx.append(drill_cell_idx[-1])
                train_idx.extend(extract_idx)
            train_idx = list(set(sorted(train_idx)))
            return train_idx
        else:
            raise ValueError

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

    # 训练集采样，从原始输入格网数据的cell的中心点集合中采样
    # 等间距采样
    def sample_eq_interval(self, interval=100):
        if self.grid_points is None:
            raise ValueError
        if self.grid_point_label is None:
            self.map_grid_vertex_labels()
        pid = None
        if self.train_idx is None:
            pid = np.arange(0, len(self.grid_points), interval)
            pid = list(set(sorted(pid)))
            self.train_idx = pid
        return pid

    # 钻孔采样, 只能在规则格网上进行钻孔采样
    def sample_with_drills(self, drill_pos=None, drill_num=10, extent=None):
        if extent is None:
            extent = [100, 100, 20]
        if self.sample_grid is None:
            self.sample_grid, self.sample_grid_extent = self.create_base_grid(extent=extent)
            self.sample_grid_outline = self.sample_grid.outline()
        horizon_slice = self.sample_grid.slice(normal='z')
        if self.grid_point_label is None:
            self.map_grid_vertex_labels()
        if drill_pos is not None:
            drill_num = len(drill_pos)
        else:
            # 获取grid水平切面，在切面上随机选取cell的中心点，打钻孔
            points = horizon_slice.cell_centers().points.tolist()
            drill_pos = random.sample(points, drill_num)

        pids = []
        ckt = spt.cKDTree(self.grid_points)
        drills = pv.MultiBlock()
        drill_map = {}
        for it, drill_id in enumerate(np.arange(drill_num)):
            pos = drill_pos[drill_id]
            extent = copy.deepcopy(self.sample_grid_extent)
            extent[0] = 1
            extent[1] = 1
            extent[2] = extent[2]  # * 10
            cell_drill = self.create_grid(extent=extent,
                                          bounds=[pos[0], pos[0], pos[1], pos[1], self.bound[4], self.bound[5]])
            # drill_line = self.sample_grid.sample_over_line(pointa=pos_a, pointb=pos_b, resolution=extent[2])
            sample_points = cell_drill.cell_centers().points
            d, pid = ckt.query(sample_points)
            pids.extend(pid)
            sample_label = np.float32(self.grid_point_label)[pid]
            drill_map[it] = np.concatenate((sample_points, sample_label), axis=1)
            cell_drill.cell_data['stratum'] = sample_label
            drills.append(cell_drill)

        train_plot_data = drill_map
        train_plot_data_type = 'drill'
        return drills, pids, train_plot_data_type, train_plot_data

    # 剖面采样           center_random={x:True, y:False}
    def sample_with_sections_along_axis(self, sample_type=None, center_random=None, section_center=None, extent=None):
        if section_center is None:
            section_center = {}
        if extent is None:
            extent = [100, 100, 20]
        if self.grid_point_label is None:
            self.map_grid_vertex_labels()
        if sample_type is None:
            sample_type = {'x': 2, 'y': 2}
        if center_random is None:
            center_random = {label: False for label in sample_type.keys()}
        else:
            # 补全  {x:True} -> {x:True, y:False}
            center_random = {label: False if label not in center_random else center_random[label]
                             for label in sample_type.keys()}
        if self.grid_point_idx is None:
            self.sample_grid, self.sample_grid_extent = self.create_base_grid(extent=extent)
            self.sample_grid_outline = self.sample_grid.outline()
        axis_labels = ['x', 'y', 'z']
        label_to_index = {label: index for index, label in enumerate(axis_labels)}

        pids = []
        ckt = spt.cKDTree(self.grid_points)
        sections = pv.MultiBlock()
        # 切片采样参数
        tmp_slice_param = {}
        for axis_label in sample_type.keys():
            ns = sample_type[axis_label]  # 切面数量
            if center_random[axis_label] is True:
                ax_index = label_to_index[axis_label.lower()]
                tolerance = (self.bound[ax_index * 2 + 1] - self.bound[ax_index * 2]) * 0.01
                ll = np.linspace(self.bound[ax_index * 2] + tolerance,
                                 self.bound[ax_index * 2 + 1] - tolerance, ns * 10).tolist()
                rng = random.sample(ll, ns)
                center = self.center
                tmp_slice_param[axis_label] = []
                tmp_slice_param[axis_label].append(ns)
                for i in range(ns):
                    center[ax_index] = rng[i]
                    sl = self.sample_grid.slice(normal=axis_label, origin=center)
                    tmp_slice_param[axis_label].append(center)
                    sample_points = sl.cell_centers().points.tolist()
                    d, pid = ckt.query(sample_points)
                    pids.extend(pid)
                    # sl.cell_data['stratum'] = np.float32(self.grid_point_label)[pid]
                    sections.append(sl)
            else:
                if axis_label in section_center.keys():
                    tmp_slice_param[axis_label] = []
                    tmp_slice_param[axis_label].append(ns)
                    for i in range(len(section_center[axis_label])):
                        center = section_center[axis_label][i]
                        sl = self.sample_grid.slice(normal=axis_label, origin=center)
                        tmp_slice_param[axis_label].append(center)
                        sample_points = sl.cell_centers().points.tolist()
                        d, pid = ckt.query(sample_points)
                        pids.extend(pid)
                        hh = sl.cell_data['scalars']
                        stratum_scalars = np.float32(self.grid_point_label)[pid]
                        # sl.cell_data['stratum'] = stratum_scalars
                        # sl.set_active_scalars('stratum')
                        sections.append(sl)
                else:
                    secs = self.sample_grid.slice_along_axis(axis=axis_label, n=ns)
                    tmp_slice_param[axis_label] = []
                    tmp_slice_param[axis_label].append(ns)
                    for sec in secs.keys():
                        sl = secs.get(sec)
                        sample_points = sl.cell_centers().points.tolist()
                        d, pid = ckt.query(sample_points)
                        pids.extend(pid)
                        # sample_label = np.float32(self.grid_point_label)[pid]
                        # sl.cell_data['stratum'] = sample_label
                        sections.append(sl)

        train_plot_data = tmp_slice_param
        train_plot_data_type = 'section'
        return sections, pids, train_plot_data_type, train_plot_data

    #  Wells: well, x, y, z, label
    #  Points: x, y, z, label
    # 功能：读取钻孔文件，计算出bound, center,
    def set_data_from_dat_file(self, dat_file_path, data_type, **kwargs):
        if not os.path.exists(dat_file_path):
            raise ValueError
        file_header = None
        file_sep = ' '
        use_cols = None
        has_air = False
        well_type = 0  # 0 为垂直钻孔，1为斜钻孔， 暂时不支持倾斜钻孔
        names = ['well', 'x', 'y', 'z', 'label']  # ['x', 'y', 'z', 'label']
        epsilon = 1
        if 'names' in kwargs.keys():
            names = kwargs['names']
        if 'use_cols' in kwargs.keys():
            use_cols = kwargs['use_cols']
        if 'file_header' in kwargs.keys():
            file_header = kwargs['file_header']
        if 'file_sep' in kwargs.keys():
            file_sep = kwargs['file_sep']
        if 'has_air' in kwargs.keys():
            has_air = kwargs['has_air']
            has_air = has_air == (data_type == 'Wells')
        if 'well_type' in kwargs.keys():
            well_type = kwargs['well_type']
        if 'epsilon' in kwargs.keys():
            epsilon = kwargs['epsilon']
        df = pd.read_table(dat_file_path, header=file_header,
                           skip_blank_lines=False, comment="#", sep=file_sep, encoding='utf-8')
        col_num = df.columns.size
        if names is not None and len(names) == col_num:
            if 'x' in names and 'y' in names and 'z' in names and 'label' in names:
                df.columns = names
            else:
                raise ValueError
        elif use_cols is not None and len(use_cols) <= col_num:
            df = df.iloc[:, use_cols]
            if data_type == 'Wells':
                if len(use_cols) == 5:
                    df.columns = ['well', 'x', 'y', 'z', 'label']
                if len(use_cols) == 4:
                    df.columns = ['x', 'y', 'z', 'label']
            elif data_type == 'Points' and len(use_cols) == 4:
                df.columns = ['x', 'y', 'z', 'label']
        else:
            raise ValueError
        # 处理空值, 坐标值不能出现空值，若出现则删除行记录
        df.dropna(subset=['x', 'y', 'z'])
        points = np.float32(df.loc[:, ['x', 'y', 'z']])
        if data_type == 'Wells':
            if 'well' in df.columns:
                df.dropna(subset=['well'])
        df['label'] = df['label'].fillna('None')
        # labels = np.array(df.loc[:, 'label'])
        self.name = os.path.splitext(os.path.basename(dat_file_path))[0]
        print('Loading Dat File {}'.format(os.path.basename(dat_file_path)))

        self.ori_points = points
        if len(points) > 3:  # 至少三个点
            x_min = min(points[:, 0])
            x_max = max(points[:, 0])
            y_min = min(points[:, 1])
            y_max = max(points[:, 1])
            z_min = min(points[:, 2])
            z_max = max(points[:, 2])
            self.bound = [x_min - epsilon, x_max + epsilon, y_min - epsilon,
                          y_max + epsilon, z_min - epsilon, z_max + epsilon]
            self.center = [(self.bound[0] + self.bound[1]) / 2, (self.bound[2] + self.bound[3]) / 2,
                           (self.bound[4] + self.bound[5]) / 2]
        # 钻孔 Well
        label_dict = {}
        ori_unique_label = sorted(df.label.unique())
        if data_type == 'Wells':
            # 如果没有钻孔标号，需要通过平面坐标(x,y)筛选，将每一个钻孔分出来
            if 'well' not in df.columns:
                df['well'] = None
                well_xy_group = df.groupby(['x', 'y'])
                well_id = 0
                for xy, well_data in well_xy_group:
                    well_name = 'well' + str(well_id)
                    well_index = well_data.index.tolist()
                    well_id += 1
                    df['well'].iloc[well_index] = well_name
            if has_air:
                # 要添加一个类别：地表之上的空气：0 所以要把0空出来
                for lv, label in enumerate(ori_unique_label):
                    label_dict[label] = lv + 1
            else:
                # 不添加空气类别，则地表与钻孔顶部间用最上部的地层填充
                for lv, label in enumerate(ori_unique_label):
                    label_dict[label] = lv
            self.train_plot_data_type = ['drill']
        elif data_type == 'Points':  # 散点 不需要添加空气的类别
            for lv, label in enumerate(ori_unique_label):
                label_dict[label] = lv
        df = df.replace({'label': label_dict})
        well_group = df.groupby(['well'])
        well_map = dict()
        for well_name, well_data in well_group:
            pos = well_data.loc[:, ['x', 'y', 'z', 'label']].values
            well_map[well_name] = pos

        self.train_plot_data = [well_map]
        self.ori_label = df.loc[:, 'label'].tolist()
        self.ori_label_num = len(ori_unique_label)

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

    def set_data_from_edge_file(self, node_file_path, edge_file_path, is_edge_node_begin_from_zero=False, **kwargs):
        if node_file_path is None or edge_file_path is None:
            return None
        file_header = None
        file_sep = '\s+'
        use_cols_e = None
        use_cols = None
        names_e = ['id', 'src', 'dst', 'other']  # ['src', 'dst']
        names = ['id', 'x', 'y', 'z']  # [id, x, y, z]
        if 'names_e' in kwargs.keys():
            names_e = kwargs['names_e']
        if 'names_n' in kwargs.keys():
            names = kwargs['names']
        if 'use_cols_e' in kwargs.keys():
            use_cols_e = kwargs['use_cols_e']
        if 'use_cols' in kwargs.keys():
            use_cols = kwargs['use_cols']
        if 'file_header' in kwargs.keys():
            file_header = kwargs['file_header']
        if 'file_sep' in kwargs.keys():
            file_sep = kwargs['file_sep']
        df_e = pd.read_table(edge_file_path, header=file_header, error_bad_lines=False,
                             skip_blank_lines=False, comment="#", sep=file_sep, encoding='utf-8', low_memory=True)
        df_n = pd.read_table(node_file_path, header=file_header, skip_blank_lines=False, comment="#",
                             sep=file_sep, encoding='utf-8')
        col_num_e = df_e.columns.size
        if names_e is not None and len(names_e) == col_num_e:
            if 'src' in names_e and 'dst' in names_e:
                df_e.columns = names_e
            else:
                raise ValueError
        elif use_cols_e is not None and len(use_cols_e) <= col_num_e:
            df_e = df_e.iloc[:, use_cols_e]
            if len(use_cols_e) == 2:
                df_e.columns = ['src', 'dst']
            if len(use_cols_e) == 3:
                df_e.columns = ['id', 'src', 'dst']
        else:
            raise ValueError

        col_num = df_n.columns.size
        if names is not None and len(names) == col_num:
            if 'x' in names and 'y' in names and 'z' in names:
                df_n.columns = names
            else:
                raise ValueError
        elif use_cols is not None and len(use_cols) <= col_num:
            df_n = df_n.iloc[:, use_cols]
            if len(use_cols) == 3:
                df_n.columns = ['x', 'y', 'z']
            if len(use_cols) == 4:
                df_n.columns = ['id', 'x', 'y', 'z']
        else:
            raise ValueError

        points = np.float32(df_n.loc[:, ['x', 'y', 'z']])
        self.unregular_grid_points = points
        ckt = spt.cKDTree(self.unregular_grid_points)
        d, pid = ckt.query(self.ori_points)
        self.unregular_grid_idx = pid.tolist()
        edge_list = df_e.loc[:, ['src', 'dst']].values.tolist()
        new_edge_list = list(set(tuple(sorted(sub)) for sub in edge_list))
        for e_i, edge in enumerate(new_edge_list):
            edge = list(edge)
            # 若从1开始，需要减1
            if not is_edge_node_begin_from_zero:
                edge[0] -= 1
                edge[1] -= 1
            new_edge_list[e_i] = edge
        self.edge_list = new_edge_list

    def generate_section2d_with_drills_test(self, sample_axis='x', scroll_scale=0.5, drill_num=5):
        axis_label = sample_axis.lower()
        sample_type = {axis_label: 1}
        axis_labels = ['x', 'y', 'z']
        label_to_index = {label: index for index, label in enumerate(axis_labels)}
        ax_index = label_to_index[axis_label]
        if scroll_scale <= 0 or scroll_scale >= 1:
            raise ValueError('scroll must be larger than 0 and less than 1.')
        # min + (max - min) * scale
        pos_axis = self.bound[ax_index * 2] + (self.bound[ax_index * 2 + 1] - self.bound[ax_index * 2]) * scroll_scale
        center = self.center
        center[ax_index] = pos_axis
        section, sec_pids, train_plot_data_type, train_plot_data = \
            self.sample_with_sections_along_axis(sample_type=sample_type, section_center={axis_label: [center]})
        section_point = self.grid_points[sec_pids]
        section_point_label = self.grid_point_label[sec_pids]
        # 在剖面上采样钻孔
        if ax_index == 0:
            p1d = list(section_point[:, 1])
        elif ax_index == 1:
            p1d = list(section_point[:, 0])
        else:
            raise ValueError
        p1d_unique = list(np.unique(p1d))
        # p1d_list = random.sample(p1d_unique, drill_num)
        pid_a = np.arange(0, len(p1d_unique), int(len(p1d_unique) / (drill_num - 1)))
        pid_a = np.int64(pid_a)
        p1d_list = np.array(p1d_unique)[pid_a]

        drill_pos_idx = list(p1d.index(dp) for dp in p1d_list)
        drill_pos = section_point[drill_pos_idx]

        drills, drill_pids, train_plot_data_type, train_plot_data = self.sample_with_drills(drill_pos=drill_pos)
        drill_sample_point = self.grid_points[drill_pids]
        ckt = spt.cKDTree(section_point)
        d, drill_pid = ckt.query(drill_sample_point)
        drill_pid = list(set(sorted(drill_pid)))
        return section, section_point, section_point_label, drills, drill_pid, train_plot_data_type, train_plot_data

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

    # 用来导出钻孔信息， dict{drill_key: [drill_points, drill_points_labels]} 将所有依次导出到一个文件
    def export_drill_dict_dat_file(self, file_path, input_drill_map=None):
        if input_drill_map is None:
            drills_map = self.get_drill_points_labels_map(sample_data_it=0)
        else:
            drills_map = input_drill_map
        out_path = file_path
        pd_list = []
        for k, v in drills_map.items():
            pd_n = pd.DataFrame(v)
            pd_list.append(pd_n)
        pd_file = pd.concat(pd_list, axis=0)
        pd_file.dropna(axis=0, how='any')
        pd_file.to_csv(out_path, index=False, header=False, sep='\t')

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

    # 钻孔统一地层编码
    # def drills_uniform_stratum_labels_process(self):
    #     drill_map = self.get_drill_points_labels_map(sample_data_it=0)
    #     for drill_key in drill_map.keys():

    # 补全钻井数据中，为地层分界点之间的格网单元添加标签
    # well_bottom_extension 钻孔底部延展
    # 暂时不支持以下
    # radius_extension 钻孔径向延展，认为紧挨着钻孔的cell的地层属性与钻孔采样一致
    # def fill_dat_well_grid_point(self, well_bottom_extension=False, radius_extension=0):
    #     if 'drill' in self.train_plot_data_type:
    #         well_data, well_num = self.train_plot_data[0]
    #         # 对钻孔进行径向加密，假设以采样点为中心，半径为radius_extension的圆柱都是钻孔的采样范围
    #         if radius_extension > 0:
    #             rand_point = self.grid_points[0]
    #             rand_point_z = rand_point[2]
    #             plane_points_idx_list = []
    #             for point_idx, point in enumerate(self.grid_points):
    #                 if point[2] == rand_point_z:
    #                     plane_points_idx_list.append(point_idx)
    #             drill_extension_points_id = []
    #             for well_id, well_point in enumerate(well_data):
    #                 for point_id in plane_points_idx_list:
    #                     pos = self.grid_points[point_id]
    #                     point_a = np.array([pos[0], pos[1]], dtype=float)
    #                     point_b = np.array([well_point[0], well_point[1]], dtype=float)
    #                     dist_2d = spt.distance.cdist(point_a, point_b, metric='euclidean', p=2)
    #                     if dist_2d < radius_extension:
    #                         drill_extension_points_id.append(point_id)
    #             well_data.extend(list(self.grid_points[drill_extension_points_id]))
    #
    #         for well_pos in well_data:
    #             pos_a = copy.deepcopy(well_pos)
    #             pos_a[2] = self.bound[5]  # z_max
    #             pos_b = copy.deepcopy(well_pos)
    #             pos_b[2] = self.bound[4]  # z_min
    #             drill_cell_idx = self.sample_grid.find_cells_along_line(pointa=pos_a, pointb=pos_b)
    #             if len(drill_cell_idx) == 0:
    #                 continue
    #             drill_cell_label = self.grid_point_label[drill_cell_idx]
    #             fill_label = [0]
    #             # 获取钻孔的底部标签
    #             bottom_idx = 0
    #             for l_idx in range(0, len(drill_cell_label))[::-1]:
    #                 if drill_cell_label[l_idx] == [-1]:
    #                     continue
    #                 else:
    #                     bottom_idx = l_idx
    #                     break
    #             for idx_tmp, cell_label in enumerate(drill_cell_label):
    #                 if cell_label == [-1]:
    #                     if idx_tmp < bottom_idx or well_bottom_extension is True:
    #                         cell_idx = drill_cell_idx[idx_tmp]
    #                         self.grid_point_label[cell_idx] = fill_label
    #                         self.train_idx.append(cell_idx)
    #                 else:
    #                     fill_label = cell_label
    #         self.train_idx = list(set(sorted(self.train_idx)))
