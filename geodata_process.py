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


class GeoMeshParse(object):
    def __init__(self, mesh, normalize=False, pre_train=True):  # , index_path, create_flann=False
        self.data = mesh

        self.normalize = normalize  # 坐标是否归一化
        self.bound = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        self.center = mesh.center   # 输入mesh的中心点坐标

        # ori 即输入的原始地质格网数据
        self.ori_data_param = None  # 原始数据的维度 (nx, ny, nz)
        self.ori_points = mesh.cell_centers().points
        self.ori_scalar = mesh.active_scalars
        self.ori_label, self.ori_label_num = self.get_ori_label()   # 获取 ori_label

        # sample数据，是一个规则grid格网，方便训练数据的可视化
        self.sample_label = None        # np.array
        self.sample_label_num = None    # int
        self.sample_point = None        # list
        # self.sample_idx 是与 self.sample_grid 的每一个单元格一一对应的，是对ori_points的规则栅格采样，允许有重复，长度与sample_grid
        # 的单元格数目一致
        self.sample_idx = None          # list
        self.sample_grid = None         # mesh
        #
        self.edge_list = None           # list 边集（采样数据，包括训练数据与测试数据）
        self.output_grid_param = None   # 输出grid的extern[3]: nx,ny,nz

        # 训练数据样本构建，随机散点、钻孔、剖面，作为带标签数据输入模型进行训练
        self.pre_train = pre_train     # 是否为预训练，True则不进行钻孔、剖面采样，False采样设置train_idx
        self.train_idx = None          # list 训练数据的idx索引
        self.train_plot_data = None    # 用于可视化采样数据，这里只存储采样参数，不存模型数据
        self.train_plot_data_type = None   # 采样数据类型(散点、钻孔、剖面)
        self.train_sample_operator = None     # 采样操作类型

        # 图特征
        self.node_feat = None  # 图节点特征    np.float32
        self.edge_feat = None  # 图边特征      np.float32

    def execute(self, sample_operator=None, extent=None, edge_feat=None, node_feat=None, **kwargs):
        if node_feat is None:
            node_feat = ['position']
        if extent is None:
            extent = [100, 100, 20]
        if sample_operator is None:
            sample_operator = ['regu_matrix']
        if edge_feat is None:
            edge_feat = ['euclidean']
        if self.normalize is True:
            self.points_normalize()
        sample_op_type = ['rand_pro', 'eq_interval', 'rand_drills', 'axis_sections']
        sample_op_to_idx = {sample_op: index for index, sample_op in enumerate(sample_op_type)}

        # 创建栅格格网格架
        self.sample_grid, self.output_grid_param = self.create_base_grid(extent=extent)
        # 获取采样点的标签
        self.get_sample_labels_standard()
        # 选择测试模型的样本形式
        if self.pre_train is False:
            for sample_op in sample_operator:
                if sample_op not in sample_op_type:
                    break
                # if sample_op == 'rand_pro':
                #     self.sample_rand_pro()
                # if sample_op == 'eq_interval':
                #     self.sample_eq_interval()
                if sample_op == 'rand_drills':
                    drill_pos = None
                    drill_num = 10
                    if 'drill_pos' in kwargs.keys():
                        drill_pos = kwargs['drill_pos']
                    if 'drill_num' in kwargs.keys():
                        drill_num = kwargs['drill_num']
                    self.sample_with_drills(drill_pos=drill_pos, drill_num=drill_num, extent=extent)
                if sample_op == 'axis_sections':
                    sample_type = None
                    center_random = False
                    if 'sample_type' in kwargs.keys():
                        sample_type = kwargs['sample_type']
                    if 'center_random' in kwargs.keys():
                        center_random = kwargs['center_random']
                    self.sample_with_sections_along_axis(sample_type=sample_type, center_random=center_random,
                                                         extent=extent)
        else:
            x = np.arange(len(self.sample_idx))
            self.train_idx, _ = train_test_split(x, test_size=0.2)
        # 生成三角网剖分
        self.get_triangulate_edges()
        # 生成边权重，以距离作为边权
        for node_feat_type in node_feat:
            self.get_node_feat(node_feat=node_feat_type)
        # for edge_feat_type in edge_feat:
        #     self.get_edge_weight_feat(edge_feat=edge_feat_type, normalize=self.normalize)
        return self.create_dgl_graph(edge_list=np.int64(self.edge_list).transpose(), node_feat=self.node_feat,
                                     edge_feat=self.edge_feat, node_label=np.int64(self.sample_label),
                                     self_loop=False, add_inverse_edge=True)

    # 要保证 edge_list 图中没有自环
    def create_dgl_graph(self, edge_list=None, node_feat=None, edge_feat=None, node_label=None,
                         self_loop=False, add_inverse_edge=False):

        # 为每一个节点添加一个固定属性-坐标 coord
        # edge_list  node_feat edge_feat node_label add_inverse_edge

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
        sample_points = np.float32(self.sample_point)
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

    # 这个函数必须要先调用
    # 在原始grid中进行规则三维矩阵点采样，构建训练数据的格架，所有训练均在这个格架上进行
    def create_base_grid(self, extent=None, set_scalar=True):
        if extent is not None:
            model_extent = extent
        elif self.output_grid_param is not None:
            model_extent = self.output_grid_param
        else:
            return None
        nx = model_extent[0]
        ny = model_extent[1]
        nz = model_extent[2]
        xrng = np.linspace(start=self.bound[0], stop=self.bound[1], num=nx)
        yrng = np.linspace(start=self.bound[2], stop=self.bound[3], num=ny)
        zrng = np.linspace(start=self.bound[4], stop=self.bound[5], num=nz)
        grid = pv.RectilinearGrid(xrng, yrng, zrng)
        if set_scalar is True:
            sample_points = grid.cell_centers().points
            ckt = spt.cKDTree(self.ori_points)
            d, pid = ckt.query(sample_points)
            self.sample_idx = pid
            self.sample_point = self.ori_points[self.sample_idx]
            grid.cell_data['scalars'] = np.array(self.ori_label)[self.sample_idx]

        self.output_grid_param = extent
        return grid, extent

    def is_connected_graph(self):
        if self.sample_point is None or self.edge_list is None:
            return 'error'
        else:
            graph = nx.Graph(self.edge_list)
            is_connected = nx.is_connected(graph)
            return is_connected

    # 如果normalize为False， 临时输出归一化坐标，但是对self.sample_point不做更新
    def points_normalize(self, normalize=True, index=None):
        minmax_pnt = preprocessing.MinMaxScaler().fit_transform(self.ori_points)
        # index 是采样的索引index, 外部传入或内部设置
        if index is not None and isinstance(index, list):
            minmax_pnt = minmax_pnt[index]
        # 如果已采样
        elif self.sample_idx is not None:
            minmax_pnt = minmax_pnt[self.sample_idx]
            if normalize is True:
                self.sample_point = minmax_pnt
                self.normalize = normalize
        else:
            raise
        return minmax_pnt

    # 将scalar转换为label, 并将其处理为从0开始的连续自然数
    def get_ori_label(self):
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
    def get_sample_labels_standard(self, index=None):
        if self.ori_label is None:
            raise ValueError
        if index is not None:
            label = self.ori_label[index]
        elif self.sample_idx is not None:
            label = np.array(self.ori_label)[self.sample_idx]
        else:
            raise ValueError
        unique_label = np.unique(label)
        label_num = len(unique_label)
        self.sample_label = label
        self.sample_label_num = label_num
        return label, label_num

    # 构建图结构边集合，采用delaunay三角剖分
    def get_triangulate_edges(self, pid=None):
        edge_list = []
        if pid is not None:
            vertex = self.ori_point[pid]
        elif self.sample_idx is not None:
            vertex = self.sample_point
        else:
            vertex = self.ori_points
        tri = spt.Delaunay(vertex)
        tet_list = tri.simplices
        for tet in tet_list:
            for n_i in np.arange(len(tet)):
                for n_j in np.arange(n_i + 1, len(tet)):
                    edge = [tet[n_i], tet[n_j]]
                    edge_list.append(edge)
        # 去重，除了重复，[0,1] [1,0]只保留[0,1]
        new_edge_list = list(set(tuple(sorted(sub)) for sub in edge_list))

        self.edge_list = new_edge_list
        return new_edge_list

    # 分配边权重
    def get_edge_weight_feat(self, edge_feat='euclidean', normalize=False):
        if self.edge_list is None:
            raise 'call get_triangulate_edges function'
        edge_weight_dist = []
        if normalize is True:
            sample_points = self.points_normalize()
        else:
            sample_points = self.ori_points[self.sample_idx]
        for item in self.edge_list:
            if edge_feat == 'euclidean':
                coord_i = sample_points[item[0]]
                coord_j = sample_points[item[1]]
                dist = np.sqrt(np.sum(np.square(coord_i - coord_j)))
                edge_weight_dist.append(dist)  # 边权重
        if self.edge_feat is not None:
            np.vstack(self.edge_feat, np.float32(edge_weight_dist))
        else:
            self.edge_feat = np.float32(edge_weight_dist)
        return edge_weight_dist

    # 获取节点特征
    def get_node_feat(self, node_feat='stratum', has_train=True, default_value=0):
        node_feat_data = None
        if node_feat == 'position':
            node_feat_data = np.float32(self.sample_point)
        if node_feat == 'stratum':
            node_feat_data = np.float32(self.sample_label)
            if has_train is True and self.train_idx is not None:
                if default_value == 0:
                    # 原来label为从0开始，若缺省值为0，则所有label+1
                    node_feat_data = np.array(list(map(lambda x: [x[0]+1], node_feat_data)))
                devalues_idx = list(set(np.arange(len(self.sample_idx))) - set(self.train_idx))
                if len(devalues_idx) > 0:
                    node_feat_data[devalues_idx][0] = default_value
                node_feat_data = np.float32(node_feat_data)
                node_feat_data.reshape(-1, 1)
        if self.node_feat is not None:
            self.node_feat = np.hstack(self.node_feat, node_feat_data)
        else:
            self.node_feat = node_feat_data
        return self.node_feat

    # 训练集采样，从原始输入格网数据的cell的中心点集合中采样
    # 等间距采样
    def sample_eq_interval(self, interval=100):
        if self.sample_label is None:
            self.get_sample_labels_standard()
        pid = np.arange(0, len(self.sample_point), interval)
        pid = list(set(sorted(pid)))
        if self.train_idx is not None:
            self.train_idx.extend(pid)
            self.train_idx = list(set(sorted(self.train_idx)))
        else:
            self.train_idx = pid
        return pid

    # 钻孔采样
    def sample_with_drills(self, drill_pos=None, drill_num=10, extent=None):
        if extent is None:
            extent = [100, 100, 20]
        if self.sample_idx is None:
            self.sample_grid, self.output_grid_param = self.create_base_grid(extent=extent)
        horizon_slice = self.sample_grid.slice(normal='z')
        if self.sample_label is None:
            self.get_sample_labels_standard()
        if drill_pos is not None:
            drill_num = len(drill_pos)
        else:
            # 获取grid水平切面，在切面上随机选取cell的中心点，打钻孔
            points = horizon_slice.cell_centers().points.tolist()
            drill_pos = random.sample(points, drill_num)

        pids = []
        ckt = spt.cKDTree(self.sample_point)
        drills = pv.MultiBlock()
        for drill_id in np.arange(drill_num):
            pos = drill_pos[drill_id]
            pos_a = copy.deepcopy(pos)
            pos_a[2] = self.bound[5]
            pos_b = copy.deepcopy(pos)
            pos_b[2] = self.bound[4]
            drill = self.sample_grid.sample_over_line(pointa=pos_a, pointb=pos_b, resolution=extent[2])

            sample_points = drill.points.tolist()
            d, pid = ckt.query(sample_points)
            pids.extend(pid)
            sample_label = np.float32(self.sample_label)[pid]
            drill.point_data['stratum'] = sample_label
            drills.append(drill)
        if self.train_plot_data is None:
            self.train_plot_data = []
        self.train_plot_data.append((drill_pos, drill_num))

        if self.train_plot_data_type is None:
            self.train_plot_data_type = []
        self.train_plot_data_type.append('drill')

        pids = list(set(sorted(pids)))
        if self.train_idx is not None:
            self.train_idx.extend(pids)
            self.train_idx = list(set(sorted(self.train_idx)))
        else:
            self.train_idx = pids
        return pids

    # 剖面采样
    def sample_with_sections_along_axis(self, sample_type=None, center_random=False, extent=None):
        if extent is None:
            extent = [100, 100, 20]
        if self.sample_label is None:
            self.get_sample_labels_standard()
        if sample_type is None:
            sample_type = {'x': 2, 'y': 2}
        if center_random is None:
            center_random = {label: False for label in sample_type.keys()}
        else:
            center_random = {label: False if label not in center_random else center_random[label]
                             for label in sample_type.keys()}
        if self.sample_idx is None:
            self.sample_grid, self.output_grid_param = self.create_base_grid(extent=extent)
        axis_labels = ['x', 'y', 'z']
        label_to_index = {label: index for index, label in enumerate(axis_labels)}

        pids = []
        ckt = spt.cKDTree(self.sample_point)

        sections = pv.MultiBlock()

        # 切片采样参数
        tmp_slice_param = {}
        for axis_label in sample_type.keys():
            ns = sample_type[axis_label]
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
                    sl.cell_data['stratum'] = np.float32(self.sample_label)[pid]
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
                    sample_label = np.float32(self.sample_label)[pid]
                    sl.cell_data['stratum'] = sample_label
                    sections.append(sl)
        if self.train_plot_data is None:
            self.train_plot_data = []
        self.train_plot_data.append(tmp_slice_param)

        if self.train_plot_data_type is None:
            self.train_plot_data_type = []
        self.train_plot_data_type.append('section')

        pids = list(set(sorted(pids)))

        if self.train_idx is not None:
            self.train_idx.extend(pids)
            self.train_idx = list(set(sorted(self.train_idx)))
        else:
            self.train_idx = pids
        return pids

    # # 按比例随机采样
    # def sample_rand_pro(self, prop=0.8):
    #     if self.sample_label is None:
    #         self.get_sample_labels_standard()
    #     if prop >= 1 or prop <= 0:
    #         prop = 0.8
    #     points_sum = len(self.ori_points)
    #     sample_num = int(points_sum * prop)
    #     pid = np.random.sample(np.arange(0, points_sum), sample_num)
    #     pid = list(set(sorted(pid)))
    #     if self.train_idx is not None:
    #         self.train_idx.extend(pid)
    #         self.train_idx = list(set(sorted(self.train_idx)))
    #     else:
    #         self.train_idx = pid
    #     return pid
    #
    # # 分层标签按比例采样，保证各标签的比例不变
    # def sample_rand_pro_label(self, prop=0.8, label=None):
    #     pid = []
    #     if self.sample_label is None:
    #         self.get_sample_labels_standard()
    #     label_dict = {}
    #     for item in np.unique(self.sample_label):
    #         label_dict[item] = []
    #     for idx, item in enumerate(self.sample_label):
    #         label_dict[item].append(idx)
    #     for key in label_dict:
    #         sample_num = int(len(label_dict[key]) * prop)
    #         pid_key = np.random.sample(label_dict[key], sample_num)
    #         pid.extend(pid_key)
    #
    #     pid = list(set(sorted(pid)))
    #     if self.train_idx is not None:
    #         self.train_idx.extend(pid)
    #         self.train_idx = list(set(sorted(self.train_idx)))
    #     else:
    #         self.train_idx = pid
    #     return pid
