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


class GeoMeshParse(object):
    def __init__(self, mesh=None, name=None, normalize=False, pre_train=True, is_noddy=True):
        self.data = mesh
        self.name = name
        self.normalize = normalize  # 坐标是否归一化
        if mesh is not None:
            self.bound = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
            self.center = mesh.center  # 输入mesh的中心点坐标
            # ori 即输入的原始地质格网数据, 原始模型数据均不做更改，坐标、标签变换在 sample数据中进行
            self.ori_points = mesh.cell_centers().points
            self.ori_scalar = mesh.active_scalars
            self.ori_label, self.ori_label_num = self.convert_noddy_scalar_to_ori_label()  # 获取 ori_label
            self.is_noddy = True
        else:
            self.bound = None
            self.center = None
            self.ori_points = None
            self.ori_scalar = None
            self.ori_label, self.ori_label_num = None, None
            self.is_noddy = False
        self.known_pro = 0  # 已知标签数据比例
        self.ori_data_param = None  # 原始数据的维度 (nx, ny, nz)
        # sample数据，是一个规则grid格网，方便训练数据的可视化
        self.grid_point_label = None  # np.array
        self.grid_point_label_num = None  # int
        self.grid_points = None  # list 格网点阵
        # self.grid_point_idx 是与 self.sample_grid 的每一个单元格一一对应的，是对ori_points的规则栅格采样，允许有重复
        # 当is_nodddy=True时，是从ori_points场数据中进行稀疏采样，当is_noddy=False时，则是将不规则离散点映射到建模网格上，
        # 前者是多采少，后者是少采多。对于前者，grid_point_idx的主要作用是作为格网数据与原始数据的映射索引，ori_points[grid_point_idx]是网格点
        # 阵列；而对于后者，ori_points相对于格网点阵是少量的，是唯一知道标签的点数据，需要将其与一一映射到各网点上，网格点阵没有映射到的点的
        # 标签是未知的
        # train_idx，其余为None
        self.grid_point_idx = None  # list
        self.grid_cell_idx = None  # idx
        self.sample_grid = None  # mesh
        #
        self.edge_list = None  # list 边集（采样数据，包括训练数据与测试数据）
        self.output_grid_param = None  # 输出grid的extern[3]: nx,ny,nz

        # 训练数据样本构建，随机散点、钻孔、剖面，作为带标签数据输入模型进行训练
        self.pre_train = pre_train  # 是否为预训练，True则不进行钻孔、剖面采样，False则采样设置train_idx
        self.train_idx = None  # list 训练数据的idx索引
        self.train_plot_data = None  # 用于可视化采样数据，这里只存储采样参数，不存模型数据
        self.train_plot_data_type = None  # 采样数据类型(散点、钻孔、剖面)
        self.train_sample_operator = None  # 采样操作类型

        # 图特征  下面两个变量用来装数据
        self.node_feat = None  # 图节点特征    np.float32
        self.edge_feat = None  # 图边特征      np.float32

    def execute(self, sample_operator=None, extent=None, edge_feat=None, node_feat=None, **kwargs):
        if node_feat is None:
            node_feat = ['position']
        if edge_feat is None:
            edge_feat = []
        if extent is None:
            extent = [100, 100, 30]
        sample_op_type = ['rand_pro', 'eq_interval', 'rand_drills', 'axis_sections']
        sample_op_to_idx = {sample_op: index for index, sample_op in enumerate(sample_op_type)}

        # 创建栅格格网格架
        if self.ori_scalar is not None:
            self.sample_grid, self.output_grid_param = self.create_base_grid(extent=extent, set_label=True)
        else:  # dat_file
            self.sample_grid, self.output_grid_param = self.create_base_grid(extent=extent, set_label=False)
        if self.normalize is True:
            self.grid_points_normalize()
        # 对标签标准化处理
        self.map_grid_vertex_labels()
        # 选择测试模型的样本形式
        if self.pre_train is False:
            if self.is_noddy:
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
            if self.is_noddy:
                x = np.arange(len(self.grid_points))
                self.train_idx, _ = train_test_split(x, test_size=0.2)
        # 生成三角网剖分
        self.get_triangulate_edges()
        # 生成边权重，以距离作为边权
        for node_feat_type in node_feat:
            self.get_node_feat(node_feat=node_feat_type)
        for edge_feat_type in edge_feat:
            self.get_edge_weight_feat(edge_feat=edge_feat_type, normalize=self.normalize)
        if self.train_idx is not None and len(self.train_idx) > 0:
            self.known_pro = len(self.train_idx) / len(self.grid_points)
        return self.create_dgl_graph(edge_list=np.int64(self.edge_list).transpose(), node_feat=self.node_feat,
                                     edge_feat=self.edge_feat, node_label=np.int64(self.grid_point_label),
                                     self_loop=False, add_inverse_edge=True)

    # 要保证 edge_list 图中没有自环
    def create_dgl_graph(self, edge_list=None, node_feat=None, edge_feat=None, node_label=None,
                         self_loop=False, add_inverse_edge=False):

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

    # 这个函数必须要先调用
    # 在原始grid中进行规则三维矩阵点采样，构建训练数据的格架，所有训练均在这个格架上进行
    def create_base_grid(self, extent=None, set_label=False):
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
        if set_label is True and self.is_noddy is True:
            sample_points = grid.cell_centers().points
            # grid_point_idx将ori_points映射到了grid节点上
            ckt = spt.cKDTree(self.ori_points)
            d, pid = ckt.query(sample_points)
            self.grid_point_idx = pid
            self.grid_points = self.ori_points[self.grid_point_idx]
            grid.cell_data['scalars'] = np.array(self.ori_label)[self.grid_point_idx]
        else:
            # 处理 .dat 数据
            self.grid_points = grid.cell_centers().points
            pid = grid.find_containing_cell(self.ori_points)  # 将带标签节点映射到网格上，设置为训练数据
            self.grid_cell_idx = pid.tolist()
        self.output_grid_param = model_extent
        return grid, self.output_grid_param

    def is_connected_graph(self):
        if self.grid_points is None or self.edge_list is None:
            raise ValueError
        else:
            graph = nx.Graph(self.edge_list)
            is_connected = nx.is_connected(graph)
            return is_connected

    # 如果normalize为False， 临时输出归一化坐标，但是对self.grid_matrix_point不做更新
    def grid_points_normalize(self, normalize=False):
        if self.grid_points is None:
            raise ValueError
        if self.normalize is True:
            return self.grid_points
        minmax_pnt = preprocessing.MinMaxScaler().fit_transform(self.grid_points)
        if normalize:
            self.grid_points = minmax_pnt
            self.normalize = normalize
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
    def map_grid_vertex_labels(self):
        if self.ori_label is None:
            raise ValueError
        if self.is_noddy is True:
            if self.grid_point_idx is not None:
                label = np.array(self.ori_label)[self.grid_point_idx]
                self.grid_point_label = label
        elif self.is_noddy is False:
            grid_point_label = []
            for lid in np.arange(len(self.grid_points)):
                if lid not in self.grid_cell_idx:
                    grid_point_label.append([-1])
                else:
                    o_id = self.grid_cell_idx.index(lid)
                    o_label = self.ori_label[o_id]
                    grid_point_label.append([o_label])
            self.train_idx = sorted(self.grid_cell_idx)
            self.grid_point_label = np.array(grid_point_label)
            if self.train_plot_data_type is not None:
                self.fill_dat_well_grid_point()
            # unique_grid_label = np.unique(grid_point_label)
            # self.grid_point_label_num = len(unique_grid_label)
        else:
            raise ValueError
        if self.grid_point_label is not None:
            unique_label = np.unique(self.grid_point_label)
            label_num = len(unique_label)
            if -1 in unique_label:
                label_num -= 1
            self.grid_point_label_num = label_num
        else:
            raise ValueError
        return self.grid_point_label, self.grid_point_label_num

    # 构建图结构边集合，采用delaunay三角剖分
    def get_triangulate_edges(self, pid=None):
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

        self.edge_list = new_edge_list
        return new_edge_list

    # 分配边权重
    def get_edge_weight_feat(self, edge_feat, normalize=False):
        if self.edge_list is None:
            raise 'call get_triangulate_edges function'
        if edge_feat is None:
            return None
        edge_weight_dist = []
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
        if self.edge_feat is not None:
            np.vstack(self.edge_feat, np.float32(edge_weight_dist))
        else:
            self.edge_feat = np.float32(edge_weight_dist)
        return edge_weight_dist

    # 获取节点特征
    def get_node_feat(self, node_feat='stratum', has_train=True, default_value=0):
        node_feat_data = None
        if node_feat == 'position':
            node_feat_data = np.float32(self.grid_points)
        if node_feat == 'stratum':
            node_feat_data = np.float32(self.grid_point_label)
            if has_train is True and self.train_idx is not None:
                if default_value == 0:
                    # 原来label为从0开始，若缺省值为0，则所有label+1
                    node_feat_data = np.array(list(map(lambda x: [x[0] + 1], node_feat_data)))
                devalues_idx = list(set(np.arange(len(self.grid_points))) - set(self.train_idx))
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

    # 钻孔采样
    def sample_with_drills(self, drill_pos=None, drill_num=10, extent=None):
        if extent is None:
            extent = [100, 100, 20]
        if self.grid_point_idx is None:
            self.sample_grid, self.output_grid_param = self.create_base_grid(extent=extent)
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
            sample_label = np.float32(self.grid_point_label)[pid]
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
        if self.grid_point_label is None:
            self.map_grid_vertex_labels()
        if sample_type is None:
            sample_type = {'x': 2, 'y': 2}
        if center_random is None:
            center_random = {label: False for label in sample_type.keys()}
        else:
            center_random = {label: False if label not in center_random else center_random[label]
                             for label in sample_type.keys()}
        if self.grid_point_idx is None:
            self.sample_grid, self.output_grid_param = self.create_base_grid(extent=extent)
        axis_labels = ['x', 'y', 'z']
        label_to_index = {label: index for index, label in enumerate(axis_labels)}

        pids = []
        ckt = spt.cKDTree(self.grid_points)
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
                    sl.cell_data['stratum'] = np.float32(self.grid_point_label)[pid]
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
                    sample_label = np.float32(self.grid_point_label)[pid]
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

    #  Wells: well, x, y, z, label
    #  Points: x, y, z, label
    def set_data_from_dat_file(self, dat_file_path, data_type, **kwargs):
        if not os.path.exists(dat_file_path):
            raise ValueError
        file_header = None
        file_sep = ' '
        use_cols = None
        has_air = False
        well_type = 0  # 0 为垂直钻孔，1为斜钻孔， 暂时不支持倾斜钻孔
        names = None  # ['x', 'y', 'z', 'label']
        epsilon = 0.01
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
            self.bound = [min(points[:, 0])-epsilon, max(points[:, 0])+epsilon, min(points[:, 1])-epsilon,
                          max(points[:, 1]+epsilon), min(points[:, 2])-epsilon, max(points[:, 2])+epsilon]
            self.center = [(self.bound[0] + self.bound[1]) / 2, (self.bound[2] + self.bound[3]) / 2,
                           (self.bound[4] + self.bound[5]) / 2]
        # 钻孔 Well
        label_dict = {}
        ori_unique_label = df.label.unique()

        if data_type == 'Wells':
            if 'well' not in df.columns:
                df['well'] = None
                well_xy_group = df.groupby(['x', 'y'])
                well_id = 0
                for xy, well_data in well_xy_group:
                    well_name = 'well' + str(well_id)
                    well_index = well_data.index.tolist()
                    well_id += 1
                    df['well'].iloc[well_index] = well_name
            well_group = df.groupby(['well'])
            well_pos = []
            well_num = 0
            for well_name, well_data in well_group:
                if well_type == 0:
                    pos = well_data.loc[:, ['x', 'y', 'z']].values.tolist()
                    well_pos.append(pos[0])
                well_num += 1
            self.train_plot_data = [well_pos, well_num]
            if has_air:
                # 要添加一个类别：地表之上的空气：0 所以要把0空出来
                for lv, label in enumerate(ori_unique_label):
                    label_dict[label] = lv + 1
            else:
                for lv, label in enumerate(ori_unique_label):
                    label_dict[label] = lv

            self.train_plot_data_type = ['drill']
        elif data_type == 'Points':  # 散点 不需要添加空气的类别
            for lv, label in enumerate(ori_unique_label):
                label_dict[label] = lv
        df = df.replace({'label': label_dict})
        self.ori_label = df.loc[:, 'label'].tolist()
        self.ori_label_num = len(ori_unique_label)

    # 补全钻井数据中，为地层分界点之间的格网单元添加标签
    # well_bottom_extension 钻孔底部延展
    # 暂时不支持以下
    # radius_extension 钻孔径向延展，认为紧挨着钻孔的cell的地层属性与钻孔采样一致
    def fill_dat_well_grid_point(self, well_bottom_extension=False, radius_extension=0):
        if 'drill' in self.train_plot_data_type:
            well_data, well_num = self.train_plot_data
            for well_pos in well_data:
                pos_a = copy.deepcopy(well_pos)
                pos_a[2] = self.bound[5]  # z_max
                pos_b = copy.deepcopy(well_pos)
                pos_b[2] = self.bound[4]  # z_min
                drill_cell_idx = self.sample_grid.find_cells_along_line(pointa=pos_a, pointb=pos_b)
                drill_cell_label = self.grid_point_label[drill_cell_idx]
                fill_label = [0]
                # 获取钻孔的底部标签
                bottom_idx = 0
                for l_idx in range(0, len(drill_cell_label))[::-1]:
                    if drill_cell_label[l_idx] == [-1]:
                        continue
                    else:
                        bottom_idx = l_idx
                        break
                for idx_tmp, cell_label in enumerate(drill_cell_label):
                    if cell_label == [-1]:
                        if idx_tmp < bottom_idx or well_bottom_extension is True:
                            cell_idx = drill_cell_idx[idx_tmp]
                            self.grid_point_label[cell_idx] = fill_label
                            self.train_idx.append(cell_idx)
                    else:
                        fill_label = cell_label
            self.train_idx = sorted(self.train_idx)

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
