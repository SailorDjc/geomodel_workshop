import copy
import pickle

import pandas as pd
from sklearn import preprocessing
import numpy as np
import pyvista as pv
import os
import dgl
import dgl.backend as F
import torch
from dgl.data.utils import load_graphs, save_graphs
from sklearn.model_selection import train_test_split
from geodata_process import GeoMeshParse
import time
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import model_visual_kit as mvk
import torchmetrics.functional as MF
from xgboost import XGBClassifier
# from scipy.interpolate import griddata
from scipy.interpolate import RBFInterpolator, NearestNDInterpolator, LinearNDInterpolator
from scipy.stats.qmc import Halton


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


class GmeModelList(object):
    def __init__(self, name, root, pre_train_model_list=None, train_model_list=None,
                 sample_operator=None, noddy_data=None, self_loop=False, add_inverse_edge=True,
                 model_extern=None, dgl_graph_param=None, data_type='Noddy', dat_file_path=None,
                 **kwargs):
        # data_type数据集类型，'Noddy': 来自Noddy数据集， 'Wells': 钻井 .dat文件1格式， 'Points': 散点 .dat文件格式
        # 注：.dat文件格式与Voxler软件一致
        if dgl_graph_param is None:
            dgl_graph_param = [['position'], None]  # [[node_feat], [edge_feat]]
        if train_model_list is None:  # 一般只装一个model
            train_model_list = []
        if pre_train_model_list is None:  # 预训练模型， 列表中有多个模型
            pre_train_model_list = []
        # 外部传入参数
        self.noddy_data = noddy_data
        self.pre_train_model_list = pre_train_model_list  # 预训练模型数据
        self.train_model_list = train_model_list  # 训练任务模型数据
        self.model_extern = model_extern
        self.dgl_graph_param = dgl_graph_param  # 图节点特征、边特征类型参数
        self.sample_operator = sample_operator  # ['rand_drills', 'axis_sections'] 设置已知数据采样方式
        self.self_loop = self_loop  # 添加自环
        self.add_inverse_edge = add_inverse_edge  # 无向图
        self.root = root  # 代码工作空间根目录， 会默认将处理后数据存放在 root/processed目录下
        self.name = name  # 数据集名称

        self.data_type = data_type
        self.dat_file_path = dat_file_path
        # file_header  # 有无表头 ,header=None没有表头， header=0 第1行是表头
        # 数据存储参数
        self.geodata = []
        self.result_model = {}
        self.graph_log = {}
        self.num_classes = None  # 数据集分类数目
        self.predict_num_classes = None
        # 文件存储路径，在processed文件夹下共存储5个数据文件，两个模型训练checkpoint存储文件
        processed_dir = os.path.join(self.root, 'processed')
        self.processed_file_path = os.path.join(processed_dir, 'gme_data_processed')  # 存储与训练的dgl_graph图数据
        self.processed_predict_graph_path = os.path.join(processed_dir, 'gme_data_predict')  # 存储用来工程预测的数据
        self.processed_geodata_path = os.path.join(processed_dir, 'geodata.pkl')  # 存储geodata模型数据
        self.graph_log_data_path = os.path.join(processed_dir, 'graph_log.pkl')  # 存储dgl_graph的日志信息
        # 记录 dgl_graph, 以及图生成参数，包括node_feat类型, edge_feat类型,类别数, 节点数, 边数.
        self.result_model_path = os.path.join(processed_dir, 'result_gme_model.pkl')  # 存储输出结果地质模型，以及相应模型参数
        self.graph = None
        self.predict_graph = None
        super(GmeModelList, self).__init__()
        self.kwargs = kwargs
        self.access_model_data()

    def access_model_data(self):
        # 如果存在处理后数据，则直接从文件加载
        if os.path.exists(self.graph_log_data_path):
            self.graph_log = self.load_graph_log()
        # 加载dgl_graph和geodata, 其中dgl_graph是图神经网络训练的数据源, geodata主要存储地质数据源用来可视化分析
        if os.path.exists(self.processed_geodata_path):  # geodata
            self.geodata = self.load_geodata()
            print('Loading {} GeoModel Data ...'.format(len(self.geodata)))
        if os.path.exists(self.processed_file_path):
            self.graph, self.num_classes = load_graphs(self.processed_file_path)  # dgl_graph
            print('Loading {} Saved Pretrain Graphs Files Data ...'.format(len(self.graph)))
            # dgl_graph 处理后dgl图数据日志字典, 用来判断是否添加新的数据
            if os.path.exists(self.graph_log_data_path):
                # self.graph_log已经加载过了
                for idx, _ in enumerate(self.graph):
                    if idx in self.graph_log['pre_train_graph'].keys():
                        graph_name = self.graph_log['pre_train_graph'][idx][0]  # graph_name图名：model_name+'_'+timecode
                        cur_model_name = graph_name.split('_')[0]
                        model_extern = self.graph_log['pre_train_graph'][idx][1]  # 采样格网尺寸
                        node_feat = self.graph_log['pre_train_graph'][idx][2]
                        edge_feat = self.graph_log['pre_train_graph'][idx][3]

                        cur_extern = self.get_model_extern(cur_model_name)
                        cur_node_feat = self.dgl_graph_param[0]
                        cur_edge_feat = self.dgl_graph_param[1]
                        # 删除重复加载的数据
                        for model_name in self.pre_train_model_list:
                            if model_extern == cur_extern and cur_node_feat == node_feat \
                                    and edge_feat == cur_edge_feat and model_name == cur_model_name:
                                self.pre_train_model_list.remove(model_name)
        # 预测数据集
        if os.path.exists(self.processed_predict_graph_path):
            self.predict_graph, self.predict_num_classes = load_graphs(self.processed_predict_graph_path)
            print('Loading {} Saved Predict Graphs Files Data ...'.format(len(self.predict_graph)))
            if os.path.exists(self.graph_log_data_path):
                for idx, _ in enumerate(self.predict_graph):
                    if idx in self.graph_log['predict_graph'].keys():
                        graph_name = self.graph_log['predict_graph'][idx][0]  # graph_name图名：model_name+'_'+timecode
                        cur_model_name = graph_name.split('_')[0]
                        model_extern = self.graph_log['predict_graph'][idx][1]  # 采样格网尺寸
                        node_feat = self.graph_log['predict_graph'][idx][2]
                        edge_feat = self.graph_log['predict_graph'][idx][3]

                        cur_extern = self.get_model_extern(cur_model_name)
                        cur_node_feat = self.dgl_graph_param[0]
                        cur_edge_feat = self.dgl_graph_param[1]
                        # 删除重复加载的数据
                        for model_name in self.train_model_list:
                            if model_extern == cur_extern and cur_node_feat == node_feat \
                                    and edge_feat == cur_edge_feat and model_name == cur_model_name:
                                self.train_model_list.remove(model_name)
        # 容器初始化
        if self.graph is None:
            self.graph = []
        if self.num_classes is None:
            self.num_classes = {'labels': []}
        if self.predict_graph is None:
            self.predict_graph = []
        if self.predict_num_classes is None:
            self.predict_num_classes = {'labels': []}

        save_flag = 0
        if self.data_type == 'Noddy':
            if len(self.pre_train_model_list) > 0 or len(self.train_model_list) > 0:
                self.process_noddy_data()
                save_flag = 1
        elif self.data_type == 'Wells' or self.data_type == 'Points':
            if self.dat_file_path is not None:
                self.process_dat_file(**self.kwargs)
                save_flag = 1
        # 存储geodata和graph_log
        if save_flag == 1:
            self.save_graph_log()
            self.save_geodata()
            self.geodata = self.load_geodata()
            self.graph_log = self.load_graph_log()
            if len(self.predict_graph) > 0:
                print('Saving...')
                save_graphs(self.processed_predict_graph_path, self.predict_graph, self.predict_num_classes)
                self.predict_graph, self.predict_num_classes = load_graphs(self.processed_predict_graph_path)
            if len(self.graph) > 0:
                print('Saving...')
                save_graphs(self.processed_file_path, self.graph, self.num_classes)
                self.graph, self.num_classes = load_graphs(self.processed_file_path)

    # 如果edge_list_path为None则使用geodata内部算法三角化，若不为None，则外部传入已经构建好的网格边集
    def set_dat_file_regular_grid(self, dat_file_path, is_save=False, data_type='Wells', **kwargs):
        self.dat_file_path = dat_file_path
        self.data_type = data_type
        self.kwargs.update(kwargs)
        self.process_dat_file(**self.kwargs)
        if is_save is True:
            self.save_graph_log()
            self.save_geodata()
            self.geodata = self.load_geodata()
            self.graph_log = self.load_graph_log()
            print('Saving...')
            save_graphs(self.processed_predict_graph_path, self.predict_graph, self.predict_num_classes)
            self.predict_graph, self.predict_num_classes = load_graphs(self.processed_predict_graph_path)

    def set_dat_file_unregular_grid(self, dat_file_path, edge_list_path=None, grid_node_file_path=None, is_save=False,
                                    data_type='Wells',
                                    **kwargs):
        self.dat_file_path = dat_file_path
        self.data_type = data_type
        self.kwargs.update(kwargs)
        geodata = GeoMeshParse(is_regular_grid=False)
        geodata.set_data_from_dat_file(dat_file_path=self.dat_file_path, data_type=self.data_type, **kwargs)
        geodata.set_data_from_edge_file(node_file_path=grid_node_file_path, edge_file_path=edge_list_path, **kwargs)
        geodata.sample_grid, geodata.sample_grid_outline = geodata.match_unregular_grid_to_regular_grid(cell_density=2)
        geodata.sample_grid_extent = None
        geodata.map_grid_vertex_labels()

        geodata.get_node_feat(node_feat='position', is_regular_grid=False)
        dgl_graph = geodata.create_dgl_graph(edge_list=np.int64(geodata.edge_list).transpose(),
                                             node_feat=geodata.node_feat,
                                             node_label=np.int64(geodata.unregular_grid_point_label), add_inverse_edge=True,
                                             normalize=True, is_regular_grid=False)

        label_num = geodata.grid_point_label_num
        is_connected = geodata.is_connected_graph()
        print('is_connected:', is_connected)
        self.geodata.append(geodata)
        save_idx = len(self.predict_graph)
        self.update_graph_log(model_name=geodata.name,
                              save_idx=save_idx, extern=geodata.sample_grid_extent,
                              node_feat=self.dgl_graph_param[0], edge_feat=self.dgl_graph_param[1],
                              is_pre_train=False)
        self.predict_graph.append(dgl_graph)
        if torch.is_tensor(self.predict_num_classes['labels']):
            self.predict_num_classes['labels'] = self.predict_num_classes['labels'].numpy().tolist()
        self.predict_num_classes['labels'].append(label_num)
        self.predict_num_classes = {'labels': torch.tensor(self.predict_num_classes['labels']).to(torch.long)}
        if is_save:
            print('Saving...')
            self.save_geodata()
            self.save_graph_log()
            self.geodata = self.load_geodata()
            save_graphs(self.processed_predict_graph_path, self.predict_graph, self.predict_num_classes)
            self.predict_graph, self.predict_num_classes = load_graphs(self.processed_predict_graph_path)

    def process_noddy_data(self):
        # 批量处理每一个地质模型数据文件
        dgl_graph_list = []
        labels_num_list = []
        for model_index in np.arange(len(self.pre_train_model_list)):
            mesh = self.noddy_data.get_grid_model(self.pre_train_model_list[model_index])
            model_extern = self.get_model_extern(self.pre_train_model_list[model_index])

            geodata = GeoMeshParse(mesh, name=self.pre_train_model_list[model_index])
            dgl_graph = geodata.execute(sample_operator=self.sample_operator, extent=model_extern,
                                        edge_feat=self.dgl_graph_param[1], node_feat=self.dgl_graph_param[0],
                                        normalize=True)
            # 对标签进行处理
            label_num = geodata.grid_point_label_num
            labels_num_list.append(label_num)
            # 已存储预训练数据集
            pre_save_graph_num = len(self.graph)
            is_connected = geodata.is_connected_graph()
            print('is_connected:', is_connected)
            if len(self.predict_graph) == 0:
                self.geodata.append(geodata)
            else:
                geo_idx = pre_save_graph_num + model_index
                self.geodata.insert(geo_idx, geodata)
            dgl_graph_list.append(dgl_graph)
            self.update_graph_log(model_name=self.pre_train_model_list[model_index],
                                  save_idx=model_index + pre_save_graph_num, extern=model_extern,
                                  node_feat=self.dgl_graph_param[0], edge_feat=self.dgl_graph_param[1],
                                  is_pre_train=True)
        self.graph.extend(dgl_graph_list)
        if torch.is_tensor(self.num_classes['labels']):
            self.num_classes['labels'] = self.num_classes['labels'].numpy().tolist()
        self.num_classes['labels'].extend(labels_num_list)
        self.num_classes = {'labels': torch.tensor(self.num_classes['labels']).to(torch.long)}

        # 最后的工程预测数据
        predict_dgl_graph_list = []
        predict_labels_num_list = []
        for train_mdl_index in np.arange(len(self.train_model_list)):
            mesh = self.noddy_data.get_grid_model(self.train_model_list[train_mdl_index])
            model_extern = self.get_model_extern(self.train_model_list[train_mdl_index])

            geodata = GeoMeshParse(mesh, name=self.train_model_list[train_mdl_index])
            drill_num = 300  # self.geodata[0].train_plot_data[0][1]
            drill_pos = None  # self.geodata[0].train_plot_data[0][0]
            sample_type = {'x': 2, 'y': 3}
            center_random = {'x': True, 'y': True}
            if 'drill_num' in self.kwargs.keys():
                drill_num = self.kwargs['drill_num']
            if 'sample_type' in self.kwargs.keys():
                sample_type = self.kwargs['sample_type']
            if 'center_random' in self.kwargs.keys():
                center_random = self.kwargs['center_random']
            if 'drill_pos' in self.kwargs.keys():
                drill_pos = self.kwargs['drill_pos']
            dgl_graph = geodata.execute(sample_operator=self.sample_operator, extent=model_extern,
                                        edge_feat=self.dgl_graph_param[1], node_feat=self.dgl_graph_param[0],  #
                                        center_random=center_random,
                                        sample_type=sample_type, drill_num=drill_num,
                                        normalize=True)  # , drill_pos=drill_pos
            # 对标签进行处理
            label_num = geodata.grid_point_label_num
            predict_labels_num_list.append(label_num)
            is_connected = geodata.is_connected_graph()
            print('is_connected:', is_connected)
            # 直接尾部追加
            self.geodata.append(geodata)
            predict_dgl_graph_list.append(dgl_graph)
            self.update_graph_log(model_name=self.train_model_list[train_mdl_index],
                                  save_idx=0, extern=model_extern,
                                  node_feat=self.dgl_graph_param[0], edge_feat=self.dgl_graph_param[1],
                                  is_pre_train=False)
        self.predict_graph.extend(predict_dgl_graph_list)
        if torch.is_tensor(self.predict_num_classes['labels']):
            self.predict_num_classes['labels'] = self.predict_num_classes['labels'].numpy().tolist()
        self.predict_num_classes['labels'].extend(predict_labels_num_list)
        self.predict_num_classes = {'labels': torch.tensor(self.predict_num_classes['labels']).to(torch.long)}

    def process_dat_file(self, **kwargs):
        geodata = GeoMeshParse()
        regular_grid_type = None
        if 'regular_grid_type' in kwargs.keys():
            regular_grid_type = kwargs['regular_grid_type']
        geodata.set_data_from_dat_file(dat_file_path=self.dat_file_path, data_type=self.data_type, **kwargs)
        dgl_graph = geodata.execute(sample_operator=self.sample_operator,
                                    edge_feat=self.dgl_graph_param[1], node_feat=self.dgl_graph_param[0],  #
                                    center_random={'x': True, 'y': True},
                                    sample_type={'x': 2, 'y': 3}, drill_num=15, normalize=True,
                                    regular_grid_type=regular_grid_type)
        # 对标签进行处理
        label_num = geodata.grid_point_label_num

        is_connected = geodata.is_connected_graph()
        print('is_connected:', is_connected)
        self.geodata.append(geodata)
        save_idx = len(self.predict_graph)
        self.update_graph_log(model_name=geodata.name,
                              save_idx=save_idx, extern=geodata.sample_grid_extent,
                              node_feat=self.dgl_graph_param[0], edge_feat=self.dgl_graph_param[1],
                              is_pre_train=False)
        self.predict_graph.append(dgl_graph)
        if torch.is_tensor(self.predict_num_classes['labels']):
            self.predict_num_classes['labels'] = self.predict_num_classes['labels'].numpy().tolist()
        self.predict_num_classes['labels'].append(label_num)
        self.predict_num_classes = {'labels': torch.tensor(self.predict_num_classes['labels']).to(torch.long)}

    def get_split_idx(self, idx):
        train, valid, test = None, None, None
        if idx < len(self.graph):
            node_num = self.graph[idx].num_nodes()
        else:
            save_idx = idx - len(self.graph)
            node_num = self.predict_graph[save_idx].num_nodes()
        x = np.arange(node_num)
        if self.geodata[idx].train_idx is None:
            # random_state 确保每次切分是确定的
            train, val_test = train_test_split(x, test_size=0.4, random_state=2)
            valid, test = train_test_split(val_test, test_size=0.5, random_state=2)
        else:
            print('get split Dataset')
            train_val = np.array(self.geodata[idx].train_idx)
            train, valid = train_test_split(train_val, test_size=0.1)  # , random_state=2
            test = np.array(list(set(x) - set(self.geodata[idx].train_idx)))
            # train = train_val
            # val_test = np.array(list(set(x) - set(self.geodata[idx].train_idx)))
            # valid, test = train_test_split(val_test, test_size=0.1, random_state=2)
        train_idx = torch.from_numpy(train)
        valid_idx = torch.from_numpy(valid)
        test_idx = torch.from_numpy(test)
        return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    # 更新self.graph_log 添加dgl_graph数据信息       extern,      node_feat,   edge_feat
    # {'pre_train_graph': {save_idx: [graph_name, [nx, ny, nz], ['position'], ['euclidean']]}
    #  'predict_graph': [......]}
    # save_idx 与 goedata 中的索引 geo_idx 对应，也与 graph中的索引对应
    # pre_train_graph中的save_idx=geo_idx, 与geodata的索引直接对应，而predict_graph中的save_idx=geo_idx-len(pre_train_graph)
    def update_graph_log(self, model_name=None, save_idx=0, extern=None, node_feat=None,
                         edge_feat=None, is_pre_train=True):
        graph_name = model_name + '_' + str(int(time.time()))
        if is_pre_train is True:
            if 'pre_train_graph' not in self.graph_log.keys():
                self.graph_log['pre_train_graph'] = {}
            if save_idx not in self.graph_log['pre_train_graph']:
                self.graph_log['pre_train_graph'][save_idx] = []
            self.graph_log['pre_train_graph'][save_idx] = [graph_name, extern, node_feat, edge_feat]
        else:
            if 'predict_graph' not in self.graph_log.keys():
                self.graph_log['predict_graph'] = {}
            if save_idx not in self.graph_log['predict_graph']:
                self.graph_log['predict_graph'][save_idx] = []
            self.graph_log['predict_graph'][save_idx] = [graph_name, extern, node_feat, edge_feat]

    # 修改训练集比例
    def change_train_idx_pro(self, model_index, sample_operator, replace=False, **kwargs):
        graph_num = len(self.graph) + len(self.predict_graph)
        if model_index < graph_num:
            if self.geodata[model_index].is_noddy:
                x = np.arange(len(self.geodata[model_index].grid_points))
                known_pro = self.geodata[model_index].known_pro
                print('Changing Graph Data train_idx ...')
                print('The known_pro before is {}.'.format(known_pro))
                extent = self.geodata[model_index].output_grid_param

                self.geodata[model_index].set_visual_geo_sample(extent=extent, sample_operator=sample_operator,
                                                                is_noddy=True, is_update=True, **kwargs)
                # 替换并存储
                if replace:
                    self.save_geodata()
                    self.geodata = self.load_geodata()
            else:
                raise ValueError

    # 获取采样格网尺寸
    def get_model_extern(self, model_name, default_extern=None):
        if default_extern is None:
            default_extern = [80, 80, 40]
        if self.model_extern is None:
            model_param_list_log = self.noddy_data.model_param_list_log
            if model_name in model_param_list_log.keys():
                model_extern = model_param_list_log[model_name][:3]
            else:
                model_extern = default_extern
        else:
            model_extern = self.model_extern
        total_size = model_extern[0] * model_extern[1] * model_extern[2]
        if total_size > 1000000:
            model_extern = default_extern
        return model_extern

    # 生成二维剖面网格，
    def generate_section_grid(self, model_idx, sample_axis='x', scroll_scale=0.5, drill_num=6, is_save=False):
        graph_num = len(self.graph) + len(self.predict_graph)
        if model_idx < graph_num:
            geodata = self.geodata[model_idx]
            section, section_point, section_point_label, drills, drill_pid, train_plot_data_type, train_plot_data = \
                geodata.generate_section2d_with_drills_test(sample_axis=sample_axis, scroll_scale=scroll_scale,
                                                            drill_num=drill_num)
            for sec in section:
                axis_labels = ['x', 'y', 'z']
                label_to_index = {label: index for index, label in enumerate(axis_labels)}
                ax_index = label_to_index[sample_axis.lower()]
                section_geodata = GeoMeshParse(sec, name=geodata.name + '_sec2d')
                section_geodata.sample_grid_extent = copy.deepcopy(geodata.sample_grid_extent)
                section_geodata.sample_grid_extent[ax_index] = 1

                section_geodata.sample_grid = sec
                section_geodata.grid_points = section_point
                section_geodata.grid_point_label = section_point_label
                section_geodata.train_idx = drill_pid
                # 获取二维剖面三角网格
                section_geodata.get_triangulate_edges_2d(axis_label=sample_axis)
                # 获取节点特征
                section_geodata.get_node_feat(node_feat='position')
                # 将采样数据参数记录下来
                section_geodata.train_plot_data_type = []
                section_geodata.train_plot_data = []
                section_geodata.train_plot_data_type.append(train_plot_data_type)
                section_geodata.train_plot_data.append(train_plot_data)
                # 构建dgl图结构网格
                sectiona_graph = section_geodata.create_dgl_graph(
                    edge_list=np.int64(section_geodata.edge_list).transpose(),
                    node_feat=section_geodata.node_feat,
                    edge_feat=section_geodata.edge_feat,
                    node_label=np.int64(section_geodata.grid_point_label),
                    self_loop=False, add_inverse_edge=True, normalize=True)
                # 将数据存入对象
                self.geodata.append(section_geodata)
                self.predict_graph.append(sectiona_graph)
                save_idx = len(self.predict_graph)
                self.update_graph_log(model_name=section_geodata.name,
                                      save_idx=save_idx, extern=section_geodata.sample_grid_extent,
                                      node_feat=self.dgl_graph_param[0], edge_feat=self.dgl_graph_param[1],
                                      is_pre_train=False)
                # 处理好图
                if torch.is_tensor(self.predict_num_classes['labels']):
                    self.predict_num_classes['labels'] = self.predict_num_classes['labels'].numpy().tolist()
                if model_idx < len(self.graph):
                    if torch.is_tensor(self.num_classes['labels']):
                        self.num_classes['labels'] = self.num_classes['labels'].numpy().tolist()
                    label_num = self.num_classes['labels'][model_idx]
                else:
                    label_num = self.predict_num_classes['labels'][model_idx - len(self.graph)]
                self.predict_num_classes['labels'].append(label_num)
                self.predict_num_classes = {'labels': torch.tensor(self.predict_num_classes['labels']).to(torch.long)}
                if is_save:
                    print('Saving...')
                    save_graphs(self.processed_predict_graph_path, self.predict_graph, self.predict_num_classes)
                    self.predict_graph, self.predict_num_classes = load_graphs(self.processed_predict_graph_path)
                    self.save_graph_log()
                    self.save_geodata()
                    self.geodata = self.load_geodata()
                    self.graph_log = self.load_graph_log()

    # 支持向量机
    def predict_with_machine_learning_method(self, model_idx, method='svm', cv=None, param=None, save_path=None):
        graph_num = len(self.graph) + len(self.predict_graph)
        if model_idx < graph_num:
            if param is None and method is 'svm':
                param = [{'kernel': ['rbf'], 'C': [1, 10, 100, 200, 500]}]  #
            if param is None and method is 'rf':
                param = [{'n_estimators': [50, 120, 160, 200, 250, 280, 300, 350, 400], 'max_depth': [2, 4, 6, 8, 10, 12, 14]}]
            if param is None and method is 'xgboost':
                param = [{'n_estimators': [50, 120, 160, 200, 250], 'max_depth': [2, 4, 6, 8, 10],
                          'learning_rate': [0.001, 0.01, 0.03]}]
            geodata = self.geodata[model_idx]
            label = np.int64(geodata.grid_point_label)
            label = np.squeeze(label)
            prediction = copy.deepcopy(label)

            split = self.get_split_idx(model_idx)
            train_idx = split['train']

            train_x = geodata.grid_points[train_idx]
            train_y = label[train_idx]
            test_idx = list(set(np.arange(len(geodata.grid_points))) - set(geodata.train_idx))

            # test_x = geodata.grid_points[test_idx]
            test_x = geodata.grid_points
            # 测试集真实标签
            # test_y = label[test_idx]
            test_y = label
            clf = None
            if method == 'svm':
                clf = GridSearchCV(estimator=svm.SVC(), param_grid=param, cv=cv)
            elif method == 'rf':
                clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param)
            elif method == 'xgboost':
                clf = GridSearchCV(estimator=XGBClassifier(), param_grid=param)
            print('Classifier computing ...')
            clf.fit(train_x, train_y)            # 输出测试集的预测结果
            clf_best = clf.best_estimator_
            predict_test_y = clf_best.predict(test_x)
            # 获得预测出的模型类别值集合，可用于可视化
            # prediction[test_idx] = predict_test_y
            prediction = predict_test_y
            print(classification_report(test_y, predict_test_y))
            accuracy = MF.accuracy(torch.tensor(predict_test_y[test_idx]), torch.tensor(test_y[test_idx]))
            print('================Test Accuracy {:.4f}================'.format(accuracy.item()))

            gen_mesh = mvk.generate_model_on_base_grid(geodata, prediction, save_path=save_path)
            mvk.visual_multiple_model(geodata.sample_grid, gen_mesh, camera=[1, 0, 0])
        else:
            raise ValueError

    # method = {'rbf', 'nearest', 'idw'}
    def predict_with_interpolate(self, model_idx, method='rbf', iso_list=None, stratum_match=None,
                                 save_path=None, **kwargs):
        graph_num = len(self.graph) + len(self.predict_graph)
        if model_idx < graph_num:
            geodata = self.geodata[model_idx]
            label = np.int64(geodata.grid_point_label)
            label = np.squeeze(label)
            prediction = np.float32(copy.deepcopy(label))
            # train_idx = geodata.extract_drills_layer_points()
            train_idx = geodata.train_idx
            train_x = geodata.grid_points[train_idx]
            train_y = label[train_idx]
            test_idx = list(set(np.arange(len(geodata.grid_points))) - set(train_idx))
            test_x = geodata.grid_points[test_idx]
            test_y = label[test_idx]
            predict_test_y = None
            if method == 'rbf':
                neighbors = None
                smoothing = 0
                degree = None
                if 'neighbors' in kwargs.keys():
                    neighbors = kwargs['neighbors']
                    smoothing = kwargs['neighbors']
                    degree = kwargs['degree']
                    # , neighbors=neighbors, smoothing=smoothing, degree=degree
                yflat = RBFInterpolator(train_x, train_y)(test_x)
                predict_test_y = torch.tensor(np.array(yflat))
            elif method == 'idw':
                # 暂时还不支持三维空间散点的IDW插值，将数据以文件形式到处，使用voxler软件处理
                return train_x, train_y, test_x, test_y, geodata.bound
            elif method == 'nearest':
                interp = NearestNDInterpolator(train_x, train_y)
                predict_value = interp(test_x)
                predict_test_y = torch.tensor(np.array(predict_value))
            if predict_test_y is None:
                raise ValueError
            prediction[test_idx] = predict_test_y
            sample_grid, contour, cell_mesh, cell_stratum = mvk.process_point_field_model(geodata.sample_grid,
                                                                                          cell_labels=prediction,
                                                                                          iso_list=iso_list,
                                                                                          stratum_match=stratum_match,
                                                                                          save_path=save_path)
            predict_labels = np.int64(cell_stratum)
            predict_labels_test = torch.tensor(predict_labels[test_idx])
            accuracy = MF.accuracy(predict_labels_test, torch.tensor(test_y))
            print('================Test Accuracy {:.4f}================'.format(accuracy.item()))

            tmp_drill_param = geodata.train_plot_data[0]
            drill_pos, drill_num = tmp_drill_param
            drills, _, _, _ = geodata.sample_with_drills(drill_pos=drill_pos)
            drills = mvk.drill_construct_tube(drills, drill_radius=25)
            mvk.visual_multiple_model([drills, contour], geodata.sample_grid, cell_mesh, sample_grid)

            ori_path = os.path.join(self.root, 'processed', 'ori.vtk')
            geodata.sample_grid.save(filename=ori_path)

            slice_x = mvk.clip_section_along_axis(cell_mesh, sample_axis='x')
            slice_y = mvk.clip_section_along_axis(cell_mesh, sample_axis='y')
            mvk.visual_multiple_model([drills, slice_x, slice_y])

        else:
            raise ValueError

    # 地质模型重采样，以提高分辨率
    ## 未写完
    def resample_geomodel(self, vtk_model_path, grid_extent):
        model = mvk.get_vtk_mesh_from_file(vtk_model_path)
        train_x = model.cell_centers().points
        train_y = model.active_scalars
        mvk.export_points_labels_dat_file(
            file_path=r'E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\points_data.dat',
            grid_points=train_x, grid_points_label=train_y)

    # 处理传入的插值属性场模型文件数据  文件类型是vtk
    def process_external_fields_model(self, intput_file_path, model_idx, iso_list, stratum_match=None, save_path=None):
        graph_num = len(self.graph) + len(self.predict_graph)
        if model_idx < graph_num:
            geodata = self.geodata[model_idx]
            if intput_file_path.endswith('.csv'):
                label = np.int64(geodata.grid_point_label)
                label = np.squeeze(label)
                cell_field_mesh = geodata.map_external_csv_data_to_base_grid(csv_path=intput_file_path,
                                                                             extent=[1, 120, 50])
                prediction = cell_field_mesh.active_scalars
                sample_grid, contour, cell_mesh, cell_stratum = mvk.process_point_field_model(cell_field_mesh,
                                                                                              cell_labels=prediction,
                                                                                              iso_list=iso_list,
                                                                                              stratum_match=stratum_match,
                                                                                              save_path=None)
            if intput_file_path.endswith('.vtk'):
                external_model = pv.read(intput_file_path)
                label = np.int64(geodata.grid_point_label)
                label = np.squeeze(label)
                cell_field_mesh = geodata.map_external_model_to_base_grid(external_model=external_model)
                prediction = cell_field_mesh.active_scalars
                sample_grid, contour, cell_mesh, cell_stratum = mvk.process_point_field_model(cell_field_mesh,
                                                                                          cell_labels=prediction,
                                                                                          iso_list=iso_list,
                                                                                          stratum_match=stratum_match,
                                                                                          save_path=None)
            drills = mvk.visual_sample_data(geodata, is_show=False, drill_radius=25)
            mvk.visual_multiple_model([drills[0], contour], geodata.sample_grid, cell_mesh, sample_grid, camera=[1,0,0])
            # geodata.sample_grid.save(filename=save_path)

            cell_mesh.save(filename=r'E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\idw_model.vtk')

            train_idx = geodata.extract_drills_layer_points()
            test_idx = list(set(np.arange(len(geodata.grid_points))) - set(train_idx))
            test_y = label[test_idx]
            predict_labels = np.int64(cell_stratum)
            predict_labels_test = torch.tensor(predict_labels[test_idx])
            accuracy = MF.accuracy(predict_labels_test, torch.tensor(test_y))
            print('================Test Accuracy {:.4f}================'.format(accuracy.item()))

    # 存储和加载函数
    def save_geodata(self):
        out_put = open(self.processed_geodata_path, 'wb')
        out_str = pickle.dumps(self.geodata)
        out_put.write(out_str)
        out_put.close()

    def load_geodata(self):
        with open(self.processed_geodata_path, 'rb') as file:
            geodata = pickle.loads(file.read())
            return geodata

    def save_graph_log(self):
        out_put = open(self.graph_log_data_path, 'wb')
        out_str = pickle.dumps(self.graph_log)
        out_put.write(out_str)
        out_put.close()

    def load_graph_log(self):
        with open(self.graph_log_data_path, 'rb') as file:
            out_put = pickle.loads(file.read())
            return out_put

    def save_result_model(self):
        out_put = open(self.result_model_path, 'wb')
        out_str = pickle.dumps(self.result_model)
        out_put.write(out_str)
        out_put.close()

    def load_result_model(self):
        with open(self.result_model_path, 'rb') as file:
            out_put = pickle.loads(file.read())
            return out_put

    def __getitem__(self, idx):
        if idx < len(self.graph):
            return self.graph[idx]
        else:
            idx = idx - len(self.graph)
            return self.predict_graph[idx]

    def __len__(self):
        if self.graph is None:
            self.graph = []
        if self.predict_graph is None:
            self.predict_graph = []
        return len(self.graph) + len(self.predict_graph)

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))
