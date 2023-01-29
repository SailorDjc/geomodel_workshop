import pickle

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
            dgl_graph_param = [['stratum'], None]  # [[node_feat], [edge_feat]]
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
        # 如果 model_extern为None,则使用原始Noddy数据集中的格网尺寸
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
        self.graph, self.predict_graph = None, None
        super(GmeModelList, self).__init__()
        self.access_model_data(**kwargs)

    def access_model_data(self, **kwargs):
        # 如果存在处理后数据，则直接从文件加载
        if os.path.exists(self.graph_log_data_path):
            self.graph_log = self.load_graph_log()
        # 加载dgl_graph和geodata, 其中dgl_graph是图神经网络训练的数据源, geodata主要存储地质数据源用来可视化分析
        if os.path.exists(self.processed_file_path):
            self.graph, self.num_classes = load_graphs(self.processed_file_path)  # dgl_graph
            print('Loading {} Saved Pretrain Graphs Files Data ...'.format(len(self.graph)))
            if os.path.exists(self.processed_geodata_path):  # geodata
                self.geodata = self.load_geodata()
                print('Loading {} GeoModel Data ...'.format(len(self.geodata)))
            # dgl_graph 处理后dgl图数据日志字典, 用来判断是否添加新的数据
            if os.path.exists(self.graph_log_data_path):
                # self.graph_log已经加载过了
                for idx, _ in enumerate(self.graph):
                    if idx in self.graph_log['pre_train_graph'].keys():
                        graph_name = self.graph_log['pre_train_graph'][idx][0]  # graph_name图名：model_name+'_'+timecode
                        model_name = graph_name.split('_')[0]
                        model_extern = self.graph_log['pre_train_graph'][idx][1]  # 采样格网尺寸
                        node_feat = self.graph_log['pre_train_graph'][idx][2]
                        edge_feat = self.graph_log['pre_train_graph'][idx][3]

                        cur_extern = self.get_model_extern(model_name)
                        cur_node_feat = self.dgl_graph_param[0]
                        cur_edge_feat = self.dgl_graph_param[1]
                        # 删除重复加载的数据
                        for model_name in self.pre_train_model_list:
                            if model_extern == cur_extern and cur_node_feat == node_feat \
                                    and edge_feat == cur_edge_feat:
                                self.pre_train_model_list.remove(model_name)
        # 预测数据集
        if os.path.exists(self.processed_predict_graph_path):
            self.predict_graph, self.predict_num_classes = load_graphs(self.processed_predict_graph_path)
            print('Loading {} Saved Predict Graphs Files Data ...'.format(len(self.predict_graph)))
            if os.path.exists(self.graph_log_data_path):
                for idx, _ in enumerate(self.predict_graph):
                    if idx in self.graph_log['predict_graph'].keys():
                        graph_name = self.graph_log['predict_graph'][idx][0]  # graph_name图名：model_name+'_'+timecode
                        model_name = graph_name.split('_')[0]
                        model_extern = self.graph_log['predict_graph'][idx][1]  # 采样格网尺寸
                        node_feat = self.graph_log['predict_graph'][idx][2]
                        edge_feat = self.graph_log['predict_graph'][idx][3]

                        cur_extern = self.get_model_extern(model_name)
                        cur_node_feat = self.dgl_graph_param[0]
                        cur_edge_feat = self.dgl_graph_param[1]
                        # 删除重复加载的数据
                        for model_name in self.train_model_list:
                            if model_extern == cur_extern and cur_node_feat == node_feat \
                                    and edge_feat == cur_edge_feat:
                                self.train_model_list.remove(model_name)
        if self.data_type == 'Noddy':
            if len(self.pre_train_model_list) > 0 or len(self.train_model_list) > 0:
                self.process_noddy_data()
        elif self.data_type == 'Wells' or self.data_type == 'Points':
            if self.dat_file_path is not None:
                self.process_dat_file(**kwargs)
        # 存储geodata和graph_log
        self.save_graph_log()
        self.save_geodata()
        self.geodata = self.load_geodata()
        self.graph_log = self.load_graph_log()

    def process_noddy_data(self):
        # 批量处理每一个地质模型数据文件
        dgl_graph_list = []
        labels_num_list = []
        # 容器初始化
        if self.graph is None:
            self.graph = []
            self.num_classes = {'labels': []}
        if self.predict_graph is None:
            self.predict_graph = []
            self.predict_num_classes = {'labels': []}
        for model_index in np.arange(len(self.pre_train_model_list)):
            mesh = self.noddy_data.get_grid_model(self.pre_train_model_list[model_index])
            model_extern = self.get_model_extern(self.pre_train_model_list[model_index])

            geodata = GeoMeshParse(mesh, name=self.pre_train_model_list[model_index], normalize=False)
            dgl_graph = geodata.execute(sample_operator=self.sample_operator, extent=model_extern,
                                        edge_feat=self.dgl_graph_param[1], node_feat=self.dgl_graph_param[0],
                                        center_random={'x': True, 'y': True},
                                        sample_type={'x': 2, 'y': 3}, drill_num=15)
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
        labels_num_dict = {'labels': torch.tensor(self.num_classes['labels']).to(torch.long)}
        if len(self.graph) > 0:
            print('Saving...')
            save_graphs(self.processed_file_path, self.graph, labels_num_dict)
            self.graph, self.num_classes = load_graphs(self.processed_file_path)
        # 最后的工程预测数据
        predict_dgl_graph_list = []
        predict_labels_num_list = []
        for train_mdl_index in np.arange(len(self.train_model_list)):
            mesh = self.noddy_data.get_grid_model(self.train_model_list[train_mdl_index])
            model_extern = self.get_model_extern(self.train_model_list[train_mdl_index])

            geodata = GeoMeshParse(mesh, name=self.train_model_list[train_mdl_index], normalize=False, pre_train=False)
            dgl_graph = geodata.execute(sample_operator=self.sample_operator, extent=model_extern,
                                        edge_feat=self.dgl_graph_param[1], node_feat=self.dgl_graph_param[0],  #
                                        center_random={'x': True, 'y': True},
                                        sample_type={'x': 2, 'y': 3}, drill_num=15)
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
        predict_labels_num_dict = {'labels': torch.tensor(self.predict_num_classes['labels']).to(torch.long)}
        if len(self.predict_graph) > 0:
            print('Saving...')
            save_graphs(self.processed_predict_graph_path, self.predict_graph, predict_labels_num_dict)
            self.predict_graph, self.predict_num_classes = load_graphs(self.processed_predict_graph_path)

    def process_dat_file(self, **kwargs):
        geodata = GeoMeshParse(normalize=False, pre_train=False)
        if self.data_type != 'Noddy':
            geodata.set_data_from_dat_file(dat_file_path=self.dat_file_path, data_type=self.data_type, **kwargs)
        dgl_graph = geodata.execute(sample_operator=self.sample_operator,
                                    edge_feat=self.dgl_graph_param[1], node_feat=self.dgl_graph_param[0],  #
                                    center_random={'x': True, 'y': True},
                                    sample_type={'x': 2, 'y': 3}, drill_num=15)
        # 对标签进行处理
        label_num = geodata.grid_point_label_num

        is_connected = geodata.is_connected_graph()
        print('is_connected:', is_connected)
        self.geodata.append(geodata)
        save_idx = len(self.predict_graph)
        self.update_graph_log(model_name=geodata.name,
                              save_idx=save_idx, extern=geodata.output_grid_param,
                              node_feat=self.dgl_graph_param[0], edge_feat=self.dgl_graph_param[1],
                              is_pre_train=False)
        if self.predict_graph is None:
            self.predict_graph = []
            self.predict_num_classes = {'labels': []}
        self.predict_graph.append(dgl_graph)
        if torch.is_tensor(self.predict_num_classes['labels']):
            self.predict_num_classes['labels'] = self.predict_num_classes['labels'].numpy().tolist()
        self.predict_num_classes['labels'].append(label_num)
        predict_labels_num_dict = {'labels': torch.tensor(self.predict_num_classes['labels']).to(torch.long)}
        print('Saving...')
        save_graphs(self.processed_predict_graph_path, self.predict_graph, predict_labels_num_dict)
        self.predict_graph, self.predict_num_classes = load_graphs(self.processed_predict_graph_path)

    def get_split_idx(self, idx):
        train, valid, test = None, None, None
        if idx < len(self.graph):
            node_num = self.graph[idx].num_nodes()
        else:
            save_idx = idx - len(self.graph)
            node_num = self.predict_graph[save_idx].num_nodes()
        x = np.arange(node_num)
        if self.geodata[idx].train_idx is None:
            train, val_test = train_test_split(x, test_size=0.4)
            valid, test = train_test_split(val_test, test_size=0.5)
        else:
            train_val = np.array(self.geodata[idx].train_idx)
            train, valid = train_test_split(train_val, test_size=0.2)
            test = np.array(list(set(x) - set(self.geodata[idx].train_idx)))
        train_idx = torch.from_numpy(train)
        valid_idx = torch.from_numpy(valid)
        test_idx = torch.from_numpy(test)
        return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    # 更新self.graph_log 添加dgl_graph数据信息       extern,      node_feat,   edge_feat
    # {'pre_train_graph': {save_idx: [graph_name, [nx, ny, nz], ['stratum'], ['euclidean']]}
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

    # 获取采样格网尺寸
    def get_model_extern(self, model_name, default_extern=None):
        if default_extern is None:
            default_extern = [120, 120, 50]
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
