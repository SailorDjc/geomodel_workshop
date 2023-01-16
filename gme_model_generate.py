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
                 model_extern=None):
        #
        if train_model_list is None:   # 一般只装一个model
            train_model_list = []
        if pre_train_model_list is None:  # 预训练模型， 列表中有多个模型
            pre_train_model_list = []

        if sample_operator is None:
            sample_operator = ['regu_matrix']
        self.noddy_data = noddy_data
        self.geodata = []
        self.pre_train_model_list = pre_train_model_list  # 预训练
        self.train_model_list = train_model_list  # 训练任务数据
        self.model_extern = model_extern
        self.sample_operator = sample_operator  # ['rand_drills', 'axis_sections']
        self.self_loop = self_loop
        self.add_inverse_edge = add_inverse_edge
        self.root = root
        self.name = name
        self.num_classes = None
        super(GmeModelList, self).__init__()
        self.access_model_data()

    def access_model_data(self):
        self.process()

    def process(self):
        processed_dir = os.path.join(self.root, 'processed')
        processed_file_path = os.path.join(processed_dir, 'gme_data_processed')
        processed_geodata_path = os.path.join(self.root, 'processed', 'geodata.pkl')
        # 如果存在处理后数据，则直接从文件加载
        if os.path.exists(processed_file_path):
            self.graph, self.num_classes = load_graphs(processed_file_path)
            if os.path.exists(processed_geodata_path):
                self.geodata = self.load_geodata()
        else:
            # 批量处理每一个地质模型数据文件
            dgl_graph_list = []
            if self.model_extern is None:
                self.model_extern = [100, 100, 30]
            labels_num_list = []
            for model_idex in np.arange(len(self.pre_train_model_list)):
                mesh = self.noddy_data.get_grid_model(self.pre_train_model_list[model_idex])

                geodata = GeoMeshParse(mesh, normalize=False)

                dgl_graph = geodata.execute(sample_operator=self.sample_operator, extent=self.model_extern,
                                            edge_feat=['euclidean'], node_feat=['stratum'],  #
                                            center_random={'x': True, 'y': True},
                                            sample_type={'x': 2, 'y': 3}, drill_num=15)
                # 对标签进行处理
                label_num = geodata.sample_label_num
                labels_num_list.append(label_num)

                is_connected = geodata.is_connected_graph()
                print('is_connected:', is_connected)
                self.geodata.append(geodata)
                dgl_graph_list.append(dgl_graph)
            if self.train_model_list is not None:
                mesh = self.noddy_data.get_grid_model(self.train_model_list[0])
                geodata = GeoMeshParse(mesh, normalize=False, pre_train=False)
                dgl_graph = geodata.execute(sample_operator=self.sample_operator, extent=self.model_extern,
                                            edge_feat=['euclidean'], node_feat=['stratum'],  #
                                            center_random={'x': True, 'y': True},
                                            sample_type={'x': 2, 'y': 3}, drill_num=15)
                # 对标签进行处理
                label_num = geodata.sample_label_num
                labels_num_list.append(label_num)

                is_connected = geodata.is_connected_graph()
                print('is_connected:', is_connected)
                self.geodata.append(geodata)
                dgl_graph_list.append(dgl_graph)

            labels_num_dict = {'labels': torch.tensor(labels_num_list).to(torch.long)}
            print('Saving...')
            save_graphs(processed_file_path, dgl_graph_list, labels_num_dict)
            self.save_geodata()
            self.graph, self.num_classes = load_graphs(processed_file_path)
            self.geodata = self.load_geodata()

    def get_split_idx(self, idx):
        train, valid, test = None, None, None
        node_num = self.graph[idx].num_nodes()
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

    def save_geodata(self):
        out_put = open(os.path.join(self.root, 'processed', 'geodata.pkl'), 'wb')
        out_str = pickle.dumps(self.geodata)
        out_put.write(out_str)
        out_put.close()

    def load_geodata(self):
        with open(os.path.join(self.root, 'processed', 'geodata.pkl'), 'rb') as file:
            geodata = pickle.loads(file.read())
            return geodata

    def __getitem__(self, idx):
        return self.graph[idx]

    def __len__(self):
        return len(self.graph)

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))
