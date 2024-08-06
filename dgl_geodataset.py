"""Dataset adapters for re-purposing a dataset for a different kind of training task."""

import os
import json
import numpy as np
import torch
from dgl.data import DGLDataset, utils
import dgl.backend as F
from geomodel_analysis import GmeModelGraphList


# DGL 图数据集封装，将图处理为dgl数据集
class DglGeoDataset(DGLDataset):

    def __init__(self, dataset: GmeModelGraphList, graph_id=0, split_ratio=None, **kwargs):
        self.graph = []
        self.dataset = dataset
        self.split_ratio = split_ratio
        if split_ratio is None:
            self.split_ratio = self.dataset.split_ratio
        else:
            if self.split_ratio != self.dataset.split_ratio:
                self.split_ratio = self.dataset.split_ratio  # 自定义验证集比例参数
        # self.split_ratio = None  # 这是dgl封装的数据集切分参数
        self.target_ntype = None
        self.num_classes = getattr(self.dataset, 'num_classes', None)
        self.graph_id = graph_id
        # self.predict_num_classes = getattr(self.dataset, 'predict_num_classes', None)
        if self.num_classes is not None:
            if torch.is_tensor(self.num_classes['labels']):
                self.num_classes['labels'] = self.num_classes['labels'].numpy().tolist()
            self.num_classes = {'labels': torch.tensor(self.num_classes['labels']).to(torch.long)}
        self.grid_data = []
        grid_data_num = len(self.dataset.geograph)
        #
        self.dataset.load_geograph(graph_id=self.graph_id)
        self.labels_count_map = self.dataset.geograph[self.graph_id].get_labels_count_map()
        self.processed_path = self.dataset.processed_dir
        self.grid_data_path = None
        if grid_data_num > self.graph_id:
            self.dataset.load_geograph(graph_id=self.graph_id)
            file_name = self.dataset.geograph[self.graph_id].data.name
            self.grid_data_path = os.path.join(self.dataset.geograph[self.graph_id].data.dir_path, file_name
                                               , file_name + '.dat')
        self.train_idx = []
        self.val_idx = []
        self.test_idx = []
        super().__init__(self.dataset.name + '-as-nodepred', **kwargs)
        # super().__init__(self.dataset.name + '-as-nodepred',
        #                  hash_key=(self.split_ratio, self.target_ntype, dataset.name, 'nodepred'), **kwargs)
        self.dataset = None

    def process(self):
        self.graph.clear()
        if self.graph_id < self.dataset.__len__():
            self.dataset.load_dglgraph(graph_id=self.graph_id)
            g = self.dataset[self.graph_id].clone()
            self.graph.append(g)
            train_idx = []
            val_idx = []
            test_idx = []
            self.train_idx.append(train_idx)
            self.val_idx.append(val_idx)
            self.test_idx.append(test_idx)

            if self.num_classes is None:
                for g_idx in np.arange(self.dataset.__len__()):
                    class_num = len(F.unique(self.dataset[g_idx].nodes[self.target_ntype].data['label']))
                    self.num_classes.append(class_num)

            if 'label' not in self.graph[0].nodes[self.target_ntype].data:
                raise ValueError("Missing node labels. Make sure labels are stored "
                                 "under name 'label'.")
            if self.split_ratio is not None:
                split = self.dataset.get_split_idx(self.graph_id, split_ratio=self.split_ratio)
                train_idx, val_idx, test_idx = split['train'], split['valid'], split['test']
                n = self.graph[0].num_nodes()
                train_mask = utils.generate_mask_tensor(utils.idx2mask(train_idx, n))
                val_mask = utils.generate_mask_tensor(utils.idx2mask(val_idx, n))
                test_mask = utils.generate_mask_tensor(utils.idx2mask(test_idx, n))
                self.graph[0].ndata['train_mask'] = train_mask
                self.graph[0].ndata['val_mask'] = val_mask
                self.graph[0].ndata['test_mask'] = test_mask
            else:
                # if self.verbose:
                #     print('Generating train/val/test masks...')
                # utils.add_nodepred_split(self, self.split_ratio, self.target_ntype)
                pass
            self._set_split_index(0)

    def __getitem__(self, idx):
        return self.graph[idx]

    def __len__(self):
        return len(self.graph)

    def save(self):
        pass

    def load(self):
        with open(os.path.join(self.save_path, 'info_{}.json'.format(self.hash)), 'r') as f:
            info = json.load(f)
            if (info['split_ratio'] != self.split_ratio
                    or info['target_ntype'] != self.target_ntype):
                raise ValueError('Provided split ratio is different from the cached file. '
                                 'Re-process the dataset.')
            self.split_ratio = info['split_ratio']
            self.target_ntype = info['target_ntype']
            self.num_classes = {'labels': torch.Tensor(info['num_classes']).to(torch.long)}
        gs, _ = utils.load_graphs(os.path.join(self.save_path, 'graph_{}.bin'.format(self.hash)))
        self.graph = gs
        for idx in np.arange(len(self.graph)):
            self._set_split_index(idx)

    def has_cache(self):
        return os.path.isfile(os.path.join(self.save_path, 'graph_{}.bin'.format(self.hash)))

    def _set_split_index(self, idx):
        """Add train_idx/val_idx/test_idx as dataset attributes according to corresponding mask."""
        ndata = self.graph[idx].nodes[self.target_ntype].data
        self.train_idx[idx] = F.nonzero_1d(ndata['train_mask'])
        self.val_idx[idx] = F.nonzero_1d(ndata['val_mask'])
        self.test_idx[idx] = F.nonzero_1d(ndata['test_mask'])
