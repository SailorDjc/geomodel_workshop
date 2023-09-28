"""Dataset adapters for re-purposing a dataset for a different kind of training task."""

import os
import json
import numpy as np
import torch
from dgl.data import DGLDataset, utils
import dgl.backend as F


class DglGeoDataset(DGLDataset):

    def __init__(self, dataset, split_ratio=None, target_ntype=None, **kwargs):
        self.graph = []
        self.dataset = dataset
        self.split_ratio = split_ratio
        self.target_ntype = target_ntype
        self.num_classes = getattr(self.dataset, 'num_classes', None)
        self.predict_num_classes = getattr(self.dataset, 'predict_num_classes', None)
        if self.predict_num_classes is not None and self.num_classes is not None:
            if torch.is_tensor(self.num_classes['labels']):
                self.num_classes['labels'] = self.num_classes['labels'].numpy().tolist()
            if torch.is_tensor(self.predict_num_classes['labels']):
                self.predict_num_classes['labels'] = self.predict_num_classes['labels'].numpy().tolist()
            self.num_classes['labels'].extend(self.predict_num_classes['labels'])
            self.num_classes = {'labels': torch.tensor(self.num_classes['labels']).to(torch.long)}
            self.predict_num_classes = {'labels': torch.tensor(self.predict_num_classes['labels']).to(torch.long)}
        self.train_idx = []
        self.val_idx = []
        self.test_idx = []
        super().__init__(self.dataset.name + '-as-nodepred',
                         hash_key=(split_ratio, target_ntype, dataset.name, 'nodepred'), **kwargs)

    def process(self):
        self.graph.clear()
        for g_idx in np.arange(self.dataset.__len__()):
            g = self.dataset[g_idx].clone()
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

        for g_idx in np.arange(self.dataset.__len__()):
            if 'label' not in self.graph[g_idx].nodes[self.target_ntype].data:
                raise ValueError("Missing node labels. Make sure labels are stored "
                                 "under name 'label'.")
            if self.split_ratio is None:
                split = self.dataset.get_split_idx(g_idx)
                train_idx, val_idx, test_idx = split['train'], split['valid'], split['test']
                n = self.graph[g_idx].num_nodes()
                train_mask = utils.generate_mask_tensor(utils.idx2mask(train_idx, n))
                val_mask = utils.generate_mask_tensor(utils.idx2mask(val_idx, n))
                test_mask = utils.generate_mask_tensor(utils.idx2mask(test_idx, n))
                self.graph[g_idx].ndata['train_mask'] = train_mask
                self.graph[g_idx].ndata['val_mask'] = val_mask
                self.graph[g_idx].ndata['test_mask'] = test_mask
            else:
                if self.verbose:
                    print('Generating train/val/test masks...')
                utils.add_nodepred_split(self, self.split_ratio, self.target_ntype)
            self._set_split_index(g_idx)

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
