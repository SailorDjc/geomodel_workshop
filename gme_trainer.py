import copy
import os
import math
import logging
import time

import dgl
import pyvista
from tqdm import tqdm
import numpy as np
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
import torchmetrics
from utils.plot_utils import visual_loss_picture, visual_acc_picture
import utils.plot_utils as mvk
from data_structure.geodata import load_object
# import model_visual_kit as mvk
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from models.loss import FocalLoss

logger = logging.getLogger(__name__)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class GraphTransConfig:

    # config.vocab_size, config.n_embd  config.embd_pdrop  n_layer out_size resid_pdrop n_head attn_pdrop
    # , vocab_size
    def __init__(self, in_size, out_size, embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, n_layer=12,
                 gnn_layer_num=4, coors=3, n_head=2, gnn_n_head=2, n_embd=512):
        self.in_size = in_size  # 输入特征维度
        # self.vocab_size = vocab_size  # 句子长度
        self.coors = coors
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.n_layer = n_layer  # Transformer层数
        self.gnn_n_layer = gnn_layer_num  # gnn层数
        self.gnn_n_head = gnn_n_head
        self.n_head = n_head  # Transformer中多头注意力的head_num
        self.n_embd = n_embd  # 隐藏层维度
        self.out_size = out_size  # 输出维度，分类数


class Rmse(torchmetrics.Metric):

    def __int__(self):
        self.add_state("sum_squared_errors", torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_observations", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        self.sum_squared_errors += torch.sum((preds - target) ** 2)
        self.n_observations += preds.numel()

    def compute(self):
        return torch.sqrt(self.sum_squared_errors / self.n_observations)


# ckpt_path: 模型训练参数的保存路径

class GmeTrainerConfig:
    # optimization parameters
    device = 'cpu'

    def __init__(self, max_epochs=10, batch_size=512, learning_rate=1e-3, weight_decay=1e-4, lr_decay=False,
                 ckpt_path=None, tokens=0, num_workers=4, output_dir=None, sample_neigh=None, gpu_num=1, **kwargs):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # learning rate decay params: linear warmup followed by cosine decay to 10% of original
        self.weight_decay = weight_decay  # L2正则化项
        self.lr_decay = lr_decay
        # checkpoint settings
        self.ckpt_path = ckpt_path
        self.tokens = tokens
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.sample_neigh = sample_neigh
        self.gpu_num = gpu_num
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if torch.cuda.is_available() and gpu_num > 0:
            self.device = torch.cuda.current_device()
        self.kwargs = kwargs


# 早停
class EarlyStopping:
    def __init__(self, patience=25, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}.')


class GmeTrainer:

    # gme_dataset: GeoDataset
    # model
    # config: GmeTrainerConfig

    def __init__(self, model, gme_dataset, config: GmeTrainerConfig):
        self.custom_loss = None
        self.model = model
        self.optimizer = None
        self.gme_dataset = gme_dataset
        self.label_dict = None
        # 处理标签
        self.labels_count_map = None
        self.preprocess_labels()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        # config param
        self.config = config
        self.lr = self.config.learning_rate
        self.batch_size = self.config.batch_size

        self.ckpt_path = self.config.ckpt_path
        self.sample_neigh = config.sample_neigh
        self.weight_decay = config.weight_decay
        self.max_epochs = config.max_epochs
        #
        self.best_loss = float('inf')
        self.log_name = None
        self.iter_record_path = None
        # take over whatever gpus are on the system
        self.device = config.device
        if 1 < config.gpu_num <= torch.cuda.current_device():
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device).cuda()
        else:
            self.model = self.model.to(self.device)

    def preprocess_labels(self, default_value=-1):
        labels = self.gme_dataset[0].ndata['label']

        unique_labels = torch.unique(labels, sorted=True)
        self.label_dict = {}
        idx = 0
        for item in unique_labels:
            if item == default_value:  # 对于默认未知值则不改变
                continue
            self.label_dict[item] = idx
            idx += 1
        labels_count_map = self.gme_dataset.labels_count_map
        if labels_count_map is not None:
            new_labels_count_map = {}
            for k, v in self.label_dict.items():
                kk = k.item()
                new_labels_count_map[v] = labels_count_map[kk]
            self.labels_count_map = copy.deepcopy(new_labels_count_map)
        self.gme_dataset[0].ndata['label'] = self.updata_labels_by_dict(labels=labels, label_dict=self.label_dict)

    @staticmethod
    def updata_labels_by_dict(labels, label_dict):
        indices_dict = {}
        for k, v in label_dict.items():
            indices = torch.nonzero(torch.eq(labels, k))
            indices_dict[k] = torch.flatten(indices)
        for k, v in label_dict.items():
            labels.index_fill_(0, indices_dict[k], v)
        return labels

    def preprocess_input(self, data, idx=0):
        # if idx >= data.__len__():
        #     raise ValueError("索引超出范围")
        g = data[idx]
        g = g.to(self.device)
        # 获取 训练集与验证集的 节点索引
        train_idx = data.train_idx[idx].to(self.device)
        val_idx = data.val_idx[idx].to(self.device)
        test_idx = data.test_idx[idx].to(self.device)
        # 采样器
        sampler = NeighborSampler(self.sample_neigh,  # fanout for [layer-0, layer-1, layer-2]
                                  prefetch_node_feats=['feat'],
                                  prefetch_labels=['label'])

        train_dataloader = dgl.dataloading.DataLoader(g, train_idx, sampler, device=self.device,
                                                      batch_size=self.config.batch_size, shuffle=True,
                                                      drop_last=False)
        valid_dataloader = dgl.dataloading.DataLoader(g, val_idx, sampler, device=self.device,
                                                      batch_size=self.batch_size, shuffle=True,
                                                      drop_last=False)  # num_workers=0, use_uva=False
        test_dataloader = None
        if test_idx is not None and len(test_idx) > 0:
            test_dataloader = dgl.dataloading.DataLoader(g, test_idx, sampler, device=self.device,
                                                         batch_size=self.batch_size, shuffle=True,
                                                         drop_last=False)
        return train_dataloader, valid_dataloader, test_dataloader

    def load_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        optimizer = torch.optim.Adam(raw_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if os.path.exists(self.config.ckpt_path):
            pretrain_dict = torch.load(self.config.ckpt_path, map_location='cpu')
            # 当前网络模型参数
            model_dict = raw_model.state_dict()
            # 更新参数，保存的模型参数与当前网络参数有部分不同
            # pretrain_dict = {k: v for k, v in pretrain_dict.items() if (k in model_dict and 'p_layer' not in k)}
            model_dict.update(pretrain_dict['model_state_dict'])
            self.best_loss = pretrain_dict['best_loss']
            raw_model.load_state_dict(model_dict)
            print("loading ", self.config.ckpt_path)
            optimizer.load_state_dict(pretrain_dict['optimizer_state_dict'])
        return raw_model, optimizer

    def save_checkpoint(self, optimizer):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        print("saving ", self.config.ckpt_path)
        torch.save({
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': self.best_loss
        }, self.config.ckpt_path)

    # data
    def inference(self, data, idx, has_test_label=True, is_show=False, save_path=None):
        _, _, self.test_dataset = self.preprocess_input(self.gme_dataset, idx)
        model = self.model.module if hasattr(self.model, "module") else self.model
        graph = data[0].to(self.device)
        nodes = torch.arange(graph.number_of_nodes())
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(model.config.gnn_n_layer,
                                                                prefetch_node_feats=['feat'],
                                                                prefetch_labels=['label'])
        total_dataloader = dgl.dataloading.DataLoader(
            graph, nodes.to(graph.device), sampler, device=self.device,
            batch_size=self.config.batch_size, shuffle=True,
            drop_last=False
        )
        model.eval()
        with torch.no_grad():
            pred = torch.empty(graph.number_of_nodes(), model.config.out_size, device=graph.device)
            for input_nodes, output_nodes, blocks in tqdm(total_dataloader):
                x = blocks[0].srcdata['feat']
                y_hat = model(blocks, x)
                pred[output_nodes] = y_hat.to(graph.device)
            grid_data = load_object(data.grid_data_path)
            grid_data.label_dict = self.label_dict
            mvk.visual_predicted_values_model(grid_data, pred, is_show=is_show, save_path=save_path)
            if has_test_label and self.test_dataset is not None:
                # test_idx = data.test_idx[idx]
                # pred_test = pred[test_idx]
                # label_test = graph.ndata['label'][test_idx].to(pred_test.device)
                # accuracy = MF.accuracy(pred_test, label_test, task='multiclass'
                #                        , num_classes=int(self.gme_dataset.num_classes['labels'][idx]))
                test_loss, test_acc, test_rmse = self.test(data_idx=idx)

                message = '==============Test Accuracy {:.4f} Loss {: .4f} Rmse {: .4f}=============' \
                    .format(test_acc, test_loss, test_rmse)
                return message
            else:
                # 用验证集精度替代
                val_idx = data.val_idx[0]
                pred_val = pred[val_idx]
                label_val = graph.ndata['label'][val_idx].to(pred_val.device)
                accuracy = MF.accuracy(pred_val, label_val, task='multiclass'
                                       , num_classes=int(self.gme_dataset.num_classes['labels'][idx]))
                message = '================Test Accuracy {:.4f}================' \
                    .format(accuracy.item())
                return message

    def test(self, data_idx=0):
        if self.test_dataset is not None:
            model = self.model
            model.train(False)
            loader = self.test_dataset
            losses = []
            ys = []
            y_hats = []
            pbar = enumerate(loader)
            with torch.set_grad_enabled(False):
                for it, (input_nodes, output_nodes, blocks) in pbar:
                    x = blocks[0].srcdata['feat']
                    y = blocks[-1].dstdata['label']
                    y_hat = model(blocks, x)
                    loss = F.cross_entropy(y_hat, y, ignore_index=-1)
                    # loss = self.custom_loss(y_hat, y)
                    losses.append(loss.item())
                    # 计算epoch 的总体 accuracy
                    ys.append(y)
                    y_hats.append(y_hat)
                test_acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys), task='multiclass'
                                       , num_classes=int(self.gme_dataset.num_classes['labels'][data_idx]))
                preds = torch.cat(y_hats).cpu().detach().numpy()
                preds = np.argmax(preds, axis=1)
                targets = torch.cat(ys).cpu().detach().numpy()
                test_rms = mean_squared_error(targets, preds)
                test_rms = math.sqrt(test_rms)
                test_loss = float(np.mean(losses))
                logger.info("test loss: ", test_loss)
                return test_loss, test_acc.item(), test_rms

    def train(self, data_split_idx=0, has_test_label=False, early_stop_patience=10, only_inference=False):
        start_time = datetime.now()
        # if not only_inference:
        labels_count_map = self.labels_count_map
        key_num = self.gme_dataset.num_classes['labels'][data_split_idx]
        if self.gme_dataset.num_classes['labels'][data_split_idx] != key_num:
            raise ValueError('Data type error.')
        alpha_list = np.array([labels_count_map[a] for a in range(key_num)])
        item_sum = np.sum(alpha_list)
        alpha_list = alpha_list / item_sum
        self.custom_loss = FocalLoss(classes_ratio=alpha_list, gamma=2, ignore_index=-1)

        model, config = self.model, self.config
        raw_model, optimizer = self.load_checkpoint()

        # split = 'train' | 'valid' | 'test'
        def run_epoch(split, data_idx):
            # 判断dataset是train()传入，还是self.dataset，self.datatset用于作预训练，tran()传入作为实际应用。
            is_train = split == 'train'
            self.train_dataset, self.val_dataset, _ = self.preprocess_input(self.gme_dataset, data_idx)
            model.train(is_train)  # train(false) 等价于 eval()
            loader = self.train_dataset if is_train else self.val_dataset

            losses = []
            ys = []
            y_hats = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

            for it, (input_nodes, output_nodes, blocks) in pbar:  # pbar:
                torch.cuda.empty_cache()
                # forward the model
                with torch.set_grad_enabled(is_train):  # torch.set_grad_enabled(False)与torch.no_grad()等价
                    x = blocks[0].srcdata['feat']
                    y = blocks[-1].dstdata['label']
                    y_hat = model(blocks, x)
                    # ignore_index=-1, 计算跳过填充值-1
                    # loss = F.cross_entropy(y_hat, y, ignore_index=-1)
                    loss = self.custom_loss(y_hat, y)
                    losses.append(loss.item())
                    # 计算epoch 的总体 accuracy
                    ys.append(y)
                    y_hats.append(y_hat)
                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    # lr = optimizer.state_dict()['param_groups'][0]['lr']  # 学习率
                    lr = optimizer.param_groups[0]['lr']
                    acc = MF.accuracy(preds=torch.cat(y_hats), target=torch.cat(ys), task='multiclass'
                                      , num_classes=int(self.gme_dataset.num_classes['labels'][data_idx]))
                    # .num_classes['labels'][model_idx]
                    # report progress
                    pbar.set_description(
                        f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. "
                        f"lr {lr:e}. acc {acc:.5f}")

            train_acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys), task='multiclass'
                                    , num_classes=int(self.gme_dataset.num_classes['labels'][data_idx]))
            preds = torch.cat(y_hats).cpu().detach().numpy()
            preds = np.argmax(preds, axis=1)
            targets = torch.cat(ys).cpu().detach().numpy()
            train_rms = mean_squared_error(targets, preds)
            train_rms = math.sqrt(train_rms)
            if not is_train:
                val_loss = float(np.mean(losses))
                logger.info("valid loss: ", val_loss)
                return val_loss, train_acc.item(), train_rms
            else:
                train_loss = float(np.mean(losses))
                return train_loss, train_acc.item(), train_rms

        self.tokens = 0  # counter used for learning rate decay
        self.log_name = os.path.join(os.path.dirname(self.ckpt_path), 'train_loss_log.txt')
        self.iter_record_path = os.path.join(os.path.dirname(self.ckpt_path), 'train_iter.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('# ================ Training Loss (%s) ================\n' % now)
        try:
            self.first_epoch, self.tokens = np.loadtxt(
                self.iter_record_path, delimiter=',', dtype=int)
            print('Resuming from epoch %d at token %d' % (self.first_epoch, self.tokens))
        except Exception as e:
            print(e)
            self.first_epoch = 1
            self.tokens = 0
            print('Could not load iteration record at %s. Starting from beginning.' %
                  self.iter_record_path)

        # train epoch
        train_loss_list = []
        val_loss_list = []
        train_acc_list = []
        var_acc_list = []

        #
        early_stopping = EarlyStopping(patience=early_stop_patience)
        for epoch in range(self.first_epoch - 1, self.max_epochs):

            train_loss, train_acc, train_rmse = run_epoch('train', data_split_idx)
            val_loss = 0
            val_acc = 0
            val_rmse = 0
            if self.val_dataset is not None:
                val_loss, val_acc, val_rmse = run_epoch('test', data_split_idx)

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            var_acc_list.append(val_acc)
            early_stopping(val_loss)
            message = f"Epoch {epoch + 1}, Train loss: {train_loss}, Train acc: {train_acc}, Train rmse: {train_rmse}, Val loss: {val_loss}, Val acc: {val_acc}, Val rmse: {val_rmse}, Stop Count: {early_stopping.counter}"
            print(message)
            with open(self.log_name, "a") as log_file:
                message_write = f"{epoch + 1},{train_loss},{train_acc},{train_rmse},{val_loss},{val_acc},{val_rmse}"
                log_file.write('%s\n' % message_write)

            np.savetxt(self.iter_record_path, (epoch + 1, self.tokens), delimiter=',', fmt='%d')
            # supports early stopping based on the test loss, or just save always if no test set is provided
            # good_model = self.val_dataset is None or val_loss < best_loss
            good_model = self.train_dataset is None or val_loss < self.best_loss
            if self.ckpt_path is not None and good_model:
                self.best_loss = val_loss
                self.save_checkpoint(optimizer=optimizer)
            if early_stopping.early_stop:
                break
        vtk_file_path = None
        if 'out_put_grid_file_name' in self.config.kwargs.keys():
            vtk_file = self.config.kwargs['out_put_grid_file_name']
            vtk_file_path = os.path.join(self.config.output_dir, vtk_file)
        # visual_loss_picture(train_loss=train_loss_list, test_loss=val_loss_list, save_path=self.config.output_dir)
        # visual_acc_picture(train_acc=train_acc_list, test_acc=var_acc_list, save_path=self.config.output_dir)
        print('Testing...')

        message = self.inference(self.gme_dataset, idx=data_split_idx, has_test_label=has_test_label,
                                 save_path=vtk_file_path)
        print(message)
        print('This round of training takes: {}s'.format(datetime.now() - start_time))
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
