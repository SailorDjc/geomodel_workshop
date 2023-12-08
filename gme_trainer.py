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
# import model_visual_kit as mvk
from datetime import datetime
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


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


class GmeTrainerConfig:
    # optimization parameters
    device = 'cuda'

    def __init__(self, max_epochs=10, batch_size=512, learning_rate=1e-3, weight_decay=1e-4, lr_decay=False,
                 ckpt_path=None, tokens=0, num_workers=4, output_dir=None, sample_neigh=None, **kwargs):
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
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if torch.cuda.is_available():
            setattr(self, 'device', torch.cuda.current_device())
        self.kwargs = kwargs


class GmeTrainer:

    # gme_dataset: GeoDataset
    # model
    # config: GmeTrainerConfig

    def __init__(self, model, gme_dataset, config):
        self.model = model
        self.gme_dataset = gme_dataset
        self.train_dataset = None
        self.val_dataset = None
        self.config = config
        self.sample_neigh = config.sample_neigh
        self.log_name = None
        self.iter_record_path = None
        # take over whatever gpus are on the system
        self.device = 'cpu'

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def preprocess_input(self, data, idx):
        if idx >= data.__len__():
            raise ValueError("索引超出范围")
        g = data[idx]
        g = g.to(self.device)
        # 获取 训练集与验证集的 节点索引
        train_idx = data.train_idx[idx].to(self.device)
        val_idx = data.val_idx[idx].to(self.device)
        # 采样器
        sampler = NeighborSampler(self.sample_neigh,  # fanout for [layer-0, layer-1, layer-2]
                                  prefetch_node_feats=['feat'],
                                  prefetch_labels=['label'])

        train_dataloader = DataLoader(g, train_idx, sampler, device=self.device,
                                      batch_size=self.config.batch_size, shuffle=True,
                                      drop_last=False,
                                      use_uva=False)  # num_workers=0,

        val_dataloader = DataLoader(g, val_idx, sampler, device=self.device,
                                    batch_size=self.config.batch_size, shuffle=True,
                                    drop_last=False,
                                    use_uva=False)  # num_workers=0,

        return train_dataloader, val_dataloader

    def load_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if os.path.exists(self.config.ckpt_path):
            pretrain_dict = torch.load(self.config.ckpt_path, map_location='cpu')
            # 当前网络模型参数
            model_dict = raw_model.state_dict()
            # 更新参数，保存的模型参数与当前网络参数有部分不同
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if (k in model_dict and 'p_layer' not in k)}

            model_dict.update(pretrain_dict)
            raw_model.load_state_dict(model_dict)
            print("loading ", self.config.ckpt_path)
        return raw_model

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        print("saving ", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    # data
    def inference(self, data, idx, has_test_label=True, is_show=True, save_path=None):
        model = self.model.module if hasattr(self.model, "module") else self.model

        graph = data[idx].to(self.device)
        geodata = data.dataset.geodata[idx]
        nodes = torch.arange(graph.number_of_nodes())
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(model.config.gnn_n_layer,
                                                                prefetch_node_feats=['feat'],
                                                                prefetch_labels=['label'])
        test_dataloader = dgl.dataloading.NodeDataLoader(
            graph, nodes.to(graph.device), sampler, device=self.device,
            batch_size=self.config.batch_size, shuffle=True,
            drop_last=False
        )
        model.eval()
        with torch.no_grad():
            pred = torch.empty(graph.number_of_nodes(), model.config.out_size, device=graph.device)
            for input_nodes, output_nodes, blocks in tqdm(test_dataloader):
                x = blocks[0].srcdata['feat']
                y = model(blocks, x)
                pred[output_nodes] = y.to(graph.device)
            scalars = np.argmax(pred.cpu().numpy(), axis=1)
            mvk.visual_predicted_values_model(geodata, pred, is_show=is_show, save_path=save_path)

            if has_test_label:
                test_idx = data.test_idx[idx]
                pred_test = pred[test_idx]
                label_test = graph.ndata['label'][test_idx].to(pred_test.device)
                accuracy = MF.accuracy(pred_test, label_test, task='multiclass'
                                       , num_classes=int(self.model.config.out_size))
                return accuracy.item()
            return 0

    def train(self, data_split_idx=0, has_test_label=True):
        model, config = self.model, self.config
        raw_model = self.load_checkpoint()
        lr = config.learning_rate
        optimizer = torch.optim.Adam(raw_model.parameters(), lr=lr, weight_decay=config.weight_decay)

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
        #
        # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=True,
        #                                            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-8,
        #                                            eps=1e-08)

        def run_epoch(split, idx):
            # 判断dataset是train()传入，还是self.dataset，self.datatset用于作预训练，tran()传入作为实际应用。
            is_train = split == 'train'
            # if self.train_dataset is None or self.val_dataset is None:
            self.train_dataset, self.val_dataset = self.preprocess_input(self.gme_dataset, idx)
            model.train(is_train)  # train(false) 等价于 eval()
            loader = self.train_dataset if is_train else self.val_dataset

            losses = []
            ys = []
            y_hats = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

            for it, (input_nodes, output_nodes, blocks) in pbar:
                torch.cuda.empty_cache()
                # forward the model
                with torch.set_grad_enabled(is_train):  # torch.set_grad_enabled(False)与torch.no_grad()等价
                    x = blocks[0].srcdata['feat']
                    y = blocks[-1].dstdata['label']
                    y_hat = model(blocks, x)
                    # ignore_index=-1, 计算跳过填充值-1
                    loss = F.cross_entropy(y_hat, y, ignore_index=-1)
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
                                      , num_classes=int(self.model.config.out_size))
                    #
                    # report progress
                    pbar.set_description(
                        f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. "
                        f"lr {lr:e}. acc {acc:.5f}")

            train_acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys), task='multiclass'
                                    , num_classes=int(self.model.config.out_size))
            preds = torch.cat(y_hats).cpu().detach().numpy()
            preds = np.argmax(preds, axis=1)
            targets = torch.cat(ys).cpu().detach().numpy()
            train_rms = mean_squared_error(targets, preds)
            train_rms = math.sqrt(train_rms)
            if not is_train:
                val_loss = float(np.mean(losses))
                logger.info("test loss: ", val_loss)
                return val_loss, train_acc.item(), train_rms
            else:
                train_loss = float(np.mean(losses))
                return train_loss, train_acc.item(), train_rms

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        self.log_name = os.path.join(os.path.dirname(self.config.ckpt_path), 'tran_loss_log.txt')
        self.iter_record_path = os.path.join(os.path.dirname(self.config.ckpt_path), 'tran_iter.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
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

        start_time = datetime.now()
        for epoch in range(self.first_epoch - 1, config.max_epochs):

            train_loss, train_acc, train_rmse = run_epoch('train', data_split_idx)
            val_loss = 0
            val_acc = 0
            val_rmse = 0
            if self.val_dataset is not None:
                val_loss, val_acc, val_rmse = run_epoch('test', data_split_idx)
            message = f"Epoch {epoch + 1}, Train loss: {train_loss}, Train acc: {train_acc}, Train rmse: {train_rmse}, Val loss: {val_loss}, Val acc: {val_acc}, Val rmse: {val_rmse}"
            print(message)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            var_acc_list.append(val_acc)

            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)

            np.savetxt(self.iter_record_path, (epoch + 1, self.tokens), delimiter=',', fmt='%d')
            # supports early stopping based on the test loss, or just save always if no test set is provided
            # good_model = self.val_dataset is None or val_loss < best_loss
            good_model = self.train_dataset is None or train_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = val_loss
                self.save_checkpoint()
        vtk_file_path = None
        if 'out_put_grid_file_name' in self.config.kwargs.keys():
            vtk_file = self.config.kwargs['out_put_grid_file_name']
            vtk_file_path = os.path.join(self.config.output_dir, vtk_file)
        # visual_loss_picture(train_loss=train_loss_list, test_loss=val_loss_list, save_path=self.config.output_dir)
        # visual_acc_picture(train_acc=train_acc_list, test_acc=var_acc_list, save_path=self.config.output_dir)
        print('Testing...')

        acc = self.inference(self.gme_dataset, idx=data_split_idx, has_test_label=has_test_label,
                             save_path=vtk_file_path)
        message = '================Test Accuracy {:.4f}================' \
            .format(acc)
        print(message)
        print('This round of training takes: {}s'.format(datetime.now() - start_time))
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
