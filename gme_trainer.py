import os
import math
import logging
import time

import dgl
from tqdm import tqdm
import numpy as np
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
from model_visual_kit import visual_comparison_mesh, visual_loss_picture, visual_acc_picture, visual_sample_data

logger = logging.getLogger(__name__)


class GraphTransConfig:

    # config.vocab_size, config.n_embd  config.embd_pdrop  n_layer out_size resid_pdrop n_head attn_pdrop
    def __init__(self, vocab_size, in_size, embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, n_layer=12,
                 gnn_layer_num=4, coors=3, n_head=2, gnn_n_head=2, n_embd=512, out_size=4):
        self.in_size = in_size  # 输入特征维度
        self.vocab_size = vocab_size  # 句子长度
        self.coors = coors
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.n_layer = n_layer   # Transformer层数
        self.gnn_n_layer = gnn_layer_num    # gnn层数
        self.gnn_n_head = gnn_n_head
        self.n_head = n_head    # Transformer中多头注意力的head_num
        self.n_embd = n_embd    # 隐藏层维度
        self.out_size = out_size   # 输出维度，分类数


class GmeTrainerConfig:
    # optimization parameters
    device = 'cuda'
    def __init__(self, max_epochs=10, batch_size=512, learning_rate=1e-5, weight_decay=0.001, lr_decay=False,
                 ckpt_path=None, tokens=0, num_workers=4, output_dir=None):
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
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if torch.cuda.is_available():
            setattr(self, 'device', torch.cuda.current_device())


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
        self.sample_neigh = [10, 15, 15, 20]
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
                                      drop_last=False, num_workers=0,
                                      use_uva=False)

        val_dataloader = DataLoader(g, val_idx, sampler, device=self.device,
                                    batch_size=self.config.batch_size, shuffle=True,
                                    drop_last=False, num_workers=0,
                                    use_uva=False)

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
        model = self.model.module

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

        # sampler = NeighborSampler(self.sample_neigh,
        #                           prefetch_node_feats=['feat'],
        #                           prefetch_labels=['label'])
        #
        # test_dataloader = DataLoader(graph, nodes.to(graph.device), sampler,
        #                              device=self.device,
        #                              batch_size=self.config.batch_size, shuffle=False,
        #                              drop_last=False, num_workers=0,
        #                              use_uva=False)
        ys = []
        labels = []
        model.eval()
        with torch.no_grad():
            pred = torch.empty(graph.number_of_nodes(), model.config.out_size, device=graph.device)
            for input_nodes, output_nodes, blocks in tqdm(test_dataloader):
                x = blocks[0].srcdata['feat']
                y = model(blocks, x)
                ##
                # label = blocks[-1].dstdata['label']
                # ys.append(y)
                # labels.append(label)
                ##
                pred[output_nodes] = y.to(graph.device)
            visual_comparison_mesh(geodata=geodata, prediction=pred, label=graph.ndata['label'], is_show=is_show,
                                   save_path=save_path)
            visual_sample_data(geodata=geodata, plotter=None, camera=None, plot_points=False)
            # acc = MF.accuracy(torch.cat(ys), torch.cat(labels))
            # print('total test accuracy: {}'.format(acc))
            if has_test_label:
                test_idx = data.test_idx[idx]
                pred_test = pred[test_idx]
                label_test = graph.ndata['label'][test_idx].to(pred_test.device)
                accuracy = MF.accuracy(pred_test, label_test)
            return accuracy

    def train(self, data_split_idx=0):
        model, config = self.model, self.config
        raw_model = self.load_checkpoint()
        optimizer = torch.optim.Adam(raw_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

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
                    loss = F.cross_entropy(y_hat, y)
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
                    lr = config.learning_rate  # 学习率

                    acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys))
                    # report progress
                    pbar.set_description(
                        f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. "
                        f"lr {lr:e}. acc {acc:.5f}")

            if not is_train:
                val_loss = float(np.mean(losses))
                logger.info("test loss: ", val_loss)
                val_acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys))
                return val_loss, val_acc.item()
            else:
                train_loss = float(np.mean(losses))
                train_acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys))
                return train_loss, train_acc.item()

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
        for epoch in range(self.first_epoch - 1, config.max_epochs):

            train_loss, train_acc = run_epoch('train', data_split_idx)
            val_loss = 0
            val_acc = 0
            if self.val_dataset is not None:
                val_loss, val_acc = run_epoch('test', data_split_idx)
            message = 'Epoch {}, Train loss: {}, Train acc: {}, Val loss: {}, ' \
                      'Val acc: {}'.format(epoch + 1, train_loss, train_acc, val_loss, val_acc)
            print(message)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            var_acc_list.append(val_acc)
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)

            np.savetxt(self.iter_record_path, (epoch + 1, self.tokens), delimiter=',', fmt='%d')
            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.val_dataset is None or val_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = val_loss
                self.save_checkpoint()
            # if epoch % 10 == 0 and epoch > 0:
            #     acc = self.inference(self.gme_dataset, idx=data_split_idx, is_show=False)
            #     print('Test Accuracy {:.4f}'.format(acc.item()))
        # test the model
        visual_loss_picture(train_loss=train_loss_list, test_loss=val_loss_list, save_path=self.config.output_dir)
        visual_acc_picture(train_acc=train_acc_list, test_acc=var_acc_list, save_path=self.config.output_dir)
        print('Testing...')

        acc = self.inference(self.gme_dataset, idx=data_split_idx)
        message = 'Test Accuracy {:.4f}'.format(acc.item())
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
