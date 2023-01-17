import os
import math
import logging
import time
from tqdm import tqdm
import numpy as np
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
from model_visual_kit import visual_comparison_mesh

logger = logging.getLogger(__name__)


class GraphTransConfig:

    # config.vocab_size, config.n_embd  config.embd_pdrop  n_layer out_size resid_pdrop n_head attn_pdrop
    def __init__(self, vocab_size, in_size, embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, n_layer=12,
                 gnn_layer_num=3, coors=3, n_head=2, gnn_n_head=2, n_embd=512, out_size=4):
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
    def __init__(self, max_epochs=10, batch_size=512, learning_rate=3e-4, weight_decay=5e-4, lr_decay=False,
                 ckpt_path=None, tokens=0, num_workers=4):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # learning rate decay params: linear warmup followed by cosine decay to 10% of original
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        # checkpoint settings
        self.ckpt_path = ckpt_path
        self.tokens = tokens
        self.num_workers = num_workers

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
        self.sample_neigh = [10, 20, 20]
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
    def inference(self, data, idx=0):
        model = self.model.module
        test_idx = data.test_idx[idx]
        graph = data[idx].to(self.device)
        geodata = data.dataset.geodata[idx]

        feat = graph.ndata['feat']
        sampler = NeighborSampler(self.sample_neigh,
                                  prefetch_node_feats=['feat'],
                                  prefetch_labels=['label'])

        test_dataloader = DataLoader(graph, torch.arange(graph.num_nodes()).to(graph.device), sampler,
                                     device=self.device,
                                     batch_size=self.config.batch_size, shuffle=False,
                                     drop_last=False, num_workers=0,
                                     use_uva=False)
        model.eval()
        with torch.no_grad():
            pred = torch.empty(graph.num_nodes(), model.config.out_size, device=graph.device)

            for input_nodes, output_nodes, blocks in tqdm(test_dataloader):
                x = blocks[0].srcdata['feat']
                y = model(blocks, x)
                pred[output_nodes[0]:output_nodes[-1] + 1] = y.to(graph.device)
            visual_comparison_mesh(geodata=geodata, prediction=pred, label=graph.ndata['label'])
            pred_test = pred[test_idx]
            label_test = graph.ndata['label'][test_idx].to(pred_test.device)
            return MF.accuracy(pred_test, label_test)

    def train(self, dataset=None, data_split_idx=0):
        model, config = self.model, self.config
        raw_model = self.load_checkpoint()
        optimizer = torch.optim.Adam(raw_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        def run_epoch(split, idx=0):
            # 判断dataset是train()传入，还是self.dataset，self.datatset用于作预训练，tran()传入作为实际应用。
            is_train = split == 'train'
            if dataset is not None:
                # reinit dataset, shuffle and select from large dataset
                self.train_dataset, self.val_dataset = self.preprocess_input(dataset)
            else:
                self.train_dataset, self.val_dataset = self.preprocess_input(self.gme_dataset, idx)
            model.train(is_train)  # train(false) 等价于 eval()
            loader = self.train_dataset if is_train else self.val_dataset

            losses = []
            ys = []
            y_hats = []
            total_loss = 0  # 每一epoch的总loss
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
                    total_loss += loss.item()
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
                        f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. epoch total loss {total_loss:.5f} "
                        f"lr {lr:e}. acc {acc:.5f}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: ", test_loss)
                return test_loss
            else:
                train_loss = float(np.mean(losses))
                return train_loss

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
        for epoch in range(self.first_epoch - 1, config.max_epochs):

            train_loss = run_epoch('train', data_split_idx)
            val_loss = 0
            if self.val_dataset is not None:
                val_loss = run_epoch('test', data_split_idx)
            message = 'Epoch {}, Train loss: {}, Val loss: {}'.format(epoch + 1, train_loss, val_loss)
            print(message)
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)

            np.savetxt(self.iter_record_path, (epoch + 1, self.tokens), delimiter=',', fmt='%d')
            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.val_dataset is None or val_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = val_loss
                self.save_checkpoint()
        # test the model
        print('Testing...')
        if dataset is not None:
            acc = self.inference(dataset, idx=data_split_idx)
        else:
            acc = self.inference(self.gme_dataset, idx=data_split_idx)

        message = 'Test Accuracy {:.4f}'.format(acc.item())
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
