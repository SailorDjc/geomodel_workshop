import os.path
import math
import logging

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from models.layers import MultiHeadSpatialLayer, MultiHeadSageLayer, GraphTransformerLayer, MLPReadout
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from tqdm import tqdm


class GNN(nn.Module):
    def __init__(self, coors, in_feats, out_feats, num_head, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.layers.append(MultiHeadSpatialLayer(coors, in_feats, out_feats, num_head))
        for l in range(1, n_layers):
            self.layers.append(
                MultiHeadSpatialLayer(coors, out_feats, out_feats, num_head))
        self.dropout = nn.Dropout(0.5)
        # self.last_layer = nn.Linear(n_layers * out_feats, out_feats)

    def forward(self, blocks, x):
        h = x
        layer_outputs = []
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.leaky_relu(h)
                h = self.dropout(h)
            layer_outputs.append(h[:blocks[-1].number_of_dst_nodes()])
        h = torch.cat(layer_outputs, dim=1)
        # h = self.last_layer(h)
        return h


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, num_head, n_layers):  # out_size
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        # self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        self.layers.append(MultiHeadSageLayer(in_size, hid_size, num_head))
        for l in range(1, n_layers):
            # self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
            self.layers.append(MultiHeadSageLayer(hid_size, hid_size, num_head))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        # self.last_layer = nn.Linear(n_layers * hid_size, hid_size)

    def forward(self, blocks, x):
        h = x
        layer_outputs = []
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.leaky_relu(h)
                h = self.dropout(h)
            layer_outputs.append(h[:blocks[-1].number_of_dst_nodes()])
        h = torch.cat(layer_outputs, dim=1)
        # h = self.last_layer(h)
        return h


class SelfAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, attn_pdrop=0.5, resid_drop=0.5):
        super(SelfAttention, self).__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.att_head_size = int(out_dim / num_heads)
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim, bias=True)
            self.K = nn.Linear(in_dim, out_dim, bias=True)
            self.V = nn.Linear(in_dim, out_dim, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim, bias=False)
            self.K = nn.Linear(in_dim, out_dim, bias=False)
            self.V = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_drop = nn.Dropout(attn_pdrop)

        self.dense = nn.Linear(out_dim, out_dim)
        self.pro_x = nn.Linear(in_dim, out_dim)
        self.LayerNorm = nn.LayerNorm(out_dim, out_dim)
        self.resid_drop = nn.Dropout(resid_drop)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.att_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)
        x = self.pro_x(x)
        q_layer = self.transpose_for_scores(q)
        k_layer = self.transpose_for_scores(k)
        v_layer = self.transpose_for_scores(v)

        att_scores = torch.matmul(q_layer, k_layer.transpose(-1, -2))
        att_scores = att_scores / math.sqrt(self.att_head_size)

        att_probs = nn.Softmax(dim=-1)(att_scores)
        att_probs = self.attn_drop(att_probs)
        context_layer = torch.matmul(att_probs, v_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.out_dim,)
        context_layer = context_layer.view(new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.resid_drop(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + x)
        return hidden_states


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config, gnn_has_heads=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.config = config
        if gnn_has_heads:
            self.n_embd = config.n_embd * config.gnn_n_layer  # * config.gnn_n_head
        else:
            self.n_embd = config.n_embd
        self.key = nn.Linear(self.n_embd, self.n_embd)
        self.query = nn.Linear(self.n_embd, self.n_embd)
        self.value = nn.Linear(self.n_embd, self.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        # mask = torch.tril(torch.ones(config.block_size, config.block_size))
        # undo masking for the unpredicted part
        # mask[:config.block_size // 2, :config.block_size // 2] = 1
        # mask = mask.view(1, 1, config.block_size, config.block_size)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", mask)
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = nn.Linear(config.n_embd, config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class BlockCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd  # * config.gnn_n_layer  # * config.gnn_n_head
        self.ln1 = nn.LayerNorm(self.n_embd)
        self.ln2 = nn.LayerNorm(self.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(self.n_embd, 4 * self.n_embd),
            nn.GELU(),
            nn.Linear(4 * self.n_embd, self.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        for l in range(0, config.n_layer):
            self.blocks.append(BlockCNN(config))
        self.last_layer = nn.Linear(config.n_embd * config.gnn_n_layer, config.n_embd)  # config.n_layer *

    def forward(self, x):
        h = x
        # layer_outputs = []
        for l, block in enumerate(self.blocks):
            h = block(h)
            # layer_outputs.append(h)
        # h = torch.cat(layer_outputs, dim=2)
        h = self.last_layer(h)
        return h


#  ===============================================================================================================   #
class GraphTransfomer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # coors, in_feats, out_feats, n_hidden, num_heads, n_layers
        self.gnn = GNN(config.coors, config.in_size, config.n_embd, config.gnn_n_head, config.gnn_n_layer)
        # self.blocks = nn.Sequential(*[BlockCNN(config) for _ in range(config.n_layer)])
        # self.ln_f = nn.LayerNorm(config.n_embd)
        # self.head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        self.blocks = nn.Sequential()
        self.blocks.add_module('at_1', SelfAttention(config.n_embd * config.gnn_n_layer
                                                     , int(config.n_embd * config.gnn_n_layer / 2)
                                                     , config.n_head, use_bias=True))
        self.blocks.add_module('at_2', SelfAttention(int(config.n_embd * config.gnn_n_layer / 2)
                                                     , config.n_embd, config.n_head
                                                     , use_bias=True))
        self.blocks.add_module('at_3', SelfAttention(config.n_embd, config.n_embd
                                                     , config.n_head, use_bias=True))

        self.p_layer = nn.Linear(config.n_embd, config.out_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, blocks, x):
        gh = self.gnn(blocks, x)  # [512, gnn_n_layer * 512]
        gh = gh.unsqueeze(1)
        h = self.blocks(gh)
        # h = self.ln_f(h)
        # logits = self.head(h)
        logits = h.squeeze(1)
        logits_result = self.p_layer(logits)
        return logits_result


class SAGETransfomer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embd_pdrop)
        self.gnn = SAGE(config.in_size, config.n_embd, config.gnn_n_head, config.gnn_n_layer)
        self.blocks = nn.Sequential()
        self.blocks.add_module('at_1', SelfAttention(config.n_embd * config.gnn_n_layer
                                                     , int(config.n_embd * config.gnn_n_layer / 2)
                                                     , config.n_head, use_bias=True))
        self.blocks.add_module('at_2', SelfAttention(int(config.n_embd * config.gnn_n_layer / 2)
                                                     , config.n_embd, config.n_head
                                                     , use_bias=True))
        self.blocks.add_module('at_3', SelfAttention(config.n_embd, config.n_embd
                                                     , config.n_head, use_bias=True))
        self.p_layer = nn.Linear(config.n_embd, config.out_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, blocks, x):
        gh = self.gnn(blocks, x)  # [1024, 1024]
        gh = gh.unsqueeze(1)
        h = self.blocks(gh)
        logits = h.squeeze(1)
        logits_result = self.p_layer(logits)
        return logits_result


class GraphModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embd_pdrop)
        # coors, in_feats, out_feats, n_hidden, num_heads, n_layers
        self.gnn = GNN(config.coors, config.in_size, config.n_embd, config.gnn_n_head, config.gnn_n_layer)
        # self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        self.p_layer = nn.Linear(config.vocab_size, config.out_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, blocks, x):
        gh = self.gnn(blocks, x)  # [1024, 1024]
        h = self.ln_f(gh)
        logits = self.head(h)
        logits_result = self.p_layer(logits)
        return logits_result


class SAGEModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embd_pdrop)
        # coors, in_feats, out_feats, n_hidden, num_heads, n_layers
        self.gnn = SAGE(config.in_size, config.n_embd, config.gnn_n_layer)
        # self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        self.p_layer = nn.Linear(config.vocab_size, config.out_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, blocks, x):
        gh = self.gnn(blocks, x)  # [1024, 1024]
        h = self.ln_f(gh)
        logits = self.head(h)
        logits_result = self.p_layer(logits)
        return logits_result


class GraphTransfomerNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GraphTransformerLayer(config.coors, config.in_size, config.n_embd, config.gnn_n_head,
                                                     dropout=0.5))
        for l in range(1, config.gnn_n_layer):
            self.gnn_layers.append(
                GraphTransformerLayer(config.coors, config.n_embd, config.n_embd, config.gnn_n_head, dropout=0.5))
        # mlp_layer
        self.p_layer = MLPReadout(config.n_embd * config.gnn_n_layer, config.out_size)
        # # self.concat_layer = nn.Linear(n_layers * out_feats, out_feats)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, blocks, x):
        h = x
        layer_outputs = []
        for l, (layer, block) in enumerate(zip(self.gnn_layers, blocks)):
            h = layer(block, h)
            layer_outputs.append(h[:blocks[-1].number_of_dst_nodes()])
        h = torch.cat(layer_outputs, dim=1)
        h = self.p_layer(h)
        return h
