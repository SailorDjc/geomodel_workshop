"""Torch Module for GraphSAGE layer"""
import math

# pylint: disable= no-member, arguments-differ, invalid-name
import torch
from torch import nn
from torch.nn import functional as F
from dgl.nn.functional import edge_softmax
import dgl.function as fn
import dgl.nn as dglnn
from dgl.base import DGLError
from dgl.utils import expand_as_pair, check_eq_shape, dgl_warning
import numpy as np
from dgl.nn.pytorch.utils import Identity
import dgl.nn.pytorch.conv.gcn2conv


# 原始SpacialConv
class SpacialConv(nn.Module):

    def __init__(self,
                 coors,
                 in_feats,
                 out_feats,
                 aggregator_type='mean',
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=F.leaky_relu,
                 eps=1e-7, ):
        super(SpacialConv, self).__init__()
        valid_aggre_types = {'cat', 'mean'}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                'Invalid aggregator_type. Must be one of {}. '
                'But got {!r} instead.'.format(valid_aggre_types, aggregator_type)
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        # self._hidden_size = hidden_size
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.eps = eps

        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=True)
        # self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)

        self.fc_spatial = nn.Linear(coors, in_feats)
        self.fc_neigh = nn.Linear(in_feats, out_feats)
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)
        self.res_fc = None
        self.reset_parameters()

    def reset_parameters(self):

        gain = nn.init.calculate_gain('leaky_relu')

        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        # nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_spatial.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def edge_weight_spatial_process(self, edges):
        relative_pos = edges.dst['position'] - edges.src['position']
        return {'e': relative_pos}

    #
    def edges_weight_process(self, edges):
        relative_pos = edges.dst['position'] - edges.src['position']
        spatial_scale = torch.norm(relative_pos, dim=1, p=2) + self.eps
        spatial_scale = spatial_scale.reshape(-1, 1)
        spatial_att = torch.add(relative_pos, 1)
        spatial_coeff = torch.div(spatial_att, spatial_scale)
        return {'e': spatial_coeff}

    # 高斯核
    def edges_weight_gauss_process(self, edges):
        relative_pos = edges.dst['position'] - edges.src['position']
        euclidean_dist = torch.norm(relative_pos, dim=1, p=2)
        variance = torch.var(euclidean_dist)
        div_value = torch.pow(variance, 2)
        exponential = - torch.divide(euclidean_dist, div_value)
        gauss_kernel = torch.exp(exponential).reshape(-1, 1)
        return {'e': gauss_kernel}

    def forward(self, graph, feat):  # , edge_weight=None
        # self._compatibility_check()
        with graph.local_scope():
            feat_src = feat_dst = self.feat_drop(feat)
            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]
            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Message Passing
            graph.srcdata['h'] = feat_src
            graph.apply_edges(self.edges_weight_process)
            # graph.apply_edges(self.edges_weight_gauss_process)
            # graph.apply_edges(self.edge_weight_spatial_process)
            graph.edata['e'] = self.fc_spatial(graph.edata['e'])
            graph.edata['e'] = F.leaky_relu(graph.edata['e'])
            graph.update_all(fn.u_mul_e('h', 'e', 'em'), fn.mean('em', 'h_mean'))

            h_neigh = self.fc_neigh(graph.dstdata['h_mean'])  # [n_nodes, out_feats]
            self_hidden = self.fc_self(h_self)
            rst = self_hidden + h_neigh

            # bias term
            if self.bias is not None:
                rst = rst + self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


class MultiHeadSpatialLayer(nn.Module):
    def __init__(self, coors, in_dim, out_dim, num_heads, merge='mean'):
        super(MultiHeadSpatialLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(SpacialConv(coors, in_dim, out_dim))
        self.merge = merge

    def forward(self, graph, h):
        head_outs = [attn_head(graph, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            rst = torch.cat(head_outs, dim=1)
        else:
            # merge using average
            rst = torch.stack(head_outs)
            rst = torch.mean(rst, dim=0)
        return rst


class MultiHeadSageLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='mean'):
        super(MultiHeadSageLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(dglnn.SAGEConv(in_dim, out_dim, 'mean'))
        self.merge = merge

    def forward(self, graph, h):
        head_outs = [attn_head(graph, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            rst = torch.cat(head_outs, dim=1)
        else:
            # merge using average
            rst = torch.stack(head_outs)
            rst = torch.mean(rst, dim=0)
        return rst



