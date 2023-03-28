"""Torch Module for GraphSAGE layer"""
import math

# pylint: disable= no-member, arguments-differ, invalid-name
import torch
from torch import nn
from torch.nn import functional as F
from dgl.nn.functional import edge_softmax
import dgl.function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair, check_eq_shape, dgl_warning
import numpy as np
from dgl.nn.pytorch.utils import Identity


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

    # def edges_weight_func(self, edges):
    #     relative_pos = edges.dst['position'] - edges.src['position']
    #     spatial_scal = torch.norm(relative_pos, dim=1) + self.eps
    #     spatial_scal = spatial_scal.reshape(-1, 1)
    #     w = torch.add(relative_pos, 1)
    #     ws = torch.div(w, spatial_scal)
    #     spatial_scaling = F.leaky_relu(self.fc_spatial(ws))  # [n_edges, coors, hidden_size * in_feats]
    #     n_edges = spatial_scaling.size(0)
    #     result = spatial_scaling.reshape(n_edges, self._in_src_feats, -1) * edges.src['h'].unsqueeze(-1)
    #     return {'e_sp_wight': result.view(n_edges, -1)}

    def edges_wight_process(self, edges):
        relative_pos = edges.dst['position'] - edges.src['position']
        spatial_scale = torch.norm(relative_pos, dim=1) + self.eps
        spatial_scale = spatial_scale.reshape(-1, 1)
        spatial_att = torch.add(relative_pos, 1)
        spatial_coeff = torch.div(spatial_att, spatial_scale)
        return {'e': spatial_coeff}

    # def message_func(self, edges):
    #     edge_weight_neigh = edges.data['e_sp_wight']
    #     return {'e': edge_weight_neigh}
    # def reduce_func(self, nodes):
    #     neigh_embed = nodes.mailbox['e']
    #     if self._aggre_type == 'cat':
    #         neigh_h = torch.cat(neigh_embed, dim=1)
    #     else:
    #         neigh_h = torch.mean(neigh_embed, dim=1)
    #     return {'neigh_h': neigh_h}

    def forward(self, graph, feat, edge_weight=None):

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
            # graph.apply_edges(self.edges_weight_func)
            graph.apply_edges(self.edges_wight_process)
            graph.edata['e'] = self.fc_spatial(graph.edata['e'])
            graph.edata['e'] = F.leaky_relu(graph.edata['e'])
            graph.update_all(fn.u_mul_e('h', 'e', 'em'), fn.mean('em', 'h_mean'))

            h_neigh = self.fc_neigh(graph.dstdata['h_mean'])  # [n_nodes, out_feats]
            # g.update_all(fn.e_add_v('theta', 'phi', 'e'), fn.max('e', 'x'))

            # graph.update_all(self.message_func, self.reduce_func)
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


# 修改后的SpacialConv
class SpacialConv_1(nn.Module):

    def __init__(self,
                 coors,
                 in_feats,
                 out_feats,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=F.leaky_relu,
                 eps=1e-7, ):
        super(SpacialConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        # self._hidden_size = hidden_size
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

    def edges_wight_process(self, edges):
        relative_pos = edges.dst['position'] - edges.src['position']
        spatial_scale = torch.norm(relative_pos, dim=1) + self.eps
        spatial_scale = spatial_scale.reshape(-1, 1)
        spatial_att = torch.add(relative_pos, 1)
        spatial_coeff = torch.div(spatial_att, spatial_scale)
        return {'e': spatial_coeff}

    def forward(self, graph, feat, edge_weight=None):

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
            # 边权重乘以源节点特征，赋给目标节点
            msg_fn = fn.u_mul_e('h', '_edge_weight', 'm')
            graph.apply_edges(self.edges_wight_process)

            graph.edata['_edge_weight'] = self.fc_spatial(graph.edata['e'])
            graph.edata['_edge_weight'] = F.leaky_relu(graph.edata['_edge_weight'])

            graph.update_all(msg_fn, fn.mean('m', 'h_mean'))

            h_neigh = self.fc_neigh(graph.dstdata['h_mean'])  # [n_nodes, out_feats]
            # g.update_all(fn.e_add_v('theta', 'phi', 'e'), fn.max('e', 'x'))

            # graph.update_all(self.message_func, self.reduce_func)
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


"""
    Util functions
"""


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}

    return func


# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """

    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}

    return func


# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}

    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, coors, in_dim, out_dim, num_heads, use_bias, feat_drop=0, eps=1e-7):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.feat_drop = nn.Dropout(feat_drop)
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_e = nn.Linear(coors, out_dim * num_heads, bias=True)
            self.fc_self = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(coors, out_dim * num_heads, bias=False)
            self.fc_self = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.eps = eps

    # 主要工作：计算新的边权重，注意力分数， 新边权乘上原有的边权， 然后聚合节点特征
    # 以下操作只是进行了边权中计算与节点特征传播，特征的shape不变
    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)

        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e'))

        # copy edge features as e_out to be passed to FFN_e
        # 输出边权重
        g.apply_edges(out_edge_features('score'))

        # Send weighted values to target nodes
        eids = g.edges()
        # 源节点特征乘以边权， 然后求和
        # g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.update_all(fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        # 将边权重加到节点特征上
        # g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
        g.update_all(fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

    def edges_spa_coeff_compute(self, edges):
        relative_pos = edges.dst['position'] - edges.src['position']
        spatial_scale = torch.norm(relative_pos, dim=1) + self.eps
        spatial_scale = spatial_scale.reshape(-1, 1)
        spatial_att = torch.add(relative_pos, 1)
        spatial_coeff = torch.div(spatial_att, spatial_scale)
        return {'e_spcf': spatial_coeff}

    def reduce_func(self, nodes):
        pass

    def forward(self, g, feat):
        with g.local_scope():
            feat_src = feat_dst = self.feat_drop(feat)
            if g.is_block:
                feat_dst = feat_src[:g.number_of_dst_nodes()]
            h_self = feat_dst

            # 计算空间系数
            g.apply_edges(self.edges_spa_coeff_compute)

            # g.edata['_edge_weight'] = self.fc_spatial(graph.edata['e'])
            # graph.edata['_edge_weight'] = F.leaky_relu(graph.edata['_edge_weight'])
            #
            # graph.update_all(msg_fn, fn.mean('m', 'h_mean'))
            e = g.edata['e_spcf']
            # Message Passing
            Q_h = self.Q(feat_src)  # h
            K_h = self.K(feat_src)
            V_h = self.V(feat_src)

            proj_e = self.proj_e(e)

            # Reshaping into [num_nodes, num_heads, feat_dim] to
            # get projections for multi-head attention
            g.srcdata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
            g.dstdata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)[:g.number_of_dst_nodes()]
            g.srcdata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
            g.dstdata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)[:g.number_of_dst_nodes()]
            g.srcdata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
            g.dstdata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)[:g.number_of_dst_nodes()]
            g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.out_dim)

            self.propagate_attention(g)
            # 这里分母加上一个极小值矩阵，防止分母为0]
            h_neigh = g.dstdata['wV'] / (g.dstdata['z'] + torch.full_like(g.dstdata['z'], 1e-6))
            h_self = self.fc_self(h_self).view(-1, self.num_heads, self.out_dim)
            h_out = h_self + h_neigh
            return h_out


class GraphTransformerLayer(nn.Module):
    """
        Param:
    """

    def __init__(self, coors, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True,
                 use_bias=False, feat_drop=0, eps=1e-6):
        super().__init__()

        self.coors = coors
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(coors, in_dim, out_dim // num_heads, num_heads, use_bias, feat_drop, eps)
        self.self_fc = nn.Linear(in_dim, out_dim)
        self.O_h = nn.Linear(out_dim, out_dim)
        # self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            # self.layer_norm1_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        # FFN for e
        # self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
        # self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            # self.layer_norm2_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            # self.batch_norm2_e = nn.BatchNorm1d(out_dim)

    def forward(self, g, h):
        h_in1 = h[:g.number_of_dst_nodes()]  # for first residual connection
        ##
        # multi-head attention out
        h_attn_out = self.attention(g, h)  # , e_attn_out

        h = h_attn_out.view(-1, self.out_channels)
        # e = e_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)
        # e = F.dropout(e, self.dropout, training=self.training)

        # e = self.O_e(e)

        if self.residual:
            h = self.self_fc(h_in1) + self.O_h(h)  # residual connection
            # e 的初始残差连接没有，因为e是通过神经网络计算得到的，不是初始给定的

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            # e = self.batch_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            # e = self.batch_norm1_e(e)

        h_in2 = h  # for second residual connection
        # e_in2 = e

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
        # e = self.FFN_e_layer1(e)
        # e = F.relu(e)
        # e = F.dropout(e, self.dropout, training=self.training)
        # e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h  # residual connection
            # e = e_in2 + e

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            # e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            # e = self.batch_norm2_e(e)

        return h  # , e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads,
                                                                                   self.residual)


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.leaky_relu(y)
        y = self.FC_layers[self.L](y)
        return y
