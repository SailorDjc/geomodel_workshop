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


class SpacialConv(nn.Module):

    def __init__(self,
                 coors,
                 in_feats,
                 out_feats,
                 hidden_size,
                 aggregator_type='mean',
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None,
                 eps=1e-7,):
        super(SpacialConv, self).__init__()
        valid_aggre_types = {'cat', 'mean'}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                'Invalid aggregator_type. Must be one of {}. '
                'But got {!r} instead.'.format(valid_aggre_types, aggregator_type)
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._hidden_size = hidden_size
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.eps = eps

        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        # self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)

        self.fc_spatial = nn.Linear(coors, hidden_size * in_feats)
        self.fc_neigh = nn.Linear(hidden_size * in_feats, out_feats)
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

    def edges_weight_func(self, edges):
        relative_pos = edges.dst['position'] - edges.src['position']
        spatial_scal = torch.norm(relative_pos, dim=1) + self.eps
        spatial_scal = spatial_scal.reshape(-1, 1)
        w = torch.add(relative_pos, 1)
        ws = torch.div(w, spatial_scal)
        spatial_scaling = F.leaky_relu(self.fc_spatial(ws))  # [n_edges, coors, hidden_size * in_feats]
        n_edges = spatial_scaling.size(0)
        result = spatial_scaling.reshape(n_edges, self._in_src_feats, -1) * edges.src['h'].unsqueeze(-1)
        return {'e_sp_wight': result.view(n_edges, -1)}

    def message_func(self, edges):
        edge_weight_neigh = edges.data['e_sp_wight']
        return {'e': edge_weight_neigh}

    def reduce_func(self, nodes):
        neigh_embed = nodes.mailbox['e']
        if self._aggre_type == 'cat':
            neigh_h = torch.cat(neigh_embed, dim=1)
        else:
            neigh_h = torch.mean(neigh_embed, dim=1)
        return {'neigh_h': neigh_h}

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
            graph.apply_edges(self.edges_weight_func)

            graph.update_all(self.message_func, self.reduce_func)
            h_neigh = graph.dstdata['neigh_h']
            # if self._aggre_type == 'mean':
            h_neigh = self.fc_neigh(h_neigh)  # [n_nodes, out_feats]
            # self._aggre_type == 'spatial':
            # else:
            #     raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            # GraphSAGE GCN does not require fc_self.
            rst = self.fc_self(h_self) + h_neigh

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
    def __init__(self, coors, in_dim, out_dim, hidden_size, num_heads, residual=False, merge='cat'):
        super(MultiHeadSpatialLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(SpacialConv(coors, in_dim, out_dim, hidden_size))
        self.merge = merge
        self.res_fc = None
        if residual:
            if num_heads * out_dim != in_dim:
                self.res_fc = nn.Linear(in_dim, num_heads*out_dim, bias=False)
            else:
                self.res_fc = nn.Identity()

    def forward(self, graph, h):
        head_outs = [attn_head(graph, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            rst = torch.cat(head_outs, dim=1)
        else:
            # merge using average
            rst = torch.mean(torch.stack(head_outs))
        # if self.res_fc is not None:
        #     resval = self.res_fc(h)
        #     rst = resval + rst
        return rst

    # def _compatibility_check(self):
    #     """Address the backward compatibility issue brought by #2747"""
    #     if not hasattr(self, 'bias'):
    #         dgl_warning("You are loading a GraphSAGE model trained from a old version of DGL, "
    #                     "DGL automatically convert it to be compatible with latest version.")
    #         bias = self.fc_neigh.bias
    #         self.fc_neigh.bias = None
    #         if hasattr(self, 'fc_self'):
    #             if bias is not None:
    #                 bias = bias + self.fc_self.bias
    #                 self.fc_self.bias = None
    #         self.bias = bias
