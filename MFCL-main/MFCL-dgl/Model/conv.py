"""
Time: 2024.4.7
Author: Yiran Shi
"""
"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name

import torch as th
from torch import nn
import torch
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from Filter import Filter, IFilter
from Normal import prob
import numpy as np
from scipy import signal

class DropLearner(nn.Module):
    def __init__(self, node_dim, edge_dim = None, mlp_edge_model_dim = 64):
        super(DropLearner, self).__init__()
        self.mlp_src = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.mlp_dst = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.mlp_con = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.concat = False
        
        if edge_dim is not None:
            self.mlp_edge = nn.Sequential(
                nn.Linear(edge_dim, mlp_edge_model_dim),
                nn.ReLU(),
                nn.Linear(mlp_edge_model_dim, 1)
            )
        else:
            self.mlp_edge = None
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def get_weight(self, head_emb, tail_emb, temperature = 0.5, relation_emb = None, edge_type = None):
        if self.concat:
            weight = self.mlp_con(head_emb + tail_emb)
            w_src = self.mlp_src(head_emb)
            w_dst = self.mlp_dst(tail_emb)
            weight += w_src + w_dst
        else:
            w_src = self.mlp_src(head_emb)
            w_dst = self.mlp_dst(tail_emb)
            weight = w_src + w_dst
        if relation_emb is not None and self.mlp_edge is not None:
            e_weight = self.mlp_edge(relation_emb)
            weight += e_weight
        weight = weight.squeeze()
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * th.rand(weight.size()) + (1 - bias)
        gate_inputs = th.log(eps) - th.log(1 - eps)
        gate_inputs = gate_inputs.to(head_emb.device)
        gate_inputs = (gate_inputs + weight) / temperature
        aug_edge_weight = th.sigmoid(gate_inputs).squeeze()
        edge_drop_out_prob = 1 - aug_edge_weight
        reg = edge_drop_out_prob.mean()
        #print(aug_edge_weight.size())
        return reg.detach(), aug_edge_weight.detach()
    
    def forward(self, node_emb, graph, temperature = 0.5, relation_emb = None, edge_type = None):
        if self.concat:
            w_con = node_emb
            graph.srcdata.update({'in': w_con})
            graph.apply_edges(fn.u_add_v('in', 'in', 'con'))
            n_weight = graph.edata.pop('con')
            weight = self.mlp_con(n_weight)
            w_src = self.mlp_src(node_emb)
            w_dst = self.mlp_dst(node_emb)
            graph.srcdata.update({'inl': w_src})
            graph.dstdata.update({'inr': w_dst})
            graph.apply_edges(fn.u_add_v('inl', 'inr', 'ine'))
            weight += graph.edata.pop('ine')
            #print(weight.size())
        else:
            # FFT
            w_src_tmp = Filter(node_emb)
            w_src = w_src_tmp
            fs1 = w_src.shape[0]
            t1 = np.arange(0, 1, 1 / fs1)
            w_src1 = w_src.cpu()
            w_src2 = w_src1.detach().numpy()
            w_src22 = prob(w_src2)
            # MPF cutoff frequency
            fch1 = 10000
            fcl1 = 5000
            bandwidth1 = fch1 - fcl1
            rp1 = 5
            order1 = 5
            b1, a1 = signal.cheby2(order1, rp1, fch1, fs=fs1, btype='high')
            b11, a11 = signal.cheby1(order1, rp1, fcl1, fs=fs1, btype='low')
            fch1_butter = fch1 / fs1
            fcl1_butter = fcl1 / fs1
            bp1, ap1 = signal.butter(order1, [fcl1_butter, fch1_butter], btype='band')
            filtered_x1 = signal.lfilter(b1, a1, w_src2)
            filtered_x11 = signal.lfilter(b11, a11, w_src2)
            filtered_xp1 = signal.lfilter(bp1, ap1, w_src2)

            # MPF dropout rate
            filtered_w_src_high = np.float32(filtered_x1)
            size_high = int(0.2 * filtered_w_src_high.shape[0])
            filtered_w_src_high_id = filtered_w_src_high[
                                     np.random.choice(filtered_w_src_high.shape[0], size_high, replace=False), :]
            w_src_high = torch.tensor(filtered_w_src_high_id)
            w_src_high = w_src_high.cuda()

            filtered_w_src_low = np.float32(filtered_x11)
            size_low = int(0.2 * filtered_w_src_low.shape[0])
            filtered_w_src_low_id = filtered_w_src_low[
                                    np.random.choice(filtered_w_src_low.shape[0], size_low, replace=False), :]
            w_src_low = torch.tensor(filtered_w_src_low_id)
            w_src_low = w_src_low.cuda()

            filtered_w_src_band = np.float32(filtered_xp1)
            size_band = filtered_w_src_band.shape[0] - size_high - size_low
            filtered_w_src_band_id = filtered_w_src_band[
                                     np.random.choice(filtered_w_src_band.shape[0], size_band, replace=False), :]
            w_src_band = torch.tensor(filtered_w_src_band_id)
            w_src_band = w_src_band.cuda()

            w_src = torch.cat([w_src_low, w_src_band, w_src_high], 0)
            # MLP
            if w_src.dtype != self.mlp_src[0].weight.dtype:
                w_src = w_src.to(self.mlp_src[0].weight.dtype)
            w_src = self.mlp_src(w_src)
            w_src = IFilter(w_src)

            w_dst_tmp = Filter(node_emb)
            w_dst = w_dst_tmp
            fs2 = w_dst.shape[0]
            t2 = np.arange(0, 1, 1 / fs2)
            w_dst1 = w_dst.cpu()
            w_dst2 = w_dst1.detach().numpy()
            # MPF cutoff frequency
            fch2 = 10000
            fcl2 = 5000
            bandwidth2 = fch2 - fcl2
            rp2 = 5
            order2 = 5
            b2, a2 = signal.cheby2(order2, rp2, fch2, fs=fs2, btype='high')
            b22, a22 = signal.cheby1(order2, rp2, fcl2, fs=fs2, btype='low')
            fch2_butter = fch2 / fs2
            fcl2_butter = fcl2 / fs2
            bp2, ap2 = signal.butter(order2, [fcl2_butter, fch2_butter], btype='band')
            filtered_x2 = signal.lfilter(b2, a2, w_dst2)
            filtered_x22 = signal.lfilter(b22, a22, w_dst2)
            filtered_xp2 = signal.lfilter(bp2, ap2, w_src2)

            filtered_w_dst_high = np.float32(filtered_x2)
            filtered_w_dst_high_id = filtered_w_dst_high[
                                     np.random.choice(filtered_w_dst_high.shape[0], size_high, replace=False), :]
            w_dst_high = torch.tensor(filtered_w_dst_high_id)
            w_dst_high = w_dst_high.cuda()

            filtered_w_dst_low = np.float32(filtered_x22)
            filtered_w_dst_low_id = filtered_w_dst_low[
                                    np.random.choice(filtered_w_dst_low.shape[0], size_low, replace=False), :]
            w_dst_low = torch.tensor(filtered_w_dst_low_id)
            w_dst_low = w_dst_low.cuda()

            filtered_w_dst_band = np.float32(filtered_xp2)
            filtered_w_dst_band_id = filtered_w_dst_band[
                                     np.random.choice(filtered_w_dst_band.shape[0], size_band, replace=False), :]
            w_dst_band = torch.tensor(filtered_w_dst_band_id)
            w_dst_band = w_dst_band.cuda()

            w_dst = torch.cat([w_dst_low, w_dst_band, w_dst_high], 0)
            # MLP
            if w_dst.dtype != self.mlp_dst[0].weight.dtype:
                w_dst = w_dst.to(self.mlp_dst[0].weight.dtype)
            w_dst = self.mlp_dst(w_dst)
            w_dst = IFilter(w_dst)

            graph.srcdata.update({'inl': w_src})
            graph.dstdata.update({'inr': w_dst})
            graph.apply_edges(fn.u_add_v('inl', 'inr', 'ine'))
            n_weight = graph.edata.pop('ine')
            weight = n_weight
        if relation_emb is not None and self.mlp_edge is not None:
            w_edge = self.mlp_edge(relation_emb)
            graph.edata.update({'ee': w_edge})
            e_weight = graph.edata.pop('ee')
            weight += e_weight
        weight = weight.squeeze()
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * th.rand(weight.size()) + (1 - bias)
        gate_inputs = th.log(eps) - th.log(1 - eps)
        gate_inputs = gate_inputs.to(node_emb.device)
        gate_inputs = (gate_inputs + weight) / temperature
        aug_edge_weight = th.sigmoid(gate_inputs).squeeze()
        edge_drop_out_prob = 1 - aug_edge_weight
        reg = edge_drop_out_prob.mean()
        aug_edge_weight = aug_edge_weight.unsqueeze(-1).unsqueeze(-1)
        #print(aug_edge_weight.size())
        return reg, aug_edge_weight

# pylint: enable=W0235
class myGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0., attn_drop=0.,
                 negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False, bias=False, alpha=0.):
        super(myGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, res_attn=None, edge_weight = None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if edge_weight is not None:
                graph.edata['a'] = graph.edata['a'] * edge_weight
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()

class DropLearner1(nn.Module):
    def __init__(self, node_dim, edge_dim=None, mlp_edge_model_dim=64):
        super(DropLearner1, self).__init__()
        self.mlp_src1 = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.mlp_dst1 = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.mlp_con1 = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.concat1 = False

        if edge_dim is not None:
            self.mlp_edge1 = nn.Sequential(
                nn.Linear(edge_dim, mlp_edge_model_dim),
                nn.ReLU(),
                nn.Linear(mlp_edge_model_dim, 1)
            )
        else:
            self.mlp_edge1 = None
        self.init_emb1()

    def init_emb1(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    def get_weight(self, head_emb, tail_emb, temperature=0.5, relation_emb=None, edge_type=None):
        if self.concat1:
            weight = self.mlp_con1(head_emb + tail_emb)
            w_src = self.mlp_src1(head_emb)
            w_dst = self.mlp_dst1(tail_emb)
            weight += w_src + w_dst
        else:
            w_src = self.mlp_src1(head_emb)
            w_dst = self.mlp_dst1(tail_emb)
            weight = w_src + w_dst
        if relation_emb is not None and self.mlp_edge1 is not None:
            e_weight = self.mlp_edge1(relation_emb)
            weight += e_weight
        weight = weight.squeeze()
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * th.rand(weight.size()) + (
                    1 - bias)
        gate_inputs = th.log(eps) - th.log(1 - eps)
        gate_inputs = gate_inputs.to(head_emb.device)
        gate_inputs = (gate_inputs + weight) / temperature
        aug_edge_weight = th.sigmoid(gate_inputs).squeeze()
        edge_drop_out_prob = 1 - aug_edge_weight
        reg = edge_drop_out_prob.mean()
        # print(aug_edge_weight.size())
        return reg.detach(), aug_edge_weight.detach()

    def forward(self, node_emb, graph, temperature=0.5, relation_emb=None, edge_type=None):
        if self.concat1:
            w_con = node_emb
            graph.srcdata.update({'in': w_con})
            graph.apply_edges(fn.u_add_v('in', 'in', 'con'))
            n_weight = graph.edata.pop('con')
            weight = self.mlp_con1(n_weight)
            w_src = self.mlp_src1(node_emb)
            w_dst = self.mlp_dst1(node_emb)
            graph.srcdata.update({'inl': w_src})
            graph.dstdata.update({'inr': w_dst})
            graph.apply_edges(fn.u_add_v('inl', 'inr', 'ine'))
            weight += graph.edata.pop('ine')
            # print(weight.size())
        else:
            # FFT
            w_src_tmp = Filter(node_emb)
            w_src = w_src_tmp
            fs1 = w_src.shape[0]
            t1 = np.arange(0, 1, 1 / fs1)
            w_src1 = w_src.cpu()
            w_src2 = w_src1.detach().numpy()
            w_src22 = prob(w_src2)
            # MPF cutoff frequency
            fch1 = 10000
            fcl1 = 5000
            bandwidth1 = fch1 - fcl1
            rp1 = 5
            order1 = 5
            b1, a1 = signal.cheby2(order1, rp1, fch1, fs=fs1, btype='high')
            b11, a11 = signal.cheby1(order1, rp1, fcl1, fs=fs1, btype='low')
            fch1_butter = fch1 / fs1
            fcl1_butter = fcl1 / fs1
            bp1, ap1 = signal.butter(order1, [fcl1_butter, fch1_butter], btype='band')
            filtered_x1 = signal.lfilter(b1, a1, w_src2)
            filtered_x11 = signal.lfilter(b11, a11, w_src2)
            filtered_xp1 = signal.lfilter(bp1, ap1, w_src2)

            # MPF dropout rate
            filtered_w_src_high = np.float32(filtered_x1)
            size_high = int(0.2 * filtered_w_src_high.shape[0])
            filtered_w_src_high_id = filtered_w_src_high[
                                     np.random.choice(filtered_w_src_high.shape[0], size_high, replace=False), :]
            w_src_high = torch.tensor(filtered_w_src_high_id)
            w_src_high = w_src_high.cuda()

            filtered_w_src_low = np.float32(filtered_x11)
            size_low = int(0.2 * filtered_w_src_low.shape[0])
            filtered_w_src_low_id = filtered_w_src_low[
                                    np.random.choice(filtered_w_src_low.shape[0], size_low, replace=False), :]
            w_src_low = torch.tensor(filtered_w_src_low_id)
            w_src_low = w_src_low.cuda()

            filtered_w_src_band = np.float32(filtered_xp1)
            size_band = filtered_w_src_band.shape[0] - size_high - size_low
            filtered_w_src_band_id = filtered_w_src_band[
                                     np.random.choice(filtered_w_src_band.shape[0], size_band, replace=False), :]
            w_src_band = torch.tensor(filtered_w_src_band_id)
            w_src_band = w_src_band.cuda()

            w_src = torch.cat([w_src_low, w_src_band, w_src_high], 0)
            # MLP
            if w_src.dtype != self.mlp_src1[0].weight.dtype:
                w_src = w_src.to(self.mlp_src1[0].weight.dtype)
            w_src = self.mlp_src1(w_src)
            w_src = IFilter(w_src)

            w_dst_tmp = Filter(node_emb)
            w_dst = w_dst_tmp
            fs2 = w_dst.shape[0]
            t2 = np.arange(0, 1, 1 / fs2)
            w_dst1 = w_dst.cpu()
            w_dst2 = w_dst1.detach().numpy()
            # MPF cutoff frequency
            fch2 = 10000
            fcl2 = 5000
            bandwidth2 = fch2 - fcl2
            rp2 = 5
            order2 = 5
            b2, a2 = signal.cheby2(order2, rp2, fch2, fs=fs2, btype='high')
            b22, a22 = signal.cheby1(order2, rp2, fcl2, fs=fs2, btype='low')
            fch2_butter = fch2 / fs2
            fcl2_butter = fcl2 / fs2
            bp2, ap2 = signal.butter(order2, [fcl2_butter, fch2_butter], btype='band')
            filtered_x2 = signal.lfilter(b2, a2, w_dst2)
            filtered_x22 = signal.lfilter(b22, a22, w_dst2)
            filtered_xp2 = signal.lfilter(bp2, ap2, w_src2)

            filtered_w_dst_high = np.float32(filtered_x2)
            filtered_w_dst_high_id = filtered_w_dst_high[
                                     np.random.choice(filtered_w_dst_high.shape[0], size_high, replace=False), :]
            w_dst_high = torch.tensor(filtered_w_dst_high_id)
            w_dst_high = w_dst_high.cuda()

            filtered_w_dst_low = np.float32(filtered_x22)
            filtered_w_dst_low_id = filtered_w_dst_low[
                                    np.random.choice(filtered_w_dst_low.shape[0], size_low, replace=False), :]
            w_dst_low = torch.tensor(filtered_w_dst_low_id)
            w_dst_low = w_dst_low.cuda()

            filtered_w_dst_band = np.float32(filtered_xp2)
            filtered_w_dst_band_id = filtered_w_dst_band[
                                     np.random.choice(filtered_w_dst_band.shape[0], size_band, replace=False), :]
            w_dst_band = torch.tensor(filtered_w_dst_band_id)
            w_dst_band = w_dst_band.cuda()

            w_dst = torch.cat([w_dst_low, w_dst_band, w_dst_high], 0)
            # MLP
            if w_dst.dtype != self.mlp_dst1[0].weight.dtype:
                w_dst = w_dst.to(self.mlp_dst1[0].weight.dtype)
            w_dst = self.mlp_dst1(w_dst)
            w_dst = IFilter(w_dst)

            graph.srcdata.update({'inl': w_src})
            graph.dstdata.update({'inr': w_dst})
            graph.apply_edges(fn.u_add_v('inl', 'inr', 'ine'))
            n_weight = graph.edata.pop('ine')
            weight = n_weight
        if relation_emb is not None and self.mlp_edge1 is not None:
            w_edge = self.mlp_edge1(relation_emb)
            graph.edata.update({'ee': w_edge})
            e_weight = graph.edata.pop('ee')
            weight += e_weight
        weight = weight.squeeze()
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * th.rand(weight.size()) + (1 - bias)
        gate_inputs = th.log(eps) - th.log(1 - eps)
        gate_inputs = gate_inputs.to(node_emb.device)
        gate_inputs = (gate_inputs + weight) / temperature
        aug_edge_weight = th.sigmoid(gate_inputs).squeeze()
        edge_drop_out_prob = 1 - aug_edge_weight
        reg = edge_drop_out_prob.mean()
        aug_edge_weight = aug_edge_weight.unsqueeze(-1).unsqueeze(-1)
        # print(aug_edge_weight.size())
        return reg, aug_edge_weight

class DropLearner2(nn.Module):
    def __init__(self, node_dim, edge_dim=None, mlp_edge_model_dim=64):
        super(DropLearner2, self).__init__()
        self.mlp_src2 = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.mlp_dst2 = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.mlp_con2 = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.concat2 = False

        if edge_dim is not None:
            self.mlp_edge2 = nn.Sequential(
                nn.Linear(edge_dim, mlp_edge_model_dim),
                nn.ReLU(),
                nn.Linear(mlp_edge_model_dim, 1)
            )
        else:
            self.mlp_edge2 = None
        self.init_emb2()

    def init_emb2(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, node_emb, graph, temperature=0.5, relation_emb=None, edge_type=None):
        if self.concat2:
            w_con = node_emb
            graph.srcdata.update({'in': w_con})
            graph.apply_edges(fn.u_add_v('in', 'in', 'con'))
            n_weight = graph.edata.pop('con')
            weight = self.mlp_con2(n_weight)
            w_src = self.mlp_src2(node_emb)
            w_dst = self.mlp_dst2(node_emb)
            graph.srcdata.update({'inl': w_src})
            graph.dstdata.update({'inr': w_dst})
            graph.apply_edges(fn.u_add_v('inl', 'inr', 'ine'))
            weight += graph.edata.pop('ine')
            # print(weight.size())
        else:
            # FFT
            w_src_tmp = Filter(node_emb)
            w_src = w_src_tmp
            fs1 = w_src.shape[0]
            t1 = np.arange(0, 1, 1 / fs1)
            w_src1 = w_src.cpu()
            w_src2 = w_src1.detach().numpy()
            w_src22 = prob(w_src2)
            # DPF cutoff frequency
            fc1 = 6000
            rp1 = 5
            order1 = 5
            b1, a1 = signal.cheby2(order1, rp1, fc1, fs=fs1, btype='high')
            b11, a11 = signal.cheby1(order1, rp1, fc1, fs=fs1, btype='low')
            filtered_x1 = signal.lfilter(b1, a1, w_src2)
            filtered_x11 = signal.lfilter(b11, a11, w_src2)

            # DPF dropout rate
            filtered_w_src_high = np.float32(filtered_x1)
            size_high = int(0.2 * filtered_w_src_high.shape[0])
            filtered_w_src_high_id = filtered_w_src_high[np.random.choice(filtered_w_src_high.shape[0], size_high, replace=False), : ]
            w_src_high = torch.tensor(filtered_w_src_high_id)
            w_src_high = w_src_high.cuda()

            filtered_w_src_low = np.float32(filtered_x11)
            size_low = filtered_w_src_low.shape[0]- size_high
            filtered_w_src_low_id = filtered_w_src_low[np.random.choice(filtered_w_src_low.shape[0], size_low, replace=False), :]
            w_src_low = torch.tensor(filtered_w_src_low_id)
            w_src_low = w_src_low.cuda()

            w_src = torch.cat([w_src_low, w_src_high], 0)
            # MLP
            if w_src.dtype != self.mlp_src2[0].weight.dtype:
                w_src = w_src.to(self.mlp_src2[0].weight.dtype)
            w_src = self.mlp_src2(w_src)
            w_src = IFilter(w_src)

            w_dst_tmp = Filter(node_emb)
            w_dst = w_dst_tmp
            fs2 = w_dst.shape[0]
            t2 = np.arange(0, 1, 1 / fs2)
            w_dst1 = w_dst.cpu()
            w_dst2 = w_dst1.detach().numpy()
            # DPF cutoff frequency
            fc2 = 6000
            rp2 = 5
            order2 = 5
            b2, a2 = signal.cheby2(order2, rp2, fc2, fs=fs2, btype='high')
            b22, a22 = signal.cheby1(order2, rp2, fc2, fs=fs2, btype='low')
            filtered_x2 = signal.lfilter(b2, a2, w_dst2)
            filtered_x22 = signal.lfilter(b22, a22, w_dst2)

            filtered_w_dst_high = np.float32(filtered_x2)
            size_high = int(0.2 * filtered_w_dst_high.shape[0])
            filtered_w_dst_high_id = filtered_w_dst_high[
                                     np.random.choice(filtered_w_dst_high.shape[0], size_high, replace=False), :]
            w_dst_high = torch.tensor(filtered_w_dst_high_id)
            w_dst_high = w_dst_high.cuda()

            filtered_w_dst_low = np.float32(filtered_x22)
            size_low = filtered_w_dst_low.shape[0] - size_high
            filtered_w_dst_low_id = filtered_w_dst_low[
                                    np.random.choice(filtered_w_dst_low.shape[0], size_low, replace=False), :]
            w_dst_low = torch.tensor(filtered_w_dst_low_id)
            w_dst_low = w_dst_low.cuda()

            w_dst = torch.cat([w_dst_low, w_dst_high], 0)
            # MLP
            if w_dst.dtype != self.mlp_dst2[0].weight.dtype:
                w_dst = w_dst.to(self.mlp_dst2[0].weight.dtype)
            w_dst = self.mlp_dst2(w_dst)
            w_dst = IFilter(w_dst)

            graph.srcdata.update({'inl': w_src})
            graph.dstdata.update({'inr': w_dst})
            graph.apply_edges(fn.u_add_v('inl', 'inr', 'ine'))
            n_weight = graph.edata.pop('ine')
            weight = n_weight

        if relation_emb is not None and self.mlp_edge2 is not None:
            w_edge = self.mlp_edge2(relation_emb)
            graph.edata.update({'ee': w_edge})
            e_weight = graph.edata.pop('ee')
            weight += e_weight
        weight = weight.squeeze()
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * th.rand(weight.size()) + (1 - bias)
        gate_inputs = th.log(eps) - th.log(1 - eps)
        gate_inputs = gate_inputs.to(node_emb.device)
        gate_inputs = (gate_inputs + weight) / temperature
        aug_edge_weight = th.sigmoid(gate_inputs).squeeze()
        edge_drop_out_prob = 1 - aug_edge_weight
        reg = edge_drop_out_prob.mean()
        aug_edge_weight = aug_edge_weight.unsqueeze(-1).unsqueeze(-1)
        # print(aug_edge_weight.size())
        return reg, aug_edge_weight
