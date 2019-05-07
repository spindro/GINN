# Copyright 2019 Indro Spinelli. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import torch
import numpy as np
import torch.nn as nn
from dgl import DGLGraph
import dgl.function as fn
import torch.nn.functional as F
import torch.autograd as autograd
from ginn.utils import proper_onehot, batch_mask
from dgl.data import register_data_args, load_data


class GCL(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation, dropout, bias=True):
        super(GCL, self).__init__()
        self.g = g
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.0
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.matmul(h, self.weight)
        # normalization by square root of src degree
        h = h * self.g.ndata["norm"]
        self.g.ndata["h"] = h
        self.g.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
        h = self.g.ndata.pop("h")
        # normalization by square root of dst degree
        h = h * self.g.ndata["norm"]
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h


class GCL_skip(nn.Module):
    def __init__(self, g, f, in_feats, out_feats, activation, dropout, bias=True):
        super(GCL_skip, self).__init__()
        self.g = g
        self.f = f
        self.wh = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.ws = nn.Parameter(torch.Tensor(out_feats, out_feats))

        if bias:
            self.bh = nn.Parameter(torch.Tensor(out_feats))
            self.bs = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bh = None
            self.bs = None

        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.0
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1.0 / math.sqrt(self.wh.size(1))
        self.wh.data.uniform_(-stdv1, stdv1)
        if self.bh is not None:
            self.bh.data.uniform_(-stdv1, stdv1)
        stdv2 = 1.0 / math.sqrt(self.ws.size(1))
        self.ws.data.uniform_(-stdv2, stdv2)
        if self.bs is not None:
            self.bs.data.uniform_(-stdv2, stdv2)

    def forward(self, h, s):
        if self.dropout:
            h = self.dropout(h)
            s = self.dropout(s)
        h = torch.matmul(h, self.wh)
        s = torch.matmul(s, self.ws)
        h = h * self.g.ndata["norm"]
        s = s * self.f.ndata["norm"]
        self.g.ndata["h"] = h
        self.f.ndata["s"] = s
        self.g.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
        self.f.update_all(fn.copy_src(src="s", out="m"), fn.sum(msg="m", out="s"))
        h = self.g.ndata.pop("h")
        s = self.f.ndata.pop("s")
        # normalization by square root of dst degree
        h = h * self.g.ndata["norm"]
        s = s * self.f.ndata["norm"]
        # bias
        if self.bh is not None:
            h = h + self.bh
            s = s + self.bs
        h = h + s
        if self.activation:
            h = self.activation(h)
        return h


class GCL_skip_global(nn.Module):
    def __init__(self, g, f, in_feats, out_feats, activation, dropout, bias=True):
        super(GCL_skip_global, self).__init__()
        self.g = g
        self.f = f
        self.wh = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.ws = nn.Parameter(torch.Tensor(out_feats, out_feats))
        self.wm = nn.Parameter(torch.Tensor(out_feats, out_feats))

        if bias:
            self.bh = nn.Parameter(torch.Tensor(out_feats))
            self.bs = nn.Parameter(torch.Tensor(out_feats))
            self.bm = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bh = None
            self.bs = None
            self.bm = None

        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.0
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1.0 / math.sqrt(self.wh.size(1))
        self.wh.data.uniform_(-stdv1, stdv1)
        if self.bh is not None:
            self.bh.data.uniform_(-stdv1, stdv1)
        stdv2 = 1.0 / math.sqrt(self.ws.size(1))
        self.ws.data.uniform_(-stdv2, stdv2)
        if self.bs is not None:
            self.bs.data.uniform_(-stdv2, stdv2)
        stdv3 = 1.0 / math.sqrt(self.wm.size(1))
        self.wm.data.uniform_(-stdv3, stdv3)
        if self.bm is not None:
            self.bm.data.uniform_(-stdv3, stdv3)

    def forward(self, h, s, m):
        if self.dropout:
            h = self.dropout(h)
        h = torch.matmul(h, self.wh)
        s = torch.matmul(s, self.ws)
        m = torch.matmul(m, self.wm)
        # normalization by square root of src degree
        h = h * self.g.ndata["norm"]
        s = s * self.f.ndata["norm"]
        self.g.ndata["h"] = h
        self.f.ndata["s"] = s
        self.g.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
        self.f.update_all(fn.copy_src(src="s", out="m"), fn.sum(msg="m", out="s"))
        h = self.g.ndata.pop("h")
        s = self.f.ndata.pop("s")
        # normalization by square root of dst degree
        h = h * self.g.ndata["norm"]
        s = s * self.f.ndata["norm"]
        # bias
        if self.bh is not None:
            h = h + self.bh
            s = s + self.bs
            m = m + self.bm
        # sum
        h = h + s + m
        if self.activation:
            h = self.activation(h)
        return h


class GCL_global(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation, dropout, bias=True):
        super(GCL_global, self).__init__()
        self.g = g
        self.wh = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.wm = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bh = nn.Parameter(torch.Tensor(out_feats))
            self.bm = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bh = None
            self.bm = None

        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.0
        self.reset_parameters()

    def reset_parameters(self):
        stdvh = 1.0 / math.sqrt(self.wh.size(1))
        self.wh.data.uniform_(-stdvh, stdvh)
        if self.bh is not None:
            self.bh.data.uniform_(-stdvh, stdvh)
        stdvm = 1.0 / math.sqrt(self.wm.size(1))
        self.wm.data.uniform_(-stdvm, stdvm)
        if self.bm is not None:
            self.bm.data.uniform_(-stdvm, stdvm)

    def forward(self, h, m):
        if self.dropout:
            h = self.dropout(h)
        h = torch.matmul(h, self.wh)
        m = torch.matmul(m, self.wm)
        print(m.shape)
        # normalization by square root of src degree
        h = h * self.g.ndata["norm"]
        self.g.ndata["h"] = h
        self.g.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
        h = self.g.ndata.pop("h")
        # normalization by square root of dst degree
        h = h * self.g.ndata["norm"]
        # bias
        if self.bh is not None:
            h = h + self.bh
            m = m + self.bm
        h = h + m
        if self.activation:
            h = self.activation(h)

        return h


class ginn_autoencoder(nn.Module):
    def __init__(self, g, mask, in_feats, h_feats, activation, dropout):
        super(ginn_autoencoder, self).__init__()
        self.mask = mask
        self.masked_gcn = GCL(g, in_feats, h_feats, activation, dropout)
        self.output_gcn = GCL(g, h_feats, in_feats, torch.sigmoid, dropout)

    def forward(self, features):
        features = torch.mul(features, self.mask)
        h = self.masked_gcn(features)
        h = self.output_gcn(h)
        return h


class ginn_autoencoder_skip(nn.Module):
    def __init__(self, g, f, mask, in_feats, h_feats, activation, dropout):
        super(ginn_autoencoder_skip, self).__init__()
        self.mask = mask
        self.masked_gcn = GCL(g, in_feats, h_feats, activation, dropout)
        self.output_gcn = GCL_skip(g, f, h_feats, in_feats, torch.sigmoid, dropout)

    def forward(self, features):
        features = torch.mul(features, self.mask)
        h = self.masked_gcn(features)
        h = self.output_gcn(h, features)
        return h


class ginn_autoencoder_global(nn.Module):
    def __init__(self, g, f, m, mask, in_feats, h_feats, activation, dropout):
        super(ginn_autoencoder_global, self).__init__()
        self.mask = mask
        self.m = m
        self.masked_gcn = GCL(g, in_feats, h_feats, activation, dropout)
        self.output_gcn = GCL_skip_global(
            g, f, h_feats, in_feats, torch.sigmoid, dropout
        )

    def forward(self, features):
        features = torch.mul(features, self.mask)
        h = self.masked_gcn(features)
        h = self.output_gcn(h, features, self.m)
        return h


class ginn_critic(nn.Module):
    def __init__(self, in_feats, h_feats, dropout):
        super(ginn_critic, self).__init__()

        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, in_feats)
        self.linear3 = nn.Linear(in_feats, 1)
        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU()
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.0

    def forward(self, features):
        h = self.relu(self.linear1(features))
        if self.dropout:
            h = self.dropout(h)
        h = self.relu(self.linear2(h))
        if self.dropout:
            h = self.dropout(h)
        h = self.linear3(h)
        return h


def gradient_penalty(net, real_data, fake_data, device):

    alpha = torch.rand(real_data.shape[0], 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    c_interpolates = net(interpolates)

    gradients = autograd.grad(
        outputs=c_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(c_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def hard_gradient_penalty(net, real_data, fake_data, device):

    mask = torch.FloatTensor(real_data.shape).to(device).uniform_() > 0.5
    inv_mask = ~mask
    mask, inv_mask = mask.float(), inv_mask.float()

    interpolates = mask * real_data + inv_mask * fake_data
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    c_interpolates = net(interpolates)

    gradients = autograd.grad(
        outputs=c_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(c_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gp = (gradients.norm(2, dim=1) - 1).pow(2).mean()
    return gp


def imputation(model, features, onehot_cat_cols):
    model.eval()
    with torch.no_grad():
        imp = model(features)
        imp = imp.cpu()
        imp = proper_onehot(imp.numpy(), onehot_cat_cols)
    return imp


def multiple_imputation(model, features, onehot_cat_cols, imputation_num):
    model.train()
    model.masked_gcn.dropout.p = 0.0
    model.output_gcn.dropout.p = 0.1
    tmp = []
    with torch.no_grad():
        for _ in range(imputation_num):
            imp = model(features)
            imp = imp.cpu()
            imp = proper_onehot(imp.numpy(), onehot_cat_cols)
            tmp.append(imp)
        m_imp = np.asarray(tmp).mean(axis=0)
    return m_imp
