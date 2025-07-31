# Copyright (c) 2025 Junyi Peng (pengjy@fit.vut.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MHFA as the backend for SSL models.

From the paper: An attention-based backend allowing efficient fine-tuning 
                of transformer models for speaker verification
Author: Junyi Peng, Oldrich Plchot, Themos Stafylakis, Ladislav Mosner, 
        Lukas Burget, Jan Cernocky
Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10022775
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import random

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

class SSL_BACKEND_MHFA(nn.Module):
    def __init__(self, head_nb=8, feat_dim=768, compression_dim=128, embed_dim=256, nb_layer=13, feature_grad_mult=1.0):
        super(SSL_BACKEND_MHFA, self).__init__()

        self.feature_grad_mult = feature_grad_mult

        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(nb_layer), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(nb_layer), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = feat_dim
        self.cmp_dim = compression_dim
        self.ous_dim = embed_dim

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        # Define linear layer to compute multi-head attention weights
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]
        x = GradMultiply.apply(x, self.feature_grad_mult)

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)

        # Compute attention weights using compressed keys
        att_k = self.att_head(k) # B, T, H

        # Adjust dimensions for computing attention output
        v = v.unsqueeze(-2) # B, T, 1

        # Compute attention output by taking weighted sum of values using softmaxed attention weights
        pooling_outs = torch.sum(v.mul(nn.functional.softmax(att_k, dim=1).unsqueeze(-1)), dim=1)

        # Reshape the tensor before passing through the fully connected layer
        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(pooling_outs)

        return outs

class SSL_BACKEND_CAMHFA(nn.Module):
    def __init__(self, head_nb=8, feat_dim=768, compression_dim=128, embed_dim=256, group_nb=64,nb_layer=13,feature_grad_mult=1.0):
        super(SSL_BACKEND_CAMHFA, self).__init__()
        # Multi Q + Single K + Single V
        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(nb_layer), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(nb_layer), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = feat_dim
        self.cmp_dim = compression_dim
        self.ous_dim = embed_dim
        self.group_nb = group_nb
        self.feature_grad_mult = feature_grad_mult

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        # Define linear layer to compute multi-head attention weights
        # group_len
        group_len = self.head_nb // self.group_nb
        self.att_head = nn.Conv2d(1,self.group_nb,(group_len+1,self.cmp_dim),bias=False,stride=1,padding=(group_len//2, 0)) # Kernel Size [G_len, F]

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.group_nb * self.cmp_dim, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]
        x = GradMultiply.apply(x, self.feature_grad_mult)
        
        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        batch_size = k.size(0)
        k = self.cmp_linear_k(k) # B, T, F
        # k = k.view(batch_size, -1, self.group_nb, self.cmp_dim).permute(2,0,1,3) # Groups, B, T, D
        v = self.cmp_linear_v(v) #B, T, F
        # v = v.view(batch_size, -1, self.group_nb, self.cmp_dim).permute(2,0,1,3) 

        k_att = self.att_head(k.unsqueeze(1)) # B, Head, T, 1
        k_att = k_att.permute(0,2,1,3) #  # B, F_len, 1, Head
        # print(k_att.shape)

        v = v.unsqueeze(-2) # B, T, 1, D

        # Compute attention output by taking weighted sum of values using softmaxed attention weights
        pooling_outs = torch.sum(v.mul(nn.functional.softmax(k_att, dim=1)), dim=1)

        # Reshape the tensor before passing through the fully connected layer
        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(pooling_outs)
        return outs
