# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

import utils

import itertools

class MyNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, use_proto_loss=True):
        super(MyNet, self).__init__( model_func,  n_way, n_support)
        self.loss_fn  = nn.CrossEntropyLoss()
        self.z_proto = None
        self.use_proto_loss = use_proto_loss
        print(f'use proto loss: {self.use_proto_loss}')


    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )


        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        # print(dists, self.n_query, self.n_way)

        self.z_proto = z_proto

        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        # pairs_list = list(itertools.combinations([0,1,2,3,4], r=2))
        proto_loss = 0.0
        for p in itertools.combinations([0,1,2,3,4], r=2):
            # print(z_proto[p[0]].size())
            proto_loss -= torch.pow(self.z_proto[p[0]] - self.z_proto[p[1]], 2).sum()
        proto_loss_lambda = 0.001

        # large margin loss
        # pos_margin = 100
        # # neg_margin = scores.max().item()
        # # margins = np.zeros(scores.size())
        # for idx, y_cur in enumerate(y_query):
        #     # print(idx, y_cur.item())
        #     # scores[idx][:] = y_query[idx][:] - neg_margin
        #     scores[idx][y_cur] -= pos_margin
        # # margins = torch.from_numpy(margins)
        # # for score in scores:
        #     # print(score.size())
        # scores = torch.clamp(scores, min=0.0)
        # print(f'classifier loss: {self.loss_fn(scores, y_query)}; proto loss: {proto_loss}')
        if self.use_proto_loss:
            loss = self.loss_fn(scores, y_query) + proto_loss_lambda * proto_loss ##########################Modified for PRETRAIN
        else:
            loss = self.loss_fn(scores, y_query)
        # loss = self.loss_fn(scores, y_query)

        return loss


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
