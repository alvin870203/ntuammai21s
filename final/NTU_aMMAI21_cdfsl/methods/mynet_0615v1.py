# This code is modified from https://github.com/jakesnell/prototypical-networks 

from torch.nn.modules import loss
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

import utils

import itertools
# from pytorch_metric_learning import losses
from utils import adjust_learning_rate
from pytorch_metric_learning import losses, miners, distances, reducers

class MyNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(MyNet, self).__init__( model_func,  n_way, n_support)
        self.loss_fn  = nn.CrossEntropyLoss().to(self.device)
        
        self.z_proto = None
        # self.metric_loss_fn = losses.TripletMarginLoss(margin=0.5)
        self.z_support = None
        self.z_query = None

        ### pytorch-metric-learning stuff ###
        # self.distance = distances.CosineSimilarity()
        # self.reducer = reducers.ThresholdReducer(low = 0)
        # self.metric_loss_fn = losses.TripletMarginLoss(margin = 0.1, distance = self.distance, reducer = self.reducer)
        self.metric_loss_fn = losses.ArcFaceLoss(self.n_way, 512).to(self.device)
        # self.metric_loss_fn = losses.ProxyNCALoss(self.n_way, 512).to(self.device)
        # self.mining_func = miners.TripletMarginMiner(margin = 0.1, distance = self.distance, type_of_triplets = "semihard")
        self.loss_optimizer = torch.optim.SGD(self.metric_loss_fn.parameters(), lr=0.01)


    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim] n_dim=512
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )


        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        # print(dists, self.n_query, self.n_way)

        self.z_proto = z_proto
        self.z_support = z_support.view(self.n_way * self.n_support, -1)
        self.z_query = z_query
        # print(self.z_support.size(), self.z_query.size())

        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        y_support = Variable(y_support.cuda())

        y_proto = torch.from_numpy(np.array(range(self.n_way)))
        y_proto = Variable(y_proto.cuda())

        # self.loss_optimizer.zero_grad()


        # embeddings = torch.cat((self.z_support, self.z_query), 0)
        # labels = torch.cat((y_support, y_query), 0)
        # embeddings = torch.cat((self.z_proto, self.z_query), 0)
        # labels = torch.cat((y_proto, y_query), 0)
        embeddings = torch.cat((self.z_support, self.z_proto, self.z_query), 0)
        labels = torch.cat((y_support, y_proto, y_query), 0)
        # embeddings = self.z_support
        # labels = y_support
        # indices_tuple = self.mining_func(embeddings, labels)
        metric_loss = self.metric_loss_fn(embeddings, labels)#, indices_tuple)

        # self.loss_optimizer.step()

        # metric_loss_pos = 0
        # metric_loss_neg = 0
        # # n_pos = 0
        # # n_neg = 0
        # for idx_a, anchor in enumerate(embeddings):
        #     for idx_q in range(idx_a+1, len(embeddings)):
        #         if labels[idx_a] == labels[idx_q]:
        #             # n_pos += 1
        #             metric_loss_pos += torch.pow(embeddings[idx_a] - embeddings[idx_q], 2).sum()
        #         else:
        #             # n_neg += 1
        #             metric_loss_neg -= torch.pow(embeddings[idx_a] - embeddings[idx_q], 2).sum()
        # # print(n_pos, n_neg)
        # metric_loss = (0 if (metric_loss_pos < 500) else metric_loss_pos) + torch.clamp(metric_loss_neg,min=-50000, max=-1000)
        # print(f'metric loss pos: {metric_loss_pos}; metric loss neg: { metric_loss_neg}')

        # print(self.z_proto.requires_grad, self.z_query.requires_grad, metric_loss.requires_grad)
        # print(self.z_support.size(), self.z_query.size(), y_support.size(), y_query.size())
        # metric_loss = self.metric_loss_fn(torch.cat((self.z_support, self.z_query), 0), torch.cat((y_support, y_query), 0))
        metric_loss_lambda = 1
        classifier_loss = self.loss_fn(scores, y_query)
        print(f'classifier loss: {classifier_loss}; metric loss: {metric_loss}')
        loss = classifier_loss
        # loss = classifier_loss + metric_loss_lambda * metric_loss
        # loss = metric_loss_lambda * metric_loss

        # # pairs_list = list(itertools.combinations([0,1,2,3,4], r=2))
        # proto_loss = 0.0
        # for p in itertools.combinations([0,1,2,3,4], r=2):
        #     # print(z_proto[p[0]].size())
        #     proto_loss -= torch.pow(self.z_proto[p[0]] - self.z_proto[p[1]], 2).sum()
        # proto_loss_lambda = 0.001
        # loss = self.loss_fn(scores, y_query) + proto_loss_lambda * proto_loss

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
        # loss = self.loss_fn(scores, y_query)

        return loss


    def train_loop(self, epoch, train_loader, optimizer ):
        print_freq = 10

        avg_loss=0
        for i, (x,_ ) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support           
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            self.loss_optimizer.zero_grad()
            adjust_learning_rate(optimizer, epoch, lr=0.0001)
            loss = self.set_forward_loss( x )
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()
            self.loss_optimizer.step()

            if i % print_freq==0:
                print(f"lr = {optimizer.state_dict()['param_groups'][0]['lr']}")
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))


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
