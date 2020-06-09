import torch.nn as nn
import torch
#rom hyptorch.nn import HyperbolicDistanceLayer
import numpy as np
import geoopt

ball = geoopt.PoincareBall(c=1.0)

class HierarchicalLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(HierarchicalLoss, self).__init__()
        self.margin = margin
        self.ball = ball

    def forward(self, parent, child):
        d_parent = self.ball.dist0(parent)
        d_child = self.ball.dist0(child)
        return torch.max(d_parent-d_child+self.margin)


class HTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(HTripletLoss, self).__init__()
        self.ball = ball
        self.margin = margin

    def forward(self, a, p, n):
        margin = self.margin
        d_a_p = self.ball.dist(a, p)
        d_a_n = self.ball.dist(a, n)
        #import ipdb as pdb
        #pdb.set_trace()
        distance = d_a_p - d_a_n +margin
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        return loss

"""
class HTripletLoss(nn.Module):
    def __init__(self):
        super(HTripletLoss, self).__init__()
        self.pdist = HyperbolicDistanceLayer(c=1.)  

    def forward(self, a, p, n, margin=0.2):
        d_a_p = self.pdist(a, p)
        d_a_n = self.pdist(a, n)
        distance = d_a_p - d_a_n +margin
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        return loss
"""

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, a, p, n):
        margin = self.margin
        # = nn.PairwiseDistance(p=2)
        # TODO: support batch size > 1
        distance = torch.norm(a-p) - torch.norm(a-n) + margin
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        return loss


