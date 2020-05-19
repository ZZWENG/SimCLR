import torch.nn as nn
import torch
from hyptorch.nn import HyperbolicDistanceLayer
import numpy as np


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


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, a, p, n, margin=0.2):
        # = nn.PairwiseDistance(p=2)
        # TODO: support batch size > 1
        distance = torch.norm(a-p) - torch.norm(a-n) + margin
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        return loss


