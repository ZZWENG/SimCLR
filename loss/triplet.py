import torch.nn as nn
import torch


class TripletLoss(nn.Module):
    def __init__(self, device):
        super(TripletLoss, self).__init__()
        self.device = device

    def forward(self, a, p, n, margin=0.2):
        d = nn.PairwiseDistance(p=2)
        distance = d(a, p) - d(a, n) + margin
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        return loss
