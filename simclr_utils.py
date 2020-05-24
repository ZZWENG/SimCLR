import numpy as np
import torch

def overlaps(m1, m2, thres):
    return (m1 * m2).sum() > thres


def overlapping_idx(anchor, masks, thres): 
    flags = np.array([overlaps(anchor, m, thres) for m in masks])
    return np.where(flags == False)[0], np.where(flags == True)[0]


def size_of(cut):
    return cut.shape[0] * cut.shape[1]

# post processing
def keep(i, masks):
    # retuns true if masks[i] overlaps with some other masks by more than x% of itself
    masks_out = []
    for j in range(len(masks)):
        if j == i: continue
        area = masks[i].sum().item() * 1.
        if area < 280:
            return False
        if (masks[j] * masks[i]).sum() / area > 0.7 and area < masks[j].sum():
#             print((masks[j] * masks[i]).sum() / masks[i].sum())
            return False
    return True
