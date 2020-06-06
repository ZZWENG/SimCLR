import os
import shutil

import cv2
import numpy as np
import skimage
import torch
import torchvision.transforms as T
from skimage.transform import rotate
from torchvision.transforms import Normalize

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def prepare_seg_triplets(masks, boxes, image, side_len=224):
    n = masks.shape[0]
    boxes = boxes.tensor.detach().cpu().numpy().astype(np.uint32)
    for i in range(n):
        m, b = masks[i], boxes[i]
        if m.sum() < 400:  continue  # skip tiny masks for now
        m = m.view(*m.shape, 1)
        full = image[b[1]:b[3], b[0]:b[2], :] / 255.
        foreground = (m * image)[b[1]:b[3], b[0]:b[2], :] / 255.
        background = ((~m) * image)[b[1]:b[3], b[0]:b[2], :] / 255.
        full = resize_tensor(full, side_len)
        foreground = resize_tensor(foreground, side_len)
        background = resize_tensor(background, side_len)
        yield full, foreground, background


# returns tensor (N, L, L, 3), (N, L, L, 3) for input to model
def prepare_object_pairs(masks, boxes, image, side_len=128):
    n = masks.shape[0]
    boxes = boxes.tensor.detach().cpu().numpy().astype(np.int)
    result = []
    result_aug = []
    for i in range(n):
        m, b = masks[i], boxes[i]
        cropped = (m.view(*m.shape, 1) * image)[b[1]:b[3], b[0]:b[2], :]
        if cropped.shape[0] < 5 or cropped.shape[1] < 5:
            continue
        try:
            cropped = cv2.resize(cropped.cpu().numpy(), (side_len, side_len))

            cropped_aug = rotate(cropped, angle=45, mode='wrap')
            cropped_aug = skimage.util.random_noise(cropped_aug)
            cropped_aug = T.RandomErasing(0.9, scale=(0.02, 0.23))(torch.tensor(cropped_aug))
        except:
            continue
        result += [cropped]
        result_aug += [cropped_aug]
    result = torch.tensor(np.stack(result)).type(torch.float).to(masks.device) / 255.
    result_aug = torch.tensor(np.stack(result_aug)).type(torch.float).to(masks.device) / 255.
    result, result_aug = result.permute(0, 3, 1, 2), result_aug.permute(0, 3, 1, 2)

    result = torch.stack([normalize(result[i]) for i in range(result.shape[0])])
    result_aug = torch.stack([normalize(result_aug[i]) for i in range(result_aug.shape[0])])
    return result, result_aug


def prepare_obj_triplets(masks, boxes, image, augment=False, side_len=224):
    n = masks.shape[0]
    boxes = boxes.tensor.detach().cpu().numpy().astype(np.uint32)
    for i in range(n):
        m1, b = masks[i], boxes[i]
        cut_a = (m1.view(*m1.shape, 1) * image)[b[1]:b[3], b[0]:b[2], :] / 255.
        if cut_a.shape[0] * cut_a.shape[1] < 10:
            continue

        neg_idx, pos_idx = overlapping_idx(m1, masks, 50)

        if len(neg_idx) == 0:  continue

        if len(pos_idx) == 0:
            if augment:
                cut_p = torch.tensor(rotate(cut_a.cpu().numpy(), angle=25, mode='wrap')).type(torch.float).to(m1.device)
            else:
                continue
        else:
            for j in range(len(pos_idx)):
                i_p = pos_idx[j]
                cut_p = apply_mask(image, masks[i_p], boxes[i_p])
                # sample 5 negative masks for each (anchor, positive) pair.
                i_ns = np.random.choice(neg_idx, min(5, len(neg_idx)), replace=False)
                for i_n in i_ns:
                    cut_n = apply_mask(image, masks[i_n], boxes[i_n])

                    if size_of(cut_p) < 10 or size_of(cut_n) < 10 or size_of(cut_a) < 10:
                        continue
                    cut_a = resize_tensor(cut_a, side_len)
                    cut_p = resize_tensor(cut_p, side_len)
                    cut_n = resize_tensor(cut_n, side_len)
                    yield cut_a, cut_p, cut_n


def apply_mask(image, m, b):
    return (m.view(*m.shape, 1) * image)[b[1]:b[3], b[0]:b[2], :] / 255.


def resize_tensor(t, side_len):
    device = t.device
    t_resized = cv2.resize(t.cpu().numpy(), (side_len, side_len))
    return torch.tensor(t_resized).type(torch.float).to(device)


def size_of(cut):
    return cut.shape[0] * cut.shape[1]


def iou(m1, m2):
    union = (m1 | m2).sum().item()
    if not union > 0: return 0.
    return (m1*m2).sum().item() * 1. / union


def overlapping_idx(anchor, masks, thres):
    def overlaps(m1, m2, thres):
        return (m1 * m2).sum() > thres
    flags = np.array([overlaps(anchor, m, thres) for m in masks])
    neg_idx = np.where(flags == False)[0]

    def is_child(m1, m2):
        m1_area, m2_area = m1.sum().item(), m2.sum().item()
        return iou(m1, m2) > 0.5 and m1_area > m2_area
    pos_flags = np.array([is_child(anchor, m) for m in masks])
    pos_idx = np.where(pos_flags == True)[0]
    return neg_idx, pos_idx


# post processing
def keep(i, masks):
    # retuns true if masks[i] overlaps with some other masks by more than x% of itself
    for j in range(i):
        area = masks[i].sum().item() * 1.
        if area < 100: return False
        #if (masks[j] * masks[i]).sum() / area > 0.7# and area < masks[j].sum():
        if (masks[j] * masks[i]).sum().item() / area > 0.7 and iou(masks[i], masks[j]) < 0.5:
            return False
    return True


def save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


def bin_to_cls_mask(labels, plot=True):
    h, w = labels.shape[1:]
    mask = np.zeros((h, w))
    for i in reversed(range(labels.shape[0])):
        mask[labels[i]] = i+1
    if plot:
        mask = mask / (labels.shape[0]+1)*255  # convert to greyscale
    return mask.astype(np.uint8)
