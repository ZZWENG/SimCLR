import cv2, os
import torch
import numpy as np
from functools import partial

from detectron2.data.build import build_detection_test_loader, build_detection_train_loader
from detectron2.structures.masks import BitMasks
from detectron2.data.dataset_mapper import DatasetMapper

if 'DETECTRON2_DATASETS' not in os.environ:
    print('Need to set DETECTRON2_DATASETS...')


# a callable which takes a sample (dict) from dataset and
# returns the format to be consumed by the model
def wrapper(d, default_m, h, w):
    d = default_m(d)
    """
    d has keys: file_name, height, width, image, instances, etc.
    """
    img = d['image'].cpu().numpy().transpose(1, 2, 0)
    img = cv2.resize(img, (h, w))  # .transpose(2, 0, 1)
    #img = F.interpolate(img, size=(h, w), mode='bilinear')

    raw_h, raw_w = d['instances'].image_size
    masks = BitMasks.from_polygon_masks(d['instances'].gt_masks, raw_h, raw_w).tensor.type(torch.uint8)
    masks_resized = np.zeros((masks.shape[0], h, w))
    for i in range(masks.shape[0]):
        #masks_resized[i] = F.upsample(masks[i], size=(h,w), mode='bilinear')
        masks_resized[i] = cv2.resize(masks[i].cpu().numpy(), (w, h))
    img = torch.tensor(img).type(torch.float)
    gt_masks = torch.tensor(masks_resized).type(torch.bool)

    """
    num_gt_masks = gt_masks.shape[0]
    ground = np.zeros_like(gt_masks[0], dtype=np.uint8)

    for j in range(num_gt_masks):
        ground[gt_masks[j]] = j + 1
    print(gt_masks.shape, np.unique(ground))
    """
    foreground = gt_masks[0]
    for i in range(1, gt_masks.shape[0]):
        foreground |= gt_masks[i]
    d['image'] = img
    d['labels'] = gt_masks
    d['background'] = ~foreground
    return d


def get_lvis_train_dataloader(cfg, h, w):
    default_mapper = DatasetMapper(cfg, is_train=True)
    mapper = partial(wrapper, default_m=default_mapper, h=h, w=w)
    dl = build_detection_train_loader(cfg, mapper=mapper)
    return dl


def get_lvis_test_dataloader(cfg, h, w):
    default_mapper = DatasetMapper(cfg, is_train=False)
    mapper = partial(wrapper, default_m=default_mapper, h=h, w=w)
    dl = build_detection_test_loader(cfg, 'lvis_v0.5_val', mapper=mapper)
    return dl


class DataSetWrapper(object):
    def __init__(self,  batch_size, num_workers, cfg, input_shape, **kwargs):

        self.cfg = cfg
        self.cfg.SOLVER.IMS_PER_BATCH = batch_size
        self.cfg.DATALOADER.NUM_WORKERS = num_workers
        self.input_shape = eval(input_shape)
        self.h = self.input_shape[0]
        self.w = self.input_shape[1]

    def get_data_loaders(self):
        train_loader = get_lvis_train_dataloader(self.cfg, self.h, self.w)
        valid_loader = None
        #valid_loader = get_lvis_test_dataloader(self.cfg, self.h, self.w)
        return train_loader, valid_loader
