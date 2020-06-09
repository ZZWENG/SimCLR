import json
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class SimCLRWriter(object):
    def __init__(self, config):
        self.config = config
        self.name = '{}_hyp={}_zdim={}_loss={}_maskloss={}'.format(
            config['desc'], config['hyperbolic'], config['model']['out_dim'],
            config['loss']['type'], config['loss']['mask_loss']
        )
        checkpoint_dir = '/scratch/users/zzweng/runs/checkpoints/{}'.format(self.name)

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.writer = SummaryWriter('runs/{}'.format(self.name))
        config_str = json.dumps(config)
        self.writer.add_text('configs', config_str, 0)

    def add_scalar(self, text, val, i):
        self.writer.add_scalar(text, val, i)

    def log_loss(self, loss_dict, n_iter):
        total_loss = loss_dict['triplet_loss'] + loss_dict['hierar_loss'] + \
                     self.config['beta'] * loss_dict['mask_loss']

        for type in loss_dict.keys():
            self.writer.add_scalar(type, loss_dict[type], global_step=n_iter)
        self.writer.add_scalar('total_loss', total_loss, global_step=n_iter)
        if loss_dict['loss_count'] > 0:
            self.writer.add_scalar('mean_loss', total_loss/loss_dict['loss_count'], global_step=n_iter)

    def visualize(self, image, image_url, masks, n_iter):
        h, w = masks[0].shape[0], masks[0].shape[1]
        # labels = batch['labels'].cpu().numpy()
        proposed_cls = bin_to_cls_mask(masks.cpu().numpy(), plot=True)
        # ground_cls = bin_to_cls_mask(labels, plot=True)
        img = image.type(torch.int32).cpu().numpy().transpose(2, 0, 1)  # [::-1,:,:]
        self.writer.add_image('input_images', img, n_iter)
        self.writer.add_text('filename', image_url, n_iter)
        # writer.add_text('ground_truth_classes', ','.join([lvis_id_cat_map[k] for k in gt_cls]), n_iter)
        self.writer.add_image('proposed_masks', make_grid(torch.tensor(proposed_cls.reshape(1, 1, h, w))), n_iter)
        # self.writer.add_image('ground_truth', make_grid(torch.tensor(ground_cls.reshape(1,1,h,w))), n_iter)


def bin_to_cls_mask(labels, plot=True):
    h, w = labels.shape[1:]
    mask = np.zeros((h, w))
    for i in reversed(range(labels.shape[0])):
        mask[labels[i]] = i+1
    if plot:
        mask = mask / (labels.shape[0]+1)*255  # convert to greyscale
    return mask.astype(np.uint8)
