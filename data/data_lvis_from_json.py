import os

import numpy as np
import skimage.io as io
import torch
from lvis import LVIS, LVISResults
from torch.utils.data import Dataset
from torchvision.transforms import Normalize


class LVISDataFromJSON(Dataset):
    def __init__(self, device, config, dt_path=r'output/inference'):
        self.device = device
        self.lvis_gt = LVIS('/scratch/users/zzweng/datasets/lvis/lvis_v0.5_val.json')
        self.dt_path = os.path.join(dt_path, 'lvis_instances_results.json')
        self.dt = LVISResults(self.lvis_gt, self.dt_path)
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if 'human_car' in config['desc']:
            self.img_ids = self._get_img_ids(cat_ids=[805, 211])
        else:
            self.img_ids = self._get_img_ids(cat_ids=None)
        print('Total number of images in the training set: {}'.format(len(self.img_ids)))

    def _get_img_ids(self, cat_ids):
        # return self.dt.get_img_ids()
        if cat_ids is None:
            return self.lvis_gt.get_img_ids()

        imgs = set([a['image_id']
                    for a in self.lvis_gt.load_anns(self.lvis_gt.get_ann_ids(cat_ids=cat_ids))])
        imgs = list(imgs)
        return imgs

    def __getitem__(self, i):
        img_id = self.img_ids[i]
        image_url = self.dt.load_imgs([img_id])[0]['coco_url']
        image = io.imread(image_url)

        if len(image.shape) == 2:
            return self.__getitem__(0)  # skip grayscale images

        image = image / 255.
        image = self.normalize(torch.tensor(image).permute(2, 0, 1)).permute(1, 2, 0)

        ann_ids = self.dt.get_ann_ids(img_ids=[img_id])
        masks = np.stack([
            self.dt.ann_to_mask(self.dt.load_anns(ids=[a_i])[0])
            for a_i in ann_ids
        ])
        boxes = np.stack([
            self.dt.load_anns(ids=[a_i])[0]['bbox']
            for a_i in ann_ids
        ]).astype(np.int)
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        masks = torch.tensor(masks).to(self.device)

        return {
            'image': image,  # normalized
            'dt_counts': masks.shape[0],
            'masks': masks,  # cuda tensor
            'boxes': boxes,
            'image_url': image_url
        }

    def __len__(self):
        return len(self.img_ids)