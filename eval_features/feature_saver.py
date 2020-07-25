# from pycocotools.mask import iou
# from matplotlib.patches import Polygon
# from pycocotools import mask as maskUtils
import multiprocessing

import cv2
import numpy as np
import os
import skimage.io as io
import torch
from tqdm import tqdm


class LvisSaver(object):
    def __init__(self, model, lvis_api, save_path, n_processes=6):
        self.lvis = lvis_api
        self.model = model
        self.model.eval()
        self.path = save_path
        self.n = len(self.lvis.get_img_ids())
        self.n_processes=n_processes

    def save(self, k=50):
        # split the files into k chunks and process each concurrently
        rng = np.linspace(0, self.n, k + 1, dtype=int).astype(int)
        print('Start saving features.')
        print(rng)
        args = list(zip(rng[:-1], rng[1:]))
        with multiprocessing.Pool(processes=self.n_processes) as pool:
            pool.starmap(self._save_part, args)
        print('Done saving all features.')

    def _save_part(self, start, end):
        print('Getting features from {} to {}'.format(start, end))
        if os.path.exists(os.path.join(self.path, 'feats_{}_{}_x.npy'.format(start, end))):
            print('feats_{}_{}_x.npy already exists'.format(start, end))
            return
         
        img_ids = self.lvis.get_img_ids()[start:end]
        feature_y = []
        feature_x = []
        feature_ann_id = []
        for img_id in tqdm(img_ids):
            img = self.lvis.load_imgs([img_id])[0]
            I = io.imread(img['coco_url'])
            if len(I.shape) == 2: continue

            for ann_id in self.lvis.get_ann_ids(img_ids=[img_id]):
                ann = self.lvis.load_anns([ann_id])[0]
                b = np.array(ann['bbox']).astype(np.int)
                try:
                    I_masked = I * np.expand_dims(self.lvis.ann_to_mask(ann), 2)
                    patch = I_masked[b[1]:b[1] + b[3], b[0]:b[0] + b[2], :] / 255.
                    patch = cv2.resize(patch, (224, 224))
                    patch_tensor = torch.tensor(patch).float()
                    feat = self.model(patch_tensor.view(1, *patch_tensor.shape).permute(0, 3, 1, 2))[
                        1].detach().numpy().flatten()
                    feature_x.append(feat)
                    feature_y.append(ann['category_id'])
                    feature_ann_id.append(ann_id)
                except:
                    print('skipping anns', b)

        feature_x_arr = np.stack(feature_x)
        feature_y_arr = np.array(feature_y)
        feature_ann_id_arr = np.array(feature_ann_id)
        print(feature_x_arr.shape, feature_y_arr.shape, feature_ann_id_arr.shape)

        np.save(os.path.join(self.path, 'feats_{}_{}_x.npy'.format(start, end)), feature_x_arr)
        np.save(os.path.join(self.path, 'feats_{}_{}_y.npy'.format(start, end)), feature_y_arr)
        np.save(os.path.join(self.path, 'feats_{}_{}_ann_id.npy'.format(start, end)), feature_ann_id)
