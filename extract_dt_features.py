from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import iou
import numpy as np
import torch, cv2
import skimage.io as io
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

import torchvision
import torch.nn as nn
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
import os
import multiprocessing
from itertools import product

cmodel_ = torchvision.models.resnet101(pretrained=True)
cmodel = nn.Sequential(*list(cmodel_.children())[:-1])
cmodel.eval()

cocoGt=COCO('/scratch/users/zzweng/datasets/coco/annotations/instances_val2017.json')
cocoDt=cocoGt.loadRes('output/inference/coco_instances_results.json')


def collect_features_from_dt(start, end, folder='features2_mask'):
    img_ids = cocoDt.getImgIds()[start:end]
    feats = []
    feats_ann = []
    count = 0
    for img_id in tqdm(img_ids):
        img = cocoDt.loadImgs([img_id])[0]
        I = io.imread(img['coco_url'])
        ann_ids = cocoDt.getAnnIds(imgIds=[img_id])
        for ann_id in ann_ids:
            try:
                ann = cocoDt.loadAnns(ids=[ann_id])[0]
                m = cocoDt.annToMask(ann)
                b = np.array(ann['bbox']).astype(np.int)
                #         if b[2]*b[3] < 1024: continue
                patch = (I*m.reshape(*m.shape, 1))[b[1]:b[1]+b[3], b[0]:b[0]+b[2], :]
                #patch = I[b[1]:b[1]+b[3], b[0]:b[0]+b[2], :]
                patch = patch/255.
                patch = cv2.resize(patch, (224,224))
                patch_tensor = torch.tensor(patch).float()
                feats.append(cmodel(patch_tensor.view(1, *patch_tensor.shape).permute(0, 3, 1, 2)).detach().numpy().flatten())
                feats_ann.append(ann_id)
            except:
                continue
        count +=1
        if count % 10 == 0:
            print(count)
    feats = np.array(feats)
    np.save(os.path.join(folder,'{}_{}.npy'.format(start, end)), feats)
    np.save(os.path.join(folder,'{}_{}_ann.npy'.format(start, end)), feats_ann)
    print('Done')
    

rng = np.linspace(0, 5000, 51, dtype=int)
args = list(zip(rng[:-1], rng[1:]))
#args = [(600,700), (700,800)]
with multiprocessing.Pool(processes=6) as pool:
    results = pool.starmap(collect_features_from_dt, args)
