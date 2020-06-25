from lvis import LVIS, LVISEval, LVISResults
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
from models.hyperbolic_resnet import HResNetSimCLR
"""
cmodel = HResNetSimCLR('resnet18', 256)
state = torch.load(r'/scratch/users/zzweng/runs/checkpoints/freeze_coco_pretrain_hyp=True_zdim=64_loss=triplet/model_50000.pth')
cmodel.load_state_dict(state)
print(cmodel)
"""
PATH = 'features_lvis_dt_hyperbolic'
os.makedirs(PATH, exist_ok=True)
checkpoint_dir = r'/scratch/users/zzweng/runs/checkpoints/all_hyp=True_zdim=2_loss=triplet_maskloss=False/'
cmodel = HResNetSimCLR('resnet101', 2)
state_dict = torch.load(os.path.join(checkpoint_dir, 'model_11500.pth')) #, map_location=device)i
cmodel.load_state_dict(state_dict)
cmodel.eval()

#cmodel_ = torchvision.models.resnet101(pretrained=True)
#cmodel = nn.Sequential(*list(cmodel_.children())[:-1])
#cmodel.eval()

lvis = LVIS('/scratch/users/zzweng/datasets/lvis/lvis_v0.5_val.json')
lvis_dt = LVISResults(lvis, r'output/inference/lvis_instances_results.json')

def collect_features_from_dt(start, end, folder=PATH):
    print('Collecting {} to {}'.format(start, end))
    img_ids = lvis_dt.get_img_ids()[start:end]
    feats = []
    feats_ann = []
    count = 0
    for img_id in tqdm(img_ids):
        img = lvis_dt.load_imgs([img_id])[0]
        I = io.imread(img['coco_url'])
        ann_ids = lvis_dt.get_ann_ids(img_ids=[img_id])
        for ann_id in ann_ids:
            try:
                ann = lvis_dt.load_anns(ids=[ann_id])[0]
                m = lvis_dt.ann_to_mask(ann)
                b = np.array(ann['bbox']).astype(np.int)
                #         if b[2]*b[3] < 1024: continue
                patch = (I*m.reshape(*m.shape, 1))[b[1]:b[1]+b[3], b[0]:b[0]+b[2], :]
                #patch = I[b[1]:b[1]+b[3], b[0]:b[0]+b[2], :]
                patch = patch/255.
                patch = cv2.resize(patch, (224,224))
                patch_tensor = torch.tensor(patch).float()
                feats.append(cmodel(patch_tensor.view(1, *patch_tensor.shape).permute(0, 3, 1, 2))[1].detach().numpy().flatten())
                feats_ann.append(ann_id)
            except:
                continue
    feats = np.array(feats)
    np.save(os.path.join(folder,'{}_{}.npy'.format(start, end)), feats)
    np.save(os.path.join(folder,'{}_{}_ann.npy'.format(start, end)), feats_ann)
    print('Done') 


rng = np.linspace(0, 5000, 51, dtype=int)
args = list(zip(rng[:-1], rng[1:]))
#args = [(1300, 1400), (1400,1500)]
with multiprocessing.Pool(processes=6) as pool:
    results = pool.starmap(collect_features_from_dt, args)
