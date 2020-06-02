from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import iou
from matplotlib.patches import Polygon
from pycocotools import mask as maskUtils

import skimage.io as io
import numpy as np
import torch, cv2
from tqdm import tqdm as tqdm

cocoGt=COCO('/scratch/users/zzweng/datasets/coco/annotations/instances_val2017.json')

feature_y = []
patch_data = []
for img_id in tqdm(cocoGt.getImgIds()):
    for ann_id in cocoGt.getAnnIds(imgIds=[img_id]):
        ann = cocoGt.loadAnns(ann_id)[0]
        b = np.array(ann['bbox']).astype(np.int)
#         if b[3] < 2 or b[2] < 2: continue
        if b[3] * b[2] < 1024: continue # only consider medium objects
            
        img = cocoGt.loadImgs(ann['image_id'])[0]
        I = io.imread(img['coco_url'])
        if len(I.shape) == 2: continue
        I = I*np.expand_dims(cocoGt.annToMask(ann), 2)
        patch = I[b[1]:b[1]+b[3], b[0]:b[0]+b[2], :]
        patch = cv2.resize(patch, (224, 224))
        patch = patch.transpose(2,0,1)
        patch_data.append(patch)
        feature_y.append(ann['category_id'])

patch_data = np.stack(patch_data)
feature_y_arr = np.array(feature_y)
print(patch_data.shape, feature_y_arr.shape)

np.save('/scratch/users/zzweng/patch_x_224_ml_mask_all.npy', patch_data)
np.save('/scratch/users/zzweng/patch_y_224_ml_mask_all.npy', feature_y_arr)

