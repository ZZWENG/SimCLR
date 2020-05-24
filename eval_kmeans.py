from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import iou
from tqdm import tqdm


cocoGt=COCO('/scratch/users/zzweng/datasets/coco/annotations/instances_val2017.json')
cocoDt=cocoGt.loadRes('output/inference/coco_instances_results.json')

import numpy as np
import torch, cv2
import skimage.io as io


cocoEval = COCOeval(cocoGt,cocoDt,'segm')

from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
import torchvision
import torch.nn as nn

generate_features = False

if generate_features:
    patch_data = np.load('patch_x_224_ml_mask.npy')/255.
    feature_y_arr = np.load('patch_y_224_ml_mask.npy')
    patch_data_tensor = torch.tensor(patch_data)
    patch_data_tensor.shape, feature_y_arr.shape
    batch_size = 32
    X_all = []
    for i in tqdm(range(len(patch_data_tensor)//32 + 1)):
        batch = patch_data_tensor[i*batch_size:(i+1)*batch_size]
        X_all.append(cmodel(batch.float()).view(len(batch), -1).detach().cpu().numpy())
    X_all = np.concatenate(X_all)
    y_all = feature_y_arr

else:
    X_all = np.load('patch_x_224_ml_mask_feats.npy')
    y_all = np.load('patch_x_224_ml_mask_feats_y.npy')
X_all.shape, y_all.shape

neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
neigh.fit(X_all, y_all)
print('KNN accuracy', neigh.score(X_all, y_all))

cmodel_ = torchvision.models.resnet101(pretrained=True)
cmodel = nn.Sequential(*list(cmodel_.children())[:-1])
cmodel.eval()

img_ids = cocoDt.getImgIds()
feats = []
feats_ann = []
for img_id in tqdm(img_ids):
    ann_ids = cocoDt.getAnnIds(imgIds=[img_id])
    for ann_id in ann_ids:
        ann = cocoDt.loadAnns(ids = [ann_id])[0]
        img = cocoDt.loadImgs([img_id])[0]
        I = io.imread(img['coco_url'])
        m = cocoDt.annToMask(ann)
        b = np.array(ann['bbox']).astype(np.int)
#         if b[2]*b[3] < 1024: continue
#         patch = (I*m.reshape(*m.shape, 1))[b[1]:b[1]+b[3], b[0]:b[0]+b[2], :]
        patch = I[b[1]:b[1]+b[3], b[0]:b[0]+b[2], :]
        patch = patch/255.
        patch = cv2.resize(patch, (224,224))
        patch_tensor = torch.tensor(patch).float()
        feats.append(cmodel(patch_tensor.view(1, *patch_tensor.shape).permute(0, 3, 1, 2)).detach().numpy().flatten())
        feats_ann.append(ann_id)

feats = np.array(feats)
print('Collected val features:', feats.shape)
print('Running KMeans')
kmeans = KMeans(100)
clusters = kmeans.fit_predict(feats)

coco_clusters = {}
cluster_to_coco = {}
for i in range(100):
    predicted = neigh.predict(feats[np.where(clusters==i)])
#     neighbors = neigh.kneighbors(feats[np.where(clusters==i)])[1]
#     distances = neigh.kneighbors(feats[np.where(clusters==i)])[0]
    votes = sorted(Counter(predicted).items(), key=lambda tup:-tup[1])
    best_ratio = votes[0][1] / len(predicted)
    if len(predicted) < 3: continue # ignore clusters with fewer than 5
    if votes[0][0] not in coco_clusters or coco_clusters[votes[0][0]][1] < best_ratio:
        coco_clusters[votes[0][0]] = (i, best_ratio, len(predicted))
        cluster_to_coco[i] = votes[0][0]

print('Update category ids')
for i in range(len(feats_ann)):
    ann_id = feats_ann[i]
    cluster_id = clusters[i]
    ann = cocoDt.loadAnns(ann_id)[0]
    if cluster_id in cluster_to_coco:
        ann['category_id'] = cluster_to_coco[cluster_id]
        break
    else:
        ann['category_id'] = -1

cocoEval.params.imgIds = img_ids
# cocoEval.params.catIds = [-1]
# cocoEval.params.useCats = 0
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()



