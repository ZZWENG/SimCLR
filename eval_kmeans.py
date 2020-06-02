from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import iou

from tqdm import tqdm
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
import torchvision
import torch.nn as nn

import numpy as np
import torch, cv2
import skimage.io as io

generate_features = False


class Eval_KMeans(object):
    def __init__(self):
        self.cocoGt=COCO('/scratch/users/zzweng/datasets/coco/annotations/instances_val2017.json')
        self.dt_path = 'output/inference/coco_instances_results.json'
        self.cocoDt=self.cocoGt.loadRes(self.dt_path)
        
        # just making sure I am not using any labels in DT annotations, let's wipe them all
#         for _, dt in self.cocoDt.anns.items(): dt['category_id'] = -1
    
    def fit_knn(self, k=5, weights='distance'):
        xf = r'/scratch/users/zzweng/patch_x_224_ml_mask_all.npy'
        yf = r'/scratch/users/zzweng/patch_y_224_ml_mask_all.npy'
        if generate_features:
            patch_data = np.load(xf)/255.
            feature_y_arr = np.load(yf)
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
        print(X_all.shape, y_all.shape)
        
        self.feats_gt = X_all
        self.feats_gt_y = y_all

        self.neigh = KNeighborsClassifier(n_neighbors=k, weights=weights)
        self.neigh.fit(X_all, y_all)
        print('KNN accuracy', self.neigh.score(X_all, y_all))
    
    def load_dt_features(self):
     
        feats = []
        feats_ann = []
        rng = np.linspace(0, 5000, 51, dtype=int)
        args = list(zip(rng[:-1], rng[1:]))
        for rng in args:
            start, end = rng[0], rng[1]
            try:
                feats.append(np.load(r'features2_mask/{}_{}.npy'.format(start, end)))
                feats_ann.extend(np.load(r'features2_mask/{}_{}_ann.npy'.format(start, end)))
            except FileNotFoundError:
                print('File features/{}_{}.npy not found. Skipped.'.format(start, end))
        self.feats = np.concatenate(feats)
        self.feats_ann = feats_ann

    def run_kmeans(self, C=100):
        feats = self.feats
        feats_ann = self.feats_ann
        print('Running KMeans ...')
        kmeans = MiniBatchKMeans(C)
        clusters = kmeans.fit_predict(feats)
        self.clusters = clusters
        
    def assign_labels(self):
        neigh = self.neigh
        cocoDt = self.cocoDt
        clusters = self.clusters
        feats = self.feats
        feats_ann = self.feats_ann
        C = len(set(clusters))
        
        coco_clusters = {}
        cluster_to_coco = {}
        print('Assigning labels using KNN ...')
        for i in tqdm(range(C)):
            idx = np.where(clusters==i)[0]
            if len(idx) == 0: continue
            predicted = neigh.predict(feats[idx])
        #     neighbors = neigh.kneighbors(feats[np.where(clusters==i)])[1]
        #     distances = neigh.kneighbors(feats[np.where(clusters==i)])[0]
            votes = sorted(Counter(predicted).items(), key=lambda tup:-tup[1])
            best_ratio = votes[0][1] / len(predicted)
#             if len(predicted) < 10: continue # ignore clusters with fewer than 5
            if best_ratio < 0.6: continue
            cluster_to_coco[i] = (votes[0][0], best_ratio, len(predicted))
                
#             if votes[0][0] not in coco_clusters or coco_clusters[votes[0][0]][1] < best_ratio:
#                 coco_clusters[votes[0][0]] = (i, best_ratio, len(predicted))
#                 cluster_to_coco[i] = votes[0][0]
                
#             for j in range(1, len(votes)):
#                 if votes[j][0] not in coco_clusters or coco_clusters[votes[j][0]][1] < votes[j][1] / len(predicted):
#                     coco_clusters[votes[j][0]] = (i, votes[j][1] / len(predicted), len(predicted))
#                     cluster_to_coco[i] = votes[j][0]
                
        self.cluster_to_coco = cluster_to_coco
        self.coco_clusters = coco_clusters
        print('Number of assigned clusters:', len(coco_clusters))
        
    def reload_annotation(self):
        self.cocoGt=COCO('/scratch/users/zzweng/datasets/coco/annotations/instances_val2017.json')
        self.cocoDt=self.cocoGt.loadRes(self.dt_path)
    
    def evaluate_plain(self):
        self.reload_annotation()
        cocoEval = COCOeval(self.cocoGt, self.cocoDt,'segm')
        img_ids = self.cocoDt.getImgIds()[:100]
        cocoEval.params.imgIds = img_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        self.cocoEval = cocoEval
        
    def evaluate(self):
        self.reload_annotation()
        cluster_to_coco = self.cluster_to_coco
        coco_clusters = self.coco_clusters
        cocoDt = self.cocoDt
        clusters = self.clusters
        feats_ann = self.feats_ann
        
        # by default everything is -1.
        for _, dt in cocoDt.anns.items(): dt['category_id'] = -1
        print('Updating category ids')
        for i in tqdm(range(len(feats_ann))):
            ann_id = int(feats_ann[i])
            cluster_id = clusters[i]
#             ann = cocoDt.loadAnns(ann_id)[0]
            if cluster_id in cluster_to_coco:
                cocoDt.anns[ann_id]['category_id'] = cluster_to_coco[cluster_id][0]
#                 print('assigned ', cluster_to_coco[cluster_id][0])
#             else:
#                 ann['category_id'] = -1
                
        print('Finally, evaluate!!')
        
        cocoEval = COCOeval(self.cocoGt, cocoDt,'segm')
        img_ids = cocoDt.getImgIds()[:100]
        cocoEval.params.catIds = [1, 2, 3, 4]# 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 33, 34, 35, 37, 40, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 54, 56, 57, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 77, 78, 79, 81, 82, 84, 85, 86, 87, 88, 90]
#         cocoEval.params.imgIds = img_ids
#         cocoEval.params.iouThrs = np.linspace(.25, 0.95, int(np.round((0.95 - .25) / .05)) + 1, endpoint=True)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
    def evaluate_class_agnostic_all(self):
        cocoDt = self.cocoDt
        cocoGt = self.cocoGt
        feats_ann = self.feats_ann
        clusters = self.clusters
        cluster_to_coco = self.cluster_to_coco
        
        for _, dt in cocoDt.anns.items(): dt['category_id'] = -1
        for _, dt in cocoGt.anns.items(): dt['category_id'] = -1
        
        cocoEval = COCOeval(cocoGt, cocoDt,'segm')
        img_ids = cocoDt.getImgIds()[:100]
        print(len(img_ids))
        cocoEval.params.imgIds = img_ids
        cocoEval.params.catIds = [-1]
        cocoEval.params.useCats = 0
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        
    def evaluate_class_agnostic(self):
        cocoDt = self.cocoDt
        cocoGt = self.cocoGt
        feats_ann = self.feats_ann
        clusters = self.clusters
        cluster_to_coco = self.cluster_to_coco
        
        for _, dt in cocoDt.anns.items(): dt['category_id'] = -2
        for _, dt in cocoGt.anns.items(): dt['category_id'] = -1
        
        for i in range(len(feats_ann)):
            ann_id = int(feats_ann[i])
            cluster_id = clusters[i]
            ann = cocoDt.loadAnns(ann_id)[0]
            if cluster_id in cluster_to_coco:
                ann['category_id'] = -1
            else:
                ann['category_id'] = -2
            
        cocoEval = COCOeval(cocoGt, cocoDt,'segm')
        img_ids = cocoDt.getImgIds()[:100]
        print(len(img_ids))
        cocoEval.params.imgIds = img_ids
        cocoEval.params.catIds = [-1]
        cocoEval.params.useCats = 0
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()


if __name__ == '__main__':
    evaluator = Eval_KMeans()
    evaluator.fit_knn()
    evaluator.load_dt_features()
    evaluator.run_kmeans()
    

