from lvis import LVIS, LVISEval, LVISResults

from tqdm import tqdm
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import torchvision
import torch.nn as nn

import numpy as np
import torch, cv2
import skimage.io as io

generate_features = False

import json
from collections import Counter
from hyperbolic_knn import HyperbolicKNN

val_feat_folder = 'features_lvis_val_hyperbolic'
dt_feat_folder = 'features_lvis_dt_hyperbolic'

class Eval_KMeans(object):
    def __init__(self):
        self.lvis = LVIS('/scratch/users/zzweng/datasets/lvis/lvis_v0.5_val.json')
        self.dt_path = 'output/inference/lvis_instances_results.json'
        self.lvis_dt = LVISResults(self.lvis, self.dt_path)
        
        coco_map = json.load(open('lvis-api/data/coco_to_synset.json'))
        synset_to_lvis = {cat['synset']: cat['id'] for cat in self.lvis.cats.values()}
        synset_to_lvis['oven.n.01'] = synset_to_lvis['toaster_oven.n.01']
        synset_to_lvis['frank.n.02'] = synset_to_lvis['sausage.n.01']

        coco_to_lvis = {}
        lvis_to_coco = {}
        for item in coco_map.values():
            coco_id, lvis_id = item['coco_cat_id'], synset_to_lvis[item['synset']]
            coco_to_lvis[coco_id] = lvis_id
            lvis_to_coco[lvis_id] = coco_id
        self.coco_to_lvis = coco_to_lvis
        self.lvis_to_coco = lvis_to_coco
        cocoEval = LVISEval(self.lvis, self.lvis_dt,'segm')
        self.freq_groups = cocoEval._prepare_freq_group()
        
        # just making sure I am not using any labels in DT annotations, let's wipe them all
#         for _, dt in self.cocoDt.anns.items(): dt['category_id'] = -1

    def reload_annotations(self):
        self.lvis = LVIS('/scratch/users/zzweng/datasets/lvis/lvis_v0.5_val.json')
        self.dt_path = 'output/inference/lvis_instances_results.json'
        self.lvis_dt = LVISResults(self.lvis, self.dt_path)
    
    def load_gt_features(self, coco_only=False, k=100, freq_groups=None):
        feats = []
        y = []
        rng = np.linspace(0, 5000, 51, dtype=int)
        args = list(zip(rng[:-1], rng[1:]))
#         args = [(400, 500), (500, 600), (600, 700), (700, 800), 
#                 (800, 900), (900, 1000), (1000, 1100), (1100, 1200)]
        for rng in args:
            start, end = rng[0], rng[1]
            try:
                feats.append(np.load(r'{}/val_feats_{}_{}_x.npy'.format(val_feat_folder, start, end)))
                y.extend(np.load(r'{}/val_feats_{}_{}_y.npy'.format(val_feat_folder, start, end)))
            except FileNotFoundError:
                print('File {}/{}_{}.npy not found. Skipped.'.format(val_feat_folder, start, end))
        feats = np.concatenate(feats)
        print(feats.shape)
        self.feats_gt = feats
        self.feats_gt_y = np.array(y)
        
        if freq_groups is not None:
            print('Filter by freq groups', freq_groups)
            freqs = (np.concatenate([self.freq_groups[i] for i in freq_groups])+1).astype(np.int)
            idx = np.isin(self.feats_gt_y, freqs)
            self.feats_gt_y = self.feats_gt_y[idx]
            self.feats_gt = self.feats_gt[idx]
            print('After:', self.feats_gt.shape)
        
        if coco_only:
            coco_cats = self.lvis_to_coco.keys() 
            idx = np.array([y in coco_cats for y in self.feats_gt_y])
            self.feats_gt = self.feats_gt[idx]
            self.feats_gt_y = self.feats_gt_y[idx]
            print('Keeping objects in COCO', self.feats_gt.shape)
            
        if k:
            print('Keeping only {} masks for each class'.format(k))
            new_feats_gt = []
            new_feats_gt_y = []
            counts = Counter(self.feats_gt_y)
            for i, c in counts.items():
                if c > k:
                    idx = np.random.choice(np.arange(len(self.feats_gt_y))[self.feats_gt_y==i], k, replace=False)
#                     print(self.feats_gt_y[idx])
                    new_feats_gt.append(self.feats_gt[idx])
                    new_feats_gt_y.extend([i]*k)
                else:
                    new_feats_gt.append(self.feats_gt[self.feats_gt_y==i])
                    new_feats_gt_y.extend([i]*c)
            self.feats_gt = np.concatenate(new_feats_gt)
            self.feats_gt_y = new_feats_gt_y
            print(self.feats_gt.shape)

    def fit_knn(self, k=5, weights='distance'):
        feats = self.feats_gt
        y = self.feats_gt_y
        if 'hyperbolic' in dt_feat_folder:
           
            self.neigh = HyperbolicKNN(k, feats, y)
            pred_y = self.neigh.predict(feats[:50])
            print('KNN accuracy', accuracy_score(y[:50], pred_y))
        else:
            self.neigh = KNeighborsClassifier(n_neighbors=k, weights=weights)
            self.neigh.fit(feats, y)
            print('KNN accuracy', self.neigh.score(feats, y))
    
    def load_dt_features(self):
        feats = []
        feats_ann = []
        rng = np.linspace(0, 5000, 51, dtype=int)
        args = list(zip(rng[:-1], rng[1:]))
        for rng in args:
            start, end = rng[0], rng[1]
            try:
                feats.append(np.load(r'{}/{}_{}.npy'.format(dt_feat_folder, start, end)))
                feats_ann.extend(np.load(r'{}/{}_{}_ann.npy'.format(dt_feat_folder, start, end)))
            except FileNotFoundError:
                print('File {}/{}_{}.npy not found. Skipped.'.format(dt_feat_folder, start, end))
        self.feats = np.concatenate(feats)
        self.feats_ann = feats_ann
        print(self.feats.shape)

    def run_kmeans(self, C=1500):
        feats = self.feats
        feats_ann = self.feats_ann
        print('Running KMeans ...')
        kmeans = MiniBatchKMeans(C)
        clusters = kmeans.fit_predict(feats)
        self.clusters = clusters
        
    def run_hyperbolic_kmeans(self, C=200):
        from poincare_kmeans import PoincareKMeans as HKMeans
        kmeans = HKMeans(self.feats.shape[1], C)
        clusters = kmeans.fit_predict(self.feats)
        self.clusters = clusters
        
    def assign_labels(self):
        neigh = self.neigh
        cocoDt = self.lvis_dt
        clusters = self.clusters
        feats = self.feats
        feats_ann = self.feats_ann
        
        C = len(set(clusters))
#         feats = self.pca.transform(feats)
        print(feats.shape)
        
        coco_clusters = {}
        cluster_to_coco = {}
        print('Assigning labels using KNN ...')
        for i in tqdm(range(C)):
            idx = np.where(clusters==i)[0]
            if len(idx) == 0: continue
            predicted = neigh.predict(feats[idx])
        #     neighbors = neigh.kneighbors(feats[np.where(clusters==i)])[1]
#             distances = neigh.kneighbors(feats[np.where(clusters==i)])[0]
            
            votes = sorted(Counter(predicted).items(), key=lambda tup:-tup[1])
            best_ratio = votes[0][1] / len(predicted)
#             if len(predicted) < 3: continue # ignore clusters with fewer than 5
            if best_ratio < 0.95: continue
            cluster_to_coco[i] = (votes[0][0], best_ratio, len(predicted))
                
#             if votes[0][0] not in coco_clusters or coco_clusters[votes[0][0]][1] < best_ratio:
#                 coco_clusters[votes[0][0]] = (i, best_ratio, len(predicted))
#                 cluster_to_coco[i] = (votes[0][0], best_ratio, len(predicted))
                
#             for j in range(1, len(votes)):
#                 if votes[j][0] not in coco_clusters or coco_clusters[votes[j][0]][1] < votes[j][1] / len(predicted):
#                     coco_clusters[votes[j][0]] = (i, votes[j][1] / len(predicted), len(predicted))
#                     cluster_to_coco[i] = votes[j][0]
                
        self.cluster_to_coco = cluster_to_coco
        self.coco_clusters = coco_clusters
        print('Number of assigned clusters:', len(cluster_to_coco))

    def evaluate_plain(self):
        cocoEval = COCOeval(self.cocoGt, self.cocoDt,'segm')
        img_ids = self.cocoDt.getImgIds()[:100]
        cocoEval.params.imgIds = img_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        self.cocoEval = cocoEval
        
    def evaluate(self):
#         self.reload_annotation()
        cluster_to_coco = self.cluster_to_coco
        coco_clusters = self.coco_clusters
        cocoDt = self.lvis_dt
        clusters = self.clusters
        feats_ann = self.feats_ann
        
        # by default everything is -1.
        for _, dt in cocoDt.anns.items(): dt['category_id'] = -1
        print('Updating category ids')
        for i in tqdm(range(len(feats_ann))):
            ann_id = int(feats_ann[i])
            cluster_id = clusters[i]
            if cluster_id in cluster_to_coco:
                cocoDt.anns[ann_id]['category_id'] = cluster_to_coco[cluster_id][0]
#                 print('assigned ', cluster_to_coco[cluster_id][0])
                
        print('Finally, evaluate!!')
        
        self.cocoEval = LVISEval(self.lvis, cocoDt,'segm')
        img_ids = cocoDt.get_img_ids()[:100]
#         cocoEval.params.catIds = [1, 2, 3, 4]# 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 33, 34, 35, 37, 40, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 54, 56, 57, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 77, 78, 79, 81, 82, 84, 85, 86, 87, 88, 90]
#         cocoEval.params.imgIds = img_ids
#         cocoEval.params.iouThrs = np.linspace(.25, 0.95, int(np.round((0.95 - .25) / .05)) + 1, endpoint=True)

        self.cocoEval.lvis_gt.cats[-1] = {'frequency': 'f',
          'id': -1,
          'synset': 'all',
          'image_count': 0,
          'instance_count': 0,
          'synonyms': ['all'],
          'def': 'nut from an oak tree',
          'name': 'all'}
        import pdb
        pdb.set_trace()
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        self.cocoEval.summarize()
        
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
    

