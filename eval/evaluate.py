import json
import os
from collections import Counter

import numpy as np
from lvis import LVIS, LVISEval, LVISResults
from sklearn.metrics import accuracy_score
from tqdm import tqdm

EVAL_NAME = 'evaluate_hyperbolic'  # description of the evaluation run
ANNOTATION_PATH = r'/scratch/users/zzweng/datasets/lvis/lvis_v0.5_val.json'
PREDICTION_PATH = r'output/inference/lvis_instances_results.json'
CACHE_PATH = r'/scratch/users/zzweng/inference'
GT_FEATS = os.path.join(CACHE_PATH, 'gt_feats')
DT_FEATS = os.path.join(CACHE_PATH, EVAL_NAME, 'dt_feats')
LVIS_API_PATH = r'../lvis-api'

os.makedirs(GT_FEATS, exist_ok=True)
os.makedirs(DT_FEATS, exist_ok=True)


class LVISEvaluator(object):
    def __init__(self, run_path):
        self.lvis_gt = LVIS(ANNOTATION_PATH)
        self.lvis_dt = LVISResults(self.lvis_gt, PREDICTION_PATH)
        self._build_coco_to_lvis_map()

        cocoEval = LVISEval(self.lvis_gt, self.lvis_dt, 'segm')
        self.freq_groups = cocoEval._prepare_freq_group()
        self.run_path = run_path
        import yaml
        config_path = os.path.join(self.run_path, 'config_lvis.yaml')
        self.config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    def _build_coco_to_lvis_map(self):
        coco_map = json.load(open(os.path.join(LVIS_API_PATH, 'data/coco_to_synset.json')))
        synset_to_lvis = {cat['synset']: cat['id'] for cat in self.lvis_gt.cats.values()}
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

    def reload_annotations(self):
        self.lvis_gt = LVIS(ANNOTATION_PATH)
        self.dt_path = PREDICTION_PATH
        self.lvis_dt = LVISResults(self.lvis_gt, self.dt_path)

    def _save_gt_features(self):
        # This takes a long time and shoud only be run once.
        from eval.feature_saver import LvisSaver
        config = self.config
        if self.config['hyperbolic']:
            from models.hyperbolic_resnet import HResNetSimCLR
            model = HResNetSimCLR(config['model']['base_model'], config['model']['out_dim'])
        else:
            from models.resnet_simclr import ResNetSimCLR
            model = ResNetSimCLR(config['model']['base_model'], config['model']['out_dim'])
        saver = LvisSaver(model, self.lvis_gt, GT_FEATS)
        saver.save()

    def _save_dt_features(self):
        from eval.feature_saver import LvisSaver
        config = self.config
        if self.config['hyperbolic']:
            from models.hyperbolic_resnet import HResNetSimCLR
            model = HResNetSimCLR(config['model']['base_model'], config['model']['out_dim'])
        else:
            from models.resnet_simclr import ResNetSimCLR
            model = ResNetSimCLR(config['model']['base_model'], config['model']['out_dim'])
        saver = LvisSaver(model, self.lvis_dt, GT_FEATS)
        saver.save()

    def load_gt_features(self, coco_only=False, k=100, freq_groups=None):
        """  Load gt features from GT_FEATS folder.
        :param coco_only: only load categories that are in COCO.
        :param k: only load k masks for each category.
        :param freq_groups: only load categories in the specified freq_groups. e.g. ['f', 'r']
        """
        feats = []
        y = []
        files = os.listdir(GT_FEATS)
        if len(files) == 0:
            self._save_gt_features()
        print('Found {} files.'.format(len(files)))
        for f in files:
            if f.endswith('_x.npy'):
                feats.append(np.load(os.path.join(GT_FEATS, f)))
            elif f.endswith('_y.npy'):
                y.extend(np.load(os.path.join(GT_FEATS, f)))
        feats = np.concatenate(feats)
        print(feats.shape)
        self.feats_gt = feats
        self.feats_gt_y = np.array(y)

        if freq_groups is not None:
            print('Filter by freq groups', freq_groups)
            freqs = (np.concatenate([self.freq_groups[i] for i in freq_groups]) + 1).astype(np.int)
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
                    idx = np.random.choice(np.arange(len(self.feats_gt_y))[self.feats_gt_y == i], k, replace=False)
                    #                     print(self.feats_gt_y[idx])
                    new_feats_gt.append(self.feats_gt[idx])
                    new_feats_gt_y.extend([i] * k)
                else:
                    new_feats_gt.append(self.feats_gt[self.feats_gt_y == i])
                    new_feats_gt_y.extend([i] * c)
            self.feats_gt = np.concatenate(new_feats_gt)
            self.feats_gt_y = new_feats_gt_y
            print(self.feats_gt.shape)

    def fit_knn(self, k=5, weights='distance'):
        """ Fit a KNN model on the ground truth mask features to see whether the embeddings
        makes sense.
        """
        feats = self.feats_gt
        y = self.feats_gt_y
        if self.config['hyperbolic']:
            from hyperbolic_knn import HyperbolicKNN
            self.neigh = HyperbolicKNN(k, feats, y)
            pred_y = self.neigh.predict(feats)
            print('KNN accuracy', accuracy_score(y, pred_y))
        else:
            from sklearn.neighbors import KNeighborsClassifier
            self.neigh = KNeighborsClassifier(n_neighbors=k, weights=weights)
            self.neigh.fit(feats, y)
            print('KNN accuracy', self.neigh.score(feats, y))

    def load_dt_features(self):
        feats = []
        feats_ann = []
        files = os.listdir(DT_FEATS)
        if len(files) == 0:
            self._save_dt_features()
        print('Found {} files.'.format(len(files)))
        for f in files:
            if f.endswith('_x.npy'):
                feats.append(np.load(os.path.join(DT_FEATS, f)))
            elif f.endswith('_ann_id.npy'):
                feats_ann.extend(np.load(os.path.join(DT_FEATS, f)))
        self.feats = np.concatenate(feats)
        self.feats_ann = np.array(feats_ann)
        print(self.feats.shape, self.feats_ann.shape)

    def run_kmeans(self, C=1500):
        feats = self.feats
        if self.config['hyperbolic']:
            print('Running Hyperbolic KMeans...')
            from poincare_kmeans import PoincareKMeans as HKMeans
            assert self.feats.shape[1] == 2, 'only supports hkmeans in 2d.'
            kmeans = HKMeans(self.feats.shape[1], C)
            clusters = kmeans.fit_predict(self.feats)
            self.clusters = clusters
        else:
            print('Running Euclidean KMeans...')
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(C)
            clusters = kmeans.fit_predict(feats)
            self.clusters = clusters

    def assign_labels(self):
        """
        Take the clusters assigned by kmeans and assign labels to the clusters.
        :return:
        """
        neigh = self.neigh
        clusters = self.clusters
        feats = self.feats

        C = len(set(clusters))
        coco_clusters = {}
        cluster_to_coco = {}
        print('Assigning labels using KNN ...')
        for i in tqdm(range(C)):
            idx = np.where(clusters == i)[0]
            if len(idx) == 0: continue
            predicted = neigh.predict(feats[idx])
            votes = sorted(Counter(predicted).items(), key=lambda tup: -tup[1])
            best_ratio = votes[0][1] / len(predicted)

            if len(predicted) < 3: continue  # ignore clusters with fewer than 5
            if best_ratio < 0.95: continue
            cluster_to_coco[i] = (votes[0][0], best_ratio, len(predicted))
        self.cluster_to_coco = cluster_to_coco
        self.coco_clusters = coco_clusters
        print('Number of assigned clusters:', len(cluster_to_coco))

    def evaluate(self):
        cluster_to_coco = self.cluster_to_coco
        lvis_dt = self.lvis_dt
        clusters = self.clusters
        feats_ann = self.feats_ann

        # by default everything is -1.
        for _, dt in lvis_dt.anns.items():  dt['category_id'] = -1

        print('Updating category ids')
        for i in tqdm(range(len(feats_ann))):
            ann_id = int(feats_ann[i])
            cluster_id = clusters[i]
            if cluster_id in cluster_to_coco:
                lvis_dt.anns[ann_id]['category_id'] = cluster_to_coco[cluster_id][0]
        #                 print('assigned ', cluster_to_coco[cluster_id][0])

        print('Finally, evaluate!!')

        self.lvisEval = LVISEval(self.lvis_gt, lvis_dt, 'segm')
        # img_ids = cocoDt.get_img_ids()[:100]
        #         lvisEval.params.catIds = [1, 2, 3, 4]# 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 33, 34, 35, 37, 40, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 54, 56, 57, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 77, 78, 79, 81, 82, 84, 85, 86, 87, 88, 90]
        #         lvisEval.params.imgIds = img_ids
        #         lvisEval.params.iouThrs = np.linspace(.25, 0.95, int(np.round((0.95 - .25) / .05)) + 1, endpoint=True)

        self.lvisEval.lvis_gt.cats[-1] = {'frequency': 'f',
                                          'id': -1,
                                          'synset': 'all',
                                          'image_count': 0,
                                          'instance_count': 0,
                                          'synonyms': ['all'],
                                          'def': 'dummy category',
                                          'name': 'all'}

        self.lvisEval.evaluate()
        self.lvisEval.accumulate()
        self.lvisEval.summarize()

    def evaluate_class_agnostic(self):
        """ Treat all masks as one category.
        """
        lvis_dt = self.lvis_dt
        lvis_gt = self.lvis_gt
        feats_ann = self.feats_ann
        cluster_to_coco = self.cluster_to_coco

        # by default, none of the predictions gets evaluated.
        for _, dt in lvis_dt.anns.items(): dt['category_id'] = -2
        for _, dt in lvis_gt.anns.items(): dt['category_id'] = -2

        print('Updating category ids')
        for i in tqdm(range(len(feats_ann))):
            cluster_id = self.clusters[i]
            ann_id = int(feats_ann[i])
            if cluster_id in cluster_to_coco:
                lvis_dt.anns[ann_id]['category_id'] = -1  # the assigned ones are included in the eval

        cocoEval = LVISEval(lvis_gt, lvis_dt, 'segm')
        cocoEval.params.catIds = [-1]  # only evaluate on the category -1.
        cocoEval.params.useCats = 0
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()


if __name__ == '__main__':
    out_dir = r'/scratch/users/zzweng/runs/checkpoints/all_relu_hyp=True_zdim=2_loss=triplet_maskloss=False/'
    evaluator = LVISEvaluator(out_dir)
    evaluator.fit_knn()
    evaluator.load_gt_features()
    evaluator.load_dt_features()
    evaluator.run_kmeans()
    evaluator.assign_labels()
    evaluator.evaluate()
    evaluator.evaluate_class_agnostic()


