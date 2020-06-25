from lvis import LVIS
# from pycocotools.mask import iou
# from matplotlib.patches import Polygon
# from pycocotools import mask as maskUtils
import multiprocessing
import skimage.io as io
import numpy as np
import torch, cv2, os
from tqdm import tqdm
import torchvision
import torch.nn as nn
from models.rpn import ProposalNetwork
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

PATH = 'features_lvis_val_hyperbolic'
os.makedirs(PATH, exist_ok=True)

class LvisSaver(object):
    def __init__(self, model, path):
        self.lvis = LVIS('/scratch/users/zzweng/datasets/lvis/lvis_v0.5_val.json')
        self.model = model
        self.model.eval()
        self.path = path
    
    def save(self, k=50):
        # split the files into 50 chunks and process each concurrently
#         self._save_part(0, 5)
        
        rng = np.linspace(0, 5000, k+1, dtype=int)
        args = list(zip(rng[:-1], rng[1:]))
        with multiprocessing.Pool(processes=6) as pool:
            results = pool.starmap(self._save_part, args)
        print('Done')
    
    def _save_part(self, start, end):
        print('Getting features from {} to {}'.format(start, end))
        img_ids = self.lvis.get_img_ids()[start:end]
        feature_y = []
        feature_x = []
        for img_id in tqdm(img_ids):
            img = self.lvis.load_imgs([img_id])[0]
            I = io.imread(img['coco_url'])
            if len(I.shape) == 2: continue
                
            for ann_id in self.lvis.get_ann_ids(img_ids=[img_id]):
                ann = self.lvis.load_anns([ann_id])[0]
                b = np.array(ann['bbox']).astype(np.int)
                try:
#                     import ipdb as pdb
#                     pdb.set_trace()
                    I_masked = I * np.expand_dims(self.lvis.ann_to_mask(ann), 2)
                    patch = I_masked[b[1]:b[1]+b[3], b[0]:b[0]+b[2], :] / 255.
                    patch = cv2.resize(patch, (224, 224))
                    patch_tensor = torch.tensor(patch).float()
                    feat = self.model(patch_tensor.view(1, *patch_tensor.shape).permute(0, 3, 1, 2))[1].detach().numpy().flatten()
                    feature_x.append(feat)
                    feature_y.append(ann['category_id'])
                except:
                    print('skipping anns', b)
             
        feature_x_arr = np.stack(feature_x)
        feature_y_arr = np.array(feature_y)
        print(feature_x_arr.shape, feature_y_arr.shape)

        np.save(os.path.join(self.path, 'val_feats_{}_{}_x.npy'.format(start, end)), feature_x_arr)
        np.save(os.path.join(self.path, 'val_feats_{}_{}_y.npy'.format(start, end)), feature_y_arr)
        
        
if __name__ == '__main__':
#     CFG_FILE =  r'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
#     rpn = ProposalNetwork(device='cuda')
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file(CFG_FILE))
#     cfg.INPUT.FORMAT = 'RGB'
#     cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
#     cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 100
#     cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 50
#     # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CFG_FILE)
#     cfg.MODEL.WEIGHTS = r'/scratch/users/zzweng/output/coco/classagnostic1/model_0021999.pth'
#     rpn.predictor = DefaultPredictor(cfg)
    from models.hyperbolic_resnet import HResNetSimCLR
    checkpoint_dir = r'/scratch/users/zzweng/runs/checkpoints/all_hyp=True_zdim=2_loss=triplet_maskloss=False/'
    model = HResNetSimCLR('resnet101', 2)
    state_dict = torch.load(os.path.join(checkpoint_dir, 'model_11500.pth')) #, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    #cmodel_ = torchvision.models.resnet101(pretrained=True)
    #model = nn.Sequential(*list(cmodel_.children())[:-1])
    #model.eval()

    saver = LvisSaver(model, PATH)
    saver.save()
