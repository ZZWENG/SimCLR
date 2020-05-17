#import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
#from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer

# "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
CFG_FILE = "LVIS-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
#CFG_FILE = "LVIS-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"

#CFG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

def get_modelzoo_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CFG_FILE))
    #cfg.MODEL.WEIGHTS = r'/home/users/zzweng/unsupervised_segmentation/detectron2/checkpoints/model_final_5e3439.pkl' 
    #cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = TRUE
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CFG_FILE)
    return cfg


def get_class_agnostic_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CFG_FILE))
    cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
    #cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CFG_FILE)
    cfg.MODEL.WEIGHTS = 'output/model_0002999.pth'  # TRAINED THE MASK HEAD ON TOP OF mask_rcnn_R_50_FPN_1x
    return cfg


class ProposalNetwork(nn.Module):
    def __init__(self, device, nms_thres=0.6, topk=50):
        super(ProposalNetwork, self).__init__()
        #self.cfg = get_modelzoo_config()
        self.cfg = get_class_agnostic_config()
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 200
        self.cfg.MODEL.RPN.NMS_THRESH = 0.5
        self.cfg.MODEL.DEVICE = device
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        #self.cfg.OUTPUT_DIR = '.output/coco'
        self.predictor = DefaultPredictor(self.cfg)
        print('Build Predictor using cfg')
        #print(self.cfg)

    def train_predictor(self):
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 2000
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR /= 10.
        self.predictor = DefaultPredictor(self.cfg)
        self.cfg.OUTPUT_DIR = './output/coco'
        print(self.cfg.OUTPUT_DIR)
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    def save(self, path):
        try:
            checkpoints_folder = os.path.join(path, 'checkpoints', 'model.pth')
            torch.save(self.predictor.model, checkpoints_folder)
        except:
            print("Error in saving the checkpoint.")

    def load(self, path):
        try:
            checkpoints_folder = os.path.join(path, 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            self.predictor.model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    def visualize(self, x):
        self.eval()
        # TODO: verify that the data transformation is applied within predictor.
        # e.g. im = predictor.transform_gen.get_transform(im).apply_image(im)
        outputs = self.predictor(x)
        v = Visualizer(x[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        fig, ax = plt.subplots(figsize=(40, 48))
        ax.imshow(v.get_image())
        self.train()

    def forward(self, x, is_train=False):
        """ Takes the raw image, and then outputs the boxes and the class agnostic masks
        :param x: (h, w, 3) tensor
        :return: (topk, h, w), (h, w)
        """
        x = x.cpu().numpy()
        x = x.astype(np.uint8)
        assert(x.shape[2] == 3)
        out = self.predictor(x)  # predictor takes images in the BGR format
        if is_train:
            masks = out['instances'].pred_masks
            boxes = out['instances'].pred_boxes
            #background = self._get_background(masks)
            return masks, boxes
        else:
            return [out]

    def _get_background(self, masks):
        foreground = masks[0]
        for i in range(1, len(masks)):
            foreground |= masks[i]
        return ~foreground


if __name__ == '__main__':
    rpn = ProposalNetwork('cuda')
    rpn.train_predictor()
