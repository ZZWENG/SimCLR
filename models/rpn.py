import matplotlib.pyplot as plt
import torch.nn as nn

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

# "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
CFG_FILE = "LVIS-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"


def get_predefine_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CFG_FILE))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CFG_FILE)
    return cfg


def get_class_agnostic_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CFG_FILE))
    cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CFG_FILE)
    # cfg.MODEL.WEIGHTS = 'detectron2/output/model_0013499.pth'  # TRAINED THE MASK HEAD ON TOP OF mask_rcnn_R_50_FPN_1x
    return cfg


class ProposalNetwork(nn.Module):
    def __init__(self, device, nms_thres=0.6, topk=50):
        super(ProposalNetwork, self).__init__()

        self.cfg = get_class_agnostic_config()
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = topk
        self.cfg.MODEL.RPN.NMS_THRESH = nms_thres
        self.cfg.MODEL.DEVICE = device
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        self.predictor = DefaultPredictor(self.cfg)

    def set_nms_params(self, topk, nms_thres):
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = nms_thres
        self.cfg.MODEL.RPN.NMS_THRESH = topk
        print('Updating predictor...')
        self.predictor = DefaultPredictor(self.cfg)

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

    def forward(self, x):
        """ Takes the raw image, and then outputs the boxes and the class agnostic masks
        :param x: (h, w, 3) tensor
        :return: (topk, h, w), (h, w)
        """

        out = self.predictor(x)
        masks = out['instances'].pred_masks
        background = self._get_background(masks)
        return masks, background

    def _get_background(self, masks):
        foreground = masks[0]
        for i in range(1, len(masks)):
            foreground |= masks[i]
        return ~foreground
