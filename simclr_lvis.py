import cv2
import json
import os
import shutil

import geoopt
import numpy as np
import skimage
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from skimage.transform import rotate
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize
from torchvision.utils import make_grid

from evaluate import get_evaluator
from loss.nt_xent import NTXentLoss
from loss.triplet import TripletLoss, HTripletLoss
from models.hyperbolic_resnet import HResNetSimCLR
from models.resnet_simclr import ResNetSimCLR

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

apex_support = False

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


def bin_to_cls_mask(labels, plot=True):
    h, w = labels.shape[1:]
    mask = np.zeros((h, w))
    for i in reversed(range(labels.shape[0])):
        mask[labels[i]] = i+1
    if plot:
        mask = mask / (labels.shape[0]+1)*255  # convert to greyscale
    return mask.astype(np.uint8)


def prepare_seg_triplets(masks, boxes, image, side_len=224):
    n = masks.shape[0]
    boxes = boxes.tensor.detach().cpu().numpy().astype(np.uint32)
    for i in range(n):
        m, b = masks[i], boxes[i]
        if m.sum() < 400:  continue  # skip tiny masks for now
        m = m.view(*m.shape, 1)
        full = image[b[1]:b[3], b[0]:b[2],:] / 255.
        foreground = (m * image)[b[1]:b[3], b[0]:b[2],:] / 255.
        background = ((~m) * image)[b[1]:b[3], b[0]:b[2],:] / 255.
        full = resize_tensor(full, side_len)
        foreground = resize_tensor(foreground, side_len)
        background = resize_tensor(background, side_len)
        yield full, foreground, background

def iou(m1, m2):
    union = (m1 | m2).sum().item()
    if not union > 0: return 0.
    return (m1*m2).sum().item() * 1. / union


def overlapping_idx(anchor, masks, thres):
    def overlaps(m1, m2, thres):
        return (m1 * m2).sum() > thres
    flags = np.array([overlaps(anchor, m, thres) for m in masks])
    neg_idx = np.where(flags == False)[0]

    def is_child(m1, m2):
        m1_area, m2_area = m1.sum().item(), m2.sum().item()
        return iou(m1, m2) > 0.5 and m1_area > m2_area
    pos_flags = np.array([is_child(anchor, m) for m in masks])
    pos_idx = np.where(pos_flags == True)[0]
    return neg_idx, pos_idx


def size_of(cut):
    return cut.shape[0] * cut.shape[1]


# post processing
def keep(i, masks):
    # retuns true if masks[i] overlaps with some other masks by more than x% of itself
    for j in range(i):
        area = masks[i].sum().item() * 1.
        if area < 100: return False
        #if (masks[j] * masks[i]).sum() / area > 0.7# and area < masks[j].sum():
        if (masks[j] * masks[i]).sum().item() / area > 0.7 and iou(masks[i], masks[j]) < 0.5:
            return False
    return True


# returns tensor (N, L, L, 3), (N, L, L, 3) for input to model
def prepare_object_pairs(masks, boxes, image, side_len=128):
    n = masks.shape[0]
    boxes = boxes.tensor.detach().cpu().numpy().astype(np.int)
    result = []
    result_aug = []
    for i in range(n):
        m, b = masks[i], boxes[i]
        cropped = (m.view(*m.shape, 1) * image)[b[1]:b[3], b[0]:b[2],:]
        if cropped.shape[0]<5 or cropped.shape[1] <5:
            continue
        try:
            cropped = cv2.resize(cropped.cpu().numpy(), (side_len,side_len))
 
            cropped_aug = rotate(cropped, angle=45, mode = 'wrap')
            cropped_aug = skimage.util.random_noise(cropped_aug)
            cropped_aug = T.RandomErasing(0.9, scale=(0.02, 0.23))(torch.tensor(cropped_aug))
        except:
            continue
        result += [cropped]
        result_aug += [cropped_aug]
    result = torch.tensor(np.stack(result)).type(torch.float).to(masks.device)/255.
    result_aug = torch.tensor(np.stack(result_aug)).type(torch.float).to(masks.device)/255.
    result, result_aug = result.permute(0,3,1,2), result_aug.permute(0,3,1,2)

    result = torch.stack([normalize(result[i]) for i in range(result.shape[0])])
    result_aug = torch.stack([normalize(result_aug[i]) for i in range(result_aug.shape[0])])
    return result, result_aug


def apply_mask(image, m, b):
    return (m.view(*m.shape, 1) * image)[b[1]:b[3], b[0]:b[2], :] / 255.


def prepare_obj_triplets(masks, boxes, image, augment=False, side_len=224):
    n = masks.shape[0]
    #k = 10  # number of triplets sampled for each anchor
    boxes = boxes.tensor.detach().cpu().numpy().astype(np.uint32)
    for i in range(n):
        m1, b = masks[i], boxes[i]
        cut_a = (m1.view(*m1.shape, 1) * image)[b[1]:b[3], b[0]:b[2], :]/255.
        if cut_a.shape[0] * cut_a.shape[1] < 10:
            continue

        neg_idx, pos_idx = overlapping_idx(m1, masks, 50)
        
        if len(neg_idx) == 0:  continue

        if len(pos_idx) == 0:
            if augment:
                cut_p = torch.tensor(rotate(cut_a.cpu().numpy(), angle=25, mode='wrap')).type(torch.float).to(m1.device)
            else:
                continue
        else:
           # print(i, pos_idx, neg_idx)
            for j in range(len(pos_idx)):
                i_p = pos_idx[j]
                #i_p = np.random.choice(pos_idx, 1)[0]
                cut_p = apply_mask(image, masks[i_p], boxes[i_p])
                
                i_ns = np.random.choice(neg_idx, min(5, len(neg_idx)), replace=False)
                #import pdb
                #pdb.set_trace()
                for i_n in i_ns:
                    cut_n = apply_mask(image, masks[i_n], boxes[i_n])

                    if size_of(cut_p) < 10 or size_of(cut_n) < 10 or size_of(cut_a) < 10:
                        continue

                    cut_a = cv2.resize(cut_a.cpu().numpy(), (side_len, side_len))
                    cut_p = cv2.resize(cut_p.cpu().numpy(), (side_len, side_len))
                    cut_n = cv2.resize(cut_n.cpu().numpy(), (side_len, side_len))

                    cut_a = torch.tensor(cut_a).type(torch.float).to('cuda')
                    cut_p = torch.tensor(cut_p).type(torch.float).to('cuda')
                    cut_n = torch.tensor(cut_n).type(torch.float).to('cuda')
                    yield cut_a, cut_p, cut_n

def resize_tensor(t, side_len):
    device = t.device
    t_resized = cv2.resize(t.cpu().numpy(), (side_len, side_len))
    return torch.tensor(t_resized).type(torch.float).to(device)


class SimCLR(object):
    def __init__(self, dataset, rpn, config):
        self.config = config
        self.rpn = rpn
        self.device = self._get_device()
        self.dataset = dataset
        if self.config['loss']['type'] == 'nce':
           self.loss_crit = NTXentLoss(self.device, config['batch_size'], **config['loss'])
        if self.config['hyperbolic']:
           self.triplet_loss_crit = HTripletLoss()
        else:
           self.triplet_loss_crit = TripletLoss()
        self.name = '{}_hyp={}_zdim={}_loss={}_maskloss={}'.format(
             self.config['desc'], self.config['hyperbolic'], self.config['model']['out_dim'], 
             self.config['loss']['type'], self.config['loss']['mask_loss']
        )
        checkpoint_dir = '/scratch/users/zzweng/runs/checkpoints/{}'.format(self.name)
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.writer = SummaryWriter('runs/{}'.format(self.name))
        self._write_configs()
        #self.evaluator = get_evaluator(rpn.cfg, 'coco')
        #self.evaluator.reset()

    def _write_configs(self):
        config_str = json.dumps(self.config)
        self.writer.add_text('configs', config_str, 0)

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, x_a, x_p, x_n):
        # get the representations and the projections
        r_a, z_a = model(x_a.permute(2,0,1).view(1, 3,x_a.shape[0],x_a.shape[1]))  # [N,C]
        r_p, z_p = model(x_p.permute(2,0,1).view(1, 3,x_p.shape[0],x_p.shape[1]))  # [N,C]
        r_n, z_n = model(x_n.permute(2,0,1).view(1, 3,x_n.shape[0],x_n.shape[1]))

        #import ipdb as pdb
        #pdb.set_trace()
        # normalize projection feature vectors
        if not self.config["hyperbolic"]:
            z_a = F.normalize(z_a, dim=0)
            z_p = F.normalize(z_p, dim=0)
            z_n = F.normalize(z_n, dim=0)
        
        loss = self.triplet_loss_crit(z_a, z_p, z_n)
        return loss

        # xis = [N, H, W, 3]
    def _step_nce(self, model, xis, xjs, n_iter):
        #assert (xis.max() <= 1.)
        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]
            
        # normalize projection feature vectors
        try:
            zis = F.normalize(zis, dim=1)
            zjs = F.normalize(zjs, dim=1)
        except:
            print(zis.shape, zjs.shape)
        loss = self.loss_crit(zis, zjs)
        return loss

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loaders()
        if self.config['hyperbolic']:
            model = HResNetSimCLR(**self.config["model"]).to(self.device)
        else:
            model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model, loaded_iter = self._load_pre_trained_weights(model)
        print('Freezing rpn weights')
        for p in self.rpn.predictor.model.parameters():
            p.requires_grad = False

        #segmentation_params = self.rpn.predictor.model.roi_heads.parameters()
        #optimizer = torch.optim.Adam(list(segmentation_params)+list(model.parameters()), 3e-4, weight_decay=eval(self.config['weight_decay']))

        if self.config['hyperbolic']:
            optimizer = geoopt.optim.RiemannianAdam(
                [p for p in model.parameters() if p.requires_grad],
                1e-4, weight_decay=eval(self.config['weight_decay']))
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), 1e-4,
                weight_decay=eval(self.config['weight_decay']))
        
        num_train = len(train_loader.dataset.dataset)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_train, eta_min=0,
                                                               last_epoch=-1)
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = loaded_iter
        for epoch_counter in range(self.config['epochs']):
            for _, batch in enumerate(train_loader):
                batch = batch[0]
                image = batch['image'].to(self.device)
                assert (image.shape[2] == 3)  # the image is in BGR format
                
                masks, boxes = self.rpn(image, is_train=True)
                if self.config["mask_nms"]:
                    idx = [i for i in range(masks.shape[0]) if keep(i, masks)]
                    masks, boxes = masks[idx], boxes[idx]

                loss = 0.
                mean_loss = 0.
                loss_count = 0
                if self.config["loss"]["mask_loss"]:
                    seg_triplets = prepare_seg_triplets(masks, boxes, image)
                    for x_a, x_p, x_n in seg_triplets:
                        curr_loss = self._step(model, x_a, x_p, x_n)
                        loss += self.config['beta'] * curr_loss
                        mean_loss += self.config['beta'] * curr_loss
                        loss_count += 1

                if self.config["loss"]["object_loss"]:

                    if self.config['loss']['type'] == 'triplet':
                        obj_triplets = prepare_obj_triplets(masks, boxes, image, augment=self.config["augment"])
                        for x_a, x_p, x_n in obj_triplets:
                            curr_loss = self._step(model, x_a, x_p, x_n)
                            loss += curr_loss
                            mean_loss += curr_loss
                            loss_count += 1

                    elif self.config['loss']['type'] == 'nce' and masks.shape[0] > 1:
                       xis, xjs = prepare_object_pairs(masks, boxes, image)
                       loss += self._step_nce(model, xis, xjs, n_iter)
                       mean_loss += loss
                       loss_count += 1
                if loss_count > 0:
                    mean_loss /= loss_count
                    optimizer.zero_grad()
                    mean_loss.backward()
                    optimizer.step()

                if n_iter % self.config['log_loss_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', mean_loss, global_step=n_iter)
                    self.writer.add_scalar('num_triplets', loss_count, global_step=n_iter)
 
                if n_iter % self.config['log_every_n_steps'] == 0 and masks.shape[0] > 1:
                    h, w = masks[0].shape[0], masks[0].shape[1] 
                    labels = batch['labels'].cpu().numpy()
                    proposed_cls = bin_to_cls_mask(masks.cpu().numpy(), plot=True)
                    ground_cls = bin_to_cls_mask(labels, plot=True)
                    img = image.type(torch.int32).cpu().numpy().transpose(2,0,1) #[::-1,:,:]
                    self.writer.add_image('input_images',img, n_iter)
                    self.writer.add_text('filename', batch['file_name'], n_iter)
                    #writer.add_text('ground_truth_classes', ','.join([lvis_id_cat_map[k] for k in gt_cls]), n_iter)
                    self.writer.add_image('proposed_masks', make_grid(torch.tensor(proposed_cls.reshape(1,1,h,w))), n_iter)
                    self.writer.add_image('ground_truth', make_grid(torch.tensor(ground_cls.reshape(1,1,h,w))), n_iter)
                    # build embeddings (N, D)
                    """
                    bb = boxes.tensor.detach().cpu().numpy().astype(np.uint32)
                    embs = torch.stack([self._get_feature(model, masks[i].view(*masks[i].shape,1)*image, bb[i]) for i in range(len(masks)) if masks[i].sum()>1024])
                    image_labels = np.array([cv2.resize((masks[i].view(*masks[i].shape, 1) * image).cpu().numpy(), (64,64)) for i in range(len(masks)) if masks[i].sum()>1024]) 
                    image_labels = torch.tensor(image_labels)
                    self.writer.add_embedding(embs, label_img=image_labels.permute(0,3,1,2), global_step=n_iter)
                    """

                if n_iter % self.config['save_checkpoint_every_n_steps'] == 0 and n_iter > 0:
                    print('Saving model..')
                    #self.rpn.save(self.checkpoint_dir, n_iter)
                    torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, 'model_'+str(n_iter)+'.pth'))   
                n_iter +=1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()

            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_files = os.listdir(self.checkpoint_dir)
            saved_iters = [int(c.strip('.pth')[6:]) for c in checkpoints_files]
            loaded_iter = max(saved_iters)
            print(saved_iters)
            #loaded_iter = int(checkpoints_files[-1].strip('.pth')[6:])
            state_dict = torch.load(os.path.join(self.checkpoint_dir, checkpoints_files[-1]))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model at iteration {} with success.".format(loaded_iter))
        except FileNotFoundError:
            loaded_iter = 0
            print("Pre-trained weights not found. Training from scratch.")

        return model, loaded_iter

    def _get_feature(self, model, x, b):
        # x is already m * image
        x = x[b[1]:b[3], b[0]:b[2],:][b[1]:b[3], b[0]:b[2],:]
        h_, w_ = x.shape[0], x.shape[1]
        """
        if h_*w_ < 1024:
            x = cv2.resize(x.cpu().numpy(), (40,40))
            x = torch.tensor(x)
            print('resized to', x.shape)
        """
        r, _ = model(x.permute(2,0,1).view(1, 3, h_, w_))
        return r
