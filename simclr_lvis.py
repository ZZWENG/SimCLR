import torch, torchvision
from models.resnet_simclr import ResNetSimCLR
from models.hyperbolic_resnet import HResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
from loss.triplet import TripletLoss, HTripletLoss
import os, sys, cv2
import shutil
import numpy as np
from skimage.transform import rotate
from torchvision.utils import make_grid
from evaluate import get_evaluator
import time
import torchvision.transforms as T
import skimage
from torchvision.transforms import Normalize
import geoopt
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
        mask = mask /(labels.shape[0]+1)*255  # convert to greyscale
    return mask.astype(np.uint8)


def prepare_seg_triplets(masks, boxes, image):
    n = masks.shape[0]
    boxes = boxes.tensor.detach().cpu().numpy().astype(np.uint32)
    for i in range(n):
        m, b = masks[i], boxes[i]
        if m.sum() < 400:  continue  # skip tiny masks for now
        m = m.view(*m.shape, 1)
        full = image[b[1]:b[3], b[0]:b[2],:]
        foreground = (m * image)[b[1]:b[3], b[0]:b[2],:]
        background = ((~m) * image)[b[1]:b[3], b[0]:b[2],:]
        yield full, foreground, background


def overlaps(m1, m2, thres):
    return (m1 * m2).sum() > thres


def overlapping_idx(anchor, masks, thres): 
    flags = np.array([overlaps(anchor, m, thres) for m in masks])
    return np.where(flags == False)[0], np.where(flags == True)[0]


def size_of(cut):
    return cut.shape[0] * cut.shape[1]

# post processing
"""
def keep(i, masks):
    # retuns true if masks[i] overlaps with some other masks by more than x% of itself
    masks_out = []
    for j in range(len(masks)):
        if j == i: continue
        area = masks[i].sum().item() * 1.
        if area < 400:
            return False  # take out small objects
        if (masks[j] * masks[i]).sum() / area > 0.6 and area < masks[j].sum():
#             print((masks[j] * masks[i]).sum() / masks[i].sum())
            return False
    return True
"""
def keep(i, masks):
    # retuns true if masks[i] overlaps with some other masks by more than x% of itself
    masks_out = []
    for j in range(i):
        if j == i: continue
        area = masks[i].sum().item() * 1.
#         if area < 400: return False
        if (masks[j] * masks[i]).sum() / area > 0.7: # and area < masks[j].sum():
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
    result_aug = torch.tensor(np.stack(result_aug)).type(torch.float).to(masks.device)
    result, result_aug = result.permute(0,3,1,2), result_aug.permute(0,3,1,2)

    result = torch.stack([normalize(result[i]) for i in range(result.shape[0])])
    result_aug = torch.stack([normalize(result_aug[i]) for i in range(result_aug.shape[0])])
    return result, result_aug


def prepare_obj_triplets(masks, boxes, image):
    n = masks.shape[0]
    boxes = boxes.tensor.detach().cpu().numpy().astype(np.uint32)
    for i in range(n):
        m1, b = masks[i], boxes[i]
        cut_a = (m1.view(*m1.shape, 1) * image)[b[1]:b[3], b[0]:b[2],:]
        if cut_a.shape[0] * cut_a.shape[1] < 10:
            continue
        neg_idx, pos_idx = overlapping_idx(m1, masks, 50)
        if len(neg_idx) == 0:
            continue
        if len(pos_idx) == 0:
            # apply augmentation
            cut_p = torch.tensor(rotate(cut_a.cpu().numpy(), angle=25, mode = 'wrap')).type(torch.float).to(m1.device)
        else:
            i_p = np.random.choice(pos_idx, 1)[0]
            m2, b2 = masks[i_p], boxes[i_p]
            cut_p = (m2.view(*m2.shape, 1) * image)[b2[1]:b2[3], b2[0]:b2[2],:]
        
        i_n = np.random.choice(neg_idx, 1)[0]
        m3, b_n = masks[i_n], boxes[i_n]
        cut_n = (m3.view(*m3.shape, 1) * image)[b_n[1]:b_n[3], b_n[0]:b_n[2],:]
        if size_of(cut_p) < 10 or size_of(cut_n) < 10 or size_of(cut_a) < 10:
            continue
        
        side_len = 128
        cut_a = cv2.resize(cut_a.cpu().numpy(), (side_len,side_len))
        cut_p = cv2.resize(cut_p.cpu().numpy(), (side_len,side_len))
        cut_n = cv2.resize(cut_n.cpu().numpy(), (side_len,side_len))

        cut_a = torch.tensor(cut_a).type(torch.float).to('cuda')
        cut_p = torch.tensor(cut_p).type(torch.float).to('cuda')
        cut_n = torch.tensor(cut_n).type(torch.float).to('cuda')
        """
        # simclr model takes float type
        cut_a = cut_a.type(torch.float)
        cut_n = cut_n.type(torch.float)
        """
        yield cut_a, cut_p, cut_n


class SimCLR(object):
    def __init__(self, dataset, rpn, config):
        self.config = config
        self.rpn = rpn
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        if self.config['loss']['type'] == 'nce':
           self.loss_crit = NTXentLoss(self.device, config['batch_size'], **config['loss'])
        if self.config['hyperbolic']:
           self.triplet_loss_crit = HTripletLoss()
        else:
           self.triplet_loss_crit = TripletLoss()
        checkpoint_dir = '/scratch/users/zzweng/runs/checkpoints/{}_hyp={}_zdim={}_loss={}'.format(
             self.config['desc'], self.config['hyperbolic'], self.config['model']['out_dim'], self.config['loss']['type']
        )
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.evaluator = get_evaluator(rpn.cfg, 'coco')
        self.evaluator.reset()

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, x_a, x_p, x_n):
        #h, w = x_a.shape[0], x_a.shape[1]
        #print(h, w)

        # get the representations and the projections
        r_a, z_a = model(x_a.permute(2,0,1).view(1, 3,x_a.shape[0],x_a.shape[1]))  # [N,C]
        r_p, z_p = model(x_p.permute(2,0,1).view(1, 3,x_p.shape[0],x_p.shape[1]))  # [N,C]
        r_n, z_n = model(x_n.permute(2,0,1).view(1, 3,x_n.shape[0],x_n.shape[1]))

        # normalize projection feature vectors
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

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loaders()
        if self.config['hyperbolic']:
            model = HResNetSimCLR(**self.config["model"]).to(self.device)
        else:
            model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)
        
        print('Freezing rpn weights')
        for p in self.rpn.predictor.model.parameters():
            p.requires_grad = False 
        #segmentation_params = self.rpn.predictor.model.roi_heads.parameters()
        #optimizer = torch.optim.Adam(list(segmentation_params)+list(model.parameters()), 3e-4, weight_decay=eval(self.config['weight_decay']))
        optimizer = geoopt.optim.RiemannianAdam([p for p in model.parameters() if p.requires_grad], 1e-4, weight_decay=eval(self.config['weight_decay']))
        #optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=eval(self.config['weight_decay']))
        
        num_train = len(train_loader.dataset.dataset)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_train, eta_min=0,
                                                               last_epoch=-1)
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        for epoch_counter in range(self.config['epochs']):
            for _, batch in enumerate(train_loader):
                batch = batch[0]
                #if 698 not in batch['instances'].gt_classes:
                #    continue 
                
                image = batch['image'].to(self.device)
                assert (image.shape[2] == 3)
                # the image is in BGR format
                
                masks, boxes = self.rpn(image, is_train=True)
                idx = [i for i in range(masks.shape[0]) if keep(i, masks)]
                masks, boxes = masks[idx], boxes[idx]
                #seg_triplets = prepare_seg_triplets(masks, boxes, image)
                #obj_triplets = prepare_obj_triplets(masks, boxes, image)
 
                loss = 0.
                mean_loss = 0.
                loss_count = 0
                """                
                seg_triplets = prepare_seg_triplets(masks, boxes, image)
                for x_a, x_p, x_n in seg_triplets:
                    curr_loss = self._step(model, x_a, x_p, x_n)
                    loss += curr_loss
                    mean_loss += curr_loss
                    loss_count += 1
                """
                if self.config['loss']['type'] == 'triplet':
                    obj_triplets = prepare_obj_triplets(masks, boxes, image)
                    for x_a, x_p, x_n in obj_triplets:
                        curr_loss = self._step(model, x_a, x_p, x_n)
                        loss += self.config['contrastive_loss_weight'] * curr_loss
                        mean_loss += self.config['contrastive_loss_weight'] * curr_loss
                        loss_count += 1
                elif self.config['loss']['type'] == 'nce' and masks.shape[0] > 1:
                   xis, xjs = prepare_object_pairs(masks, boxes, image)
                   loss += 5 * self._step_nce(model, xis, xjs, n_iter)
                   mean_loss += self.config['contrastive_loss_weight']*loss
                   loss_count += 1

                if loss_count > 0:
                    mean_loss /= loss_count
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if n_iter % self.config['log_loss_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', mean_loss, global_step=n_iter)
 
                if n_iter % self.config['log_every_n_steps'] == 0 and masks.shape[0] > 1:
                    print(mean_loss)
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
                # do test using the detectron2 script. This is too slow.
                #do_test(self.evaluator, self.rpn.cfg, self.rpn.predictor.model)
                #torch.save(model.state_dict(), 'model.pth')
                if (n_iter) % self.config['save_checkpoint_every_n_steps'] == 0:
                    self.rpn.save(self.checkpoint_dir, n_iter)
                if (n_iter % self.config['save_checkpoint_every_n_steps']) == 0:
                    print('Saving simclr model..')
                    torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, 'model_'+str(n_iter)+'.pth'))   
                n_iter +=1
            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            #do_test(self.rpn.cfg, self.rpn.predictor.model)
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model):
        try:
            state_dict = torch.load(os.path.join(self.checkpoint_dir, 'model_60000.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

"""
def do_test(evaluator, cfg, model):
    from detectron2.data.build import build_detection_test_loader
    from detectron2.evaluation import inference_on_dataset

    dataset_name = cfg.DATASETS.TEST[0] 
    data_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(model, data_loader, evaluator)
    return results
"""
