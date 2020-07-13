import os
from collections import defaultdict

import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.data_lvis_from_json import LVISDataFromJSON, my_collate_fn
from loss.triplet import TripletLoss, HTripletLoss, HierarchicalLoss
from models.hyperbolic_resnet import HResNetSimCLR
from models.resnet_simclr import ResNetSimCLR
from utils.simclr_utils import *
from utils.writer import SimCLRWriter

torch.manual_seed(0)


class SimCLR(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.train_loader = self._load_lvis_results()
        if self.config['loss']['type'] == 'nce':
            from loss.nt_xent import NTXentLoss
            self.loss_crit = NTXentLoss(self.device, config['batch_size'], **config['loss'])
        if self.config['loss']['include_hierarchical']:
            self.hierarchical_loss_crit = HierarchicalLoss(margin=config['loss']['margin'])
        if self.config['hyperbolic']:
            self.triplet_loss_crit = HTripletLoss(margin=config['loss']['margin'])
        else:
            self.triplet_loss_crit = TripletLoss(margin=config['loss']['margin'])

    def _load_lvis_results(self):
        dataset = LVISDataFromJSON(self.device, self.config)
        return DataLoader(dataset=dataset, shuffle=True, batch_size=self.config["batch_size"], collate_fn=my_collate_fn)

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, z_a, z_p, z_n, has_hierarchy=True, type=None):
        if type == 'mask':
            return {'mask_loss': self.triplet_loss_crit(z_a, z_p, z_n)}

        res = defaultdict(float)
        res['loss_count'] = 1
        res["triplet_loss"] = self.triplet_loss_crit(z_a, z_p, z_n)

        if self.config["loss"]["include_hierarchical"] and has_hierarchy:
            res["hierar_loss"] = self.hierarchical_loss_crit(z_a, z_p)
        return res

    # def _step(self, x_a, x_p, x_n, has_hierarchy=True, type=None):
    #     model = self.model
    #     # get the representations and the projections
    #     r_a, z_a = model(x_a.permute(2,0,1).view(1, 3, x_a.shape[0],x_a.shape[1]))  # [N,C]
    #     r_p, z_p = model(x_p.permute(2,0,1).view(1, 3, x_p.shape[0],x_p.shape[1]))  # [N,C]
    #     r_n, z_n = model(x_n.permute(2,0,1).view(1, 3, x_n.shape[0],x_n.shape[1]))
    #
    #     # normalize projection feature vectors
    #     if not self.config["hyperbolic"]:
    #         z_a = F.normalize(z_a, dim=0)
    #         z_p = F.normalize(z_p, dim=0)
    #         z_n = F.normalize(z_n, dim=0)
    #
    #     if type == 'mask':
    #         return {'mask_loss': self.triplet_loss_crit(z_a, z_p, z_n)}
    #
    #     res = defaultdict(float)
    #     res['loss_count'] = 1
    #     res["triplet_loss"] = self.triplet_loss_crit(z_a, z_p, z_n)
    #
    #     if self.config["loss"]["include_hierarchical"] and has_hierarchy:
    #         res["hierar_loss"] = self.hierarchical_loss_crit(z_a, z_p)
    #     return res

    def _init_model_and_optimizer(self):
        if self.config['hyperbolic']:
            model = HResNetSimCLR(**self.config["model"]).to(self.device)
        else:
            model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model, loaded_iter = self._load_pre_trained_weights(model)

        # if self.config['hyperbolic']:
        #     optimizer = geoopt.optim.RiemannianAdam(
        #         [p for p in self.model.parameters() if p.requires_grad],
        #         1e-4, weight_decay=eval(self.config['weight_decay']))
        # else:
        lr = float(self.config['lr'])
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr,
            weight_decay=eval(self.config['weight_decay']))

        self.model = model
        self.optimizer = optimizer
        return loaded_iter

    def train(self):
        self.writer = SimCLRWriter(self.config)
        train_loader = self.train_loader
        loaded_iter = self._init_model_and_optimizer()
        mask_size, augment = self.config['mask_size'], self.config['augment']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=60000, eta_min=0, last_epoch=-1)

        n_iter = loaded_iter + 1
        for epoch_counter in range(self.config['epochs']):
            for _, batch in enumerate(train_loader):
                image, masks, boxes = batch['image'], batch['masks'], batch['boxes']
                image_url = batch['image_url']
                assert (image[0].shape[2] == 3)  # the image is in BGR format

                # if self.config["mask_nms"]:  # not necessary for the json version
                #     idx = [i for i in range(masks.shape[0]) if keep(i, masks)]
                #     masks, boxes = masks[idx], boxes[idx]

                loss_dict = {
                    'mask_loss': 0.,
                    'triplet_loss': 0.,
                    'hierar_loss': 0.,
                    'loss_count': 0
                }
                mask_tensors = self._get_mask_tensors(image, masks, boxes)  # N, H, W, 3
                features = self._get_mask_features(mask_tensors)
                if self.config["loss"]["mask_loss"]:
                    seg_triplets = prepare_seg_triplets_batched(masks, boxes, image, mask_size)
                    for x_a, x_p, x_n in seg_triplets:
                        res = self._step(x_a, x_p, x_n, type='mask')
                        loss_dict = {k: v + res[k] for k, v in loss_dict.items()}

                if self.config["loss"]["object_loss"]:
                    if self.config['loss']['type'] == 'triplet':
                        obj_triplets = prepare_obj_triplets_batched(masks, boxes, image, augment)
                        for b, i_a, i_p, i_n, is_hierar in obj_triplets:
                            if i_a == i_p:
                                pos_features = self.model(
                                    mask_tensors[b][i_a].permute(2, 0, 1).view(1, 3, mask_size, mask_size))[1]
                            else:
                                pos_features = features[b][i_p]
                            res = self._step(
                                features[b][i_a],
                                pos_features,
                                features[i_n[0]][i_n[1]],
                                has_hierarchy=is_hierar)
                            loss_dict = {k: v + res[k] for k, v in loss_dict.items()}

                    # elif self.config['loss']['type'] == 'nce' and masks.shape[0] > 1:
                    #    xis, xjs = prepare_object_pairs(masks, boxes, image)
                    #    loss += self._step_nce(xis, xjs, n_iter)
                    #    mean_loss += loss
                    #    loss_count += 1
                # print(loss_dict)
                if loss_dict['loss_count'] > 0:
                    total_loss = loss_dict['triplet_loss'] + loss_dict['hierar_loss'] + \
                                 self.config['beta'] * loss_dict['mask_loss']
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                if n_iter % self.config['log_loss_every_n_steps'] == 1:
                    self.writer.log_loss(loss_dict, n_iter)
                if n_iter % self.config['log_every_n_steps'] == 1 and masks[0].shape[0] > 1:
                    self.writer.visualize(image[0], image_url[0], masks[0], n_iter)
                if n_iter % self.config['save_checkpoint_every_n_steps'] == 0 and n_iter > 0:
                    print('Saving model..')
                    torch.save(self.model.state_dict(), os.path.join(self.writer.checkpoint_dir, 'model_'+str(n_iter)+'.pth'))

                n_iter +=1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()

            self.writer.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    def _get_mask_tensors(self, image, masks, boxes):
        mask_size = self.config['mask_size']
        mask_tensors = [
            torch.stack([
                resize_tensor(apply_mask(image[b], masks[b][i], boxes[b][i]), mask_size)
                for i in range(masks[b].shape[0])
            ])
            for b in range(len(image))
        ]
        return mask_tensors

    def _get_mask_features(self, mask_tensors):
        features = []
        #import ipdb as pdb; pdb.set_trace()
        for b in range(len(mask_tensors)):
            r, z = self.model(mask_tensors[b].permute(0, 3, 1, 2))
            # normalize projection feature vectors
            if not self.config["hyperbolic"]:
                z = F.normalize(z, dim=0)
            features.append(z)
        return features

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_files = os.listdir(self.writer.checkpoint_dir)
            saved_iters = [int(c.strip('.pth')[6:]) for c in checkpoints_files]
            loaded_iter = max(saved_iters) if len(saved_iters) > 0 else 0
            print('Found saved checkpoints at iter:', saved_iters)
            state_dict = torch.load(os.path.join(self.writer.checkpoint_dir, 'model_{}.pth'.format(loaded_iter)))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model at iteration {} with success.".format(loaded_iter))
        except FileNotFoundError:
            loaded_iter = 0
            print("Pre-trained weights not found. Training from scratch.")
        return model, loaded_iter

        # xis = [N, H, W, 3]
    # def _step_nce(self, xis, xjs, n_iter):
    #     #assert (xis.max() <= 1.)
    #     model = self.model
    #     # get the representations and the projections
    #     ris, zis = model(xis)  # [N,C]
    #     # get the representations and the projections
    #     rjs, zjs = model(xjs)  # [N,C]
    #     try:  # normalize projection feature vectors
    #         zis = F.normalize(zis, dim=1)
    #         zjs = F.normalize(zjs, dim=1)
    #     except:
    #         print(zis.shape, zjs.shape)
    #     loss = self.loss_crit(zis, zjs)
    #     return loss
