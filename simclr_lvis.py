import json

import geoopt
import skimage.io as io
import torch.nn.functional as F
from lvis import LVIS, LVISResults
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize
from torchvision.utils import make_grid

from loss.nt_xent import NTXentLoss
from loss.triplet import TripletLoss, HTripletLoss
from models.hyperbolic_resnet import HResNetSimCLR
from models.resnet_simclr import ResNetSimCLR
from simclr_utils import *

apex_support = False
torch.manual_seed(0)


class SimCLR(object):
    def __init__(self, dataset, rpn, config):
        self.config = config
        self.rpn = rpn
        self.device = self._get_device()
        # self.dataset = dataset
        self.dataset = self._load_lvis_results()

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

    def _load_lvis_results(self):
        # for each image

        class DummyLoader(object):
            def __init__(self, dt_path=r'output/inference'):
                self.lvis_gt = LVIS('/scratch/users/zzweng/datasets/lvis/lvis_v0.5_val.json')
                self.dt_path = os.path.join(dt_path, 'lvis_instances_results.json')
                self.dt = LVISResults(self.lvis_gt, self.dt_path)
                self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            def __getitem__(self, i):
                img_id = self.dt.get_img_ids()[i]
                image_url = self.dt.load_imgs([img_id])[0]['coco_url']
                image = io.imread(image_url)
                if len(image.shape) == 2:
                    return None  # skip grayscale images
                image = image / 255.
                image = self.normalize(torch.tensor(image).permute(2, 0, 1)).permute(1, 2, 0)

                ann_ids = self.dt.get_ann_ids(img_ids=[img_id])
                masks = np.stack([
                    self.dt.ann_to_mask(self.dt.load_anns(ids=[a_i])[0])
                    for a_i in ann_ids
                ])
                boxes = np.stack([
                    self.dt.load_anns(ids=[a_i])[0]['bbox']
                    for a_i in ann_ids
                ]).astype(np.uint32)
                boxes[:, 2] += boxes[:, 0]
                boxes[:, 3] += boxes[:, 1]

                return image, masks, boxes

            def __len__(self):
                return len(self.dt.get_img_ids())
        return DummyLoader()

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

        try:  # normalize projection feature vectors
            zis = F.normalize(zis, dim=1)
            zjs = F.normalize(zjs, dim=1)
        except:
            print(zis.shape, zjs.shape)
        loss = self.loss_crit(zis, zjs)
        return loss

    def _get_optimizer(self):
        #segmentation_params = self.rpn.predictor.model.roi_heads.parameters()
        #optimizer = torch.optim.Adam(list(segmentation_params)+list(model.parameters()), 3e-4, weight_decay=eval(self.config['weight_decay']))

        if self.config['hyperbolic'] and False: # TODO: we don't need this
            optimizer = geoopt.optim.RiemannianAdam(
                [p for p in self.model.parameters() if p.requires_grad],
                1e-4, weight_decay=eval(self.config['weight_decay']))
        else:
            optimizer = torch.optim.Adam(
                [p for p in self.model.parameters() if p.requires_grad], 1e-4,
                weight_decay=eval(self.config['weight_decay']))
        return optimizer

    def train(self):
        # train_loader, valid_loader = self.dataset.get_data_loaders()
        train_loader = self.dataset

        if self.config['hyperbolic']:
            model = HResNetSimCLR(**self.config["model"]).to(self.device)
        else:
            model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model, loaded_iter = self._load_pre_trained_weights(model)
        self.model = model
        optimizer = self._get_optimizer()

        # print('Freezing rpn weights')
        # for p in self.rpn.predictor.model.parameters():
        #     p.requires_grad = False
        
        # num_train = len(train_loader.dataset.dataset)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60000, eta_min=0,
                                                               last_epoch=-1)
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        save_config_file(model_checkpoints_folder)

        n_iter = loaded_iter + 1
        for epoch_counter in range(self.config['epochs']):
            for _, batch in enumerate(train_loader):
                # batch = batch[0]
                image = batch[0].to(self.device)
                assert (image.shape[2] == 3)  # the image is in BGR format
                masks, boxes = batch[1], batch[2]

                # masks, boxes = self.rpn(image, is_train=True)
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
            state_dict = torch.load(os.path.join(self.checkpoint_dir, checkpoints_files[-1]))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model at iteration {} with success.".format(loaded_iter))
        except FileNotFoundError:
            loaded_iter = 0
            print("Pre-trained weights not found. Training from scratch.")
        return model, loaded_iter
