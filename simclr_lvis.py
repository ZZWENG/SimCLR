import torch
from models.resnet_simclr import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
# from loss.nt_xent import NTXentLoss
from loss.triplet import TripletLoss
import os
import shutil
import sys

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


def prepare_seg_triplets(masks, boxes):
    return []


def prepare_obj_triplets(masks, boxes, image):
    n = 10
    # TODO: dummy code
    anchors = torch.randint(high=masks.shape[0], size=(n,))
    neg = torch.randint(high=masks.shape[0], size=(n,))
    pos = torch.randint(high=masks.shape[0], size=(n,))
    for i in range(n):
        m1,m2,m3 = masks[anchors[i]], masks[pos[i]], masks[neg[i]]
        yield m1 * image, m2 * image, m3 * image


class SimCLR(object):
    def __init__(self, dataset, rpn, config):
        self.config = config
        self.rpn = rpn
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        # self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])
        self.triplet_criterion = TripletLoss(self.device)

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    # def _step(self, model, xis, xjs, n_iter):
    #
    #     # get the representations and the projections
    #     ris, zis = model(xis)  # [N,C]
    #
    #     # get the representations and the projections
    #     rjs, zjs = model(xjs)  # [N,C]
    #
    #     # normalize projection feature vectors
    #     zis = F.normalize(zis, dim=1)
    #     zjs = F.normalize(zjs, dim=1)
    #
    #     loss = self.nt_xent_criterion(zis, zjs)
    #     return loss

    def _step(self, model, x_a, x_p, x_n):

        # get the representations and the projections
        r_a, z_a = model(x_a)  # [N,C]
        r_p, z_p = model(x_p)  # [N,C]
        r_n, z_n = model(x_n)

        # normalize projection feature vectors
        z_a = F.normalize(z_a, dim=1)
        z_p = F.normalize(z_p, dim=1)
        z_n = F.normalize(z_n, dim=1)

        loss = self.triplet_criterion(z_a, z_p, z_n)
        # loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):
        import pdb
        pdb.set_trace()
        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        for epoch_counter in range(self.config['epochs']):
            for batch, _ in train_loader:
                image = batch['image'].to(self.device)
                assert (image.shape[2] == 3)
                masks, boxes = self.rpn(image)

                optimizer.zero_grad()

                # seg_triplets = prepare_seg_triplets(masks, boxes)
                obj_triplets = prepare_obj_triplets(masks, boxes, image)

                mean_loss = 0.
                trip_count = 0
                for x_a, x_p, x_n in obj_triplets:
                    x_a = x_a.to(self.device)
                    x_p = x_p.to(self.device)
                    x_n = x_n.to(self.device)
                    loss = self._step(model, x_a, x_p, x_n)
                    mean_loss += loss
                    trip_count += 1

                    if apex_support and self.config['fp16_precision']:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()

                mean_loss /= trip_count
                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', mean_loss, global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs), _ in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
