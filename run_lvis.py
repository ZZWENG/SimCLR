import torch
import yaml

from simclr_lvis import SimCLR

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

def main():
    config = yaml.load(open("config_lvis.yaml", "r"), Loader=yaml.FullLoader)

    # from models.rpn import ProposalNetwork
    # rpn = ProposalNetwork(device)
    # cfg = rpn.cfg
    #
    # dataset = DataSetWrapper(config['batch_size'], cfg=cfg, **config['dataset'])
    print('Start Training...')
    simclr = SimCLR(config)
    simclr.train()


if __name__ == "__main__":
    main()
