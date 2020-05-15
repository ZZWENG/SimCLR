from simclr_lvis import SimCLR
import yaml
from data_aug.data_lvis import DataSetWrapper
#from models.rpn import ProposalNetwork
import torch 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

def main():
    config = yaml.load(open("config_lvis.yaml", "r"), Loader=yaml.FullLoader)

    from models.rpn import ProposalNetwork
    rpn = ProposalNetwork(device)
    cfg = rpn.cfg

    dataset = DataSetWrapper(config['batch_size'], cfg=cfg, **config['dataset'])
    print('Start Training...')
    simclr = SimCLR(dataset, rpn, config)
    simclr.train()


if __name__ == "__main__":
    main()
