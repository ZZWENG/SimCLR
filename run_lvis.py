from simclr_lvis import SimCLR
import yaml
from data_aug.data_lvis import DataSetWrapper
from models.rpn import ProposalNetwork


def main():
    config = yaml.load(open("config_lvis.yaml", "r"), Loader=yaml.FullLoader)

    rpn = ProposalNetwork()
    cfg = rpn.cfg

    dataset = DataSetWrapper(config['batch_size'], **config['dataset'], cfg=cfg)

    simclr = SimCLR(dataset, rpn, config)
    simclr.train()


if __name__ == "__main__":
    main()
